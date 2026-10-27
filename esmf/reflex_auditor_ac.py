"""G0-AC: budgeted reflex/auditor learning from an unrevealed Mat2(R) state.

The learner lives in raw sensor coordinates. It receives neither injected
sections nor physical chart/transport adapters. Its diagonal-RLS recurrent
ensemble is an ordinary real learner, not an exact geometric reconstruction.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import time

from .associated22_u import flatten
from .geometry_k import inverse, matmul, matvec


REGISTRATION = Path(__file__).resolve().parents[1] / "research/G0_AC_PREREGISTRATION.json"
with REGISTRATION.open(encoding="utf-8") as source:
    SPEC = json.load(source)
WORLDS = tuple(SPEC["worlds"])
POLICIES = tuple(SPEC["policies"])
COST = SPEC["costs"]
GATE = SPEC["gate"]
MODEL = SPEC["learner"]
HORIZON = SPEC["horizon"]
BURN_IN = SPEC["burn_in"]
CIRCUITS = ("quiet", "focus", "audit")
COMMANDS = ("forward", "inverse", "rest")


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def command_at(tick):
    return COMMANDS[(tick // 4) % 3]


def sl2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s), (s, c))


class MatrixWorld:
    """Only the experiment scorer may inspect state, charts and reset labels."""

    def __init__(self, seed, world):
        if world not in WORLDS or seed < 0:
            raise ValueError("invalid world or seed")
        self.seed, self.world = seed, world
        rng = random.Random(seed*19381 + 337)
        self.state = tuple(tuple(rng.gauss(0, .7) for _ in range(2))
                           for _ in range(2))
        self.charts = (
            ((1., .2), (-.1, .9)),
            matmul(((1.1, .3), (0., .8)), sl2(rng.uniform(-.6, .6))),
            matmul(((.9, 0.), (.2, 1.2)), sl2(rng.uniform(-.6, .6))))
        self.tick = -1

    def transports(self, command, tick):
        if command not in COMMANDS:
            raise ValueError("unknown intervention")
        drift = .001*tick if self.world == "drift" else 0.
        # Nonorthogonal SL2 factors, not another one-dimensional shear orbit
        # or an SO2-only rotation model.
        left = matmul(sl2(.065 + drift),
                      ((math.exp(.015), .025), (0., math.exp(-.015))))
        right = matmul(((math.exp(-.012), 0.), (.02, math.exp(.012))),
                       sl2(-.045 + drift/2))
        # Alternate noncommuting transport families between complete
        # forward/inverse/rest cycles, rather than a single repeated operator.
        if (tick // 12) % 2:
            left = tuple(zip(*left))
        if (tick // 12) % 3:
            right = tuple(zip(*right))
        if command == "inverse":
            left, right = inverse(left), inverse(right)
        if command == "rest":
            left = right = ((1., 0.), (0., 1.))
        return left, right

    def step(self, command):
        self.tick += 1
        left, right = self.transports(command, self.tick)
        self.state = matmul(matmul(left, self.state), right)
        reset = (self.world == "independent" or
                 self.world == "reset" and self.tick in (16, 32))
        if reset:
            rng = random.Random(self.seed*17749 + self.tick*193 + 17)
            self.state = tuple(tuple(rng.gauss(0, .7) for _ in range(2))
                               for _ in range(2))
        if self.world == "dissipation":
            self.state = tuple(tuple(.97*x for x in row) for row in self.state)
        return reset

    def readings(self, noisy=True):
        x = flatten(self.state)
        result = {}
        for sensor, chart in enumerate(self.charts):
            local = matvec(chart, x[sensor:sensor+2])
            rng = random.Random(self.seed*104729 + self.tick*65537 + sensor*997)
            result[sensor] = tuple(v + (rng.gauss(0, MODEL["sensor_sd"])
                                       if noisy else 0.) for v in local)
        return result


@dataclass
class Energy:
    cap: float
    spent: float = 0.
    charges: dict = field(default_factory=dict)
    held: float = 0.

    def __post_init__(self):
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError("positive finite energy cap required")

    @property
    def remaining(self):
        return max(0., self.cap - self.spent - self.held)

    def can_pay(self, amount):
        return math.isfinite(amount) and 0 <= amount <= self.remaining + 1e-12

    def pay(self, name, count=1):
        amount = COST[name]*count
        if not self.can_pay(amount):
            return False
        self.spent += amount
        self.charges[name] = self.charges.get(name, 0.) + amount
        if not 0 <= self.spent <= self.cap + 1e-10:
            raise ArithmeticError("metabolic budget violation")
        return True


class DiagonalRLS:
    """Bounded-capacity diagonal covariance approximation, including its cost."""

    def __init__(self, inputs=10, outputs=6, forgetting=.95):
        self.weights = [[0.]*inputs for _ in range(outputs)]
        self.cov = [[MODEL["initial_covariance"]]*inputs for _ in range(outputs)]
        self.variance = [MODEL["initial_variance"]]*outputs
        self.forgetting = forgetting
        self.operations = 0

    def predict(self, features):
        self.operations += 2*len(features)*len(self.weights)
        return [dot(row, features) for row in self.weights]

    def update(self, features, observed):
        prediction = self.predict(features)
        for output, value in observed.items():
            residual = value - prediction[output]
            row, cov = self.weights[output], self.cov[output]
            denom = 1. + sum(p*x*x for p, x in zip(cov, features))
            for j, x in enumerate(features):
                gain = cov[j]*x/denom
                row[j] += gain*residual
                cov[j] = min(20., max(1e-6,
                    (cov[j] - gain*x*cov[j])/self.forgetting))
            self.variance[output] = max(MODEL["variance_floor"],
                .95*self.variance[output] + .05*residual*residual)
            self.operations += 12*len(features) + 8
        if not all(math.isfinite(x) for row in self.weights for x in row):
            raise ArithmeticError("nonfinite recurrent learner")


class RecurrentEnsemble:
    def __init__(self, mixing=True):
        self.models = [DiagonalRLS(forgetting=f) for f in MODEL["forgetting"]]
        self.weights = [1/3]*3
        self.last = [0.]*6
        self.mixing = mixing
        self.hazard = .02
        self.slow_updates = 0
        self.child = None
        self.events = []

    def forecast(self, command):
        features = self.last + [float(command == c) for c in COMMANDS] + [1.]
        predictions = [m.predict(features) for m in self.models]
        mean = [sum(w*p[j] for w, p in zip(self.weights, predictions))
                for j in range(6)]
        variance = sum(w*(m.variance[0] + (p[0]-mean[0])**2)
                       for w, m, p in zip(self.weights, self.models, predictions))
        return features, predictions, mean, max(MODEL["variance_floor"], variance)

    def observe(self, tick, features, predictions, observed):
        values = {2*s+j: value for s, pair in observed.items()
                  for j, value in enumerate(pair)}
        logs = []
        for model, prediction in zip(self.models, predictions):
            logs.append(-.5*sum(math.log(2*math.pi*model.variance[j]) +
                         (value-prediction[j])**2/model.variance[j]
                         for j, value in values.items()))
        if self.child is not None:
            self.child["log_bayes"] += logs[2] - logs[0]
        if self.mixing and values:
            unnorm = [math.log(max(1e-12, (1-self.hazard)*w+self.hazard/3)) + ll
                      for w, ll in zip(self.weights, logs)]
            peak = max(unnorm)
            self.weights = [math.exp(v-peak) for v in unnorm]
            total = sum(self.weights)
            self.weights = [w/total for w in self.weights]
        for model in self.models:
            model.update(features, values)
        mean = [sum(w*p[j] for w, p in zip(self.weights, predictions))
                for j in range(6)]
        self.last = [values.get(j, mean[j]) for j in range(6)]

    def slow_update(self, failure_rate):
        self.hazard = min(.3, max(.005, .8*self.hazard + .2*failure_rate))
        self.slow_updates += 1

    def bud(self, tick, evidence, energy):
        if self.child is not None or not energy.pay("bud"):
            return False
        self.child = {"tick": tick, "log_bayes": 0.,
                      "saved": deepcopy(self.models[2])}
        energy.held = COST["prune"]
        self.models[2] = DiagonalRLS(forgetting=MODEL["forgetting"][2])
        self.weights = [.45, .45, .10]
        self.events.append({"event": "bud", "tick": tick, "parent": "root",
                            "child": f"temporary-{tick}", "evidence": evidence,
                            "typed_remainder": "[R]_id"})
        return True

    def prune(self, tick, energy, force=False):
        if self.child is None:
            return False
        contradicted = self.child["log_bayes"] < GATE["prune_log_bayes"]
        expired = tick-self.child["tick"] >= GATE["bud_lifetime"]
        if not (contradicted or expired or force):
            return False
        energy.held = 0.
        if not energy.pay("prune"):
            raise ArithmeticError("lost reserved pruning energy")
        old_tick = self.child["tick"]
        self.models[2] = self.child["saved"]
        self.weights = [.45, .45, .10]
        self.child = None
        self.events.append({"event": "prune", "tick": tick,
                            "child": f"temporary-{old_tick}",
                            "reason": "contradicted" if contradicted else (
                                "expired" if expired else "episode_end"),
                            "typed_remainder": "[R]_id"})
        return True


class SensorAuditor:
    """Lossy directed maps, never a proof of identity or global orientability."""

    def __init__(self):
        self.maps = {(a, b): DiagonalRLS(3, 2) for a in range(3)
                     for b in range(3) if a != b}
        self.records = []
        self.audits = 0

    def audit(self, tick, readings):
        if set(readings) != {0, 1, 2}:
            raise ValueError("audit requires three independent partial sensors")
        start = list(readings[0])
        returned = start
        for a, b in ((0, 1), (1, 2), (2, 0)):
            returned = self.maps[a, b].predict(returned + [1.])
        residual = [x-y for x, y in zip(start, returned)]
        record = {"tick": tick, "path": [0, 1, 2, 0],
                  "source_reading": start, "returned_reading": returned,
                  "prior_audits": self.audits,
                  "typed_remainder": "[R]_transport", "residual": residual,
                  "positive_square": dot(residual, residual),
                  "identifies_full_fiber": False}
        self.records.append(record)
        self.audits += 1
        for (a, b), model in self.maps.items():
            model.update(list(readings[a])+[1.], dict(enumerate(readings[b])))
        return record


class ReflexAuditorGate:
    def __init__(self):
        self.failures = 1.
        self.successes = 3.
        self.audit_mode = False
        self.switched_at = -GATE["minimum_dwell"]
        self.last_value = -math.inf

    @property
    def failure_rate(self):
        return self.failures/(self.failures+self.successes)

    def evidence(self, standardized_residual):
        if standardized_residual > GATE["failure_residual"]:
            self.failures += 1
        else:
            self.successes += 1

    def choose(self, tick, energy):
        total = self.failures+self.successes
        mean = self.failure_rate
        sd = math.sqrt(mean*(1-mean)/(total+1))
        credible = max(0., mean-GATE["credible_z"]*sd)
        self.last_value = (GATE["consequence"]*credible -
                           (COST["audit"]-COST["focus"]))
        if tick-self.switched_at >= GATE["minimum_dwell"]:
            old = self.audit_mode
            if self.last_value > GATE["on_value"]:
                self.audit_mode = True
            elif self.last_value < GATE["off_value"]:
                self.audit_mode = False
            if old != self.audit_mode:
                self.switched_at = tick
        preferred = "audit" if self.audit_mode else "focus"
        # Reserve the current update and one future cheap prediction.
        for name in (preferred, "focus", "quiet"):
            if energy.can_pay(COST[name]+COST["update"]+COST["predict"]):
                return name
        return None


def probability(mean, variance, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite calibration temperature required")
    p = .5*(1+math.erf(mean/math.sqrt(2*variance*temperature)))
    return min(1-1e-12, max(1e-12, p))


def simulate(seed, world, policy, cap=20., temperature=1., trace=False):
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    energy = Energy(cap)
    environment = MatrixWorld(seed, world)
    learner = RecurrentEnsemble(mixing=policy != "real_recurrent")
    auditor, gate = SensorAuditor(), ReflexAuditorGate()
    rng = random.Random(seed*1543 + 71)
    effective = "adaptive" if policy == "equivalent_real" else policy
    clock = time.perf_counter()
    cpu = time.process_time()
    scores, history, resets = [], [], []
    cumulative_operations = 0
    # Prepay the common intervention schedule so sensing choices cannot change
    # the paired physical task when a policy exhausts its sensory budget.
    interventions_funded = energy.pay("intervention", HORIZON)
    prediction = ([0.]*10, [[0.]*6 for _ in range(3)], [0.]*6, 1.)
    for tick in range(HORIZON):
        command = command_at(tick)
        # Intervention schedule is shared across policies, not privileged feedback.
        applied = command if interventions_funded else "rest"
        reset = environment.step(applied)
        if reset:
            resets.append(tick)
        if energy.pay("predict"):
            prediction = learner.forecast(applied)
        else:
            # Exhaustion is abstention, NOT removal from the scored population.
            prediction = ([0.]*10, [[0.]*6 for _ in range(3)], [0.]*6, 1.)
        features, predictions, mean, variance = prediction
        p = probability(mean[0], variance, temperature)
        truth = environment.readings(noisy=False)[0][0]
        target = int(truth >= 0)
        covered = max(p, 1-p) >= MODEL["confidence"]
        if tick >= BURN_IN:
            scores.append({"brier": (p-target)**2,
                           "logloss": -math.log(p if target else 1-p),
                           "mse": (mean[0]-truth)**2,
                           "covered": covered,
                           "correct": (p >= .5) == bool(target),
                           # A local predictive-continuity declaration, NOT
                           # a topological orientation/gluing certificate.
                           "false_gluing": covered and reset})
        if effective in ("fixed_focus", "real_recurrent", "real_imm"):
            circuit = "focus"
        elif effective == "fixed_quiet":
            circuit = "quiet"
        elif effective == "random":
            circuit = rng.choice(CIRCUITS)
        else:
            circuit = gate.choose(tick, energy)
        if circuit is not None and not energy.can_pay(COST[circuit]+COST["update"]):
            circuit = None
        observed = {}
        evidence = 0.
        return_evidence = 0.
        if circuit is not None:
            energy.pay(circuit)
            readings = environment.readings()
            sensors = (0,) if circuit == "quiet" else (
                (0, 1, 2) if circuit == "audit" else (0, 1+tick % 2))
            observed = {s: readings[s] for s in sensors}
            if effective == "blinded_chart":
                labels = list(range(3))
                random.Random(seed*883+tick*139).shuffle(labels)
                observed = {labels[s]: pair for s, pair in observed.items()}
            if 0 in observed:
                evidence = abs(observed[0][0]-mean[0])/math.sqrt(variance)
                gate.evidence(evidence)
            if circuit == "audit":
                record = auditor.audit(tick, observed)
                return_evidence = math.sqrt(
                    record["positive_square"]/(2*MODEL["variance_floor"]))
                # The lossy return map supplies a second, explicitly typed
                # warning; it is not evidence that the full fiber is glued.
                gate.evidence(return_evidence)
            energy.pay("update")
            learner.observe(tick, features, predictions, observed)
        if (effective in ("adaptive", "no_budding", "blinded_chart") and
                (tick+1) % GATE["slow_period"] == 0 and observed and
                energy.pay("slow")):
            learner.slow_update(gate.failure_rate)
        # Meter before a structural edit can retire a model.
        cumulative_operations += sum(m.operations for m in learner.models)
        for model in learner.models:
            model.operations = 0
        if effective in ("adaptive", "no_slow", "blinded_chart"):
            # Always keep enough reserve to undo a newly created wrong bud.
            learner.prune(tick, energy)
            if ((tick+1) % GATE["structural_period"] == 0 and
                    circuit == "audit" and
                    min(evidence, return_evidence) > GATE["bud_evidence"] and
                    energy.can_pay(COST["bud"]+COST["prune"])):
                learner.bud(tick, min(evidence, return_evidence), energy)
        if trace:
            history.append({"tick": tick, "command": applied, "circuit": circuit,
                            "p": p, "energy": energy.spent,
                            "hazard": learner.hazard, "gate_value": gate.last_value
                            if math.isfinite(gate.last_value) else None})
    learner.prune(HORIZON, energy, force=True)
    buds = [e for e in learner.events if e["event"] == "bud"]
    # One-to-one event matching; the same reset cannot reward multiple buds.
    unmatched = set(t for t in resets if BURN_IN <= t < HORIZON)
    true_buds = 0
    for bud in buds:
        candidates = sorted(t for t in unmatched if 0 <= bud["tick"]-t <= 7)
        if candidates:
            unmatched.remove(candidates[0])
            true_buds += 1
    covered_count = sum(s["covered"] for s in scores)
    result = {
        "seed": seed, "world": world, "policy": policy, "cap": cap,
        "temperature": temperature, "scored_ticks": len(scores),
        "brier": sum(s["brier"] for s in scores)/len(scores),
        "logloss": sum(s["logloss"] for s in scores)/len(scores),
        "mse": sum(s["mse"] for s in scores)/len(scores),
        "coverage": covered_count/len(scores),
        "covered": covered_count,
        "correct_covered": sum(s["covered"] and s["correct"] for s in scores),
        "false_confident_gluing": sum(s["false_gluing"] for s in scores),
        "buds": len(buds), "true_buds": true_buds,
        "reset_events": sum(BURN_IN <= t < HORIZON for t in resets),
        "energy": energy.spent, "reserve": cap-energy.spent,
        "charges": energy.charges, "audits": auditor.audits,
        "slow_updates": learner.slow_updates,
        "operations": cumulative_operations + sum(m.operations for m in auditor.maps.values()),
        "operations_scope": "RLS arithmetic proxy only; use CPU/wall time for total computer cost",
        "wall_seconds": time.perf_counter()-clock,
        "cpu_seconds": time.process_time()-cpu,
        "actual_joules": None,
        "capacity": {"experts": 3, "recurrent_weights": 180,
                     "diagonal_covariance": 180, "return_map_weights": 36},
    }
    if trace:
        result.update(trace=history, events=learner.events, remainders=auditor.records)
    return result


def summary(rows):
    if not rows:
        raise ValueError("empty experiment")
    result = {key: rows[0][key] for key in ("world", "policy", "cap", "temperature")}
    for key in ("brier", "logloss", "mse", "coverage", "energy", "operations",
                "audits", "buds", "false_confident_gluing"):
        result[key] = sum(r[key] for r in rows)/len(rows)
    covered = sum(r["covered"] for r in rows)
    buds = sum(r["buds"] for r in rows)
    resets = sum(r["reset_events"] for r in rows)
    result.update(
        episodes=len(rows),
        selective_accuracy=sum(r["correct_covered"] for r in rows)/covered if covered else None,
        bud_precision=sum(r["true_buds"] for r in rows)/buds if buds else None,
        bud_recall=sum(r["true_buds"] for r in rows)/resets if resets else None,
        wall_seconds=sum(r["wall_seconds"] for r in rows),
        cpu_seconds=sum(r["cpu_seconds"] for r in rows),
        actual_joules=None)
    return result

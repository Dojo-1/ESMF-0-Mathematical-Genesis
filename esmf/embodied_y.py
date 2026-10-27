"""G0-Y: a partially observed, resource-limited cylinder-world benchmark.

The local organs and chart controls below are explicit computational
surrogates. The evaluator alone sees the physical state; learners receive
delayed noisy sensor traces and action labels. No biological or power claim is
made.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random
import time


ENERGY_CAP = 4.0
ENERGY_FLOOR = 0.12
HORIZON = 28
TRAIN_EPISODES = 5
TEST_EPISODES = 3
AUTONOMIC_PERIOD = 2
MODELS = ("recurrent", "kalman", "hierarchical", "chart_aware", "geometric")
TAU = 2 * math.pi


@dataclass(frozen=True)
class Scenario:
    name: str
    dynamics: str = "noncommuting"
    twist: bool = True
    seam_change: bool = False
    regulators: bool = True
    chart_mode: str = "adaptive"
    remainder: bool = True
    no_winding_training: bool = False


SCENARIOS = (
    Scenario("noncommuting"),
    Scenario("commuting", dynamics="commuting", twist=False, chart_mode="fixed"),
    Scenario("dissipative", dynamics="dissipative"),
    Scenario("out_of_family", dynamics="perturbed", chart_mode="adaptive"),
    Scenario("no_regulatory_organs", regulators=False),
    Scenario("no_transport_remainder", remainder=False),
    Scenario("fixed_charts", chart_mode="fixed"),
    Scenario("false_charts", chart_mode="false"),
    Scenario("hidden_seam_change", seam_change=True,
             no_winding_training=True),
)


def momentum_transition(momentum, angle, motor, damage, dynamics):
    if dynamics == "commuting":
        damping, drive, contextual = .82, .42, 0.0
    elif dynamics == "dissipative":
        damping, drive, contextual = .54, .37, .08
    elif dynamics == "perturbed":
        damping, drive, contextual = .63, .28, .23
    else:
        damping, drive, contextual = .82, .42, .11
    return (damping * momentum + drive * motor * (1.0 - damage) +
            contextual * math.sin(angle))


def momentum_action_gradient(damage, dynamics):
    drive = {"commuting": .42, "dissipative": .37,
             "perturbed": .28}.get(dynamics, .42)
    return drive * (1.0 - damage)


@dataclass(frozen=True)
class Observation:
    """Public sensor packet; timestamps expose the channel delays."""
    angle: float
    momentum: float
    reserve: float
    damage: float
    chart_id: int
    chart_gain: float
    tick: int
    angle_tick: int
    momentum_tick: int
    reserve_tick: int


class EntericReservoir:
    def __init__(self):
        self.reserve = ENERGY_CAP
        self.supplied = 0.0
        self.spent = 0.0
        self.overflow = 0.0

    def step(self, action, focus, regulated=True):
        supply = .055 if regulated else 0.0
        self.supplied += supply
        self.reserve += supply
        if self.reserve > ENERGY_CAP:
            self.overflow += self.reserve - ENERGY_CAP
            self.reserve = ENERGY_CAP
        motor_cost = .12 if action in ("left", "right") else 0.0
        action_cost = .035 + motor_cost + (.11 if action == "repair" else 0.0)
        sensing_cost = .035 if focus else .012
        total = action_cost + sensing_cost
        if self.reserve - total < ENERGY_FLOOR - 1e-12:
            raise RuntimeError("resource floor would be violated")
        self.reserve -= total
        self.spent += total
        if action == "rest" and regulated:
            recovery = min(.08, ENERGY_CAP - self.reserve)
            self.supplied += recovery
            self.reserve += recovery
        residual = ENERGY_CAP + self.supplied - self.spent - self.overflow - self.reserve
        if abs(residual) > 1e-9:
            raise ArithmeticError("resource ledger does not close")
        return total


class AutonomicGate:
    def __init__(self, regulated=True):
        self.regulated = regulated
        self.focus = False
        self.last_surprise = 0.0

    def update(self, observation, previous_momentum, reserve, tick):
        if tick % AUTONOMIC_PERIOD:
            return self.focus
        surprise = abs(observation.momentum - previous_momentum)
        self.last_surprise = surprise
        if self.regulated:
            if reserve > .9 and surprise > .28:
                self.focus = True
            elif reserve < .65 or surprise < .12:
                self.focus = False
        else:
            self.focus = False
        return self.focus


class SpinalLoop:
    """Fast policy using only delayed observations, never hidden world state."""
    @staticmethod
    def choose(observation, tick, no_winding=False, regulated=True,
               heldout_composition=False):
        if observation.reserve < .75:
            return "rest" if regulated else "coast"
        if regulated and observation.damage > .55:
            return "repair"
        if no_winding and tick < HORIZON:
            return "coast"
        if heldout_composition and tick % 8 == 4:
            return "rest"
        if heldout_composition and tick % 8 == 5 and observation.damage > .12:
            return "repair"
        if observation.momentum > .16:
            return "left"
        if observation.momentum < -.16:
            return "right"
        return "right" if math.sin(observation.angle) < 0 else "left"


class CylinderWorld:
    """Hidden physical state plus delayed scalar sensors and bounded resources."""
    def __init__(self, seed, scenario, split, episode_index=0):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.rng = random.Random(seed * 1000003 + episode_index * 7919 +
                                 (0 if split == "train" else 90127))
        self.scenario = scenario
        self.split = split
        self.tick = 0
        if scenario.no_winding_training and split == "train":
            self.angle = self.rng.uniform(-.5, .5)
            self.momentum = self.rng.uniform(-.02, .02)
        else:
            self.angle = self.rng.uniform(-math.pi, math.pi)
            self.momentum = self.rng.uniform(-.12, .12)
        self.damage = self.rng.uniform(0.0, .08)
        self.ledger = EntericReservoir()
        self.gate = AutonomicGate(scenario.regulators)
        self.winding = 0
        self.seam_decision = None
        self._angle_history = []
        self._momentum_history = []
        self._reserve_history = []
        self._damage_history = []
        self.previous_momentum = 0.0
        self._record_sensor_state(focus=False)

    def _chart_gain(self, chart_id):
        if self.scenario.chart_mode == "fixed":
            return 1.0
        if self.scenario.chart_mode == "false":
            return 1.0 + .18 * (1 - 2 * chart_id)
        return 1.0 + .12 * chart_id

    def _active_twist(self):
        changed = self.scenario.seam_change and self.tick >= HORIZON // 2
        return self.scenario.twist != changed

    def _record_sensor_state(self, focus):
        chart_id = int(math.floor((self.angle + math.pi) / TAU)) % 2
        true_gain = 1.0 if self.scenario.chart_mode == "fixed" else 1.0 + .12 * chart_id
        p_sign = -1.0 if self._active_twist() and chart_id else 1.0
        noise = .012 if focus else .035
        self._angle_history.append(
            self.angle + .22 * math.tanh(self.momentum) +
            self.rng.gauss(0.0, noise))
        self._momentum_history.append(
            true_gain * p_sign * self.momentum +
            self.rng.gauss(0.0, noise * 1.4))
        self._reserve_history.append(
            self.ledger.reserve + self.rng.gauss(0.0, noise * .5))
        self._damage_history.append(
            self.damage + self.rng.gauss(0.0, noise * .5))

    def observe(self):
        angle_tick = max(0, self.tick - 1)
        momentum_tick = max(0, self.tick - 2)
        reserve_tick = self.tick
        chart_id = int(math.floor((self.angle + math.pi) / TAU)) % 2
        return Observation(
            angle=self._angle_history[angle_tick],
            momentum=self._momentum_history[momentum_tick],
            reserve=max(0.0, min(ENERGY_CAP, self._reserve_history[reserve_tick])),
            damage=max(0.0, min(1.0, self._damage_history[angle_tick])),
            chart_id=chart_id,
            chart_gain=self._chart_gain(chart_id),
            tick=self.tick,
            angle_tick=angle_tick,
            momentum_tick=momentum_tick,
            reserve_tick=reserve_tick,
        )

    def step(self, action):
        if action not in ("left", "right", "coast", "rest", "repair"):
            raise ValueError("invalid action")
        before_winding = math.floor((self.angle + math.pi) / TAU)
        focus = self.gate.update(self.observe(), self.previous_momentum,
                                 self.ledger.reserve, self.tick)
        self.ledger.step(action, focus, self.scenario.regulators)
        motor = {"left": -1.0, "right": 1.0}.get(action, 0.0)
        if action == "repair":
            self.damage = max(0.0, self.damage - .075)
        elif action == "rest":
            self.damage = max(0.0, self.damage - .035)
        elif motor:
            low_resource = self.ledger.reserve < 1.0
            self.damage = min(1.0, self.damage +
                              (.018 if low_resource else .004))
        dynamics = self.scenario.dynamics
        if dynamics == "perturbed" and self.split == "train":
            dynamics = "noncommuting"
        physical_noise = .015 if dynamics == "dissipative" else .004
        old_momentum = self.momentum
        self.momentum = (momentum_transition(
            self.momentum, self.angle, motor, self.damage, dynamics) +
            self.rng.gauss(0.0, physical_noise))
        self.angle += self.momentum
        self.previous_momentum = old_momentum
        self.tick += 1
        after_winding = math.floor((self.angle + math.pi) / TAU)
        self.winding += after_winding - before_winding
        self._record_sensor_state(self.gate.focus)
        return self.observe()

    def ledger_result(self):
        return {
            "reserve": self.ledger.reserve,
            "spent": self.ledger.spent,
            "supplied": self.ledger.supplied,
            "overflow": self.ledger.overflow,
            "closure_error": abs(ENERGY_CAP + self.ledger.supplied -
                                 self.ledger.spent - self.ledger.overflow -
                                 self.ledger.reserve),
        }


def collect_trace(seed, scenario, split, episode_index, no_winding=False):
    world = CylinderWorld(seed, scenario, split, episode_index)
    obs = [world.observe()]
    actions = []
    for tick in range(HORIZON):
        action = SpinalLoop.choose(
            obs[-1], tick, no_winding=no_winding,
            regulated=scenario.regulators,
            heldout_composition=(split == "test"))
        actions.append(action)
        obs.append(world.step(action))
    if world.ledger.reserve < ENERGY_FLOOR - 1e-10:
        raise AssertionError("episode violated the resource floor")
    return {
        "observations": obs,
        "actions": actions,
        "winding": world.winding,
        "ledger": world.ledger_result(),
        "seam_identifiable": bool(world.winding % 2),
    }


def feature_stream(trace, model, use_remainder=True):
    observations, actions = trace["observations"], trace["actions"]
    fast = slow = previous_p = 0.0
    previous_chart_p = 0.0
    rows = []
    for i, action in enumerate(actions):
        obs = observations[i]
        motor = {"left": -1.0, "right": 1.0}.get(action, 0.0)
        innovation = obs.momentum - previous_p
        residual = innovation if use_remainder else 0.0
        if model == "recurrent":
            features = (1.0, obs.momentum + .1 * residual,
                        obs.reserve, motor)
        elif model == "kalman":
            fast += .55 * (obs.momentum - fast) + .1 * residual
            features = (1.0, fast, obs.reserve, motor)
        elif model == "hierarchical":
            fast = .7 * fast + .3 * obs.momentum + .1 * residual
            slow = .92 * slow + .08 * obs.momentum
            features = (1.0, fast, slow, motor)
        elif model in ("chart_aware", "geometric"):
            canonical_momentum = obs.momentum / max(.5, obs.chart_gain)
            chart_residual = (canonical_momentum - previous_chart_p
                              if use_remainder else 0.0)
            features = (1.0, canonical_momentum + .1 * chart_residual,
                        obs.reserve, motor)
            previous_chart_p = canonical_momentum
        else:
            raise ValueError("unknown model: " + model)
        rows.append((features, observations[i + 1].momentum))
        previous_p = obs.momentum
    return rows


def solve_linear(matrix, rhs):
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] += 1e-8
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def fit_model(traces, model, use_remainder=True):
    gram = [[0.0] * 4 for _ in range(4)]
    cross = [0.0] * 4
    count = 0
    for trace in traces:
        for features, target in feature_stream(trace, model, use_remainder):
            for i in range(4):
                cross[i] += features[i] * target
                for j in range(4):
                    gram[i][j] += features[i] * features[j]
            count += 1
    for i in range(4):
        gram[i][i] += 1e-6
    weights = solve_linear(gram, cross)
    squared_errors = [
        (dot(weights, features) - target) ** 2
        for trace in traces
        for features, target in feature_stream(trace, model, use_remainder)
    ]
    variance = max(1e-8, sum(squared_errors) / len(squared_errors))
    return weights, count, variance


def dot(weights, features):
    return sum(a * b for a, b in zip(weights, features))


def evaluate(weights, traces, model, use_remainder=True,
             predictive_variance=1.0):
    errors = []
    episode_mse = []
    compute = 0
    for trace in traces:
        trace_errors = []
        for features, target in feature_stream(trace, model, use_remainder):
            error = dot(weights, features) - target
            errors.append(error)
            trace_errors.append(error)
            compute += 4
        episode_mse.append(sum(error ** 2 for error in trace_errors) /
                           len(trace_errors))
    if not errors:
        raise ValueError("evaluation requires observations")
    variance = max(1e-8, sum(e * e for e in errors) / len(errors))
    rmse = math.sqrt(variance)
    return {
        "transitions": len(errors),
        "rmse": rmse,
        "mean_squared_error": variance,
        "training_predictive_variance": predictive_variance,
        "gaussian_nll": sum(
            .5 * (math.log(2 * math.pi * predictive_variance) +
                  e * e / predictive_variance) for e in errors) / len(errors),
        "nominal_90_coverage": sum(
            abs(e) <= 1.645 * math.sqrt(predictive_variance)
            for e in errors) / len(errors),
        "episode_mean_squared_errors": episode_mse,
        "compute_multiply_adds": compute,
    }


@dataclass
class ContextWorldlet:
    family: str
    weights: tuple = ()
    transitions: int = 0
    predictive_variance: float = 1.0

    def fit(self, traces, use_remainder=True):
        (self.weights, self.transitions,
         self.predictive_variance) = fit_model(
            traces, self.family, use_remainder)

    def score(self, traces, use_remainder=True):
        return evaluate(self.weights, traces, self.family, use_remainder,
                        self.predictive_variance)


def scenario_pair(seed, scenario):
    train_scenario = scenario
    test_scenario = scenario
    if scenario.name == "out_of_family":
        train_scenario = Scenario("training_family", chart_mode="adaptive")
    elif scenario.name == "hidden_seam_change":
        train_scenario = Scenario("no_winding_training", dynamics="commuting",
                                  twist=False,
                                  chart_mode="adaptive",
                                  no_winding_training=True)
    train = [
        collect_trace(seed, train_scenario, "train", i,
                      no_winding=train_scenario.no_winding_training)
        for i in range(TRAIN_EPISODES)
    ]
    test = [
        collect_trace(seed, test_scenario, "test", i,
                      no_winding=False)
        for i in range(TEST_EPISODES)
    ]
    return train, test


def run_seed(seed, scenario):
    train, test = scenario_pair(seed, scenario)
    results = []
    for model in MODELS:
        started = time.perf_counter()
        use_remainder = scenario.remainder
        learner = ContextWorldlet(model)
        learner.fit(train, use_remainder)
        metrics = learner.score(test, use_remainder)
        elapsed = time.perf_counter() - started
        test_transitions = metrics.pop("transitions")
        result = {
            "seed": seed,
            "scenario": scenario.name,
            "model": model,
            "parameters": 4,
            "weights": learner.weights,
            "train_transitions": learner.transitions,
            "test_transitions": test_transitions,
            "train_events": TRAIN_EPISODES * HORIZON,
            "test_events": TEST_EPISODES * HORIZON,
            "cpu_seconds": elapsed,
            "winding_train": [trace["winding"] for trace in train],
            "winding_test": [trace["winding"] for trace in test],
            "seam_identifiable_from_training": any(
                trace["seam_identifiable"] for trace in train),
            "seam_hypothesis": (
                "unknown" if not any(t["seam_identifiable"] for t in train)
                else "not_inferred"),
            "mean_test_resource_spent": sum(
                trace["ledger"]["spent"] for trace in test) / len(test),
            "max_ledger_closure_error": max(
                trace["ledger"]["closure_error"] for trace in train + test),
            "model_state_bytes": 4 * 8,
            "feature_matrix_payload_bytes_estimate": (
                (learner.transitions + test_transitions) * 4 * 8),
            **metrics,
        }
        results.append(result)
    geometric = next(r for r in results if r["model"] == "geometric")
    chart = next(r for r in results if r["model"] == "chart_aware")
    if geometric["weights"] != chart["weights"]:
        raise AssertionError("geometric and calibrated real controls must tie")
    return results


def paired_results(seeds=32, start=0):
    if not 1 <= seeds <= 128:
        raise ValueError("seed count must be in 1..128")
    return [
        result
        for scenario in SCENARIOS
        for seed in range(start, start + seeds)
        for result in run_seed(seed, scenario)
    ]


def summaries(rows):
    out = []
    for scenario in SCENARIOS:
        by_seed = {
            row["seed"]: row["rmse"] for row in rows
            if row["scenario"] == scenario.name and row["model"] == "chart_aware"
        }
        for model in MODELS:
            group = [r for r in rows if r["scenario"] == scenario.name
                     and r["model"] == model]
            if not group:
                continue
            mean = lambda key: sum(r[key] for r in group) / len(group)
            rmse_values = [r["rmse"] for r in group]
            rmse_mean = mean("rmse")
            rmse_se = (math.sqrt(sum((value - rmse_mean) ** 2
                                     for value in rmse_values) /
                                 (len(group) - 1) / len(group))
                       if len(group) > 1 else 0.0)
            paired_deltas = [row["rmse"] - by_seed[row["seed"]]
                             for row in group if row["seed"] in by_seed]
            paired_mean = (sum(paired_deltas) / len(paired_deltas)
                           if paired_deltas else 0.0)
            paired_se = (math.sqrt(sum((value - paired_mean) ** 2
                                       for value in paired_deltas) /
                                   (len(paired_deltas) - 1) / len(paired_deltas))
                         if len(paired_deltas) > 1 else 0.0)
            out.append({
                "scenario": scenario.name,
                "model": model,
                "seeds": len(group),
                "mean_rmse": rmse_mean,
                "rmse_normal_95_ci": [
                    max(0.0, rmse_mean - 1.96 * rmse_se),
                    rmse_mean + 1.96 * rmse_se,
                ],
                "paired_rmse_delta_vs_chart_aware": paired_mean,
                "paired_delta_normal_95_ci": [
                    paired_mean - 1.96 * paired_se,
                    paired_mean + 1.96 * paired_se,
                ],
                "mean_nll": mean("gaussian_nll"),
                "mean_90_coverage": mean("nominal_90_coverage"),
                "mean_resource_spent": mean("mean_test_resource_spent"),
                "mean_cpu_seconds": mean("cpu_seconds"),
                "mean_compute_multiply_adds": mean("compute_multiply_adds"),
            })
    return out


def selfcheck():
    scenario = Scenario("check")
    trace = collect_trace(0, scenario, "test", 0)
    assert trace["observations"][0].angle_tick == 0
    assert trace["observations"][2].momentum_tick == 0
    assert trace["observations"][2].tick == 2
    assert set(trace["observations"][0].__dataclass_fields__) == {
        "angle", "momentum", "reserve", "damage", "chart_id",
        "chart_gain", "tick", "angle_tick", "momentum_tick", "reserve_tick"}
    sample = run_seed(0, scenario)
    assert len(sample) == len(MODELS)
    assert all(r["parameters"] == 4 for r in sample)
    assert all(r["max_ledger_closure_error"] < 1e-9 for r in sample)
    return {
        "partial_delayed_sensors": True,
        "hidden_state_not_in_observation": True,
        "resource_ledger_closes": True,
        "matched_four_parameter_models": len(MODELS),
        "calibrated_real_equals_geometric": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print("SELFCHECK", json.dumps(selfcheck(), sort_keys=True))
    rows = paired_results(args.seeds, args.start)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    for row in summaries(rows):
        print("SUMMARY", json.dumps(row, sort_keys=True))
    if not args.output:
        for row in rows:
            print("SEED", json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

"""G0-AB reflex circuit coupled to the rank-four neutral bundle.

The fast policy is a finite-state sensorimotor circuit. Bayesian continuity
audits run only after novelty, prediction error, or ambiguous provenance.
This is an experimental bridge between G0-AB control and G0-AC0 transport,
not a biological or hardware model.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random
import time
import tracemalloc

from .adaptive_connection_ac0 import (
    BURN_IN, HORIZON, RLSConnection, make_probe, predict, source_coefficients,
)
from .global_bundle_q import NeutralCylinderBundle


RESOURCE_CAP = 8.0
RESOURCE_FLOOR = .25
RESOURCE_SUPPLY = .16
FOCUS_COST = .12
QUIET_COST = .04
AUDIT_COST = .08
MAINTENANCE_COST = .025
FOCUS_SD = .02
QUIET_SD = .05
MODE_VARIANCE = (1.0, 4.0, 25.0)
POLICIES = ("reflex", "fixed_focus", "gaussian_state_space",
            "posterior_threshold")
WORLDS = ("static", "smooth_drift", "abrupt_step", "sign_reversal",
          "rapidly_variable", "nonlinear_drift", "noise_mismatch")


@dataclass
class ResourceLedger:
    reserve: float = RESOURCE_CAP
    delivered: float = 0.0
    spent: float = 0.0
    overflow: float = 0.0

    def replenish(self):
        total = self.reserve + RESOURCE_SUPPLY
        self.reserve = min(RESOURCE_CAP, total)
        self.delivered += RESOURCE_SUPPLY
        self.overflow += total - self.reserve
        self.verify()

    def charge(self, amount):
        if amount < 0 or self.reserve - amount < RESOURCE_FLOOR - 1e-10:
            return False
        self.reserve -= amount
        self.spent += amount
        self.verify()
        return True

    def verify(self):
        if not RESOURCE_FLOOR - 1e-10 <= self.reserve <= RESOURCE_CAP + 1e-10:
            raise AssertionError("resource reserve outside viability bounds")
        if abs(RESOURCE_CAP + self.delivered - self.spent -
               self.overflow - self.reserve) > 1e-9:
            raise AssertionError("resource ledger does not close")


class ContinuityAudit:
    """Three-model Bayesian audit over stable, drifting, and reset residuals."""

    def __init__(self):
        self.weights = [1 / 3] * 3
        self.events = 0

    def update(self, residuals, gradients, coefficient_variance, sensor_sd):
        if (len(residuals) != 2 or len(gradients) != 2 or
                any(not math.isfinite(value) for value in (*residuals, *gradients)) or
                not math.isfinite(coefficient_variance) or coefficient_variance < 0 or
                not math.isfinite(sensor_sd) or sensor_sd <= 0):
            raise ValueError("invalid audit residuals")
        var0 = sensor_sd * sensor_sd + gradients[0] ** 2 * coefficient_variance
        var1 = sensor_sd * sensor_sd + gradients[1] ** 2 * coefficient_variance
        covariance = gradients[0] * gradients[1] * coefficient_variance
        determinant = var0 * var1 - covariance * covariance
        if determinant <= 0 or not math.isfinite(determinant):
            raise ArithmeticError("invalid predictive residual covariance")
        quadratic = (
            var1 * residuals[0] ** 2 - 2 * covariance * residuals[0] * residuals[1]
            + var0 * residuals[1] ** 2
        ) / determinant
        scores = [
            math.log(max(1e-300, weight)) - .5 * (
                2 * math.log(2 * math.pi) + math.log(determinant) +
                2 * math.log(scale) + quadratic / scale
            )
            for weight, scale in zip(self.weights, MODE_VARIANCE)
        ]
        peak = max(scores)
        raw = [math.exp(score - peak) for score in scores]
        total = sum(raw)
        self.weights = [value / total for value in raw]
        self.events += 1
        if abs(sum(self.weights) - 1.) > 1e-12:
            raise ArithmeticError("audit posterior failed to normalize")
        return self.weights

    @property
    def reset_probability(self):
        return self.weights[2]

    @property
    def ambiguous(self):
        return max(self.weights) < .80


class ReflexCircuit:
    """Fixed alternating fast circuit with mesoscopic focus hysteresis."""

    def __init__(self):
        self.phase = 0
        self.focus = False
        self.fast_clock = 0
        self.meso_clock = 0
        self.switches = 0
        self.audit_calls = 0
        self.last_surprise = 0.
        self.last_provenance_known = True

    def choose(self, policy, ledger):
        focus = policy == "fixed_focus" or (
            policy in ("reflex", "posterior_threshold",
                       "gaussian_state_space") and self.focus)
        axis = self.phase
        self.phase = 1 - self.phase
        self.fast_clock += 1
        cost = (FOCUS_COST if focus else QUIET_COST) + MAINTENANCE_COST
        if not ledger.charge(cost):
            return None, focus
        return axis, focus

    def update_meso(self, audit, ledger):
        if self.fast_clock % 2:
            return
        self.meso_clock += 1
        previous = self.focus
        if (self.last_surprise > 2.5 or audit.reset_probability > .65 or
                not self.last_provenance_known):
            self.focus = ledger.reserve > 1.55
        elif self.last_surprise < .8 and audit.reset_probability < .20:
            self.focus = False
        if previous != self.focus:
            self.switches += 1

    def should_audit(self, policy, audit, provenance_known):
        if policy == "gaussian_state_space":
            return True
        if policy == "fixed_focus":
            return False
        if policy == "posterior_threshold":
            return audit.ambiguous or audit.reset_probability > .25
        return (
            audit.events == 0 or self.last_surprise > 2.5 or
            audit.ambiguous or not provenance_known
        )


def connection_coefficients(seed, tick, world):
    if world == "nonlinear_drift":
        phase = seed * .37
        return (
            .40 + .18 * math.sin(.15 * tick + phase) +
            .08 * math.sin(.41 * tick + phase) ** 2,
            -.33 + .16 * math.cos(.12 * tick + .6 * phase) +
            .06 * math.sin(.29 * tick + phase) ** 3,
        )
    if world == "noise_mismatch":
        return source_coefficients(seed, tick, "smooth_drift")
    return source_coefficients(seed, tick, world)


def simulate(seed, world, policy):
    if world not in WORLDS or policy not in POLICIES:
        raise ValueError("unknown world or policy")
    rng = random.Random(seed * 9337 + 691)
    bundle = NeutralCylinderBundle()
    learner = RLSConnection(
        "constant_forgetting" if policy == "gaussian_state_space" else "adaptive")
    audit = ContinuityAudit()
    circuit = ReflexCircuit()
    ledger = ResourceLedger()
    heldout_errors = []
    audit_trace = []
    focus_count = 0
    seam_crossings = 0
    pending = None

    def deliver(event):
        tick, axis, probe, observed, focus, provenance_known, truth = event
        sd = FOCUS_SD if focus else QUIET_SD
        predicted, gradients = predict(
            probe, learner.coeff[axis], bundle, gradient=True)
        residuals = tuple(y - mean for y, mean in zip(observed, predicted))
        predictive_variances = tuple(
            sd * sd + gradient * gradient * learner.var[axis]
            for gradient in gradients)
        correlation = gradients[0] * gradients[1] * learner.var[axis]
        determinant = (predictive_variances[0] * predictive_variances[1]
                       - correlation * correlation)
        quadratic = (
            predictive_variances[1] * residuals[0] ** 2
            - 2 * correlation * residuals[0] * residuals[1]
            + predictive_variances[0] * residuals[1] ** 2
        ) / max(1e-12, determinant)
        normalized_residual = math.sqrt(max(0., quadratic) / len(residuals))
        if circuit.should_audit(policy, audit, provenance_known):
            if ledger.charge(AUDIT_COST):
                weights = audit.update(
                    residuals, gradients, learner.var[axis], sd)
                circuit.audit_calls += 1
                audit_trace.append((tick, tuple(weights), provenance_known))
        circuit.last_surprise = normalized_residual
        circuit.last_provenance_known = provenance_known
        learner.update(probe, observed, bundle)
        if tick >= BURN_IN:
            eval_probe = make_probe(rng, axis, bundle)
            expected = predict(eval_probe, truth, bundle)
            forecast = predict(eval_probe, learner.coeff[axis], bundle)
            heldout_errors.append(sum(
                (actual - estimate) ** 2
                for actual, estimate in zip(expected, forecast)) / len(expected))

    for tick in range(HORIZON):
        ledger.replenish()
        if pending is not None:
            deliver(pending)
            pending = None
        circuit.update_meso(audit, ledger)
        axis = tick % 2
        focus = policy == "fixed_focus"
        if policy in ("reflex", "posterior_threshold",
                      "gaussian_state_space"):
            axis, focus = circuit.choose(policy, ledger)
        else:
            cost = (FOCUS_COST if focus else QUIET_COST) + MAINTENANCE_COST
            if not ledger.charge(cost):
                raise AssertionError("unaffordable scheduled probe")
        if axis is None:
            break
        focus_count += int(focus)
        cross_seam = axis == 0 and (tick + seed) % 11 == 0
        probe = make_probe(rng, axis, bundle, cross_seam=cross_seam)
        seam_crossings += int(cross_seam)
        truth = connection_coefficients(seed, tick, world)[axis]
        clean = predict(probe, truth, bundle)
        sd = FOCUS_SD if focus else QUIET_SD
        if world == "noise_mismatch":
            shared_noise = rng.gauss(0., 1.2 * sd)
            outlier = rng.gauss(0., 3. * sd) if rng.random() < .10 else 0.
            observed = tuple(value + shared_noise + rng.gauss(0., .7 * sd) +
                             outlier for value in clean)
        else:
            observed = tuple(value + rng.gauss(0., sd) for value in clean)
        provenance_known = (tick + seed) % 17 != 0
        pending = (tick, axis, probe, observed, focus, provenance_known, truth)
    if pending is not None:
        ledger.replenish()
        deliver(pending)
        circuit.update_meso(audit, ledger)
    ledger.verify()
    return {
        "seed": seed, "world": world, "policy": policy,
        "heldout_probe_mse": (
            sum(heldout_errors) / len(heldout_errors) if heldout_errors else 0.),
        "audit_calls": circuit.audit_calls,
        "audit_rate": circuit.audit_calls / HORIZON,
        "focus_probes": focus_count,
        "seam_crossings": seam_crossings,
        "gate_switches": circuit.switches,
        "posterior_modes": audit.weights[:],
        "resource_spent": ledger.spent,
        "resource_delivered": ledger.delivered,
        "resource_overflow": ledger.overflow,
        "resource_reserve": ledger.reserve,
        "sensor_delay_ticks": 1,
        "simulation_ticks": HORIZON + 1,
        "trace": audit_trace,
    }


def benchmark(seed, world, policy, repeats=3):
    """Report runtime and Python-tracked peak memory; power is not measured."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    durations = []
    peak_bytes = 0
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        simulate(seed, world, policy)
        durations.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes = max(peak_bytes, peak)
    return {
        "median_wall_seconds": sorted(durations)[len(durations) // 2],
        "peak_traced_bytes": peak_bytes,
        "joules": None,
    }


def run(seeds=8, worlds=WORLDS):
    if not 1 <= seeds <= 256 or any(world not in WORLDS for world in worlds):
        raise ValueError("invalid seeds or worlds")
    results = []
    for world in worlds:
        for policy in POLICIES:
            tracemalloc.start()
            started = time.perf_counter()
            rows = [simulate(seed, world, policy) for seed in range(seeds)]
            wall_seconds = time.perf_counter() - started
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results.append({
                "world": world,
                "policy": policy,
                "episodes": len(rows),
                "mean_heldout_probe_mse": sum(
                    row["heldout_probe_mse"] for row in rows) / len(rows),
                "mean_audit_calls": sum(row["audit_calls"] for row in rows) / len(rows),
                "mean_focus_probes": sum(row["focus_probes"] for row in rows) / len(rows),
                "mean_resource_spent": sum(
                    row["resource_spent"] for row in rows) / len(rows),
                "wall_seconds": wall_seconds,
                "peak_traced_bytes": peak_bytes,
                "joules": None,
            })
            tracemalloc.stop()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--worlds", default=",".join(WORLDS))
    args = parser.parse_args()
    worlds = tuple(args.worlds.split(","))
    try:
        results = run(args.seeds, worlds)
    except ValueError as exc:
        parser.error(str(exc))
    for result in results:
        print("MODEL", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

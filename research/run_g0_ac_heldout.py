"""G0-AC replay: calibrate on development seeds, freeze, then evaluate once."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

from esmf.reflex_auditor_ac import (
    POLICIES, REGISTRATION, SPEC, WORLDS, simulate, summary,
)


def registration_hash():
    return hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()


def calibration_key(policy, cap):
    return f"{policy}:{cap:g}"


def calibrate():
    """Same calibration capacity, data and objective for every policy."""
    first, last = SPEC["development_seeds"]
    temperatures = {}
    losses = {}
    for cap in SPEC["energy_caps"]:
        for policy in POLICIES:
            if policy == "equivalent_real":
                key = calibration_key("adaptive", cap)
                temperatures[calibration_key(policy, cap)] = temperatures[key]
                losses[calibration_key(policy, cap)] = losses[key]
                continue
            candidates = {}
            for temperature in SPEC["learner"]["temperature_candidates"]:
                rows = [simulate(seed, world, policy, cap, temperature)
                        for world in WORLDS for seed in range(first, last+1)]
                candidates[str(temperature)] = sum(r["brier"] for r in rows)/len(rows)
            best = min(candidates, key=candidates.get)
            key = calibration_key(policy, cap)
            temperatures[key] = float(best)
            losses[key] = candidates
    return {"experiment": "G0-AC", "registration_sha256": registration_hash(),
            "development_seeds": [first, last], "temperatures": temperatures,
            "development_brier": losses,
            "calibration_frozen_before_heldout": True}


def validate_calibration(calibration):
    if calibration["registration_sha256"] != registration_hash():
        raise ValueError("registration changed since calibration")
    if calibration["development_seeds"] != SPEC["development_seeds"]:
        raise ValueError("unregistered calibration seeds")
    for cap in SPEC["energy_caps"]:
        for policy in POLICIES:
            temperature = calibration["temperatures"][calibration_key(policy, cap)]
            if temperature not in SPEC["learner"]["temperature_candidates"]:
                raise ValueError("unregistered calibration temperature")
        if (calibration["temperatures"][calibration_key("adaptive", cap)] !=
                calibration["temperatures"][calibration_key("equivalent_real", cap)]):
            raise ValueError("equivalent-real calibration differs")


def paired_interval(differences):
    rng = random.Random(73159)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(2000))
    return [means[49], means[1949]]


def pareto_frontier(stats):
    """Prediction/actual metabolic cost/work frontier; never compare caps alone."""
    return [r for r in stats if not any(
        all(other[k] <= r[k] for k in ("brier", "energy", "operations")) and
        any(other[k] < r[k] for k in ("brier", "energy", "operations"))
        for other in stats)]


def compare(allrows, stats):
    statements = []
    blocked = False
    for world in WORLDS:
        controls = [r for r in stats if r["world"] == world and
                    r["policy"] in ("fixed_focus", "real_recurrent", "real_imm")]
        frontier = pareto_frontier(controls)
        complete_frontier = pareto_frontier([
            r for r in stats if r["world"] == world and
            r["policy"] not in ("equivalent_real", "blinded_chart")])
        # Include the candidate itself: a dominating adaptive policy can
        # actually remove a reflex-focused point from the full frontier.
        if any(r["policy"] == "fixed_focus" for r in complete_frontier):
            blocked = True
        for adaptive in (r for r in stats if r["world"] == world and
                         r["policy"] == "adaptive"):
            eligible = [r for r in frontier if
                        r["energy"] <= adaptive["energy"]+1e-10 and
                        r["operations"] <= adaptive["operations"]]
            if not eligible:
                blocked = True
            for control in eligible:
                a = allrows[world, "adaptive", adaptive["cap"]]
                b = allrows[world, control["policy"], control["cap"]]
                differences = [x["brier"]-y["brier"] for x, y in zip(a, b)]
                interval = paired_interval(differences)
                false_splits = sum(x["buds"]-x["true_buds"] for x in a)
                extra_gluing = sum(x["false_confident_gluing"] for x in a) > sum(
                    y["false_confident_gluing"] for y in b)
                cpu_matched = control["cpu_seconds"] <= adaptive["cpu_seconds"]
                wins = (interval[1] < 0 and not extra_gluing and
                        false_splits == 0 and cpu_matched)
                blocked = blocked or not wins
                statements.append({
                    "world": world, "adaptive_cap": adaptive["cap"],
                    "comparator": control["policy"], "comparator_cap": control["cap"],
                    "actual_energy_matched_or_lower": True,
                    "rls_work_matched_or_lower": True,
                    "comparator_cpu_seconds": control["cpu_seconds"],
                    "adaptive_cpu_seconds": adaptive["cpu_seconds"],
                    "total_cpu_matched_or_lower": cpu_matched,
                    "mean_brier_difference": sum(differences)/len(differences),
                    "paired_seed_bootstrap95": interval,
                    "extra_false_confident_gluing": extra_gluing,
                    "adaptive_false_splits": false_splits, "passes": wins})
    return statements, {
        "promotion_blocked": blocked,
        "decision": "exception-triggered only; no superiority claim" if blocked else
                    "candidate predictive improvement; no physical efficiency claim",
        "hardware_energy_measured": False}


def evaluate(calibration, seeds=None):
    validate_calibration(calibration)
    if seeds is None:
        first, last = SPEC["heldout_seeds"]
        seeds = range(first, last+1)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seeds required")
    allrows, stats = {}, []
    began, cpu = time.perf_counter(), time.process_time()
    for world in WORLDS:
        for cap in SPEC["energy_caps"]:
            for policy in POLICIES:
                temperature = calibration["temperatures"][calibration_key(policy, cap)]
                rows = [simulate(seed, world, policy, cap, temperature) for seed in seeds]
                allrows[world, policy, cap] = rows
                stats.append(summary(rows))
            for a, b in zip(allrows[world, "adaptive", cap],
                            allrows[world, "equivalent_real", cap]):
                nondeterministic = ("policy", "wall_seconds", "cpu_seconds")
                if ({k: v for k, v in a.items() if k not in nondeterministic} !=
                        {k: v for k, v in b.items() if k not in nondeterministic}):
                    raise AssertionError("ordinary real equivalence failed")
    comparisons, decision = compare(allrows, stats)
    registered = list(seeds) == list(range(
        SPEC["heldout_seeds"][0], SPEC["heldout_seeds"][1]+1))
    if not registered:
        decision["promotion_blocked"] = True
        decision["decision"] = "development diagnostic only; no held-out claim"
    return {
        "experiment": "G0-AC", "registration_sha256": registration_hash(),
        "calibration_sha256": hashlib.sha256(
            json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        "seeds": list(seeds),
        "registered_heldout": registered,
        "python": platform.python_version(), "platform": platform.platform(),
        "models": stats, "comparisons": comparisons, "kill_decision": decision,
        "equivalent_real_ties": True,
        "total_wall_seconds": time.perf_counter()-began,
        "total_cpu_seconds": time.process_time()-cpu,
        "actual_joules": None,
        "limitations": [
            "A common cap is not equal actual spend; comparisons filter actual spend and RLS work, and promotion also requires no more measured comparator CPU time.",
            "Operation counts cover only RLS arithmetic, not total work or joules; total runtime includes all policy logic, evaluation and bootstrap. CPU comparisons are noisy single-machine measurements.",
            "Gate value is a conservative consequence-risk proxy, not identified causal value of information.",
            "False gluing is a local confident predictive-continuity assertion across a true reset, not topological gluing.",
            "Sensor return maps are lossy; raw sensor recurrence is not a reconstructed neutral connection.",
            "Temporary model slots are computational morphology, not physical growth.",
            "No new nonorientable-cylinder coupling or sentience/biology claim."
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate", "smoke"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing frozen artifact")
    if args.stage == "calibrate":
        result = calibrate()
    else:
        if args.calibration is None:
            parser.error("--calibration is required")
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        seeds = [SPEC["development_seeds"][0]] if args.stage == "smoke" else None
        result = evaluate(calibration, seeds)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"stage": args.stage, "output": str(args.output),
                      "kill_decision": result.get("kill_decision")}, sort_keys=True))


if __name__ == "__main__":
    main()

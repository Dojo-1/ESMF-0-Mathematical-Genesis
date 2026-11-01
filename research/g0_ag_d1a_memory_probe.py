"""G0-AG D1a: development-only *behavioral* evaluation of contextual memory.

Predictors are committed BEFORE fresh test outcomes are generated.
Both see the same 24 observable training encounters, with the same 8
bounded integer memory slots. The evaluator's Bernoulli probabilities
are not passed to either memory model.

This isolates G0-AE's question-mode information loss. It does NOT test
glial physiology, quaternionic neural advantage or autonomous context
discovery. D1b / original G0-AG1 heldout is NOT executed.
"""
import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

from esmf.telos_joint_ae import Distinction

CAP = 3
TRAIN_PER_CELL = 6
TEST_PER_CELL = 32
CONTEXTS = (0, 1)
MODES = (0, 1)
# EVALUATOR ONLY. These are the previously specified, artificial
# environment probabilities from G0-AE's example_world (for probe mode).
EVALUATOR_P_CHANGE = ((.85, .55), (.15, .45))
MEMORY_SLOTS = 8


class PooledG0AEMemory:
    """The actual G0-AE per-context Distinction update contract."""

    def __init__(self):
        self.cells = (Distinction(), Distinction())

    def observe(self, context, mode, changed):
        check_event(context, mode, changed)
        cells = list(self.cells)
        cells[context] = cells[context].observe(not changed, CAP)
        self.cells = tuple(cells)

    def predict(self, context, mode):
        check_query(context, mode)
        cell = self.cells[context]
        return (cell.seen_different + 1) / (
            cell.seen_different + cell.seen_same + 2
        )

    def slots(self):
        return len(self.cells) * 4


class ModeConditionedMemory:
    """Ordinary capped per-(context, probe) Bernoulli estimator."""

    def __init__(self):
        self.counts = {(c, q): [0, 0] for c in CONTEXTS for q in MODES}

    def observe(self, context, mode, changed):
        check_event(context, mode, changed)
        cell = self.counts[context, mode]
        index = int(changed)
        cell[index] = min(CAP, cell[index] + 1)

    def predict(self, context, mode):
        check_query(context, mode)
        same, different = self.counts[context, mode]
        return (different + 1) / (different + same + 2)

    def slots(self):
        return len(self.counts) * 2


def check_query(context, mode):
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError("unknown observational context")
    if type(mode) is not int or mode not in MODES:
        raise ValueError("unknown interrogative mode")


def check_event(context, mode, changed):
    check_query(context, mode)
    if type(changed) is not bool:
        raise ValueError("observation must be a boolean change event")


def training_encounters(seed):
    """Public observed events only; evaluator probabilities stay internal."""
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid development seed")
    rng = random.Random(seed)
    schedule = [
        (c, q)
        for c in CONTEXTS for q in MODES
        for _ in range(TRAIN_PER_CELL)
    ]
    rng.shuffle(schedule)
    return tuple(
        (c, q, rng.random() < EVALUATOR_P_CHANGE[c][q])
        for c, q in schedule
    )


def shuffle_modes_within_context(trace, seed):
    """Randomized negative control, preserving per-context question totals."""
    rng = random.Random(seed + 912_737)
    labels = {
        c: [q for observed_context, q, _ in trace if observed_context == c]
        for c in CONTEXTS
    }
    for available in labels.values():
        rng.shuffle(available)
    indexes = {c: 0 for c in CONTEXTS}
    result = []
    for c, _, changed in trace:
        q = labels[c][indexes[c]]
        indexes[c] += 1
        result.append((c, q, changed))
    return tuple(result)


def brier(prediction, observed_change):
    if not 0. <= prediction <= 1.:
        raise ValueError("invalid probability")
    return (prediction - float(observed_change)) ** 2


def expected_brier(prediction, p_change):
    """Independent analytic Bernoulli oracle; not input to predictors."""
    if not 0. <= p_change <= 1.:
        raise ValueError("invalid evaluator probability")
    return p_change * (1. - prediction) ** 2 + (
        1. - p_change) * prediction ** 2


def evaluate_one(seed):
    trace = training_encounters(seed)
    models = {
        "pooled_g0ae": PooledG0AEMemory(),
        "mode_conditioned": ModeConditionedMemory(),
        "mode_shuffled": ModeConditionedMemory(),
    }
    scrambled = shuffle_modes_within_context(trace, seed)
    for c, q, changed in trace:
        models["pooled_g0ae"].observe(c, q, changed)
        models["mode_conditioned"].observe(c, q, changed)
    for c, q, changed in scrambled:
        models["mode_shuffled"].observe(c, q, changed)
    if any(model.slots() != MEMORY_SLOTS for model in models.values()):
        raise AssertionError("memory slot parity broken")
    # Commit and store all predictions BEFORE drawing test outcomes.
    committed = {
        name: {(c, q): model.predict(c, q)
               for c in CONTEXTS for q in MODES}
        for name, model in models.items()
    }
    # Separate RNG, and the same future response for all predictors.
    future_rng = random.Random(seed + 1_000_003)
    future = {
        (c, q): [
            future_rng.random() < EVALUATOR_P_CHANGE[c][q]
            for _ in range(TEST_PER_CELL)
        ]
        for c in CONTEXTS for q in MODES
    }
    empirical = {}
    analytic = {}
    for name, predictions in committed.items():
        empirical[name] = {
            f"{c}:{q}": statistics.mean(
                brier(predictions[c, q], changed)
                for changed in future[c, q]
            )
            for c in CONTEXTS for q in MODES
        }
        analytic[name] = {
            f"{c}:{q}": expected_brier(
                predictions[c, q], EVALUATOR_P_CHANGE[c][q]
            )
            for c in CONTEXTS for q in MODES
        }
    return {
        "seed": seed,
        "observed_training": [
            {"context": c, "mode": q, "changed": changed}
            for c, q, changed in trace
        ],
        "shuffled_training": [
            {"context": c, "mode": q, "changed": changed}
            for c, q, changed in scrambled
        ],
        "predictions_committed_before_outcomes": {
            name: {f"{c}:{q}": predictions[c, q]
                   for c in CONTEXTS for q in MODES}
            for name, predictions in committed.items()
        },
        "heldout_from_training_test_outcomes": {
            f"{c}:{q}": future[c, q] for c in CONTEXTS for q in MODES
        },
        "empirical_brier_by_cell": empirical,
        "analytic_expected_brier_by_cell": analytic,
        "empirical_brier": {
            name: statistics.mean(scores.values())
            for name, scores in empirical.items()
        },
        "analytic_expected_brier": {
            name: statistics.mean(scores.values())
            for name, scores in analytic.items()
        },
        "memory_slots": {name: model.slots() for name, model in models.items()},
    }


def paired_interval(values):
    values = list(values)
    if not values:
        raise ValueError("at least one paired seed required")
    avg = statistics.mean(values)
    error = (1.96 * statistics.stdev(values) / math.sqrt(len(values))
             if len(values) > 1 else 0.)
    return {"mean": avg, "normal_approx_95": [avg - error, avg + error],
            "n": len(values), "confirmatory": False}


def run(seeds=range(8150, 8164)):
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("invalid seed set")
    started = time.process_time()
    rows = [evaluate_one(seed) for seed in seeds]
    cpu = time.process_time() - started
    arms = ("pooled_g0ae", "mode_conditioned", "mode_shuffled")
    means = {
        arm: {
            "future_brier": statistics.mean(
                row["empirical_brier"][arm] for row in rows
            ),
            "analytic_expected_brier": statistics.mean(
                row["analytic_expected_brier"][arm] for row in rows
            ),
        }
        for arm in arms
    }
    comparisons = {
        "pooled_minus_mode_conditioned": paired_interval(
            row["empirical_brier"]["pooled_g0ae"] -
            row["empirical_brier"]["mode_conditioned"]
            for row in rows
        ),
        "mode_shuffled_minus_mode_conditioned": paired_interval(
            row["empirical_brier"]["mode_shuffled"] -
            row["empirical_brier"]["mode_conditioned"]
            for row in rows
        ),
    }
    return {
        "scope": "D1a development-only offline observed-response forecasting",
        "evidence": "synthetic task measurement, not G0-AG1 heldout",
        "git_sha": os.getenv("GITHUB_SHA"),
        "settings": {
            "seed_first": seeds[0], "seed_last": seeds[-1],
            "seed_count": len(seeds), "training_per_cell": TRAIN_PER_CELL,
            "future_outcomes_per_cell": TEST_PER_CELL,
            "memory_slots_per_arm": MEMORY_SLOTS, "counter_cap": CAP,
        },
        "means": means,
        "paired": comparisons,
        "cpu_seconds": cpu,
        "records": rows,
        "limitations": [
            "Context and question IDs supplied in advance; not endogenous",
            "Training stream is exogenous; no glial/neurite resource dynamics",
            "Scores are against a synthetic evaluator, not real semantics",
            "Capped slot parity does not equal entire architecture complexity",
            "Development seeds only; original G0-AG1 heldout remains sealed",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = run()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    print("G0_AG_D1A_SUMMARY=" + json.dumps(
        {k: v for k, v in results.items() if k != "records"},
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

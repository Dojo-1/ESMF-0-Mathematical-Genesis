"""G0-AG D1a verification: no oracle leak, known answers and capacity parity."""
import math
import unittest

from esmf.telos_joint_ae import Distinction
from research.g0_ag_d1a_memory_probe import (
    CAP, MEMORY_SLOTS, TEST_PER_CELL, TRAIN_PER_CELL,
    PooledG0AEMemory, ModeConditionedMemory, brier, expected_brier,
    training_encounters, shuffle_modes_within_context, evaluate_one, run,
)


class TestG0AGD1aMemoryProbe(unittest.TestCase):
    def test_known_answer_pooled_memory_loses_question_mode(self):
        pooled = PooledG0AEMemory()
        mode = ModeConditionedMemory()
        for _ in range(6):
            pooled.observe(0, 0, True)
            mode.observe(0, 0, True)
        for _ in range(6):
            pooled.observe(0, 1, False)
            mode.observe(0, 1, False)
        self.assertEqual(pooled.predict(0, 0), .5)
        self.assertEqual(pooled.predict(0, 1), .5)
        self.assertEqual(mode.predict(0, 0), .8)
        self.assertEqual(mode.predict(0, 1), .2)
        self.assertEqual(pooled.slots(), MEMORY_SLOTS)
        self.assertEqual(mode.slots(), MEMORY_SLOTS)
        self.assertEqual(pooled.cells[0], Distinction(3, 3, 3, 3))

    def test_independent_analytic_reference(self):
        self.assertAlmostEqual(brier(.8, True), .04)
        self.assertAlmostEqual(brier(.8, False), .64)
        self.assertAlmostEqual(expected_brier(.8, .85), .13)
        self.assertAlmostEqual(expected_brier(.5, .85), .25)
        self.assertAlmostEqual(expected_brier(.5, .15), .25)
        with self.assertRaises(ValueError):
            expected_brier(.4, 1.5)
        with self.assertRaises(ValueError):
            brier(-.1, True)

    def test_training_balanced_and_shuffled_without_outcome_edits(self):
        for seed in range(8150, 8153):
            trace = training_encounters(seed)
            self.assertEqual(len(trace), 4 * TRAIN_PER_CELL)
            self.assertEqual(trace, training_encounters(seed))
            self.assertEqual(
                sorted([(c, q) for c, q, changed in trace]),
                sorted([(c, q) for c in (0, 1) for q in (0, 1)
                        for _ in range(TRAIN_PER_CELL)]),
            )
            shuffled = shuffle_modes_within_context(trace, seed)
            self.assertEqual(
                [(c, changed) for c, _, changed in trace],
                [(c, changed) for c, _, changed in shuffled],
            )
            self.assertEqual(
                sorted((c, q) for c, q, _ in trace),
                sorted((c, q) for c, q, _ in shuffled),
            )

    def test_predictions_are_committed_before_future_outcomes(self):
        first = evaluate_one(8150)
        self.assertEqual(first, evaluate_one(8150))
        self.assertEqual(
            first["memory_slots"],
            {"pooled_g0ae": MEMORY_SLOTS,
             "mode_conditioned": MEMORY_SLOTS,
             "mode_shuffled": MEMORY_SLOTS},
        )
        self.assertEqual(
            sum(len(v) for v in first[
                "heldout_from_training_test_outcomes"].values()),
            4 * TEST_PER_CELL,
        )
        for arm, predictions in first[
                "predictions_committed_before_outcomes"].items():
            self.assertEqual(len(predictions), 4)
            self.assertTrue(all(0. <= p <= 1. and math.isfinite(p)
                                for p in predictions.values()))
            self.assertTrue(0 <= first["empirical_brier"][arm] <= 1.)

    def test_run_reports_paired_differences_no_heldout_promotion(self):
        r = run(range(8150, 8152))
        self.assertEqual(r["settings"]["seed_count"], 2)
        self.assertEqual(len(r["records"]), 2)
        self.assertEqual(r["settings"]["memory_slots_per_arm"], 8)
        self.assertEqual(r["settings"]["counter_cap"], CAP)
        self.assertIn("not G0-AG1", r["evidence"])
        self.assertEqual(
            r["paired"]["pooled_minus_mode_conditioned"]["n"], 2)
        for metric in r["means"].values():
            self.assertTrue(0 <= metric["future_brier"] <= 1.)
            self.assertTrue(0 <= metric["analytic_expected_brier"] <= 1.)
        self.assertFalse(
            r["paired"]["pooled_minus_mode_conditioned"]["confirmatory"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            PooledG0AEMemory().predict(2, 0)
        with self.assertRaises(ValueError):
            ModeConditionedMemory().observe(0, 2, True)
        with self.assertRaises(ValueError):
            ModeConditionedMemory().observe(1, 0, "changed")
        with self.assertRaises(ValueError):
            training_encounters(-1)
        with self.assertRaises(ValueError):
            run(())


if __name__ == "__main__":
    unittest.main()

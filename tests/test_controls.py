import random
import unittest

from esmf.controls import (
    LocalReadout, READOUT, _diagnostics, observation_gram, paired, sample,
    signed_laplacian,
)


class TestControls(unittest.TestCase):
    def test_paired_reproducibility(self):
        self.assertEqual(paired(4, 80, 3), paired(4, 80, 3))

    def test_matched_contact_update_exploration_and_cost_budgets(self):
        row = paired(4, 80, 3)
        models = row["models"]
        keys = ("active_contacts", "active_contact_exposures", "updates",
                "exploration_events", "candidate_evaluations",
                "logical_operation_units_estimate", "state_scalars",
                "memory_bytes_estimate")
        for key in keys:
            self.assertEqual(len({models[name][key] for name in models}), 1, key)
        self.assertEqual(models["fixed_informative"]["edges"], [[0, READOUT]])
        self.assertEqual(models["fixed_informative"]["active_contact_exposures"], 80)
        self.assertEqual(models["fixed_informative"]["updates"], 80)
        self.assertEqual(models["fixed_informative"]["exploration_events"], 4)

    def test_local_readout_ignores_unconnected_regions(self):
        model = LocalReadout("fixed_informative", 2)
        model.weights[2] = 1.0
        cues = [0] * 12
        before = model.predict(cues, 0)[0]
        cues[7] = 1
        self.assertEqual(model.predict(cues, 0)[0], before)
        cues[0] = 1
        self.assertNotEqual(model.predict(cues, 0)[0], before)

    def test_residual_proposals_use_local_residual_signals(self):
        model = LocalReadout("residual_driven", 3)
        for i in range(20):
            cues = [0] * 12
            cues[0] = i % 2
            cues[READOUT] = (i // 2) % 2
            model.update(cues, int(cues[0] and cues[READOUT]), 0)
        self.assertEqual(model.proposal_events, 1)
        self.assertEqual(model.candidate_evaluations, 11)
        self.assertTrue(model.valid())

    def test_holdout_has_novel_compositions_interventions_and_appearance_strata(self):
        row = paired(1, 40, 2)
        groups = row["models"]["fixed_informative"]["groups"]
        self.assertEqual(len(groups), 12)
        self.assertIn("xor/do_a/appearance-1", groups)
        self.assertEqual(row["holdout_compositions"], ["and", "xor"])
        self.assertEqual(row["holdout_interventions"], ["observe", "do_a", "do_b"])
        self.assertFalse(row["models"]["fixed_informative"]["diagnostics"]["used_for_learning"])
        self.assertFalse(row["z_plus"]["implemented"])

    def test_diagnostics_are_external_and_symmetric(self):
        observations = [[0] * 12, [1] * 12]
        gram = observation_gram(observations)
        laplacian = signed_laplacian((0, 1), observations)
        self.assertEqual(gram, [list(row) for row in zip(*gram)])
        self.assertEqual(laplacian[0][1], laplacian[1][0])
        model = LocalReadout("fixed_informative", 5)
        before = (list(model.weights), set(model.edges), list(model.residual_scores))
        diagnostics = _diagnostics(model, observations)
        after = (list(model.weights), set(model.edges), list(model.residual_scores))
        self.assertEqual(before, after)
        self.assertFalse(diagnostics["used_for_learning"])

    def test_invalid_generation_inputs(self):
        with self.assertRaises(ValueError):
            sample(random.Random(0), "nor", "observe")
        with self.assertRaises(ValueError):
            paired(0, 0, 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import replace

from esmf.embodied_y import (
    HORIZON, MODELS, SCENARIOS, AutonomicGate, CylinderWorld, Observation,
    Scenario, collect_trace, feature_stream, momentum_action_gradient,
    momentum_transition, run_seed, scenario_pair,
)


class TestEmbodiedY(unittest.TestCase):
    def test_observations_are_partial_and_delayed(self):
        world = CylinderWorld(1, Scenario("partial"), "test")
        first = world.observe()
        self.assertEqual(first.angle_tick, 0)
        self.assertEqual(first.momentum_tick, 0)
        self.assertNotIn("angle_state", first.__dataclass_fields__)
        self.assertNotIn("hidden_seam", first.__dataclass_fields__)
        world.step("coast")
        second = world.observe()
        self.assertEqual(second.tick, 1)
        self.assertEqual(second.angle_tick, 0)
        self.assertEqual(second.momentum_tick, 0)
        world.step("coast")
        third = world.observe()
        self.assertEqual(third.tick - third.momentum_tick, 2)

    def test_resource_accounting_and_damage_recovery(self):
        world = CylinderWorld(4, Scenario("resource"), "test")
        world.damage = .4
        before = world.damage
        for action in ("repair", "rest", "right", "left", "rest"):
            world.step(action)
        self.assertLess(world.damage, before)
        self.assertGreaterEqual(world.ledger.reserve, .12)
        self.assertLess(world.ledger_result()["closure_error"], 1e-10)

    def test_autonomic_gate_runs_on_its_meso_clock(self):
        gate = AutonomicGate()
        observation = Observation(0.0, 1.0, 2.0, .0, 0, 1.0,
                                 0, 0, 0, 0)
        self.assertFalse(gate.update(observation, 0.0, 2.0, tick=1))
        self.assertTrue(gate.update(observation, 0.0, 2.0, tick=2))

    def test_all_predictors_have_equal_capacity_and_same_traces(self):
        scenario = Scenario("matched")
        train_a, test_a = scenario_pair(8, scenario)
        train_b, test_b = scenario_pair(8, scenario)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual(len(train_a[0]["actions"]), HORIZON)
        self.assertIn("rest", test_a[0]["actions"])
        self.assertNotEqual(train_a[0]["actions"], test_a[0]["actions"])
        for model in MODELS:
            self.assertEqual(
                len(feature_stream(train_a[0], model, True)[0][0]), 4)

    def test_chart_aware_real_and_geometric_control_tie(self):
        result = run_seed(9, Scenario("chart_tie"))
        by_name = {row["model"]: row for row in result}
        self.assertEqual(by_name["chart_aware"]["weights"],
                         by_name["geometric"]["weights"])
        self.assertEqual(by_name["chart_aware"]["rmse"],
                         by_name["geometric"]["rmse"])
        self.assertEqual(by_name["chart_aware"]["compute_multiply_adds"],
                         by_name["geometric"]["compute_multiply_adds"])

    def test_chart_calibration_is_covariant_to_positive_rescaling(self):
        trace = collect_trace(10, Scenario("chart_covariance"), "test", 0)
        scaled = dict(trace)
        scaled["observations"] = [
            replace(obs, momentum=obs.momentum * 1.7,
                    chart_gain=obs.chart_gain * 1.7)
            for obs in trace["observations"]
        ]
        calibrated = feature_stream(trace, "chart_aware")
        rescaled = feature_stream(scaled, "chart_aware")
        for expected, actual in zip(calibrated, rescaled):
            for expected_value, actual_value in zip(expected[0], actual[0]):
                self.assertAlmostEqual(expected_value, actual_value, places=14)
        self.assertNotEqual(
            [row[0] for row in feature_stream(trace, "recurrent")],
            [row[0] for row in feature_stream(scaled, "recurrent")])

    def test_action_gradient_matches_finite_difference(self):
        values = (.21, -.3, .4, .18)
        eps = 1e-6
        upper = momentum_transition(values[0], values[1],
                                    values[2] + eps, values[3],
                                    "noncommuting")
        lower = momentum_transition(values[0], values[1],
                                    values[2] - eps, values[3],
                                    "noncommuting")
        finite_difference = (upper - lower) / (2 * eps)
        self.assertAlmostEqual(
            finite_difference,
            momentum_action_gradient(values[3], "noncommuting"),
            places=8)

    def test_no_winding_training_leaves_seam_unknown(self):
        scenario = next(s for s in SCENARIOS
                        if s.name == "hidden_seam_change")
        train, _ = scenario_pair(11, scenario)
        self.assertTrue(all(trace["winding"] == 0 for trace in train))
        result = run_seed(11, scenario)
        self.assertTrue(all(row["seam_hypothesis"] == "unknown"
                            for row in result))

    def test_all_control_scenarios_are_resource_bounded(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                result = run_seed(13, scenario)
                self.assertEqual(len(result), len(MODELS))
                self.assertTrue(all(row["max_ledger_closure_error"] < 1e-9
                                    for row in result))

    def test_seed_replay_and_matched_event_accounting(self):
        scenario = Scenario("replay")
        first = collect_trace(12, scenario, "test", 0)
        second = collect_trace(12, scenario, "test", 0)
        self.assertEqual(first, second)
        results = run_seed(12, scenario)
        self.assertEqual({row["test_events"] for row in results},
                         {3 * HORIZON})
        self.assertEqual({row["parameters"] for row in results}, {4})
        self.assertTrue(all(row["max_ledger_closure_error"] < 1e-9
                            for row in results))


if __name__ == "__main__":
    unittest.main()

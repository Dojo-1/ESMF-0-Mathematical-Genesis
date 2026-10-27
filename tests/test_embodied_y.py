import unittest

from esmf.embodied_y import (
    HORIZON, MODELS, SCENARIOS, CylinderWorld, Scenario, collect_trace,
    feature_stream, run_seed, scenario_pair,
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

    def test_all_predictors_have_equal_capacity_and_same_traces(self):
        scenario = Scenario("matched")
        train_a, test_a = scenario_pair(8, scenario)
        train_b, test_b = scenario_pair(8, scenario)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual(len(train_a[0]["actions"]), HORIZON)
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

    def test_no_winding_training_leaves_seam_unknown(self):
        scenario = next(s for s in SCENARIOS
                        if s.name == "hidden_seam_change")
        train, _ = scenario_pair(11, scenario)
        self.assertTrue(all(trace["winding"] == 0 for trace in train))
        result = run_seed(11, scenario)
        self.assertTrue(all(row["seam_hypothesis"] == "unknown"
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

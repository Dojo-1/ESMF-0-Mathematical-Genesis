"""G0-AC invariants; development seeds only, never the sealed held-out range."""
import math
import unittest

from esmf.associated22_u import flatten, neutral_square
from esmf.geometry_k import inverse, matmul
from esmf.reflex_auditor_ac import (
    BURN_IN, COST, Energy, GATE, HORIZON, MatrixWorld, POLICIES, SPEC,
    RecurrentEnsemble, ReflexAuditorGate, SensorAuditor, WORLDS,
    probability, simulate, summary,
)
from research.run_g0_ac_heldout import (
    calibration_key, evaluate, paired_interval, pareto_frontier,
    registration_hash, validate_calibration,
)


class TestReflexAuditorAC(unittest.TestCase):
    def test_full_state_and_left_right_sl2_transport(self):
        world = MatrixWorld(512, "reversible")
        original = world.state
        self.assertEqual(len(set(flatten(original))), 4)
        determinant = neutral_square(flatten(original))
        world.step("forward")
        self.assertNotEqual(world.state, original)
        self.assertAlmostEqual(neutral_square(flatten(world.state)), determinant)
        world.step("inverse")
        for a, b in zip(flatten(original), flatten(world.state)):
            self.assertAlmostEqual(a, b)
        left, right = world.transports("forward", 0)
        self.assertNotEqual(left, right)
        for transport in (left, right):
            product = matmul(transport, inverse(transport))
            self.assertAlmostEqual(product[0][0], 1.)
            self.assertAlmostEqual(product[1][1], 1.)
        other_left, other_right = world.transports("forward", 12)
        self.assertNotEqual(matmul(left, other_left), matmul(other_left, left))
        self.assertNotEqual(matmul(right, other_right), matmul(other_right, right))

    def test_separate_interventions_reset_drift_dissipation(self):
        forward = MatrixWorld(513, "reversible")
        rest = MatrixWorld(513, "reversible")
        forward.step("forward")
        rest.step("rest")
        self.assertNotEqual(forward.state, rest.state)
        reset = MatrixWorld(513, "reset")
        self.assertEqual([t for t in range(HORIZON) if reset.step("rest")], [16, 32])
        drift = MatrixWorld(513, "drift")
        self.assertNotEqual(drift.transports("forward", 0),
                            drift.transports("forward", 30))
        damped = MatrixWorld(513, "dissipation")
        before = sum(x*x for x in flatten(damped.state))
        damped.step("rest")
        self.assertAlmostEqual(sum(x*x for x in flatten(damped.state)), .97**2*before)

    def test_sensor_graph_partial_independent_noise_and_remainders(self):
        world = MatrixWorld(512, "reversible")
        world.step("rest")
        clean, noisy = world.readings(False), world.readings()
        self.assertEqual(set(clean), {0, 1, 2})
        self.assertTrue(all(len(v) == 2 for v in clean.values()))
        noise = [noisy[s][0]-clean[s][0] for s in clean]
        self.assertEqual(len(set(noise)), 3)
        self.assertEqual(noisy, world.readings())
        auditor = SensorAuditor()
        record = auditor.audit(0, noisy)
        self.assertEqual(record["path"], [0, 1, 2, 0])
        self.assertEqual(record["typed_remainder"], "[R]_transport")
        self.assertFalse(record["identifies_full_fiber"])
        self.assertGreaterEqual(record["positive_square"], 0.)
        with self.assertRaises(ValueError):
            auditor.audit(1, {0: noisy[0]})

    def test_energy_finite_nonnegative_and_exact(self):
        for cap in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                Energy(cap)
        energy = Energy(1.)
        self.assertFalse(energy.can_pay(-.1))
        self.assertFalse(energy.can_pay(math.nan))
        self.assertTrue(energy.pay("audit"))
        self.assertFalse(energy.pay("focus"))
        self.assertAlmostEqual(energy.spent, sum(energy.charges.values()))
        self.assertAlmostEqual(energy.spent+energy.remaining, energy.cap)

    def test_gate_hysteresis_credible_consequence_and_budget(self):
        gate, energy = ReflexAuditorGate(), Energy(32.)
        self.assertEqual(gate.choose(0, energy), "focus")
        for _ in range(40):
            gate.evidence(10.)
        self.assertEqual(gate.choose(4, energy), "audit")
        for _ in range(2000):
            gate.evidence(0.)
        self.assertEqual(gate.choose(5, energy), "audit")
        self.assertEqual(gate.choose(4+GATE["minimum_dwell"], energy), "focus")
        self.assertIsNone(gate.choose(10, Energy(.01)))

    def test_wrong_bud_undo_and_reserved_pruning_cost(self):
        learner, energy = RecurrentEnsemble(), Energy(1.)
        saved = [row[:] for row in learner.models[2].weights]
        self.assertTrue(learner.bud(7, 5., energy))
        self.assertEqual(energy.held, COST["prune"])
        learner.models[2].weights[0][-1] = 100.
        features, predictions, _, _ = learner.forecast("rest")
        learner.observe(8, features, predictions, {0: (0., 0.)})
        self.assertLess(learner.child["log_bayes"], GATE["prune_log_bayes"])
        while energy.pay("predict"):
            pass
        self.assertTrue(learner.prune(8, energy))
        self.assertEqual(learner.models[2].weights, saved)
        self.assertIsNone(learner.child)
        self.assertEqual(learner.events[-1]["reason"], "contradicted")
        self.assertIn("prune", energy.charges)
        self.assertEqual(energy.held, 0.)

    def test_bud_expires_and_retains_provenance(self):
        learner, energy = RecurrentEnsemble(), Energy(2.)
        learner.bud(7, 5., energy)
        self.assertFalse(learner.prune(8, energy))
        self.assertTrue(learner.prune(7+GATE["bud_lifetime"], energy))
        self.assertEqual([e["event"] for e in learner.events], ["bud", "prune"])
        self.assertEqual(learner.events[0]["child"], learner.events[1]["child"])

    def test_no_privileged_state_in_learner_interface(self):
        learner = RecurrentEnsemble()
        features, predictions, mean, variance = learner.forecast("forward")
        self.assertEqual(len(features), 10)
        self.assertEqual(len(mean), 6)
        learner.observe(0, features, predictions, {0: (.3, -.7)})
        self.assertEqual(learner.last[:2], [.3, -.7])
        self.assertTrue(all(0 <= p <= 1 for p in (probability(mean[0], variance),)))

    def test_every_policy_budget_scores_abstentions_and_real_equivalence(self):
        for policy in POLICIES:
            row = simulate(512, "reset", policy, cap=.1, trace=True)
            self.assertEqual(row["scored_ticks"], HORIZON-BURN_IN)
            self.assertLessEqual(row["energy"], .1+1e-10)
            self.assertEqual(row["coverage"], 0.)
            self.assertAlmostEqual(row["brier"], .25)
            self.assertEqual(row["actual_joules"], None)
            self.assertEqual(row["capacity"]["recurrent_weights"], 180)
        a = simulate(512, "drift", "adaptive", trace=True)
        b = simulate(512, "drift", "equivalent_real", trace=True)
        for key in ("policy", "wall_seconds", "cpu_seconds"):
            a.pop(key)
            b.pop(key)
        self.assertEqual(a, b)

    def test_world_policy_metrics_and_ablations(self):
        for world in WORLDS:
            row = simulate(513, world, "adaptive", trace=True)
            self.assertTrue(all(math.isfinite(row[k]) for k in ("brier", "logloss", "mse")))
            self.assertAlmostEqual(row["energy"]+row["reserve"], row["cap"])
            self.assertAlmostEqual(row["energy"], sum(row["charges"].values()))
            self.assertGreater(row["operations"], 0)
            stats = summary([row])
            self.assertEqual(stats["episodes"], 1)
        no_slow = simulate(513, "drift", "no_slow")
        no_bud = simulate(513, "reset", "no_budding")
        self.assertEqual(no_slow["slow_updates"], 0)
        self.assertEqual(no_bud["buds"], 0)
        for policy in ("fixed_focus", "adaptive"):
            row = simulate(513, "reversible", policy, cap=12., trace=True)
            self.assertEqual([r["command"] for r in row["trace"]],
                             [("forward", "inverse", "rest")[(t//4) % 3]
                              for t in range(HORIZON)])

    def test_frozen_calibration_validation_and_frontier(self):
        calibration = {
            "registration_sha256": registration_hash(),
            "development_seeds": SPEC["development_seeds"],
            "temperatures": {calibration_key(policy, cap): 1.
                             for policy in POLICIES for cap in SPEC["energy_caps"]}}
        validate_calibration(calibration)
        calibration["temperatures"][calibration_key("adaptive", 12.)] = 3.
        with self.assertRaises(ValueError):
            validate_calibration(calibration)
        cheap = {"brier": .1, "energy": 1., "operations": 10}
        expensive = {"brier": .2, "energy": 2., "operations": 20}
        accurate = {"brier": .05, "energy": 3., "operations": 30}
        self.assertEqual(pareto_frontier([cheap, expensive, accurate]), [cheap, accurate])
        self.assertEqual(paired_interval([0., 0.]), [0., 0.])

    def test_development_replay_all_controls_no_heldout_claim(self):
        calibration = {
            "registration_sha256": registration_hash(),
            "development_seeds": SPEC["development_seeds"],
            "temperatures": {calibration_key(policy, cap): 1.
                             for policy in POLICIES for cap in SPEC["energy_caps"]}}
        result = evaluate(calibration, seeds=(514,))
        self.assertFalse(result["registered_heldout"])
        self.assertTrue(result["equivalent_real_ties"])
        self.assertEqual(len(result["models"]),
                         len(WORLDS)*len(POLICIES)*len(SPEC["energy_caps"]))
        self.assertTrue(result["kill_decision"]["promotion_blocked"])
        self.assertIsNone(result["actual_joules"])


if __name__ == "__main__":
    unittest.main()

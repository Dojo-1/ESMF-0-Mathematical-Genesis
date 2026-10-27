import math
import random
import unittest

from esmf.epistemic_circuits_ab import (
    HORIZON, POLICIES, WORLDS, ContinuityAudit, ReflexCircuit,
    ResourceLedger, benchmark, run, simulate,
)
from esmf.adaptive_connection_ac0 import (
    boost_transport, make_probe, math_checks, predict,
)
from esmf.global_bundle_q import NeutralCylinderBundle, TAU, TWIST
from esmf.geometry_k import inverse, matmul, matvec


class TestEpistemicCircuitsAB(unittest.TestCase):
    def test_audit_is_normalized_and_reset_evidence_is_bayesian(self):
        audit = ContinuityAudit()
        prior_reset_probability = audit.reset_probability
        audit.update((0., 0.), (1., .6), 1., .2)
        stable_probability = audit.weights[0]
        audit.update((4., 4.), (1., .6), 1., .2)
        self.assertGreater(audit.reset_probability, prior_reset_probability)
        self.assertAlmostEqual(sum(audit.weights), 1.)
        self.assertLess(audit.weights[0], stable_probability)
        self.assertEqual(audit.events, 2)

    def test_audit_likelihood_uses_shared_coefficient_covariance(self):
        positive = ContinuityAudit()
        negative = ContinuityAudit()
        positive.update((.8, -.6), (1., 1.), 1., .2)
        negative.update((.8, -.6), (1., -1.), 1., .2)
        self.assertNotEqual(positive.weights, negative.weights)

    def test_reflex_circuit_is_finite_state_and_respects_resource_floor(self):
        circuit = ReflexCircuit()
        ledger = ResourceLedger()
        chosen = []
        for _ in range(HORIZON):
            ledger.replenish()
            axis, focused = circuit.choose("reflex", ledger)
            chosen.append((axis, focused))
        ledger.verify()
        self.assertEqual([axis for axis, _ in chosen], [i % 2 for i in range(HORIZON)])
        self.assertGreaterEqual(ledger.reserve, .25)
        self.assertEqual(circuit.fast_clock, HORIZON)

    def test_large_predictive_error_triggers_audit(self):
        circuit = ReflexCircuit()
        circuit.last_surprise = 3.
        self.assertTrue(circuit.should_audit(
            "reflex", ContinuityAudit(), provenance_known=True))

    def test_partial_sensors_use_full_neutral_bundle_transport(self):
        checks = math_checks()
        self.assertLess(checks["max_metric_compatibility_residual"], 1e-7)
        self.assertLess(checks["max_chart_prediction_gap"], 1e-7)
        self.assertEqual(checks["winding_det_signs"], [-1, -1, -1])

    def test_simulated_probe_applies_global_reflection_seam(self):
        rng = random.Random(887)
        bundle = NeutralCylinderBundle()
        probe = make_probe(rng, 0, bundle, cross_seam=True)
        self.assertLess(probe.theta0, TAU)
        self.assertGreater(probe.theta1, TAU)
        b0, _, _ = bundle.frame(probe.chart0, probe.theta0, probe.z0)
        b1, _, _ = bundle.frame(probe.chart1, probe.theta1, probe.z1)
        source = matvec(inverse(b0), probe.source_local)
        transport = matmul(TWIST, boost_transport(0, .37, probe.h))
        target = matvec(b1, matvec(transport, source))
        expected = tuple(sum(c * x for c, x in zip(sensor, target))
                         for sensor in probe.sensors_local)
        predicted, gradient = predict(probe, .37, bundle, gradient=True)
        for actual, want in zip(predicted, expected):
            self.assertAlmostEqual(actual, want, places=12)
        eps = 1e-6
        hi = predict(probe, .37 + eps, bundle)
        lo = predict(probe, .37 - eps, bundle)
        for j in range(2):
            self.assertAlmostEqual(
                gradient[j], (hi[j] - lo[j]) / (2 * eps), places=7)

    def test_reflex_audits_only_when_triggered_and_respects_ledger(self):
        result = simulate(8, "nonlinear_drift", "reflex")
        self.assertGreater(result["audit_calls"], 0)
        self.assertLess(result["audit_calls"], HORIZON)
        self.assertGreaterEqual(result["resource_reserve"], .25 - 1e-10)
        self.assertAlmostEqual(
            8. + result["resource_delivered"],
            result["resource_spent"] + result["resource_overflow"] +
            result["resource_reserve"], places=9)
        self.assertEqual(result["sensor_delay_ticks"], 1)
        self.assertEqual(result["simulation_ticks"], HORIZON + 1)
        self.assertGreater(result["seam_crossings"], 0)
        self.assertTrue(math.isfinite(result["heldout_probe_mse"]))

    def test_audit_every_event_and_fixed_focus_are_separate_controls(self):
        audit_every = simulate(8, "abrupt_step", "gaussian_state_space")
        focused = simulate(8, "abrupt_step", "fixed_focus")
        reflex = simulate(8, "abrupt_step", "reflex")
        self.assertEqual(audit_every["audit_calls"], HORIZON)
        self.assertEqual(focused["focus_probes"], HORIZON)
        self.assertLess(reflex["audit_calls"], audit_every["audit_calls"])

    def test_matched_controls_replay_and_measure_cost_without_claiming_joules(self):
        for world in ("static", "abrupt_step", "nonlinear_drift"):
            for policy in POLICIES:
                first = simulate(3, world, policy)
                self.assertEqual(first, simulate(3, world, policy))
                self.assertGreaterEqual(first["resource_spent"], 0.)
        report = benchmark(3, "static", "reflex", repeats=1)
        self.assertGreaterEqual(report["median_wall_seconds"], 0.)
        self.assertGreater(report["peak_traced_bytes"], 0)
        self.assertIsNone(report["joules"])

    def test_comparison_runner_reports_all_controls_and_resource_metrics(self):
        results = run(seeds=1, worlds=("static",))
        self.assertEqual({row["policy"] for row in results}, set(POLICIES))
        self.assertTrue(all(row["episodes"] == 1 for row in results))
        self.assertTrue(all(row["joules"] is None for row in results))
        self.assertTrue(all(row["peak_traced_bytes"] > 0 for row in results))

    def test_all_declared_worlds_have_valid_transport_metrics(self):
        for world in WORLDS:
            result = simulate(5, world, "fixed_focus")
            self.assertTrue(math.isfinite(result["heldout_probe_mse"]))
            self.assertEqual(result["policy"], "fixed_focus")


if __name__ == "__main__":
    unittest.main()

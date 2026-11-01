"""G0-AG D0: rate-control experiment contracts, NOT held-out task scores."""
import math
import unittest

from esmf.glial_quaternion_ag import GlialQuaternionEcology, qchain_discrepancy
from research.g0_ag_development_probe import (
    ARMS, METRICS, challenge_state, control_events, descriptive_interval,
    run, simulate,
)


def event(events, name):
    matches = [x for x in events if x.name == name]
    if len(matches) != 1:
        raise AssertionError("missing or ambiguous event " + name)
    return matches[0]


class TestG0AGDevelopmentProbe(unittest.TestCase):
    def setUp(self):
        self.m = GlialQuaternionEcology()
        self.challenge = challenge_state(self.m)

    def test_registered_challenge_is_reachable_viable_and_active(self):
        s = self.challenge
        self.assertTrue(self.m.valid(s))
        self.assertEqual(self.m._energy_balance(s), 0)
        self.assertEqual(self.m.af.diagnostics(s.tissue)[
            "material_balance_residual"], 0)
        self.assertTrue(s.glia.tripartite)
        self.assertTrue(s.tissue.links)
        self.assertGreater(qchain_discrepancy(s.fibers[0]), 0.)
        self.assertGreaterEqual(s.tissue.cells[0].atp, 1)
        self.assertGreaterEqual(s.tissue.cells[0].material, 1)

    def test_arms_only_change_rates_never_destinations_or_observations(self):
        full = control_events(self.m, self.challenge, "full")
        signature = tuple((e.name, e.destination) for e in full)
        for arm in ARMS:
            self.assertEqual(signature, tuple(
                (e.name, e.destination)
                for e in control_events(self.m, self.challenge, arm)
            ))
        # This is an analytic known-answer construction, not a discovered
        # performance effect.
        gap = qchain_discrepancy(self.challenge.fibers[0])
        f = event(full, "af/neurite_extend:0:0")
        ablated = event(control_events(
            self.m, self.challenge, "no_geometry_feedback"),
            "af/neurite_extend:0:0")
        self.assertAlmostEqual(
            ablated.rate / f.rate, 1. + .25 * gap, places=12)
        contact = event(full, "af/contact_release:0")
        other = event(control_events(
            self.m, self.challenge, "no_glial_rate_feedback"),
            "af/contact_release:0")
        self.assertAlmostEqual(other.rate / contact.rate, 4., places=12)
        sensory = event(full, "af/cognitive/encounter:0")
        other_sensory = event(control_events(
            self.m, self.challenge, "no_glial_rate_feedback"),
            "af/cognitive/encounter:0")
        self.assertAlmostEqual(
            sensory.rate / other_sensory.rate, 1.20, places=12)
        self.assertEqual(control_events(self.m, self.challenge, "full"),
                         control_events(self.m, self.challenge, "placebo"))

    def test_seeded_traces_are_deterministic_and_placebo_exact(self):
        for seed in (8100, 8101, 8102):
            a = simulate(self.m, self.challenge, "full", seed, horizon=.16)
            b = simulate(self.m, self.challenge, "placebo", seed, horizon=.16)
            for key in a:
                if key != "arm":
                    self.assertEqual(a[key], b[key])
            self.assertEqual(a, simulate(
                self.m, self.challenge, "full", seed, horizon=.16))
            self.assertTrue(a["state_valid"])
            self.assertEqual(a["energy_residual"], 0)
            self.assertEqual(a["material_residual"], 0)
            self.assertGreaterEqual(a["contact_exposure"], 0.)
            self.assertLessEqual(a["tripartite_exposure"], .16 + 1e-12)

    def test_report_contains_all_raw_and_paired_metrics(self):
        report = run(seeds=range(8100, 8102), horizon=.12)
        self.assertEqual(report["settings"]["n_seeds"], 2)
        self.assertEqual(len(report["raw"]), 2 * len(ARMS))
        self.assertEqual(set(report["arm_means"]), set(ARMS))
        self.assertEqual(set(report["paired_full_minus_ablation"]),
                         {"no_geometry_feedback", "no_glial_rate_feedback",
                          "both_disabled"})
        for arm in ARMS:
            for metric in METRICS:
                self.assertTrue(math.isfinite(
                    report["arm_means"][arm][metric]))
        self.assertTrue(all(report["check"].values()))
        self.assertIn("not G0-AG1", report["scope"])
        self.assertIn("No independent task utility",
                      "; ".join(report["unresolved"]))

    def test_invalid_configuration_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            control_events(self.m, self.challenge, "no_glia_MAGIC")
        with self.assertRaises(ValueError):
            simulate(self.m, self.challenge, "full", -3)
        with self.assertRaises(ValueError):
            simulate(self.m, self.challenge, "full", 8100, horizon=0.)
        with self.assertRaises(ValueError):
            simulate(self.m, self.challenge, "full", 8100, max_events=0)
        with self.assertRaises(ValueError):
            run(seeds=(), horizon=.25)
        with self.assertRaises(ValueError):
            descriptive_interval([])


if __name__ == "__main__":
    unittest.main()

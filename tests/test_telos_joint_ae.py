"""G0-AE contract tests; no hidden semantic or biological performance claims."""
from dataclasses import replace
import math
import unittest

from esmf.geometry_k import identity
from esmf.telos_joint_ae import (
    Channel, Distinction, EncounterView, EncounterWorld, I4,
    JointGenesis, Organism, example_world, selfcheck,
)


def rates_of(model, state):
    return {event.name: event.rate for event in model.channels(state)}


def get(model, state, name):
    matches = [c for c in model.channels(state) if c.name == name]
    if len(matches) != 1:
        raise AssertionError("expected exactly one " + name)
    return matches[0]


class TestTelosJointAE(unittest.TestCase):
    def setUp(self):
        self.m = JointGenesis(example_world())
        self.start = Organism.seed(self.m.world)

    def test_smoke_and_shared_state_generator(self):
        smoke = selfcheck()
        self.assertEqual(smoke["evidence"], "contract-only")
        self.assertFalse(smoke["semantic_emergence_proved"])
        self.assertTrue(self.m.viable(self.start))
        channels = self.m.channels(self.start)
        self.assertTrue(all(isinstance(c, Channel) and
                            c.rate > 0 and self.m.viable(c.destination)
                            for c in channels))
        self.assertEqual(self.m.row(self.start),
                         self.m.row(self.start, reverse=True))
        self.assertAlmostEqual(
            sum(self.m.row(self.start).values()) +
            self.m.diagonal(self.start), 0., places=12)
        self.assertGreater(self.m.total_rate_bound(),
                           -self.m.diagonal(self.start))

    def test_environment_validation_and_capacity(self):
        with self.assertRaises(ValueError):
            EncounterView("A", "B", (1.,), (1., 2., 3., 4.), 1.)
        with self.assertRaises(ValueError):
            EncounterView("A", "B", (1., 2., 3., 4.),
                          (1., 2., 3., 4.), -1.)
        with self.assertRaises(ValueError):
            EncounterWorld(("one",), (((
                EncounterView("x", "x", (1., 2., 3., 4.),
                              (1., 2., 3., 4.), .7),), (
                EncounterView("x", "x", (1., 2., 3., 4.),
                              (1., 2., 3., 4.), 1.),)),))
        with self.assertRaises(ValueError):
            JointGenesis(self.m.world, resource_cap=2)
        with self.assertRaises(ValueError):
            JointGenesis(self.m.world, witness_cap=1)

    def test_probe_questions_affect_observations_and_cost_resource(self):
        q0 = self.m.current_question(self.start)
        self.assertEqual((q0.mode, q0.relation), ("whether", "concordance"))
        ask = get(self.m, self.start, "interrogate")
        q1 = self.m.current_question(ask.destination)
        self.assertEqual((q1.mode, q1.relation), ("why", "contrariety"))
        self.assertEqual(ask.destination.resource, self.start.resource - 1)
        self.assertEqual(
            [o.probability for o in self.m.world.probes[0][0]],
            [.85, .15])
        self.assertEqual(
            [o.probability for o in self.m.world.probes[0][1]],
            [.55, .45])
        encountered = get(self.m, self.start, "encounter:0").destination
        self.assertEqual(encountered.distinctions[0].seen_different, 1)
        self.assertEqual(encountered.distinctions[1], Distinction())
        self.assertEqual(encountered.last, (0, 0, 0))
        self.assertEqual(encountered.resource, self.start.resource - 1)
        self.assertTrue(self.m.observed_pair(encountered).before == "six")

    def test_distinctions_survive_belief_dissipation(self):
        scene = get(self.m, self.start, "encounter:0").destination
        self.assertEqual(scene.distinctions[0].live_different, 1)
        faded = get(self.m, scene, "dissipation:0").destination
        self.assertEqual(faded.distinctions[0].live_different, 0)
        self.assertEqual(faded.distinctions[0].seen_different, 1)
        self.assertEqual(faded.distinctions[0].seen_same, 0)
        self.assertEqual(faded.resource, scene.resource)

    def test_geometry_is_observation_gated_not_oracle(self):
        self.assertIsNone(self.m.geometric_defect(self.start))
        view = replace(self.start, last=(0, 0, 0))
        self.assertAlmostEqual(self.m.geometric_defect(view), 2., places=12)
        self.assertAlmostEqual(
            self.m.geometric_defect(view, alternative=1), 0., places=12)
        self.assertAlmostEqual(self.m.neutral_diagnostic(view), 0., places=12)
        repair = get(self.m, view, "transport_repair")
        self.assertEqual(repair.destination.geometry[0], 1)
        self.assertEqual(repair.destination.resource, view.resource - 1)
        self.assertNotIn("transport_repair",
                         rates_of(self.m, self.start))
        self.assertNotIn("transport_repair",
                         rates_of(self.m, repair.destination))

    def test_nonorthogonal_frame_covariance(self):
        bs = ((2., .3, 0., 0.), (0., 1., 0., 0.),
              (0., 0., .5, 0.), (0., 0., 0., 1.))
        bt = ((1., 0., 0., 0.), (.5, 1., 0., 0.),
              (0., 0., 1.5, .2), (0., 0., 0., 1.))
        framed = JointGenesis(example_world(bs, bt))
        s = replace(self.start, last=(0, 0, 0))
        for candidate in (0, 1):
            self.assertAlmostEqual(
                framed.geometric_defect(s, candidate),
                self.m.geometric_defect(s, candidate), places=10)
        self.assertAlmostEqual(
            get(framed, s, "transport_repair").rate,
            get(self.m, s, "transport_repair").rate, places=10)

    def test_budding_requires_combined_evidence_and_paid_resources(self):
        observed = replace(self.start,
                           distinctions=(Distinction(2, 2, 2, 2), Distinction()),
                           last=(0, 0, 0), resource=5, material=3)
        bud = get(self.m, observed, "worldlet_bud").destination
        self.assertTrue(bud.buds[0])
        self.assertEqual(bud.resource, 3)
        self.assertEqual(bud.material, 1)
        self.assertEqual(bud.distinctions, observed.distinctions)
        self.assertNotIn("worldlet_bud",
                         rates_of(self.m, replace(observed,
                             distinctions=(Distinction(1, 1, 1, 1), Distinction()))))
        self.assertNotIn("worldlet_bud",
                         rates_of(self.m, replace(observed, material=1)))
        self.assertNotIn("worldlet_bud",
                         rates_of(self.m, replace(
                             observed, last=None)))
        self.assertNotIn("worldlet_bud",
                         rates_of(self.m, replace(
                             observed, distinctions=(
                                 Distinction(2, 0, 2, 0), Distinction()))))
        wilted = replace(bud, resource=1)
        pruned = get(self.m, wilted, "worldlet_prune:0").destination
        self.assertFalse(pruned.buds[0])
        self.assertEqual(pruned.material, 2)
        self.assertEqual(pruned.resource, 1)
        self.assertEqual(pruned.distinctions, observed.distinctions)

    def test_biology_and_questions_causally_couple(self):
        full = replace(self.start, resource=5)
        hungry = replace(self.start, resource=0)
        self.assertIn("encounter:0", rates_of(self.m, full))
        self.assertNotIn("encounter:0", rates_of(self.m, hungry))
        self.assertNotIn("interrogate", rates_of(self.m, hungry))
        conflict = replace(full,
                           distinctions=(Distinction(1, 1, 1, 1), Distinction()))
        self.assertGreater(
            get(self.m, conflict, "interrogate").rate,
            get(self.m, full, "interrogate").rate)
        bud = replace(full, buds=(True, False))
        self.assertGreater(
            get(self.m, bud, "encounter:0").rate,
            get(self.m, full, "encounter:0").rate)
        self.assertGreater(
            get(self.m, bud, "maintenance").rate,
            get(self.m, full, "maintenance").rate)

    def test_trajectory_finite_no_inviable_states(self):
        final, trace = self.m.trajectory(self.start, 5., seed=371,
                                         max_events=110)
        self.assertTrue(self.m.viable(final))
        self.assertTrue(trace)
        self.assertTrue(all(self.m.viable(state) and t >= 0.
                            for t, _, state in trace))
        self.assertTrue(all(trace[i][0] < trace[i + 1][0]
                            for i in range(len(trace) - 1)))
        self.assertTrue(all(t <= 5. for t, _, _ in trace))

    def test_uniformization_positive_with_explicit_tail(self):
        result, tail = self.m.distribution(
            self.start, .012, tolerance=1e-8, max_terms=30,
            max_states=30000)
        self.assertTrue(all(p >= 0 and math.isfinite(p)
                            for p in result.values()))
        self.assertLess(tail, 1e-8)
        self.assertAlmostEqual(sum(result.values()) + tail, 1., places=9)
        self.assertGreater(result.get(self.start, 0.), 0.)
        self.assertGreater(len(result), 1)
        one_step = self.m.uniformization_step({self.start: 1.})
        self.assertAlmostEqual(sum(one_step.values()), 1., places=12)
        self.assertTrue(all(p >= 0 for p in one_step.values()))

    def test_classical_generator_has_real_causal_order_sensitivity(self):
        # At zero available resource, encounter is disabled; feeding first
        # enables it. This bracket is NOT a quantum commutator.
        s = replace(self.start, resource=0)
        witness = lambda x: x.distinctions[0].seen_different
        gap = self.m.operator_bracket(
            s, "energy_supply", "encounter", witness)
        self.assertAlmostEqual(gap, .70 * .55 * .85, places=12)
        self.assertAlmostEqual(
            self.m.operator_bracket(s, "encounter", "energy_supply", witness),
            -gap, places=12)
        self.assertGreater(gap, 0.)

    def test_invalid_state_is_never_repaired_by_silent_clipping(self):
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.start, resource=-1))
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.start, geometry=(3, 0)))
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.start, last=(0, 1, 99)))
        with self.assertRaises(ValueError):
            self.m.trajectory(self.start, -1.)
        with self.assertRaises(ValueError):
            self.m.distribution(self.start, -1.)
        self.assertTrue(self.m.viable(replace(self.start, resource=0)))
        self.assertEqual(identity(4), I4)


if __name__ == "__main__":
    unittest.main()

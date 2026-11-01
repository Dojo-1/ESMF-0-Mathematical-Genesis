"""G0-AG: tests of glial resource coupling and continuous neurite connections.

These check deterministic finite mathematical contracts, NOT biological
validity or task-level predictive superiority.
"""
from dataclasses import replace
import math
import unittest

from esmf.glial_quaternion_ag import (
    ONE, GlialQuaternionEcology,
    observation_quaternion, reframe_sections,
    qmult, qconj, qnorm, qlength2, qerror, qdirection,
    qreframe, qinterpolate, qtransport_chain, qchain_discrepancy, selfcheck,
)


def real_four_coordinate_hamilton(a, b):
    """Independent R4 left-action matrix: a control, not quaternion magic."""
    w, x, y, z = a
    matrix = (
        (w, -x, -y, -z),
        (x, w, -z, y),
        (y, z, w, -x),
        (z, -y, x, w),
    )
    return tuple(sum(row[k] * b[k] for k in range(4)) for row in matrix)


def find(model, state, name):
    choices = [e for e in model.channels(state) if e.name == name]
    if len(choices) != 1:
        raise AssertionError("missing/nonunique channel " + name)
    return choices[0]


def names(model, state):
    return {e.name for e in model.channels(state)}


class TestGlialQuaternionAG(unittest.TestCase):
    def setUp(self):
        self.m = GlialQuaternionEcology()
        self.seed = self.m.seed()

    def test_smoke_joint_generator_and_local_carriers(self):
        self.assertEqual(selfcheck()["claims"], "bounded contracts only")
        self.assertTrue(self.m.valid(self.seed))
        self.assertEqual(self.m.summary(self.seed)["carrier_real_dimensions"], (8, 8))
        self.assertEqual(self.m._energy_balance(self.seed), 0)
        self.assertEqual(self.m.row(self.seed), self.m.row(self.seed, reverse=True))
        self.assertAlmostEqual(
            self.m.diagonal(self.seed)
            + sum(self.m.row(self.seed).values()), 0., places=12)
        self.assertTrue(all(self.m.valid(e.destination) and e.rate > 0.
                            for e in self.m.channels(self.seed)))

    def test_continuous_quaternion_algebra_not_just_q8(self):
        i = (0., 1., 0., 0.)
        j = (0., 0., 1., 0.)
        k = (0., 0., 0., 1.)
        self.assertEqual(qmult(i, j), k)
        self.assertEqual(qmult(j, i), tuple(-v for v in k))
        a = qdirection((1, 0, 0), .40)
        b = qdirection((0, 1, 0), .63)
        c = qdirection((0, 0, 1), .27)
        self.assertAlmostEqual(qlength2(a), 1., places=12)
        self.assertAlmostEqual(qlength2(qmult(a, b)), 1., places=12)
        self.assertGreater(abs(a[0]), .9)
        self.assertLess(a[0], 1.)
        for x, y in zip(qmult(qmult(a, b), c),
                        qmult(a, qmult(b, c))):
            self.assertAlmostEqual(x, y, places=13)
        result = qmult(a, qconj(a))
        for x, y in zip(result, ONE):
            self.assertAlmostEqual(x, y, places=12)

    def test_independent_four_real_hamilton_reference(self):
        # An algebraically equivalent real-coordinate comparator should
        # reproduce quaternion multiplication, not show a mythical gain.
        axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, -1))
        for i, direction in enumerate(axes):
            a = qdirection(direction, .17 + i * .32)
            for j, other_direction in enumerate(axes):
                b = qdirection(other_direction, .31 + j * .27)
                expected = real_four_coordinate_hamilton(a, b)
                observed = qmult(a, b)
                for x, y in zip(observed, expected):
                    self.assertAlmostEqual(x, y, places=12)

    def test_ordered_connection_composition_retains_gauge_structure(self):
        i = (0., 1., 0., 0.)
        j = (0., 0., 1., 0.)
        k = (0., 0., 0., 1.)
        self.assertEqual(qtransport_chain((i, j), ONE),
                         tuple(-x for x in k))
        self.assertEqual(qtransport_chain((j, i), ONE), k)
        a = qdirection((1, 0, 1), .6)
        b = qdirection((0, 1, 1), .5)
        g0 = qdirection((1, 0, 0), .4)
        g1 = qdirection((0, 1, 0), .7)
        g2 = qdirection((0, 0, 1), .9)
        base = qtransport_chain((a, b), ONE)
        a_new = qmult(qmult(g1, a), qconj(g0))
        b_new = qmult(qmult(g2, b), qconj(g1))
        lifted = qtransport_chain((a_new, b_new), qmult(g0, ONE))
        for x, y in zip(lifted, qmult(g2, base)):
            self.assertAlmostEqual(x, y, places=12)

    def test_paid_neurite_growth_covaries_with_motor_chart(self):
        # This fails if spatial qdirection is applied as a fixed left
        # quaternion independently of the neurite's observer/motor frame.
        gauge = qdirection((0, 1, 0), .71)
        transformed = reframe_sections(self.seed.fibers[0], (gauge,))
        s2 = replace(self.seed, fibers=(transformed, self.seed.fibers[1]))
        self.assertTrue(self.m.valid(s2))
        base = find(self.m, self.seed, "af/neurite_extend:0:0")
        changed = find(self.m, s2, "af/neurite_extend:0:0")
        self.assertAlmostEqual(base.rate, changed.rate, places=12)
        a = base.destination.fibers[0]
        b = changed.destination.fibers[0]
        self.assertEqual(a.real_carrier_dimension, b.real_carrier_dimension)
        self.assertEqual(base.destination.tissue, changed.destination.tissue)
        for x, y in zip(b.phases[-1], qmult(gauge, a.phases[-1])):
            self.assertAlmostEqual(x, y, places=12)
        expected_connection = qmult(qmult(gauge, a.forward[-1]),
                                    qconj(gauge))
        for x, y in zip(b.forward[-1], expected_connection):
            self.assertAlmostEqual(x, y, places=12)
        self.assertAlmostEqual(qchain_discrepancy(a),
                               qchain_discrepancy(b), places=12)

    def test_full_observation_plasticity_respects_active_local_frames(self):
        # Independent frame changes must co-transform *the learner's* sensor
        # chart, not merely its post-hoc qerror metric.
        s = find(self.m, self.seed, "af/neurite_extend:0:0").destination
        s = find(self.m, s, "af/cognitive/encounter:0").destination
        s = find(self.m, s, "af/metabolic_conversion:0").destination
        f = s.fibers[0]
        g0 = qdirection((1, 0, 0), .63)
        g1 = qdirection((0, 1, 0), .78)
        rotated = reframe_sections(f, (g0, g1))
        reframed = replace(s, fibers=(rotated, s.fibers[1]))
        self.assertTrue(self.m.valid(reframed))
        self.assertAlmostEqual(qchain_discrepancy(f),
                               qchain_discrepancy(rotated), places=11)
        a = find(self.m, s, "fiber_experience_update")
        b = find(self.m, reframed, "fiber_experience_update")
        self.assertAlmostEqual(a.rate, b.rate, places=12)
        new_a = a.destination.fibers[0]
        new_b = b.destination.fibers[0]
        for x, y in zip(new_b.phases[-1], qmult(g1, new_a.phases[-1])):
            self.assertAlmostEqual(x, y, places=11)
        self.assertAlmostEqual(qchain_discrepancy(new_a),
                               qchain_discrepancy(new_b), places=11)
        self.assertEqual(b.remainder[0], "[R]_memory_prior")
        a_repaid = find(self.m, a.destination,
                        "af/metabolic_conversion:0").destination
        b_repaid = find(self.m, b.destination,
                        "af/metabolic_conversion:0").destination
        rep_a = find(self.m, a_repaid, "fiber_connection_repair")
        rep_b = find(self.m, b_repaid, "fiber_connection_repair")
        self.assertAlmostEqual(rep_a.rate, rep_b.rate, places=12)
        Ua = rep_a.destination.fibers[0].forward[0]
        Ub = rep_b.destination.fibers[0].forward[0]
        expected = qmult(qmult(g1, Ua), qconj(g0))
        for x, y in zip(Ub, expected):
            self.assertAlmostEqual(x, y, places=11)

    def test_zero_observation_abstains_without_crashing_joint_generator(self):
        # The accepted environment can present a zero R4 vector; this
        # cannot be normalized into a point of S3. Do not invent its phase.
        self.assertIsNone(observation_quaternion((0., 0., 0., 0.)))
        self.assertIsNone(observation_quaternion((1e308, 1e308, 1e308, 1e308)))
        with self.assertRaises(ValueError):
            qnorm((1e308, 1e308, 1e308, 1e308))
        self.assertAlmostEqual(qlength2(qnorm((1e300, 1e300, 0., 0.))),
                               1., places=12)
        old_world = self.m.af.core.world
        old_pair = old_world.probes[0][0]
        replacement = replace(old_pair[0], target=(0., 0., 0., 0.))
        probes = (( (replacement,) + old_pair[1:], old_world.probes[0][1]),
                  old_world.probes[1])
        self.m.af.core.world = replace(old_world, probes=probes)
        encountered = find(self.m, self.seed,
                           "af/cognitive/encounter:0").destination
        self.assertTrue(self.m.valid(encountered))
        self.assertNotIn("fiber_experience_update", names(self.m, encountered))
        self.assertNotIn("fiber_connection_repair", names(self.m, encountered))
        self.assertTrue(self.m.summary(encountered)[
            "unrepresentable_active_observation"])
        self.assertEqual(self.m.summary(encountered)[
            "R_transport_observation_norm"], 0.)

        # Normalizing R4 as H is intentionally non-injective: amplitude
        # distinctions must not be silently promoted to equivalence.
        a = observation_quaternion((1., 2., 3., 4.))
        b = observation_quaternion((3., 6., 9., 12.))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)
        self.assertNotEqual((1., 2., 3., 4.), (3., 6., 9., 12.))

    def test_independent_unit_quaternion_frame_covariance(self):
        source = qdirection((1, 0, 0), .62)
        target = qdirection((0, 0, 1), .48)
        connection = qdirection((0, 1, 0), .91)
        gs = qdirection((1, 1, 0), 1.1)
        gt = qdirection((0, 1, 1), .8)
        transformed = qreframe(source, target, connection, gs, gt)
        self.assertAlmostEqual(
            qerror(source, target, connection),
            qerror(*transformed), places=12)
        self.assertAlmostEqual(
            qlength2(transformed[-1]), 1., places=12)
        u = qinterpolate(source, target, .28)
        framed_u = qinterpolate(qmult(gs, source), qmult(gs, target), .28)
        for x, y in zip(qmult(gs, u), framed_u):
            self.assertAlmostEqual(x, y, places=12)
        with self.assertRaises(ValueError):
            qnorm((0., 0., 0., 0.))
        with self.assertRaises(ValueError):
            qinterpolate(ONE, tuple(-x for x in ONE), .5)
        with self.assertRaises(ValueError):
            qdirection((0, 0, 0))

    def test_physical_growth_extends_rank_and_adds_connection(self):
        grow = find(self.m, self.seed, "af/neurite_extend:0:0")
        s = grow.destination
        self.assertTrue(self.m.valid(s))
        self.assertEqual(s.tissue.cells[0].path, ((1, 0, 0),))
        self.assertEqual(len(s.fibers[0].phases), 2)
        self.assertEqual(len(s.fibers[0].forward), 1)
        self.assertEqual(s.fibers[0].real_carrier_dimension, 12)
        self.assertEqual(s.fibers[1].real_carrier_dimension, 8)
        self.assertAlmostEqual(qerror(
            s.fibers[0].phases[0], s.fibers[0].phases[1],
            s.fibers[0].forward[0]), 0., places=12)
        self.assertLess(s.fibers[0].phases[1][0], 1.)
        self.assertEqual(self.m._energy_balance(s), 0)

    def test_pruning_records_full_continuous_lost_section(self):
        grown = find(self.m, self.seed, "af/neurite_extend:0:0").destination
        result = find(self.m, grown, "af/neurite_prune:0")
        s = result.destination
        self.assertTrue(self.m.valid(s))
        self.assertEqual(len(s.fibers[0].phases), 1)
        self.assertEqual(len(s.fibers[0].forward), 0)
        self.assertEqual(len(s.fiber_scars), 1)
        scar = s.fiber_scars[0]
        self.assertEqual((scar[0], scar[1]), (0, (1, 0, 0)))
        self.assertEqual(scar[2], grown.fibers[0].phases[-1])
        self.assertEqual(scar[3], grown.fibers[0].forward[-1])
        self.assertEqual(result.remainder[0], "[R]_transport_continuous")
        capped = GlialQuaternionEcology(max_fiber_records=0)
        cut_target = find(capped, capped.seed(),
                          "af/neurite_extend:0:0").destination
        self.assertNotIn("af/neurite_prune:0", names(capped, cut_target))

    def test_glia_has_separate_closed_metabolism(self):
        s1 = find(self.m, self.seed, "glia_supply").destination
        self.assertEqual(s1.glial_input, 1)
        self.assertEqual(s1.glia.substrate, 3)
        self.assertEqual(self.m._energy_balance(s1), 0)
        upkeep = find(self.m, self.seed, "glia_maintenance").destination
        self.assertEqual(upkeep.glia.substrate, 1)
        self.assertEqual(upkeep.glial_waste, 1)
        self.assertEqual(self.m._energy_balance(upkeep), 0)
        s2 = find(self.m, self.seed, "glia_convert").destination
        self.assertEqual((s2.glia.substrate, s2.glia.transferable), (1, 2))
        self.assertEqual(self.m._energy_balance(s2), 0)
        recipient = find(self.m, self.seed, "glia_transfer:0").destination
        self.assertEqual(recipient.glia.transferable, 0)
        self.assertEqual(recipient.tissue.cells[0].substrate, 3)
        self.assertEqual(recipient.tissue.nutrient_input, 1)
        self.assertEqual(recipient.glial_to_tissue, 1)
        self.assertEqual(recipient.glial_input, 0)
        self.assertEqual(self.m._energy_balance(recipient), 0)
        self.assertTrue(self.m.af.valid(recipient.tissue))

    def test_glial_calcium_requires_local_activity_not_an_oracle(self):
        self.assertNotIn("glia_calcium_pulse", names(self.m, self.seed))
        engaged = find(self.m, self.seed,
                       "af/cognitive/encounter:0").destination
        self.assertIn("glia_calcium_pulse", names(self.m, engaged))
        pulsed = find(self.m, engaged, "glia_calcium_pulse").destination
        self.assertEqual(pulsed.glia.calcium, 1)
        self.assertEqual(pulsed.glial_waste, 1)
        self.assertEqual(pulsed.glia.substrate, 1)
        self.assertEqual(self.m._energy_balance(pulsed), 0)
        relaxed = find(self.m, pulsed, "glia_calcium_relax").destination
        self.assertEqual(relaxed.glia.calcium, 0)
        self.assertEqual(self.m._energy_balance(relaxed), 0)
        # Persistent last-event provenance is not itself active calcium drive.
        faded = find(self.m, engaged,
                     "af/cognitive/dissipation:0").destination
        self.assertIsNotNone(faded.tissue.cognitive.last)
        self.assertNotIn("glia_calcium_pulse", names(self.m, faded))

    def test_glial_secretion_is_local_and_resource_accounted(self):
        seen = find(self.m, self.seed, "af/cognitive/encounter:0").destination
        activated = find(self.m, seen, "glia_calcium_pulse").destination
        secreted = find(self.m, activated, "glia_local_signal").destination
        loc = self.m.af.index(self.m.glial_origin)
        self.assertEqual(secreted.tissue.attract[loc], 1)
        self.assertEqual(secreted.glia.transferable, 0)
        self.assertEqual(secreted.glial_to_tissue, 1)
        self.assertEqual(secreted.tissue.nutrient_input, 1)
        self.assertEqual(self.m._energy_balance(secreted), 0)

    def test_tripartite_support_modulates_real_contact_and_releases(self):
        grown = find(self.m, self.seed,
                     "af/neurite_extend:0:0").destination
        contacted = find(self.m, grown, "af/contact_birth:0").destination
        self.assertEqual(contacted.tissue.links, ((0, 1),))
        self.assertNotIn("tripartite_bind", names(self.m, contacted))
        stimulated = find(self.m, contacted, "glia_calcium_pulse").destination
        before_rate = find(self.m, stimulated,
                           "af/contact_release:0").rate
        bound = find(self.m, stimulated, "tripartite_bind").destination
        self.assertTrue(bound.glia.tripartite)
        self.assertEqual(bound.glial_waste, 2)
        self.assertEqual(self.m._energy_balance(bound), 0)
        after_rate = find(self.m, bound,
                          "af/contact_release:0").rate
        self.assertAlmostEqual(after_rate / before_rate, .25)
        lost = find(self.m, bound, "af/contact_release:0")
        self.assertFalse(lost.destination.glia.tripartite)
        self.assertEqual(lost.remainder[0], "[R]_attr_glial_support")
        self.assertTrue(self.m.valid(lost.destination))

    def test_astrocyte_not_assumed_essential_for_all_events(self):
        depleted_g = replace(self.seed.glia, substrate=0, transferable=0)
        s = replace(self.seed, glia=depleted_g, glial_waste=3)
        self.assertTrue(self.m.valid(s))
        self.assertNotIn("glia_transfer:0", names(self.m, s))
        self.assertIn("af/cognitive/encounter:0", names(self.m, s))
        self.assertIn("af/neurite_extend:0:0", names(self.m, s))

    def test_experience_update_and_transport_repair_are_distinct(self):
        s = find(self.m, self.seed, "af/neurite_extend:0:0").destination
        s = find(self.m, s, "af/cognitive/encounter:0").destination
        self.assertEqual(s.tissue.cells[0].atp, 0)
        self.assertNotIn("fiber_experience_update", names(self.m, s))
        s = find(self.m, s, "af/metabolic_conversion:0").destination
        update = find(self.m, s, "fiber_experience_update")
        s = update.destination
        f = s.fibers[0]
        self.assertAlmostEqual(qlength2(f.phases[-1]), 1., places=12)
        self.assertGreater(qerror(f.phases[0], f.phases[-1],
                                  f.forward[-1]), 1e-8)
        self.assertEqual(len(s.repairs), 0)
        self.assertEqual(len(s.phase_history), 1)
        self.assertGreater(qchain_discrepancy(s.fibers[0]), 1e-8)
        self.assertEqual(update.remainder[0], "[R]_memory_prior")
        self.assertEqual(s.phase_history[0][2],
                         update.remainder[1][2])
        self.assertEqual(s.tissue.cells[0].atp, 0)
        s = find(self.m, s, "af/metabolic_conversion:0").destination
        repaired = find(self.m, s, "fiber_connection_repair")
        s = repaired.destination
        f = s.fibers[0]
        self.assertAlmostEqual(qerror(f.phases[0], f.phases[-1],
                                       f.forward[-1]), 0., places=12)
        self.assertEqual(len(s.repairs), 1)
        self.assertAlmostEqual(qchain_discrepancy(s.fibers[0]), 0., places=11)
        self.assertEqual(len(s.phase_history), 1)
        self.assertEqual(repaired.remainder[0], "[R]_transport_prior")
        self.assertTrue(self.m.valid(s))

    def test_continuous_transport_disagreement_modulates_growth(self):
        # One paid physical section permits a nontrivial transport defect.
        s = find(self.m, self.seed, "af/neurite_extend:0:0").destination
        s = find(self.m, s, "af/cognitive/encounter:0").destination
        s = find(self.m, s, "af/metabolic_conversion:0").destination
        s = find(self.m, s, "fiber_experience_update").destination
        s = find(self.m, s, "af/metabolic_conversion:0").destination
        mismatch = qchain_discrepancy(s.fibers[0])
        self.assertGreater(mismatch, 0.)
        slowed = find(self.m, s, "af/neurite_extend:0:0").rate
        repaired = find(self.m, s, "fiber_connection_repair").destination
        # Repair itself costs ATP; compare growth at the same ATP level.
        repaired = find(self.m, repaired, "af/nutrient_supply:0").destination
        repaired = find(self.m, repaired,
                        "af/metabolic_conversion:0").destination
        self.assertAlmostEqual(qchain_discrepancy(repaired.fibers[0]),
                               0., places=11)
        restored = find(self.m, repaired, "af/neurite_extend:0:0").rate
        self.assertAlmostEqual(restored / slowed,
                               1. + .25 * mismatch, places=11)
        self.assertTrue(self.m.valid(repaired))

    def test_phase_history_exhaustion_blocks_unlogged_updates(self):
        capped = GlialQuaternionEcology(max_fiber_records=0)
        a = find(capped, capped.seed(), "af/neurite_extend:0:0").destination
        b = find(capped, a, "af/cognitive/encounter:0").destination
        c = find(capped, b, "af/metabolic_conversion:0").destination
        self.assertNotIn("fiber_experience_update", names(capped, c))
        self.assertEqual(c.tissue.cells[0].atp, 1)
        self.assertEqual(len(c.phase_history), 0)

    def test_context_switch_does_not_see_unobserved_target(self):
        self.assertNotIn("fiber_experience_update", names(self.m, self.seed))
        observed = find(self.m, self.seed,
                        "af/cognitive/encounter:0").destination
        switched = find(self.m, observed,
                        "af/cognitive/context_shift").destination
        self.assertNotIn("fiber_experience_update", names(self.m, switched))
        self.assertNotIn("fiber_connection_repair", names(self.m, switched))

    def test_invalid_state_and_ledger_mismatches_refused(self):
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.seed, glial_to_tissue=1))
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.seed, glial_waste=1))
        with self.assertRaises(ValueError):
            self.m.trajectory(self.seed, 1.0, max_events=0)
        with self.assertRaises(ValueError):
            GlialQuaternionEcology(max_support=0)
        with self.assertRaises(ValueError):
            GlialQuaternionEcology(max_glial_input=-1)
        with self.assertRaises(ValueError):
            self.m.channels(replace(self.seed,
                phase_history=((0, 0, ONE, ONE, (0, 0, 0)),) * 33))
        bad = replace(self.seed.fibers[0],
                      phases=(qnorm((1., 1., 1., 1.))[:-1],))
        self.assertFalse(self.m.valid(
            replace(self.seed, fibers=(bad, self.seed.fibers[1]))))

    def test_2d_and_3d_are_substrate_variants_not_quaternion_dimensions(self):
        model2 = GlialQuaternionEcology(spatial_dims=2)
        model3 = GlialQuaternionEcology(spatial_dims=3)
        self.assertTrue(model2.valid(model2.seed()))
        self.assertTrue(model3.valid(model3.seed()))
        self.assertEqual(model2.seed().fibers[0].real_carrier_dimension,
                         model3.seed().fibers[0].real_carrier_dimension)
        self.assertEqual(len(model2.seed().tissue.attract), 16)
        self.assertEqual(len(model3.seed().tissue.attract), 64)

    def test_deterministic_short_trajectory_and_budget(self):
        end, path = self.m.trajectory(self.seed, .45, seed=14, max_events=150)
        twin, path2 = self.m.trajectory(self.seed, .45, seed=14, max_events=150)
        self.assertEqual(end, twin)
        self.assertEqual(path, path2)
        self.assertTrue(self.m.valid(end))
        self.assertEqual(self.m.summary(end)["joint_energy_residual"], 0)
        self.assertTrue(all(math.isfinite(t) and t > 0 for t, _, _ in path))
        self.assertTrue(all(path[i][0] < path[i + 1][0]
                            for i in range(len(path) - 1)))


if __name__ == "__main__":
    unittest.main()

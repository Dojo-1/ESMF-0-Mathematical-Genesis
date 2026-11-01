"""G0-AF contracts: true paid path sections, Q8, tissue, joint causality.

Finite fixtures are not held-out scientific evidence of cognition or biology.
"""
from dataclasses import replace
import math
import unittest

from esmf.associated22_u import neutral_square
from esmf.geometry_k import matvec
from esmf.orientation_seam_w import P
from esmf.telos_joint_ae import Distinction
from esmf.hypercomplex_tissue_af import (
    HyperNeuriteEcology, TissueEvent, Q8_NAMES,
    q8_mul, q8_inv, q8_coords, q8_reframe, phase_gap, selfcheck,
)


def event(model, state, name):
    matches = [ch for ch in model.channels(state) if ch.name == name]
    if len(matches) != 1:
        raise AssertionError("expected exactly one " + name)
    return matches[0]


def event_names(model, state):
    return {ch.name for ch in model.channels(state)}


class TestHyperNeuriteAF(unittest.TestCase):
    def setUp(self):
        self.model = HyperNeuriteEcology(spatial_dims=2)
        self.state = self.model.seed()

    def test_contract_and_state_viability(self):
        report = selfcheck()
        self.assertEqual(report["contract"], "G0-AF")
        self.assertTrue(self.model.valid(self.state))
        self.assertEqual(self.model.diagnostics(self.state)[
            "real_fiber_dimensions"], (8, 8))
        self.assertEqual(len(self.model.channels(self.state)) > 4, True)
        self.assertTrue(all(isinstance(ch, TissueEvent) and
                            ch.rate > 0 and self.model.valid(ch.destination)
                            for ch in self.model.channels(self.state)))
        self.assertEqual(self.model.row(self.state),
                         self.model.row(self.state, reverse=True))
        self.assertAlmostEqual(
            self.model.diagonal(self.state) + sum(
                self.model.row(self.state).values()), 0., places=12)

    def test_hamilton_q8_is_noncommutative_but_associative(self):
        self.assertEqual(len(Q8_NAMES), 8)
        self.assertEqual(q8_mul(1, 2), 3)
        self.assertEqual(q8_mul(2, 1), 7)
        self.assertNotEqual(q8_mul(1, 2), q8_mul(2, 1))
        for a in range(8):
            self.assertEqual(q8_mul(a, q8_inv(a)), 0)
            self.assertEqual(q8_mul(q8_inv(a), a), 0)
            self.assertEqual(sum(x*x for x in q8_coords(a)), 1)
            for b in range(8):
                for c in range(8):
                    self.assertEqual(q8_mul(q8_mul(a, b), c),
                                     q8_mul(a, q8_mul(b, c)))
        with self.assertRaises(ValueError):
            q8_mul(-1, 0)
        with self.assertRaises(ValueError):
            q8_mul(1, 8)
        with self.assertRaises(ValueError):
            q8_inv(8)

    def test_independent_local_quaternion_frames(self):
        # Independent left gauge changes plus the conjugated transport.
        for source in range(8):
            for target in range(8):
                for forward in (0, 1, 2, 3, 4, 7):
                    old_gap = phase_gap(source, target, forward)
                    for gs, gt in ((1, 2), (3, 5), (0, 0), (6, 7)):
                        shifted = q8_reframe(source, target, forward, gs, gt)
                        self.assertEqual(
                            old_gap, phase_gap(*shifted))
        self.assertEqual(phase_gap(1, 1), 0)
        self.assertEqual(phase_gap(1, 5), 4)
        self.assertEqual(phase_gap(1, 2), 2)

    def test_paid_neurite_extension_changes_real_fiber_dimension(self):
        growth = event(self.model, self.state, "neurite_extend:0:0")
        grown = growth.destination
        c = grown.cells[0]
        self.assertEqual(c.path, ((1, 0),))
        self.assertEqual(c.sections, (0, 1))
        self.assertEqual(c.fiber_real_dimension, 12)
        self.assertEqual(grown.cells[1].fiber_real_dimension, 8)
        self.assertEqual(c.atp, 1)
        self.assertEqual(c.material, 1)
        self.assertEqual(grown.waste, 1)
        self.assertEqual(self.model.diagnostics(grown)[
            "energy_balance_residual"], 0)
        self.assertEqual(self.model.diagnostics(grown)[
            "material_balance_residual"], 0)
        self.assertTrue(self.model.valid(grown))

    def test_prune_records_the_deleted_local_quaternion_section(self):
        grown = event(self.model, self.state,
                      "neurite_extend:0:0").destination
        cut = event(self.model, grown, "neurite_prune:0")
        pruned = cut.destination
        self.assertEqual(cut.typed_remainder,
                         ("[R]_transport", (0, (1, 0), 1)))
        self.assertEqual(pruned.scars, ((0, (1, 0), 1),))
        self.assertEqual(pruned.cells[0].path, ())
        self.assertEqual(pruned.cells[0].sections, (0,))
        self.assertEqual(pruned.cells[0].material, 2)
        self.assertEqual(pruned.cells[0].fiber_real_dimension, 8)
        self.assertTrue(self.model.valid(pruned))
        frozen = HyperNeuriteEcology(spatial_dims=2, max_scars=0)
        g = event(frozen, frozen.seed(), "neurite_extend:0:0").destination
        self.assertNotIn("neurite_prune:0", event_names(frozen, g))

    def test_transpose_seam_in_neutral_carrier_is_involutive(self):
        west = event(self.model, self.state,
                     "neurite_extend:0:2").destination
        initial = self.state.cells[0].neutral
        moved = west.cells[0].neutral
        self.assertNotEqual(initial, moved)
        self.assertEqual(moved, tuple(matvec(P, initial)))
        self.assertAlmostEqual(neutral_square(initial),
                               neutral_square(moved), places=14)
        returned = event(self.model, west, "neurite_prune:0").destination
        self.assertEqual(returned.cells[0].neutral, initial)

    def test_contact_uses_neurite_tissue_and_retains_payer(self):
        grown = event(self.model, self.state,
                      "neurite_extend:0:0").destination
        birth = event(self.model, grown, "contact_birth:0")
        attached = birth.destination
        self.assertEqual(attached.links, ((0, 1),))
        self.assertEqual(attached.cells[0].atp, 0)
        self.assertEqual(attached.cells[0].material, 0)
        self.assertTrue(self.model.valid(attached))
        self.assertIn("contact_release:0", event_names(self.model, attached))
        self.assertNotIn("contact_release:1", event_names(self.model, attached))
        self.assertNotIn("neurite_prune:0", event_names(self.model, attached))
        removed = event(self.model, attached, "contact_release:0")
        self.assertEqual(removed.typed_remainder,
                         ("[R]_attr", (0, 1)))
        self.assertEqual(removed.destination.links, ())
        self.assertEqual(removed.destination.cells[0].material, 1)
        self.assertTrue(self.model.valid(removed.destination))

    def test_chemical_diffusion_and_dissipation_have_closed_ledgers(self):
        s = event(self.model, self.state,
                  "secrete_attract:0").destination
        self.assertEqual(s.cells[0].substrate, 1)
        self.assertEqual(sum(s.attract), 1)
        self.assertTrue(self.model.valid(s))
        diffused = event(self.model, s, "diffuse_attract:0:0").destination
        self.assertEqual(sum(diffused.attract), 1)
        self.assertEqual(diffused.attract[self.model.index((1, 0))], 1)
        decayed = event(self.model, diffused,
                        f"decay_attract:{self.model.index((1, 0))}").destination
        self.assertEqual(sum(decayed.attract), 0)
        self.assertEqual(decayed.waste, 1)
        self.assertEqual(self.model.diagnostics(decayed)[
            "energy_balance_residual"], 0)

    def test_source_conversion_and_separate_material_supply(self):
        met = event(self.model, self.state,
                    "metabolic_conversion:0").destination
        self.assertEqual(met.cells[0].substrate, 1)
        self.assertEqual(met.cells[0].atp, 3)
        self.assertTrue(self.model.valid(met))
        food = event(self.model, self.state, "nutrient_supply:0").destination
        self.assertEqual(food.nutrient_input, 1)
        self.assertEqual(food.cells[0].substrate, 3)
        mat = event(self.model, self.state, "material_supply:0").destination
        self.assertEqual(mat.material_input, 1)
        self.assertEqual(mat.cells[0].material, 3)
        self.assertTrue(self.model.valid(food))
        self.assertTrue(self.model.valid(mat))

    def test_bidirectional_telos_tissue_coupling(self):
        base = event(self.model, self.state, "cognitive/encounter:0")
        self.assertEqual(base.destination.cells[0].atp, 1)
        self.assertEqual(base.destination.waste, 1)
        assert base.destination.cognitive.distinctions[0].seen_different == 1

        grown = event(self.model, self.state,
                      "neurite_extend:0:0").destination
        self.assertGreater(
            event(self.model, grown, "cognitive/encounter:0").rate,
            event(self.model, self.state, "cognitive/encounter:0").rate)

        cognitive = replace(
            self.state.cognitive,
            distinctions=(Distinction(2, 2, 2, 2), Distinction()))
        conflicted = replace(self.state, cognitive=cognitive)
        self.assertTrue(self.model.valid(conflicted))
        self.assertGreater(
            event(self.model, conflicted, "neurite_extend:0:0").rate,
            event(self.model, self.state, "neurite_extend:0:0").rate)
        self.assertIn("phase_pulse:0", event_names(self.model, conflicted))
        pulsed = event(self.model, conflicted, "phase_pulse:0").destination
        self.assertEqual(pulsed.cells[0].sections, (2,))
        self.assertEqual(pulsed.waste, 1)

    def test_tissue_atp_boundary_has_noncommuting_causal_generators(self):
        cell0 = replace(self.state.cells[0], atp=0)
        depleted = replace(
            self.state, cells=(cell0, self.state.cells[1]), waste=2)
        self.assertTrue(self.model.valid(depleted))
        self.assertNotIn("cognitive/encounter:0",
                         event_names(self.model, depleted))
        met = event(self.model, depleted,
                    "metabolic_conversion:0").destination
        self.assertIn("cognitive/encounter:0",
                      event_names(self.model, met))
        observer = lambda s: s.cognitive.distinctions[0].seen_different
        bracket = self.model.generator_bracket(
            depleted, "metabolic_conversion:0", "cognitive/encounter:",
            observer)
        self.assertAlmostEqual(bracket, .35 * .55 * .85, places=12)
        self.assertGreater(bracket, 0.)

    def test_capacity_and_validity_gate_no_quiet_clipping(self):
        with self.assertRaises(ValueError):
            HyperNeuriteEcology(side=3)
        with self.assertRaises(ValueError):
            HyperNeuriteEcology(max_sections=1)
        with self.assertRaises(ValueError):
            self.model.channels(replace(self.state, waste=5))
        with self.assertRaises(ValueError):
            self.model.trajectory(self.state, -1)
        with self.assertRaises(ValueError):
            self.model.trajectory(self.state, 1, max_events=0)
        with self.assertRaises(RuntimeError):
            self.model.trajectory(self.state, 100., max_events=1)
        capped = HyperNeuriteEcology(spatial_dims=2, max_sections=2)
        grown = event(capped, capped.seed(),
                      "neurite_extend:0:0").destination
        self.assertFalse(
            any(e.name.startswith("neurite_extend:0:") for e
                in capped.channels(grown)))

    def test_three_dimensional_tissue_chart_keeps_local_highdim_carrier(self):
        model = HyperNeuriteEcology(side=8, spatial_dims=3, max_sections=96)
        start = model.seed()
        self.assertTrue(model.valid(start))
        self.assertEqual(start.cells[0].soma, (0, 0, 0))
        self.assertEqual(len(start.attract), 8 ** 3)
        self.assertEqual(model.site(model.index((1, 2, 3))), (1, 2, 3))
        lifted = event(model, start, "neurite_extend:0:4").destination
        self.assertTrue(model.valid(lifted))
        self.assertEqual(lifted.cells[0].path, ((0, 0, 1),))
        self.assertEqual(lifted.cells[0].sections, (0, 3))
        self.assertEqual(lifted.cells[0].fiber_real_dimension, 12)
        self.assertEqual(lifted.cells[0].neutral, start.cells[0].neutral)
        # Theoretical dynamic-capacity bound, not achieved neural rank.
        self.assertEqual(4 + 4 * model.max_sections, 388)
        with self.assertRaises(ValueError):
            model.index((1, 2))
        with self.assertRaises(ValueError):
            model.site(512)

    def test_short_replay_is_deterministic_and_valid(self):
        final, trace = self.model.trajectory(
            self.state, 1.1, seed=17, max_events=300)
        duplicate, same_trace = self.model.trajectory(
            self.state, 1.1, seed=17, max_events=300)
        self.assertEqual(final, duplicate)
        self.assertEqual(trace, same_trace)
        self.assertTrue(self.model.valid(final))
        self.assertTrue(all(math.isfinite(t) and t > 0 for t, _, _ in trace))
        self.assertEqual(self.model.diagnostics(final)[
            "material_balance_residual"], 0)
        self.assertEqual(self.model.diagnostics(final)[
            "energy_balance_residual"], 0)


if __name__ == "__main__":
    unittest.main()

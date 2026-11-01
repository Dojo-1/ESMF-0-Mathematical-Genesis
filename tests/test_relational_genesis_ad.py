"""G0-AD mathematical contracts: no held-out performance claims."""
import unittest

from esmf.geometry_k import matvec
from esmf.remainder import Correspondence, Worldlet
from esmf.relational_genesis_ad import (
    ContextualSymmetry, Encounter, Question, questions,
    contrast_audit, selfcheck, transport_audit,
)


def sample(context, before, after, witness, transform="half-turn"):
    return Encounter(context, transform, before, after, witness)


class TestRelationalGenesisAD(unittest.TestCase):
    def test_contract_selfcheck(self):
        report = selfcheck()
        self.assertEqual(report["status"], "contract-only")
        self.assertTrue(report["kernel_contrast_audit"][
            "false_merge_of_declared_contrast"])

    def test_questions_are_typed_proposals_not_truths(self):
        generated = questions("numeral-reading", "half-turn")
        self.assertEqual(len(generated), 4)
        self.assertEqual(
            {(q.mode, q.relation) for q in generated},
            {("whether", "concordance"), ("what", "difference"),
             ("why", "contrariety"), ("how", "mediation")},
        )
        self.assertTrue(all(q.evidence == "S" for q in generated))
        with self.assertRaises(ValueError):
            Question("c", "g", "whether", "difference", "?", evidence="P")
        with self.assertRaises(ValueError):
            questions("", "g")

    def test_same_transform_different_contexts(self):
        system = ContextualSymmetry(minimum_evidence=2.)
        for i in range(3):
            system.observe(sample("numerals", "six", "nine", f"n{i}"))
            system.observe(sample("objects", "A", "A", f"o{i}"))
        self.assertEqual(system.decide("numerals", "half-turn"), "distinguish")
        self.assertEqual(system.decide("objects", "half-turn"), "preserve")
        self.assertEqual(system.decide("new-world", "half-turn"), "abstain")
        self.assertEqual(system.fast_clock, 6)

    def test_counterexample_never_silently_glued(self):
        system = ContextualSymmetry(minimum_evidence=1.)
        system.observe(sample("context", "same", "same", "a"))
        self.assertEqual(system.decide("context", "half-turn"), "preserve")
        system.observe(sample("context", "different", "outcome", "b"))
        self.assertEqual(system.decide("context", "half-turn"), "contested")
        proposed = system.propose_split("context", "half-turn")
        self.assertIsNotNone(proposed)
        self.assertEqual(proposed["status"], "H")
        self.assertEqual(system.report("context", "half-turn")["witnesses_change"], 1)
        system.dissipate(30)
        self.assertEqual(system.decide("context", "half-turn"), "contested")
        self.assertEqual(system.report("context", "half-turn")["witnesses_change"], 1)

    def test_dissipation_and_new_context_are_separate(self):
        system = ContextualSymmetry(minimum_evidence=2., decay=.5)
        for i in range(3):
            system.observe(sample("before", "same", "same", str(i)))
        self.assertEqual(system.decide("before", "half-turn"), "preserve")
        system.dissipate(3)
        self.assertEqual(system.decide("before", "half-turn"), "abstain")
        self.assertEqual(system.report("before", "half-turn")["witnesses_preserve"], 3)
        system.fork_context("before", "after")
        self.assertEqual(system.lineage["after"], "before")
        self.assertEqual(system.decide("after", "half-turn"), "abstain")
        for i in range(3):
            system.observe(sample("after", "new", "different", f"after-{i}"))
        self.assertEqual(system.decide("after", "half-turn"), "distinguish")
        self.assertEqual(system.report("before", "half-turn")["witnesses_change"], 0)

    def test_capacity_and_provenance_gate(self):
        system = ContextualSymmetry(max_pairs=1, max_events=2)
        system.observe(sample("A", "x", "x", "w1"))
        with self.assertRaises(ValueError):
            system.observe(sample("A", "x", "x", "w1"))
        system.observe(sample("A", "x", "x", "w2"))
        with self.assertRaises(RuntimeError):
            system.observe(sample("A", "x", "x", "w3"))
        self.assertEqual(len(system.provenance), 2)
        with self.assertRaises(RuntimeError):
            system.observe(sample("B", "x", "x", "w4"))
        self.assertEqual(system.fast_clock, 2)
        self.assertEqual(system.decide("A", "half-turn"), "preserve")
        with self.assertRaises(ValueError):
            system.fork_context("missing", "new")
        with self.assertRaises(ValueError):
            ContextualSymmetry(decay=1.5)

    def test_transport_audit_covariant_and_loss_aware(self):
        a = Worldlet(3)
        b = Worldlet(2)
        c = Correspondence(
            a, b,
            ((1., 0., 0.), (0., 1., 0.)),
            ((1., 0.), (0., 1.), (0., 0.)))
        source = (2., -1., 4.)
        target = (2.5, -1.)
        report = transport_audit(c, source, target)
        self.assertEqual(report["R_transport_target"], (.5, 0.))
        self.assertEqual(report["R_transport_roundtrip"], (0., 0., 4.))
        self.assertEqual(report["semantic_equivalence"], "unassessed")
        self.assertAlmostEqual(report["signed_target_quadratic"], .25)
        bs = ((2., .3, 0.), (0., 1., 0.), (0., 0., .5))
        bt = ((1., .5), (0., 2.))
        changed = c.reframe(bs, bt)
        updated = transport_audit(
            changed, matvec(bs, source), matvec(bt, target))
        self.assertAlmostEqual(
            report["signed_target_quadratic"],
            updated["signed_target_quadratic"], places=10)
        expected = matvec(bt, report["R_transport_target"])
        for actual, wanted in zip(updated["R_transport_target"], expected):
            self.assertAlmostEqual(actual, wanted, places=10)

    def test_kernel_contrast_and_noncollapse(self):
        a = Worldlet(3)
        b = Worldlet(2)
        c = Correspondence(a, b, ((1., 0., 0.), (0., 1., 0.)),
                           ((1., 0.), (0., 1.), (0., 0.)))
        killed = contrast_audit(c, (0., 0., 0.), (0., 0., 1.), True)
        self.assertTrue(killed["false_merge_of_declared_contrast"])
        retained = contrast_audit(c, (0., 0., 0.), (1., 0., 0.), True)
        self.assertFalse(retained["collapsed_at_target"])
        self.assertFalse(retained["false_merge_of_declared_contrast"])
        with self.assertRaises(ValueError):
            contrast_audit(c, (0., 0., 0.), (0., 0., 0.), True, tol=-1.)

    def test_indefinite_quadratic_not_misrepresented_as_loss(self):
        a = Worldlet(2, ((-1., 0.), (0., 1.)))
        c = Correspondence(
            a, a, ((1., 0.), (0., 1.)), ((1., 0.), (0., 1.)))
        result = transport_audit(c, (0., 0.), (1., 0.))
        self.assertEqual(result["signed_target_quadratic"], -1.)
        self.assertNotIn("nonnegative_loss", result)


if __name__ == "__main__":
    unittest.main()

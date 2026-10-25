import math
import random
import unittest

from esmf.provenance_z import (
    ProvenanceBelief,SCHEDULE,episode,selfcheck,summary
)
from esmf.active_epistemics_y import Belief,ENERGY_CAP,ENERGY_FLOOR


class TestProvenanceZ(unittest.TestCase):
    def test_one_observation_cannot_select_origin(self):
        for parity in (0,1):
            for y in (-2.1,0.,.5,1.2):
                b=ProvenanceBelief().update(parity,y,.41)
                self.assertAlmostEqual(b.w[0],.4)
                self.assertAlmostEqual(b.w[1],.4)
                self.assertAlmostEqual(b.w[2],.2)
                self.assertIsNone(b.decision)

    def test_even_can_test_continuity_but_not_orientation(self):
        belief=ProvenanceBelief()
        for y in (1.2,1.25,1.31,1.18):
            belief.update(0,y,.21)
        self.assertGreater(belief.shared,.8)
        self.assertAlmostEqual(belief.orientation_given_shared,.5)
        self.assertIsNone(belief.decision)

    def test_independent_analytic_gaussian_evidence(self):
        # Two even measurements: shared latent D vs two independent D_i.
        y1,y2=.7,1.2
        s1,s2=.31,.52
        b=ProvenanceBelief().update(0,y1,s1).update(0,y2,s2)
        a=1+s1*s1
        c=1+s2*s2
        ds=a*c-1.
        di=a*c
        logp_shared=-.5*math.log(ds)-.5*(c*y1*y1+a*y2*y2-2*y1*y2)/ds
        logp_ind=-.5*math.log(di)-.5*(y1*y1/a+y2*y2/c)
        expected=(.8/.2)*math.exp(logp_shared-logp_ind)
        actual=b.shared/b.independent
        self.assertAlmostEqual(actual,expected,places=11)

    def test_two_shared_components_are_original_bayes_filter(self):
        rng=random.Random(1927)
        b=ProvenanceBelief()
        plain=Belief()
        for i in range(12):
            parity=i%2
            y=rng.gauss(0,1)
            sd=.3+rng.random()*.4
            b.update(parity,y,sd)
            plain.update(parity,y,sd)
            self.assertLess(abs(b.orientation_given_shared-plain.w[1]),1e-10)
            self.assertAlmostEqual(sum(b.w),1.)
            self.assertTrue(all(0<v<=1 for v in b.var))

    def test_same_sensor_ledger_all_worlds(self):
        for seed in range(10):
            for world in ("shared_identity","shared_transpose","independent","change_point"):
                x=episode(seed,world)
                self.assertEqual(x["observations"],len(SCHEDULE))
                self.assertLessEqual(x["simulated_resource_spent"],
                                     ENERGY_CAP-ENERGY_FLOOR+1e-12)
                self.assertLessEqual(x["posterior_shared"],1.)
                self.assertLessEqual(x["posterior_independent"],1.)
                if world=="independent" and x["audited_orientation_decision"] is not None:
                    self.assertTrue(x["audited_confident"])

    def test_independent_conflict_quarantine(self):
        b=ProvenanceBelief()
        for y in (1.8,-1.7,1.4,-1.9,1.6):
            b.update(0,y,.22)
        self.assertTrue(b.quarantine)
        self.assertIsNone(b.decision)

    def test_replay(self):
        for world in ("shared_identity","shared_transpose","independent","change_point"):
            self.assertEqual(episode(7,world),episode(7,world))

    def test_contract(self):
        self.assertTrue(selfcheck()["one_observation_cannot_identify_continuity"])


if __name__=="__main__":
    unittest.main()

import math
import random
import unittest

from esmf.temporal_repair_aa import (
    TemporalEcology,observation_stream,episode,selfcheck,SCHEDULE
)
from esmf.provenance_z import episode as old_episode
from esmf.active_epistemics_y import (
    Belief,ENERGY_CAP,ENERGY_FLOOR,PROBE_BY_NAME
)


def marginalized_segment_loglike(readings,signs,noise):
    """External Gaussian integral / Sherman-Morrison witness."""
    precision=sum(1/(s*s) for s in noise)
    linear=sum(y*sign/(s*s) for y,sign,s in zip(readings,signs,noise))
    quadratic=sum((y/s)**2 for y,s in zip(readings,noise))
    return (-.5*sum(math.log(s*s) for s in noise)
            -.5*math.log1p(precision)
            -.5*(quadratic-linear*linear/(1+precision)))


class TestTemporalRepairAA(unittest.TestCase):
    def test_single_observation_preserves_priors(self):
        for parity in (0,1):
            model=TemporalEcology()
            weights=[h.weight for h in model.local_worlds]
            model.update(parity,.72,.31)
            for a,b in zip(weights,model.local_worlds):
                self.assertAlmostEqual(a,b.weight,places=12)
            self.assertIsNone(model.proposed_repair())

    def test_even_only_preserves_orientation_uncertainty(self):
        model=TemporalEcology()
        for y in (1.3,1.2,1.1,1.35,1.18,1.25):
            model.update(0,y,.25)
        self.assertAlmostEqual(model.p_transpose_conditional,.5,places=11)
        self.assertIsNone(model.decision())

    def test_exact_segment_evidence(self):
        ys=(.8,.95,.9,-1.2,-1.1,-1.3)
        noise=(.21,.27,.24,.29,.25,.23)
        parities=(0,1,0,1,0,1)
        model=TemporalEcology()
        for p,y,s in zip(parities,ys,noise):
            model.update(p,y,s)
        h_shared=next(h for h in model.local_worlds if h.mode=="shared" and h.twist==0)
        h_k3=next(h for h in model.local_worlds if h.mode=="split" and h.twist==0 and h.cut==3)
        signed=[1.]*6
        p_shared=marginalized_segment_loglike(ys,signed,noise)
        p_split=(marginalized_segment_loglike(ys[:3],signed[:3],noise[:3])+
                 marginalized_segment_loglike(ys[3:],signed[3:],noise[3:]))
        prior_ratio=(.35/(2*5))/(.45/2)
        inferred_ratio=h_k3.weight/h_shared.weight
        self.assertAlmostEqual(inferred_ratio,prior_ratio*math.exp(p_split-p_shared),places=8)

    def test_strong_changepoint_spawns_two_worldlets(self):
        model=TemporalEcology()
        contrasts=(1.4,1.4,1.4,-1.5,-1.5,-1.5)
        for parity,y in zip((0,1,0,1,0,1),contrasts):
            model.update(parity,y,.16)
        repair=model.proposed_repair()
        self.assertIsNotNone(repair)
        self.assertEqual(repair["boundary_after_event"],3)
        self.assertEqual(len(repair["worldlets"]),2)
        self.assertEqual(repair["worldlets"][0]["span"],[0,3])
        self.assertEqual(repair["worldlets"][1]["span"],[3,6])

    def test_stimulus_replay_matches_g0z(self):
        mapping={"shared_identity":"shared_identity",
                 "shared_transpose":"shared_transpose",
                 "independent":"independent",
                 "change_identity":"change_point"}
        for seed in (0,1,5,9):
            for new,old in mapping.items():
                stream,energy=observation_stream(seed,new)
                naive=Belief()
                from esmf.provenance_z import ProvenanceBelief
                provenance=ProvenanceBelief()
                for p,y,sd in stream:
                    naive.update(p,y,sd)
                    provenance.update(p,y,sd)
                prior=old_episode(seed,old)
                self.assertAlmostEqual(naive.w[1],prior["plain_orientation_p"],places=12)
                self.assertAlmostEqual(provenance.shared,prior["posterior_shared"],places=12)
                self.assertAlmostEqual(energy,prior["simulated_resource_spent"],places=12)

    def test_all_models_are_typed_and_normalized(self):
        model=TemporalEcology()
        self.assertEqual(len(model.local_worlds),13)
        random_generator=random.Random(8366)
        for i in range(len(SCHEDULE)):
            model.update(i%2,random_generator.gauss(0,1),.3)
            self.assertAlmostEqual(sum(h.weight for h in model.local_worlds),1.)
            self.assertTrue(all(h.variance>0 and h.variance<=1 for h in model.local_worlds))

    def test_energy_and_reproducibility(self):
        for world in ("shared_identity","shared_transpose","change_identity",
                      "change_transpose","independent"):
            a=episode(7,world)
            b=episode(7,world)
            self.assertEqual(a,b)
            self.assertLessEqual(a["energy"],ENERGY_CAP-ENERGY_FLOOR+1e-11)
            if a["repair_proposed"]:
                self.assertIn(a["repair_boundary"],(1,2,3,4,5))

    def test_selfcheck(self):
        self.assertEqual(selfcheck()["n_component_hypotheses"],13)


if __name__=="__main__":
    unittest.main()

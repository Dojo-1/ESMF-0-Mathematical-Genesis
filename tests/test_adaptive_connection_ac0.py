"""Tests: changing rank-four neutral connection and partial charted observations."""
import math
import random
import unittest

from esmf.adaptive_connection_ac0 import (
    D,X,Y,HORIZON,BURN_IN,WORLDS,MODEL_NAMES,
    boost_transport,derivative_transport,make_probe,predict,
    source_coefficients,curvature,math_checks,RLSConnection,simulate
)
from esmf.global_bundle_q import NeutralCylinderBundle,TWIST,det_sign
from esmf.bundle_p import ETA,commutator,maxdiff
from esmf.geometry_k import identity,matmul,transpose


class TestAdaptiveConnectionAC0(unittest.TestCase):
    def test_neutral_generators_and_nonflat_curvature(self):
        checks=math_checks()
        self.assertTrue(checks["curvature_nonzero"])
        self.assertLess(checks["max_curvature_gauge_residual"],1e-10)
        self.assertGreater(checks["max_wrong_chart_transport_gap"],.01)
        self.assertEqual(checks["winding_det_signs"],[-1,-1,-1])

    def test_semigroup_and_signed_metric_preservation(self):
        for axis in (0,1):
            U1=boost_transport(axis,.37,.11)
            U2=boost_transport(axis,.37,.09)
            both=boost_transport(axis,.37,.20)
            self.assertLess(maxdiff(matmul(U1,U2),both),1e-12)
            self.assertLess(maxdiff(matmul(matmul(transpose(U1),ETA),U1),ETA),1e-12)

    def test_partial_sensor_gradient_against_independent_difference(self):
        b=NeutralCylinderBundle()
        rng=random.Random(7273)
        for axis in (0,1):
            for _ in range(9):
                probe=make_probe(rng,axis,b)
                value,gradient=predict(probe,.41,b,gradient=True)
                eps=1e-6
                hi=predict(probe,.41+eps,b)
                lo=predict(probe,.41-eps,b)
                for j in (0,1):
                    numeric=(hi[j]-lo[j])/(2*eps)
                    self.assertLess(abs(numeric-gradient[j]),1e-7)
                self.assertEqual(len(value),2)
                self.assertEqual(len(probe.source_local),4)

    def test_partial_observations_invariant_between_chart_representations(self):
        b=NeutralCylinderBundle()
        rng=random.Random(8913)
        p=make_probe(rng,0,b)
        B0,_,_=b.frame(p.chart0,p.theta0,p.z0)
        B1,_,_=b.frame(p.chart1,p.theta1,p.z1)
        vref=predict(p,.35,b)
        # Replace chart coordinates with the reference chart simultaneously
        # for input section AND output dual covectors.
        from esmf.geometry_k import matvec,inverse
        from esmf.adaptive_connection_ac0 import PartialProbe
        reference_input=matvec(inverse(B0),p.source_local)
        reference_duals=tuple(matvec(transpose(B1),c)
                              for c in p.sensors_local)
        class RefBundle:
            def frame(self,chart,theta,z):
                return identity(4),None,None
        p2=PartialProbe(p.axis,p.h,p.theta0,p.theta1,p.z0,p.z1,
                        0,0,reference_input,reference_duals)
        self.assertEqual(len(vref),2)
        y=predict(p2,.35,RefBundle())
        for q,r in zip(vref,y):
            self.assertAlmostEqual(q,r,places=12)

    def test_true_connection_changes_without_losing_metric(self):
        for world in ("static","smooth_drift","abrupt_step","sign_reversal",
                      "rapidly_variable"):
            for t in (5,29,30,59):
                a,b=source_coefficients(7,t,world)
                self.assertTrue(-1<a<1 and -1<b<1)
                for axis,coef in enumerate((a,b)):
                    U=boost_transport(axis,coef,.08)
                    self.assertLess(maxdiff(matmul(matmul(transpose(U),ETA),U),ETA),1e-12)

    def test_controllers_actually_update_or_freeze(self):
        rng=random.Random(8828)
        bundle=NeutralCylinderBundle()
        ad=RLSConnection("adaptive")
        fr=RLSConnection("frozen_after_calibration")
        co=RLSConnection("constant_forgetting")
        for t in range(40):
            axis=t%2
            p=make_probe(rng,axis,bundle)
            truth=(.37,-.26) if t<20 else (-.57,.52)
            noisy=predict(p,truth[axis],bundle)
            ad.update(p,noisy,bundle)
            fr.update(p,noisy,bundle)
            co.update(p,noisy,bundle)
        self.assertEqual(fr.events,40)
        self.assertEqual(ad.events,40)
        self.assertNotEqual(fr.coeff,ad.coeff)
        self.assertNotEqual(co.coeff,ad.coeff)
        self.assertTrue(all(abs(c)<2 for c in ad.coeff))
        self.assertTrue(all(v>0 for v in ad.var))

    def test_deterministic_replay_and_exact_real_control(self):
        for world in ("static","abrupt_step","sign_reversal"):
            a=simulate(7,world,"adaptive")
            b=simulate(7,world,"equivalent_real")
            self.assertEqual({k:v for k,v in a.items() if k!="policy"},
                             {k:v for k,v in b.items() if k!="policy"})
            self.assertEqual(a,simulate(7,world,"adaptive"))

    def test_horizon_sample_budget_and_finite_loss(self):
        for world in WORLDS:
            for policy in MODEL_NAMES:
                d=simulate(11,world,policy)
                self.assertTrue(math.isfinite(d["heldout_mse"]))
                self.assertTrue(math.isfinite(d["mean_coeff_abs_error"]))
                self.assertTrue(math.isfinite(d["mean_curvature_entry_error"]))
                self.assertGreaterEqual(d["heldout_mse"],0.)
                self.assertEqual(d["learned_state_dimension"],2)


if __name__=="__main__":
    unittest.main()

import unittest
from esmf.geometry_k import (
    bilinear,change_frames,hyperbolic_distance,hyperboloid_point,
    inverse,lorentz_boost,metric_signature_21,mismatch_energy,selfcheck
)

class TestGeometryK(unittest.TestCase):
    def test_gauge(self):
        self.assertLess(selfcheck()["holonomy_conjugacy_residual"],1e-10)
    def test_hyperboloid(self):
        x=hyperboloid_point(.8,.4)
        self.assertAlmostEqual(bilinear(x,metric_signature_21(),x),-1.,places=12)
        self.assertAlmostEqual(hyperbolic_distance(x,x),0.,places=6)
    def test_nonorthogonal_charts(self):
        H=((2.,.3),(.3,3.))
        ui=(.2,.7);uj=(-.3,.8);U=((1.,.2),(.1,1.))
        a=((1.,2.),(0.,1.));b=((2.,0.),(.2,.5))
        old=mismatch_energy(ui,uj,U,H)
        new=mismatch_energy(*change_frames(ui,uj,U,H,a,b))
        self.assertAlmostEqual(old,new,places=10)
    def test_noninvertible_rejected(self):
        with self.assertRaises(ValueError):
            inverse(((1.,2.),(2.,4.)))

if __name__=="__main__":
    unittest.main()

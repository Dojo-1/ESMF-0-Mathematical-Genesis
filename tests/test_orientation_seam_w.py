import unittest
from esmf.orientation_seam_w import selfcheck,P,det4,right_factor
from esmf.associated22_u import H,lift_left
from esmf.geometry_k import matmul,transpose

class TestOrientationSeamW(unittest.TestCase):
    def test_orientation_vs_metric(self):
        self.assertAlmostEqual(det4(P),-1.)
        self.assertEqual(matmul(matmul(transpose(P),H),P),H)
    def test_action_factor_swap(self):
        J=((1.,.4),(0.,1.))
        self.assertEqual(matmul(matmul(P,lift_left(J)),P),right_factor(J))
    def test_full_selfcheck(self):
        r=selfcheck()
        self.assertLess(r["seam_equivariance_residual"],1e-12)
        self.assertGreater(r["within_factor_order_gap"],.01)
        self.assertGreater(r["chart_blind_seam_gap_max"],.05)

if __name__=="__main__":
    unittest.main()

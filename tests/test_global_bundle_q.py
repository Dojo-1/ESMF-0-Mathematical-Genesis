import unittest
import math
from esmf.global_bundle_q import NeutralCylinderBundle,det_sign
from esmf.bundle_p import maxdiff,scaled,identity

class TestGlobalBundleQ(unittest.TestCase):
    def test_full_selfcheck(self):
        r=NeutralCylinderBundle().selfcheck()
        self.assertEqual(r["winding_det_sign"],-1)
        self.assertLess(r["numeric_curvature"],1e-8)
    def test_gluing_is_not_fake(self):
        b=NeutralCylinderBundle()
        p=math.pi+.09
        g12=b.transition(1,2,p,.23)
        g01=b.transition(0,1,p,.23)
        g02=b.transition(0,2,p,.23)
        from esmf.geometry_k import matmul
        self.assertLess(maxdiff(matmul(g12,g01),g02),1e-12)
    def test_outside_chart(self):
        b=NeutralCylinderBundle()
        with self.assertRaises(ValueError):
            b.frame(2,.2,.4)
    def test_seam_twist(self):
        b=NeutralCylinderBundle()
        self.assertEqual(det_sign(b.transition(0,1,.12,.23)),-1)
        self.assertEqual(det_sign(b.transition(0,1,math.pi+.09,.23)),1)

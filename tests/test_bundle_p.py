import unittest
from esmf.bundle_p import selfcheck,metric_compatible,generators

class TestBundleP(unittest.TestCase):
    def test_bundle_invariants(self):
        r=selfcheck()
        self.assertTrue(r["broken_cocycle_detected"])
        self.assertTrue(r["curvature_nonzero"])
    def test_generators(self):
        for A in generators():
            self.assertLess(metric_compatible(A),1e-12)

if __name__=="__main__":
    unittest.main()

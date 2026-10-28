import unittest
from esmf.strata_r import selfcheck,Stratum,Restriction

class TestStrataR(unittest.TestCase):
    def test_gluing_and_defect(self):
        r=selfcheck()
        self.assertTrue(r["compatible_overlap"]["compatible"])
        self.assertFalse(r["incompatible_overlap"]["compatible"])
    def test_rank_mismatch(self):
        with self.assertRaises(ValueError):
            Restriction(Stratum("a",4),Stratum("b",2),((1.,0.),(0.,1.)))

import unittest
from esmf.neutral22 import ETA,boost,neutral_form,remainder,selfcheck
from esmf.geometry_k import matmul,transpose

class TestNeutral22(unittest.TestCase):
    def test_isometry(self):
        b=boost(.8)
        a=matmul(matmul(transpose(b),ETA),b)
        self.assertLess(max(abs(a[i][j]-ETA[i][j]) for i in range(4) for j in range(4)),1e-12)
    def test_nonzero_null(self):
        x=(2.,1.,2.,-1.)
        self.assertNotEqual(x,(0.,)*4)
        self.assertAlmostEqual(neutral_form(x,x),0.)
    def test_signed_not_cost(self):
        r=remainder((0.,0.,0.,5.))
        self.assertEqual(neutral_form(r,r),-25.)
        self.assertEqual(sum(v*v for v in r),25.)
    def test_selfcheck(self):
        self.assertTrue(selfcheck()["gauge_covariant_mismatch"])

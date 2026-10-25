import unittest
from esmf.associated22_u import H,selfcheck,neutral_square,positive_square,lift_left,flatten
from esmf.geometry_k import matmul,transpose
from esmf.causal_bundle_s import det

class TestAssociated22U(unittest.TestCase):
    def test_form(self):
        A=((.7,-.4),(.8,1.2))
        self.assertAlmostEqual(neutral_square(flatten(A)),det(A),places=12)
    def test_metric_isometry(self):
        J=((1.,.37),(0.,1.))
        L=lift_left(J)
        m=matmul(matmul(transpose(L),H),L)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(m[i][j],H[i][j],places=12)
    def test_nonzero_null_is_not_free_energy(self):
        x=(1.,0.,0.,0.)
        self.assertEqual(neutral_square(x),0.)
        self.assertGreater(positive_square(x),0.)
    def test_chart_and_dissipation(self):
        r=selfcheck()
        self.assertLess(r["max_metric_transport_gap"],1e-10)
        self.assertGreater(r["dissipative_counterexample_gap"],.01)

if __name__=="__main__":
    unittest.main()

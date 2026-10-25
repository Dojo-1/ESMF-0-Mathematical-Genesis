import unittest
from esmf.gauge import (
    change_chart,determinant,edge_energy,gram,gram_basis_change,
    inverse,rotation,selfcheck
)


class TestGauge(unittest.TestCase):
    def test_selfcheck(self):
        self.assertTrue(selfcheck()["coordinate_invariance"])

    def test_nonorthogonal_metric_adjustment(self):
        e=edge_energy((.5,.6),(.1,.9),rotation(.3),((2.,.1),(.1,1.)))
        args=change_chart((.5,.6),(.1,.9),rotation(.3),
                          ((2.,.1),(.1,1.)),
                          ((1.,2.),(0.,1.)),((2.,0.),(0.,.5)))
        self.assertAlmostEqual(e,edge_energy(*args),places=10)

    def test_gram_volume_not_gl_invariant(self):
        g=gram(((1.,0.),(0.,1.)))
        gp=gram_basis_change(g,((2.,0.),(0.,1.)))
        self.assertAlmostEqual(determinant(gp),.25)

    def test_singular_basis_rejected(self):
        with self.assertRaises(ValueError):
            inverse(((1.,2.),(2.,4.)))


if __name__=="__main__":
    unittest.main()

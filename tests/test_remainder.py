import unittest
from esmf.remainder import Correspondence,Worldlet,selfcheck

class TestRemainder(unittest.TestCase):
    def test_selfcheck(self):
        self.assertTrue(selfcheck()["frame_covariance"])
    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            Correspondence(Worldlet(3),Worldlet(2),((1.,0.),(0.,1.)),((1.,0.),(0.,1.),(0.,0.)))
    def test_exact_identity(self):
        w=Worldlet(2)
        c=Correspondence(w,w,((1.,0.),(0.,1.)),((1.,0.),(0.,1.)))
        self.assertEqual(c.remainder((.4,-2.)),(0.,0.))
    def test_indefinite_not_norm(self):
        w=Worldlet(2,((-1.,0.),(0.,1.)))
        v=Worldlet(1)
        c=Correspondence(w,v,((0.,1.),),((0.,),(1.,)))
        with self.assertRaises(ValueError):
            c.roundtrip_magnitude((1.,0.))

if __name__=="__main__":
    unittest.main()

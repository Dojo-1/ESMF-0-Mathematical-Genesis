import unittest
from esmf.representations_n import selfcheck,run,features

class TestRepresentationsN(unittest.TestCase):
    def test_isometries(self):
        self.assertTrue(selfcheck())
    def test_reproducible_scores(self):
        a,b=run(2,80),run(2,80)
        for mode in ("euclidean","neutral22","complex2"):
            self.assertEqual(a[mode]["brier"],b[mode]["brier"])
    def test_all_same_parameter_count(self):
        for mode in ("euclidean","neutral22","complex2"):
            self.assertEqual(len(features(1,1,0,mode)),4)

if __name__=="__main__":
    unittest.main()

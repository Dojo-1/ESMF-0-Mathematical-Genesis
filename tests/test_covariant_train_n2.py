import unittest
from esmf.covariant_train_n2 import train_pair

class TestCovariantTraining(unittest.TestCase):
    def test_transport(self):
        for seed in (0,2,7):
            r=train_pair(seed,80)
            self.assertLess(r["max_prediction_gap"],1e-10)
            self.assertLess(r["max_weight_transport_gap"],1e-10)

if __name__=="__main__":
    unittest.main()

import unittest
import random
from esmf.monodromy_x import selfcheck,episode,select,run,simulate
from esmf.causal_bundle_s import PI

class TestMonodromyX(unittest.TestCase):
    def test_windings_and_identifiability(self):
        self.assertTrue(selfcheck()["zero_winding_topology_unidentifiable"])
        rng=random.Random(20)
        params=(.58,-.44)
        even=episode(rng,params,True,"even",noise=0.)
        self.assertEqual(abs(even["winding"]),2)
        zero=episode(rng,params,True,"zero",noise=0.)
        answer,_=select([zero],params)
        self.assertIsNone(answer)

    def test_control_selection(self):
        for twist in (False,True):
            result=run(0,twist)
            self.assertEqual(result["selected_twist"],twist)
            self.assertFalse(result["null_probe_identifiable"])
            self.assertLess(result["double_winding_max_odd_vs_even_gap"],1e-10)

if __name__=="__main__":
    unittest.main()

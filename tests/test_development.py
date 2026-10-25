import unittest
from esmf.development import Development, run

class TestDevelopment(unittest.TestCase):
    def test_seeded(self):
        self.assertEqual(run(11,80),run(11,80))
    def test_bounds(self):
        d=Development(seed=4,n=18,max_degree=2)
        for _ in range(250):
            d.step()
            self.assertTrue(d.valid())
    def test_empty(self):
        self.assertTrue(Development(seed=1).valid())
    def test_invalid_stimulus(self):
        with self.assertRaises(ValueError):
            Development().step([0])

if __name__=="__main__":
    unittest.main()

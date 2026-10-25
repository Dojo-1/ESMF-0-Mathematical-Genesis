import unittest
from esmf.mechanochemical import ChemicalField, Organism, run

class TestMechanochemistry(unittest.TestCase):
    def test_diffusion_cfl(self):
        with self.assertRaises(ValueError):
            ChemicalField(dt=1.,diffusivity=1.)
    def test_chemical_mass(self):
        f=ChemicalField(side=8)
        for i in range(100):
            f.update({(3,4):1.0 if i<30 else 0.})
        self.assertLess(f.balance_residual,1e-9)
        self.assertGreaterEqual(min(map(min,f.c)),0)
    def test_bounded_growth(self):
        o=Organism(seed=4,n=12)
        for _ in range(150):
            o.step()
        self.assertTrue(o.valid())
    def test_replay(self):
        self.assertEqual(run(7,90),run(7,90))

if __name__=="__main__":
    unittest.main()

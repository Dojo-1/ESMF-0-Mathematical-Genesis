import unittest
from esmf.mechanochemical import (
    ChemicalField, Organism, continuum_suite, control_suite, run,
)

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
    def test_no_flux_boundary_mass_and_sampling(self):
        f=ChemicalField(side=8,decay=0.,diffusivity=.2,boundary="no_flux")
        f.c[0][0]=1.
        initial=f.mass()
        for _ in range(20):
            f.update()
        self.assertAlmostEqual(f.mass(),initial)
        f.c[-1][-1]=2.
        self.assertEqual(f.sample(16.,16.),2.)
    def test_bounded_growth(self):
        o=Organism(seed=4,n=12)
        for _ in range(150):
            o.step()
        self.assertTrue(o.valid())
    def test_replay(self):
        self.assertEqual(run(7,90),run(7,90))
    def test_independent_energy_and_material_budgets(self):
        o=Organism(seed=8,n=8)
        for _ in range(40):
            o.step()
        self.assertLess(abs(o._energy_residual()),1e-8)
        self.assertLess(abs(o._material_residual()),1e-8)
        self.assertAlmostEqual(sum(o.branch_length),o.total_growth)
        self.assertNotEqual(o.tips,o.exploratory_tips)
        self.assertAlmostEqual(o.energy_expenditure,sum((
            o.maintenance_expenditure,o.exploration_expenditure,
            o.extension_expenditure,o.contact_expenditure)))
    def test_paired_control_conditions(self):
        rows=control_suite(seeds=(2,),steps=5)
        self.assertEqual({row["condition"] for row in rows},{
            "chemistry_on","chemistry_off","chemotaxis_reversed",
            "nutrient_ablation","random_navigation"})
        self.assertTrue(all(row["valid"] for row in rows))
        random_row=next(row for row in rows if row["condition"]=="random_navigation")
        self.assertTrue(random_row["random_navigation"])
    def test_continuum_cases_match_physical_time(self):
        rows=continuum_suite(seed=1,physical_time=.4)
        self.assertEqual({row["case"] for row in rows},{
            "baseline","half_timestep","refined_grid","no_flux_boundary"})
        self.assertTrue(all(abs(row["physical_time"]-.4)<1e-12 for row in rows))
        self.assertTrue(all(row["valid"] for row in rows))
        self.assertEqual(next(row for row in rows if row["case"]=="refined_grid")["grid_side"],32)
        with self.assertRaises(ValueError):
            continuum_suite(seed=1,physical_time=.41)

if __name__=="__main__":
    unittest.main()

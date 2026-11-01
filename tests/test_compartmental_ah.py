import unittest
from dataclasses import replace
from esmf.compartmental_ah import (
  CompartmentalNeuriteEcology, qmul, qnorm
)

class TestCompartmentalAH(unittest.TestCase):
  def setUp(self):
    self.m = CompartmentalNeuriteEcology()
    self.seed = self.m.seed()

  def test_dimensions_and_exact_ledgers(self):
    self.assertTrue(self.m.valid(self.seed))
    self.assertEqual(self.m.diagnostics(self.seed)["cells"], 16)
    self.assertEqual(self.m.diagnostics(self.seed)["territories"], 4)
    self.assertEqual(self.m.ledger(self.seed),(0,0))
    with self.assertRaises(ValueError):
      CompartmentalNeuriteEcology(cells=16,contexts=2,questions=2)

  def test_real_branching_and_paid_growth(self):
    s=self.seed
    for cue in ((0,0),(0,1),(1,0),(1,1)):
      s=self.m.grow(s,0,0,cue)
      self.assertTrue(self.m.valid(s))
    bs=s.cells[0].branches
    self.assertEqual([b.parent for b in bs],[-1,0,0,1])
    self.assertEqual(len(set(b.voxel for b in bs)),4)
    self.assertEqual(s.cells[0].atp,self.seed.cells[0].atp-4)
    self.assertEqual(s.cells[0].material,self.seed.cells[0].material-4)
    self.assertEqual(self.m.ledger(s),(0,0))
    self.assertEqual(self.m.grow(s,0,0,(0,0)),s)

  def test_question_indexing_and_nonlin_coincidence(self):
    s=self.m.grow(self.seed,0,0,(1,0))
    b=s.cells[0].branches[0]
    self.assertTrue(self.m.plateau(b,(1,0)))
    self.assertFalse(self.m.plateau(b,(1,1)))
    s=self.m.observe(s,0,0,(1,0),True)
    self.assertEqual(self.m.predict(s,0,0,(1,0)),2/3)
    self.assertEqual(self.m.predict(s,0,1,(1,0)),.5)
    self.assertEqual(self.m.predict(s,1,0,(1,0)),.5)
    self.assertEqual(self.m.ledger(s),(0,0))

  def test_prediction_precedes_label_and_requires_paid_evidence(self):
    s=self.m.grow(self.seed,0,0,(0,1))
    first=self.m.predict(s,0,0,(0,1))
    self.assertEqual(first,.5)
    with self.assertRaises(ValueError):
      self.m.observe(s,0,0,(0,1),"bad")
    newer=self.m.observe(s,0,0,(0,1),True)
    self.assertGreater(self.m.predict(newer,0,0,(0,1)),first)
    self.assertEqual(newer.cells[0].atp,s.cells[0].atp-1)
    with self.assertRaises(ValueError):
      self.m.observe(s,0,0,(1,1),True)

  def test_glial_multiscale_consolidation(self):
    s=self.m.grow(self.seed,0,0,(0,0))
    s=self.m.observe(s,0,0,(0,0),True)
    no_glia=self.m.slow_step(s,glia_enabled=False)
    self.assertEqual(no_glia.cells[0].branches[0].long_changed,0)
    pulsed=self.m.glial_pulse(s,0)
    self.assertEqual(self.m.ledger(pulsed),(0,0))
    mature=self.m.slow_step(pulsed)
    self.assertEqual(mature.cells[0].branches[0].long_changed,1)
    self.assertEqual(mature.cells[0].branches[0].fast_changed,0)
    self.assertEqual(self.m.predict(mature,0,0,(0,0)),2/3)
    self.assertEqual(self.m.ledger(mature),(0,0))

  def test_sparse_synapses_and_delayed_inhibition(self):
    s=self.m.grow(self.seed,0,0,(1,1))
    s=self.m.connect(s,1,0,0,kind='inhibit',delay=2)
    s=self.m.emit(s,1)
    self.assertEqual(len(s.pending),1)
    before=self.m.predict(s,0,0,(1,1))
    self.assertEqual(before,.5)
    s=self.m.fast_step(s)
    self.assertEqual(s.cells[0].branches[0].drive,0)
    s=self.m.fast_step(s)
    self.assertEqual(s.cells[0].branches[0].drive,-1)
    self.assertEqual(self.m.choose_branch(s.cells[0],(1,1)),None)
    self.assertEqual(self.m.predict(s,0,0,(1,1)),.5)
    s=self.m.fast_step(s)
    self.assertEqual(s.cells[0].branches[0].drive,0)
    self.assertEqual(self.m.ledger(s),(0,0))

  def test_pruning_leaf_provenance_and_material(self):
    s=self.m.grow(self.seed,0,0,(0,0))
    s=self.m.grow(s,0,0,(0,1))
    with self.assertRaises(ValueError):
      self.m.prune(s,0,0,0)
    old=s.cells[0].material
    s=self.m.prune(s,0,0,1)
    self.assertEqual(s.cells[0].material,old+1)
    self.assertEqual(s.remainders[-1].kind,'[R]_memory_pruned')
    self.assertEqual(self.m.ledger(s),(0,0))

  def test_quaternion_gauge_invariance_across_actual_update(self):
    s=self.m.grow(self.seed,0,0,(1,0))
    s=self.m.observe(s,0,0,(1,0),True)
    g0=qnorm((1.,2.,0.,0.))
    g1=qnorm((1.,0.,2.,0.))
    sg=self.m.reframe(s,0,g0,(g1,))
    self.assertAlmostEqual(self.m.defects(s)[0],self.m.defects(sg)[0],places=11)
    a=self.m.observe(s,0,0,(1,0),False)
    b=self.m.observe(sg,0,0,(1,0),False)
    want=qmul(g1,a.cells[0].branches[0].phase)
    for x,y in zip(b.cells[0].branches[0].phase,want):
      self.assertAlmostEqual(x,y,places=11)
    self.assertAlmostEqual(self.m.defects(a)[0],self.m.defects(b)[0],places=11)
    self.assertEqual(self.m.ledger(b),(0,0))

  def test_capacity_and_rejections(self):
    with self.assertRaises(ValueError):
      self.m.grow(self.seed,0,0,(3,0))
    with self.assertRaises(ValueError):
      self.m.cell_id(88,0)
    with self.assertRaises(ValueError):
      self.m.connect(self.seed,0,1,0)
    with self.assertRaises(ValueError):
      self.m.fast_step(replace(self.seed,waste=15))

  def test_64_cell_4_branch_configuration_constructs(self):
    m=CompartmentalNeuriteEcology(cells=64,contexts=8,questions=8,
       branch_cap=4,territory_size=8,side=40)
    s=m.seed()
    self.assertEqual(len(s.glia),8)
    for j,cue in enumerate(((0,0),(0,1),(1,0),(1,1))):
      s=m.grow(s,0,0,cue)
    self.assertTrue(m.valid(s))
    self.assertEqual(m.diagnostics(s)['cells'],64)

if __name__=='__main__':
  unittest.main()

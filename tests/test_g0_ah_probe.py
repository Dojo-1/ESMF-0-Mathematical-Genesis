import unittest
from research.g0_ah_probe import (
  DEVELOPMENT, SEED_HOLDOUT, evaluator_probability, OrdinaryCounts, one, run
)

class TestG0AHProbe(unittest.TestCase):
  def test_evaluator_has_declared_ambiguous_and_discriminating_modes(self):
    self.assertEqual(evaluator_probability(0,0,(0,0)),.85)
    self.assertEqual(evaluator_probability(0,0,(0,1)),.15)
    self.assertEqual(evaluator_probability(1,2,(1,0)),.55)
    self.assertEqual(evaluator_probability(1,2,(0,0)),.45)

  def test_ordinary_typed_predictor_known_answer(self):
    model=OrdinaryCounts(True)
    pooled=OrdinaryCounts(False)
    for _ in range(6):
      model.observe(0,0,(0,0),True)
      model.observe(0,0,(1,1),False)
      pooled.observe(0,0,(0,0),True)
      pooled.observe(0,0,(1,1),False)
    self.assertGreater(model.predict(0,0,(0,0)),.8)
    self.assertLess(model.predict(0,0,(1,1)),.2)
    self.assertEqual(pooled.predict(0,0,(0,0)),.5)
    self.assertEqual(pooled.predict(0,0,(1,1)),.5)
    self.assertEqual(model.slots,128)
    self.assertEqual(pooled.slots,32)

  def test_seeded_independent_future_outcomes_and_equal_glial_resource(self):
    row=one(DEVELOPMENT[0])
    self.assertEqual(row['seed'],8300)
    self.assertTrue(row['all_predictors_committed_before_outcomes'])
    self.assertTrue(row['glial_resource_parity'])
    self.assertEqual(row['full_diagnostics']['compartments'],64)
    self.assertEqual(row['no_glia_diagnostics']['compartments'],64)
    self.assertEqual(row['full_diagnostics']['energy_residual'],0)
    self.assertEqual(row['full_diagnostics']['material_residual'],0)
    self.assertEqual(len(row['training_events']),384)
    self.assertEqual(len(row['future_outcomes']),64)
    self.assertEqual(sum(len(x) for x in row['future_outcomes'].values()),2048)
    self.assertEqual(row['scores']['ah_glial'],row['scores']['ah_no_consolidation'])
    self.assertEqual(row['scores'],one(DEVELOPMENT[0])['scores'])

  def test_development_result_rejects_unregistered_seeds(self):
    for s in (0,8200,8314,SEED_HOLDOUT[0]):
      with self.assertRaises(ValueError):
        one(s)
    with self.assertRaises(ValueError):
      run((8300,8301))

  def test_local_conventional_control_is_allowed_to_win(self):
    report=run()
    self.assertEqual(report['seeds'],list(DEVELOPMENT))
    self.assertEqual(report['future_per_seed'],2048)
    self.assertEqual(len(report['raw']),14)
    self.assertEqual(report['measured_coverage'],14)
    self.assertTrue(all(x['glial_resource_parity'] for x in report['raw']))
    self.assertGreater(report['means']['ah_glial']['mean'],
                       report['means']['ordinary_cue_table']['mean'])
    self.assertFalse(report['paired']['ah_glial_minus_ordinary']['confirmatory'])

if __name__=='__main__':
  unittest.main()

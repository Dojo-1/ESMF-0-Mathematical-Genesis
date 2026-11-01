import unittest
from esmf.compartmental_ah import CompartmentalNeuriteEcology
from research.export_g0_ah_microscope import SCHEMA, build, digest, snapshot

class TestG0AHMicroscope(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.data=build('a'*40)

  def test_recorded_states_are_exact_and_source_linked(self):
    self.assertEqual(self.data['schema'],SCHEMA)
    self.assertEqual(self.data['source_commit'],'a'*40)
    payload=dict(self.data)
    checksum=payload.pop('payload_sha256')
    self.assertEqual(digest(payload),checksum)
    self.assertEqual(len(self.data['runs']),2)
    for run in self.data['runs']:
      for i,frame in enumerate(run['frames']):
        self.assertEqual(frame['index'],i)
        self.assertEqual(frame['sha256'],digest(frame['state']))
        state=frame['state']
        self.assertEqual(state['energy_residual'],0)
        self.assertEqual(state['material_residual'],0)
        self.assertEqual(len(state['cells']),16)
        self.assertEqual(len(state['territories']),4)
        for cell in state['cells']:
          self.assertLessEqual(len(cell['branches']),4)
          for branch in cell['branches']:
            self.assertGreaterEqual(branch['transport_defect'],0)
            self.assertLessEqual(branch['predicted_change'],1)

  def test_demonstration_keeps_selected_events_distinct_from_random(self):
    focus,population=self.data['runs']
    self.assertEqual(len(focus['frames']),17)
    self.assertEqual(len(population['frames']),17)
    self.assertTrue(all(frame['selection']=='deliberately_selected_operator'
             for run in (focus,population) for frame in run['frames'][1:]))
    self.assertEqual(focus['frames'][-1]['state']['fast_tick'],3)
    self.assertTrue(any(frame['state']['pending'] for frame in focus['frames']))
    self.assertTrue(any(frame['state']['remainders'] for frame in focus['frames']))
    self.assertEqual(sum(len(c['branches']) for c in population['frames'][-1]['state']['cells']),64)

  def test_deterministic_without_claim_of_independent_validity(self):
    self.assertEqual(build('a'*40),self.data)
    self.assertEqual(snapshot(CompartmentalNeuriteEcology(),CompartmentalNeuriteEcology().seed())['energy_residual'],0)
    with self.assertRaises(ValueError):
      build('')
    self.assertTrue(any('not drawn' in s for s in self.data['limitations']))

if __name__=='__main__':
  unittest.main()

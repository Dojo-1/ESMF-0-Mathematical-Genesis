"""Source-backed film witnesses physical model states, not target competence."""
import unittest
from research.export_g0_ai_film import export, fingerprint, SEED
from research.g0_ai_probe import run_seed


class TestG0AIFilm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.film=export('b'*40)

    def test_source_verifiable_and_ledger_closed(self):
        d=dict(self.film)
        sha=d.pop('payload_sha256')
        self.assertEqual(sha,fingerprint(d))
        self.assertEqual(d['seed'],SEED)
        self.assertEqual(len(d['runs']),3)
        self.assertIn('not intelligence',d['evidence'])
        for run in d['runs']:
            self.assertEqual(len(run['frames']),64)
            for idx,frame in enumerate(run['frames']):
                s=frame['state']
                self.assertEqual(frame['index'],idx)
                self.assertEqual(fingerprint(s),frame['state_sha256'])
                self.assertEqual((s['energy_residual'],s['material_residual']),(0,0))
                self.assertEqual(len(s['units']),20)
                self.assertTrue(all(u['atp']>=0 and u['material']>=0 for u in s['units']))
                self.assertTrue(all(abs(sum(u['rule_mix'])-1) < .000004 for u in s['units']))
                self.assertLessEqual(len(s['contacts']),72)
                self.assertTrue(all(e['source']!=e['target'] for e in s['contacts']))
                self.assertTrue(all(r['kind']=='[R]_pruned_contact' for r in s['recent_remainders']))

    def test_same_initial_tissue_and_world_but_different_development(self):
        a,b,c=self.film['runs']
        self.assertEqual((a['mode'],b['mode'],c['mode']),('adaptive','fixed','shuffled'))
        starts=[run['frames'][0]['state'] for run in (a,b,c)]
        self.assertEqual(starts[0],starts[1]);self.assertEqual(starts[1],starts[2])
        for frames in zip(*(r['frames'] for r in (a,b,c))):
            self.assertEqual(len({f['state']['tick'] for f in frames}),1)
            self.assertEqual(len({f['state']['observed_label'] for f in frames}),1)
            self.assertEqual(len({tuple(f['state']['observed_cue'] or ()) for f in frames}),1)
        self.assertGreater(a['frames'][-1]['state']['growth'],0)
        self.assertEqual(b['frames'][-1]['state']['growth'],0)
        self.assertGreater(c['frames'][-1]['state']['growth'],0)
        self.assertEqual(a['frames'][-1]['state']['tick'],720)
        self.assertTrue(any(f['state']['recent_remainders'] for f in a['frames']))

    def test_final_score_matches_full_registered_development_stream(self):
        results=run_seed(SEED)
        for run in self.film['runs']:
            mode=run['mode']
            final=run['frames'][-1]['state']
            self.assertAlmostEqual(final['cumulative_brier_after_warmup'],
                                   results['scores'][mode]['mean'],places=12)
            self.assertEqual(final['growth'],results['cost'][-1][mode]['growth'])

    def test_determinism_and_provenance(self):
        self.assertEqual(self.film,export('b'*40))
        with self.assertRaises(ValueError):export('')
        self.assertIn('stochastic',next(x for x in self.film['limitations'] if 'stochastic' in x))

if __name__=='__main__':unittest.main()

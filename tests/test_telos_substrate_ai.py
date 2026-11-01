"""Small contractual gates, not evidence for cognitive emergence."""
import unittest
from random import Random
from esmf.telos_substrate_ai import TelosSubstrate


class TestTelosSubstrateAI(unittest.TestCase):
    def test_weakly_differentiated_seed_and_exact_resource_ledger(self):
        model=TelosSubstrate(seed=17)
        u=model.s.units
        self.assertEqual({x.rate for x in u},{.06})
        self.assertEqual({x.gain for x in u},{1.})
        self.assertEqual(len({x.sensor for x in u}),20)
        self.assertEqual(model.s.ledger(),(0,0))
        self.assertTrue(all(x.material>=0 for x in u))
        self.assertFalse(any(hasattr(x,'question') or hasattr(x,'type') for x in u))
        self.assertTrue(0 < len(model.s.contacts) <= model.max_edges)

    def test_seed_sweep_nonnegative_and_no_initial_overallocation(self):
        for seed in range(40,90):
            model=TelosSubstrate(seed=seed)
            self.assertTrue(all(u.material>=0 for u in model.s.units))
            self.assertEqual(model.s.ledger(),(0,0))

    def test_morphogenesis_not_trained_on_external_labels(self):
        a=TelosSubstrate(seed=121)
        b=TelosSubstrate(seed=121)
        rng=Random(44)
        for _ in range(160):
            x=tuple(rng.choice((-1,1)) for k in range(4))
            a.predict(x); b.predict(x)
            a.observe(True); b.observe(False)
        self.assertEqual(list(a.s.contacts),list(b.s.contacts))
        self.assertEqual([(k,e.weight) for k,e in a.s.contacts.items()],
                         [(k,e.weight) for k,e in b.s.contacts.items()])
        self.assertEqual([u.rule_mix for u in a.s.units],[u.rule_mix for u in b.s.units])
        self.assertEqual(a.s.ledger(),(0,0))
        self.assertNotEqual(a.readout,b.readout)

    def test_development_changes_contacts_and_regulatory_mixture(self):
        model=TelosSubstrate(seed=33,max_edges=45)
        initial=len(model.s.contacts)
        rng=Random(223)
        for _ in range(400):
            model.predict(tuple(rng.choice((-1,1)) for _ in range(4)))
            model.observe(rng.choice((True,False)))
            self.assertEqual(model.s.ledger(),(0,0))
        d=model.integrity()
        self.assertEqual(d['cells'],20)
        self.assertGreater(d['growth'],0)
        self.assertGreater(d['regulator_spread'],0)
        self.assertGreater(d['rule_weight_spread'],0)
        self.assertGreater(d['pruning'],0)
        self.assertEqual(d['remainders'],d['pruning'])
        self.assertTrue(all(r.kind=='[R]_pruned_contact' for r in model.s.remainders))
        self.assertEqual(d['ledger'],(0,0))
        self.assertGreaterEqual(d['contacts'],initial)
        self.assertLessEqual(d['contacts'],45)

    def test_strong_fixed_and_morphology_ablation(self):
        rng=Random(50)
        models=[TelosSubstrate(seed=88,mode=k) for k in ('adaptive','fixed','plastic_no_growth','shuffled')]
        starts=[list(m.s.contacts) for m in models]
        self.assertTrue(all(x==starts[0] for x in starts))
        for i in range(170):
            cue=tuple(rng.choice((-1,1)) for _ in range(4))
            for m in models:
                m.predict(cue);m.observe(i%3==0)
        self.assertGreater(models[0].s.growth_events,0)
        self.assertEqual(models[1].s.growth_events,0)
        self.assertEqual(models[2].s.growth_events,0)
        self.assertTrue(all(m.s.ledger()==(0,0) for m in models))
        self.assertEqual({u.rate for u in models[1].s.units},{.06})

    def test_invalid_input_and_failed_resources_cannot_advance(self):
        model=TelosSubstrate(seed=9)
        with self.assertRaises(ValueError): model.predict((2,-1,1,1))
        self.assertEqual(model.s.tick,0)
        with self.assertRaises(ValueError):model.observe(True)
        model.predict((1,-1,1,1))
        with self.assertRaises(RuntimeError):model.predict((1,-1,1,1))
        with self.assertRaises(ValueError):model.observe(1)
        model.observe(True)
        model.s.waste += model.s.units[0].atp
        model.s.units[0].atp=0
        before=model.s.tick
        with self.assertRaises(RuntimeError):model.predict((1,-1,1,1))
        self.assertEqual(model.s.tick,before)
        with self.assertRaises(ValueError):TelosSubstrate(seed='9')
        with self.assertRaises(ValueError):TelosSubstrate(seed=8,mode='oracle')


if __name__=='__main__':unittest.main()

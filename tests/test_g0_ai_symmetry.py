"""G0-AI P1: exact equivariance under global sensor-sign inversion.

This is a conditional software-level symmetry property, not an emergent
cognition or a neural physiology theorem.
"""
import unittest
from random import Random
from esmf.telos_substrate_ai import TelosSubstrate


class TestG0AISignEquivariance(unittest.TestCase):
    def test_mirrored_tissue_and_readout_preserve_predictions(self):
        for mode in ('adaptive','fixed','plastic_no_growth','shuffled'):
            positive=TelosSubstrate(seed=17,mode=mode)
            negative=TelosSubstrate(seed=17,mode=mode)
            rng=Random(144)
            for tick in range(180):
                x=tuple(rng.choice((-1,1)) for _ in range(4))
                opposite=tuple(-v for v in x)
                y=x[0]*x[1]>0  # an even nonlinear outcome
                self.assertEqual(positive.predict(x),negative.predict(opposite))
                self.assertEqual(list(positive.s.contacts),list(negative.s.contacts))
                for edge in positive.s.contacts:
                    self.assertEqual(positive.s.contacts[edge].weight,
                                     negative.s.contacts[edge].weight)
                for a,b in zip(positive.s.units,negative.s.units):
                    self.assertEqual(a.activity,-b.activity)
                    self.assertEqual(a.fast_trace,-b.fast_trace)
                    self.assertEqual(a.slow_trace,-b.slow_trace)
                    self.assertEqual(a.rate,b.rate)
                    self.assertEqual(a.gain,b.gain)
                    self.assertEqual(a.rule_mix,b.rule_mix)
                    self.assertEqual(a.material,b.material)
                self.assertEqual(positive.s.ledger(),(0,0))
                self.assertEqual(negative.s.ledger(),(0,0))
                positive.observe(y);negative.observe(y)
                self.assertEqual(positive.readout[0],negative.readout[0])
                for a,b in zip(positive.readout[1:],negative.readout[1:]):
                    self.assertEqual(a,-b)
            self.assertEqual(positive.s.growth_events,negative.s.growth_events)
            self.assertEqual(positive.s.pruning_events,negative.s.pruning_events)
            self.assertEqual(positive.s.remainders,negative.s.remainders)

    def test_nonconstant_parity_evenness(self):
        for first in (-1,1):
            for second in (-1,1):
                y=first*second>0
                self.assertEqual(y,(-first)*(-second)>0)
        self.assertNotEqual(1*1>0,1*(-1)>0)


if __name__=='__main__':unittest.main()

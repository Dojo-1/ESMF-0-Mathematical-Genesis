import unittest

from esmf.experiment import LocalLearner, run, sample
import random


class TestG0(unittest.TestCase):
    def test_reproducibility(self):
        self.assertEqual(run(4, 60), run(4, 60))

    def test_connectivity_bound(self):
        model = LocalLearner(adaptive=True, seed=3)
        rng = random.Random(9)
        for _ in range(120):
            x, y = sample(rng)
            model.update(x, y)
            self.assertTrue(model.valid())

    def test_fixed_does_not_grow(self):
        model = LocalLearner(adaptive=False, seed=3)
        rng = random.Random(9)
        for _ in range(120):
            model.update(*sample(rng))
        self.assertEqual(model.resource, 0)

    def test_predictions_are_probabilities(self):
        m = LocalLearner(adaptive=True, seed=2)
        for x in ([0]*12, [1]*12):
            p, _ = m.predict(x)
            self.assertGreater(p, 0)
            self.assertLess(p, 1)


if __name__ == "__main__":
    unittest.main()

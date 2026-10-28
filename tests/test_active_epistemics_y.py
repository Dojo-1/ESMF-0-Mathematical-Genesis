import random
import unittest

from esmf.active_epistemics_y import (
    Belief, PROBE_BY_NAME, ENERGY_CAP, ENERGY_FLOOR,
    MAX_EVENTS, CONFIDENCE, closed_form_pair, expected_information,
    expected_pair_gain, numerical_selfcheck, simulate,
)


class TestActiveEpistemicsY(unittest.TestCase):
    def test_analytic_pair_posterior_both_orders(self):
        for a,b in ((.7,1.2),(-.8,.7),(1.,-.45),(0.,1.)):
            for se,so in ((.23,.36),(.7,.7),(.13,.51)):
                exact=closed_form_pair(a,b,se,so)
                x=Belief().update(0,a,se).update(1,b,so)
                y=Belief().update(1,b,so).update(0,a,se)
                self.assertAlmostEqual(x.w[1],exact,places=12)
                self.assertAlmostEqual(y.w[1],exact,places=12)

    def test_one_parity_cannot_identify_symmetric_unknown(self):
        rng=random.Random(5931)
        for parity in (0,1):
            belief=Belief()
            for _ in range(100):
                belief.update(parity,rng.gauss(0,1),rng.uniform(.14,.9))
            self.assertLess(abs(belief.w[1]-.5),2e-12)

    def test_info_synergy_and_conditional_switch(self):
        b=Belief()
        self.assertLess(expected_information(b,0,.3),1e-12)
        self.assertLess(expected_information(b,1,.3),1e-12)
        both=expected_pair_gain(b,PROBE_BY_NAME["even_quiet"],
                                PROBE_BY_NAME["odd_quiet"],ENERGY_CAP)
        same=expected_pair_gain(b,PROBE_BY_NAME["even_quiet"],
                                PROBE_BY_NAME["even_quiet"],ENERGY_CAP)
        self.assertGreater(both,.15)
        self.assertLess(same,1e-10)

        after_odd=Belief().update(1,.8,.4)
        self.assertAlmostEqual(after_odd.w[1],.5)
        self.assertLess(expected_information(after_odd,1,.3),1e-11)
        self.assertGreater(expected_information(after_odd,0,.3),.1)

        after_even=Belief().update(0,.8,.4)
        self.assertLess(expected_information(after_even,0,.3),1e-11)
        self.assertGreater(expected_information(after_even,1,.3),.1)

    def test_known_contrast_even_likelihood_is_identical(self):
        # Analytic conditional invariance: y_even=D+eps has no T argument.
        D=1.23
        y=.9
        sd=.42
        logprob_identity=-.5*((y-D)/sd)**2
        logprob_twist=-.5*((y-D)/sd)**2
        self.assertEqual(logprob_identity,logprob_twist)

    def test_budget_and_abstention(self):
        for seed in range(8):
            for twist in (0,1):
                for policy in ("anticipatory","greedy_stop",
                               "alternating_quiet","random",
                               "odd_only","even_only"):
                    r=simulate(seed,twist,policy)
                    self.assertGreaterEqual(r["reserve"],ENERGY_FLOOR-1e-10)
                    self.assertLessEqual(r["n_observations"],MAX_EVENTS)
                    self.assertGreaterEqual(r["spent"],0.)
                    self.assertLessEqual(r["spent"],ENERGY_CAP-ENERGY_FLOOR+1e-10)
                    self.assertGreaterEqual(r["posterior_twist"],0.)
                    self.assertLessEqual(r["posterior_twist"],1.)
                    if r["confident"]:
                        self.assertGreaterEqual(max(r["posterior_twist"],1-r["posterior_twist"]),CONFIDENCE-1e-12)
                    else:
                        self.assertIsNone(r["decision"])
                    if policy in ("odd_only","even_only"):
                        self.assertAlmostEqual(r["posterior_twist"],.5,places=11)
                        self.assertFalse(r["confident"])

    def test_reproducibility_and_exact_real_baseline(self):
        for seed in (0,1,2,5,13):
            for twist in (0,1):
                a=simulate(seed,twist,"anticipatory")
                b=simulate(seed,twist,"matched_real")
                self.assertEqual(a,b|{"policy":"anticipatory"})
                self.assertEqual(a,simulate(seed,twist,"anticipatory"))

    def test_no_myopic_progress_without_complementarity(self):
        a=simulate(0,0,"greedy_stop")
        self.assertEqual(a["n_observations"],0)
        self.assertEqual(a["posterior_twist"],.5)
        self.assertFalse(a["confident"])

    def test_numerical_contract(self):
        self.assertTrue(numerical_selfcheck()["single_parity_50_observations_nonidentifying"])


if __name__=="__main__":
    unittest.main()

"""Independent invariant and behavior checks for G0-AB adaptive ecology."""
import math
import random
import unittest

from esmf.adaptive_ecology_ab import (
    ENERGY_CAP,ENERGY_FLOOR,HORIZON,POLICIES,WORLDS,
    AdaptiveContinuityFilter,AutonomicGate,ResourceLedger,
    transition_rows,neutral_invariant,fiber,fiber_transport,
    simulate,
)
from esmf.causal_bundle_s import mm


class TestAdaptiveEcologyAB(unittest.TestCase):
    def test_stochastic_transition_normalization_and_bounds(self):
        for h in (.005,.045,.16,.30):
            rows=transition_rows(h)
            for row in rows:
                self.assertAlmostEqual(sum(row),1.,places=13)
                self.assertTrue(all(0<=x<=1 for x in row))

    def test_fiber_carries_genuine_neutral_determinant_invariant(self):
        for d0,d1 in ((-.7,.3),(1.2,-2.1),(0.,0.)):
            F=fiber(d0)
            J=fiber_transport(d0,d1)
            actual=mm(J,F)
            expected=fiber(d1)
            for i in range(2):
                for j in range(2):
                    self.assertAlmostEqual(actual[i][j],expected[i][j])
            for twist in (0,1):
                for parity in (0,1):
                    self.assertAlmostEqual(neutral_invariant(d1,twist,parity),1.)

    def test_unobserved_or_one_parity_does_not_identify_orientation(self):
        for parity in (0,1):
            rng=random.Random(9137+parity)
            model=AdaptiveContinuityFilter()
            for i in range(12):
                observation=rng.gauss(0,1)
                model.predictive_step(observation,parity=parity,sd=.4)
                self.assertAlmostEqual(model.probability_twist,.5,places=11)
                self.assertIsNone(model.label())
                self.assertAlmostEqual(
                    model.p_persistent+model.diffuse_weight,1.,places=12)

    def test_joint_provenance_orientation_gate(self):
        model=AdaptiveContinuityFilter()
        # Artificially concentrate conditional orientation on transpose,
        # while classifying most observations as fresh-source noise.
        for k in range(3):
            model.states[0][k].weight=.001
            model.states[1][k].weight=.999*(.10/3)
        model.diffuse_weight=1-sum(h.weight for row in model.states for h in row)
        self.assertGreater(model.probability_twist,.95)
        self.assertIsNone(model.label())
        self.assertGreater(model.diffuse_weight,.8)

    def test_budding_changes_future_dynamics_but_not_orientation_posterior(self):
        f=AdaptiveContinuityFilter()
        for i in range(6):
            f.predictive_step((.3 if i%2 else -.9),parity=i%2,sd=.31)
        previous=f.probability_twist
        old_h=f.hazard
        old_reset=f.p_reset
        event=f.recontextualize(5)
        self.assertAlmostEqual(f.probability_twist,previous,places=12)
        self.assertGreaterEqual(f.hazard,old_h)
        self.assertGreater(f.p_reset,old_reset)
        self.assertTrue(event["orientation_mass_preserved"])
        self.assertEqual(event["typed_remainder"],"[R]_id")

    def test_slow_plasticity_and_static_ablation(self):
        a=AdaptiveContinuityFilter(plastic=True)
        b=AdaptiveContinuityFilter(plastic=False)
        for i in range(12):
            obs=2.4 if i in (5,8) else .23
            a.predictive_step(obs,parity=i%2,sd=.2)
            b.predictive_step(obs,parity=i%2,sd=.2)
        self.assertEqual(a.slow_clock,3)
        self.assertEqual(b.slow_clock,0)
        self.assertNotAlmostEqual(a.hazard,.045,places=5)
        self.assertAlmostEqual(b.hazard,.045,places=12)

    def test_autonomic_mode_and_source_aware_action_change(self):
        filt=AdaptiveContinuityFilter()
        resources=ResourceLedger()
        gate=AutonomicGate("adaptive")
        filt.surprise=2.
        gate.update(2,filt,resources)
        self.assertTrue(gate.focus)
        filt.diffuse_weight=.91
        x=gate.choose(4,filt,resources,random.Random(3))
        y=gate.choose(7,filt,resources,random.Random(3))
        self.assertEqual(x.name,"even_quiet")
        self.assertEqual(y.name,"odd_quiet")
        self.assertEqual(gate.provenance_checks,2)

    def test_resource_ledger_tracks_supply_cost_overflow(self):
        l=ResourceLedger()
        for i in range(40):
            l.replenish()
            l.charge(.1 if i%3==0 else .39)
            l.verify()
        self.assertGreaterEqual(l.reserve,ENERGY_FLOOR)
        self.assertAlmostEqual(
            ENERGY_CAP+l.delivered,l.spent+l.overflow+l.reserve,places=10)

    def test_replay_and_real_equivalence(self):
        for world in ("stable","jumps","diffuse","switch"):
            for twist in (0,1):
                a=simulate(7,twist,world,"adaptive")
                b=simulate(7,twist,world,"equivalent_real")
                self.assertEqual({k:v for k,v in a.items() if k!="policy"},
                                 {k:v for k,v in b.items() if k!="policy"})
                self.assertEqual(a,simulate(7,twist,world,"adaptive"))

    def test_all_regimes_and_policies_respect_budgets(self):
        for world in WORLDS:
            for policy in POLICIES:
                r=simulate(11,1,world,policy)
                self.assertTrue(0<=r["probes"]<=HORIZON)
                self.assertGreaterEqual(r["reserve"],ENERGY_FLOOR-1e-10)
                self.assertLessEqual(r["reserve"],ENERGY_CAP+1e-10)
                self.assertGreaterEqual(r["hazard_final"],.005)
                self.assertLessEqual(r["hazard_final"],.30)
                self.assertGreaterEqual(r["p_independent"],0.)
                self.assertLessEqual(r["p_independent"],1.)
                self.assertTrue(math.isfinite(r["moment_match_dispersion"]))
                self.assertEqual(len(r["trace"]),HORIZON)

    def test_event_gated_developmental_bud_is_reachable(self):
        rows=[simulate(seed,twist,"jumps","adaptive")
              for seed in range(16) for twist in (0,1)]
        self.assertGreater(sum(r["bud_count"] for r in rows),0)
        for r in rows:
            for event in r["buds"]:
                self.assertGreaterEqual(event["boundary_after_tick"],2)
                self.assertEqual(event["typed_remainder"],"[R]_id")
                self.assertTrue(event["orientation_mass_preserved"])


if __name__=="__main__":
    unittest.main()

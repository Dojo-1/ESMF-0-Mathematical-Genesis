import unittest

from esmf.causal_bundle_s import (
    selfcheck, physical_world, predict, wrap, tangent_step,
    induced_tangent, det, run
)

class CausalBundleSTests(unittest.TestCase):
    def test_signature_and_gauge(self):
        r=selfcheck()
        self.assertEqual(r["fiber_signature"],"(2,2) from determinant")
        self.assertLess(r["chart_metamorphic"]["max_fiber_chart_gap"],1e-12)
        self.assertTrue(r["chart_metamorphic"]["nontrivial_frame_change_seen"])

    def test_action_inverse(self):
        state=(2.95,.6)
        theta=(.61,-.52)
        for action in ("X","Y"):
            y=physical_world(state,((action,.7),(action,-.7)),theta,"noncommuting")
            self.assertLess(abs(wrap(y[0]-state[0])),1e-12)
            self.assertLess(abs(y[1]-state[1]),1e-12)

    def test_det_jacobian(self):
        state=(.4,.6)
        for action in ("X","Y"):
            self.assertAlmostEqual(det(tangent_step(state,action,.7,(.6,-.5))),1.)
        end,F=induced_tangent(state,(("X",.7),("Y",-.4)),(.6,-.5))
        self.assertAlmostEqual(det(F),1.,places=12)

    def test_ordinary_real_recurrence_ties(self):
        theta=(.61,-.52)
        actions=(("X",.5),("Y",.8),("X",-.4),("Y",.2))
        state=(.4,-.35)
        plain=physical_world(state,actions,theta,"noncommuting")
        predicted=predict(state,actions,theta,"nonlinear")
        self.assertAlmostEqual(plain[0],predicted[0])
        self.assertAlmostEqual(plain[1],predicted[1])

    def test_finite_diff_parameter_gradients(self):
        state=(.7,.4)
        theta=(.42,-.57)
        path=(("X",.6),("Y",-.5),("X",-.2))
        predicted,partials=predict(state,path,theta,"nonlinear",True)
        eps=1e-6
        for j in (0,1):
            plus=list(theta);minus=list(theta)
            plus[j]+=eps;minus[j]-=eps
            hi=predict(state,path,plus,"nonlinear")
            lo=predict(state,path,minus,"nonlinear")
            self.assertAlmostEqual(partials[0][j],wrap(hi[0]-lo[0])/(2*eps),places=6)
            self.assertAlmostEqual(partials[1][j],(hi[1]-lo[1])/(2*eps),places=6)

    def test_replay(self):
        a=run(0,"noncommuting",train_size=40,epochs=40)
        b=run(0,"noncommuting",train_size=40,epochs=40)
        for model in ("flat","linear","nonlinear"):
            self.assertEqual(a["scores"][model]["heldout_loss"],
                             b["scores"][model]["heldout_loss"])

if __name__=="__main__":
    unittest.main()

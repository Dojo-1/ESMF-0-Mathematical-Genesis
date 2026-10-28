import unittest
from esmf.context_bundle_t import encode,decode,calibrate,gauge_check,chart_jac,run
from esmf.causal_bundle_s import wrap

class TestContextBundleT(unittest.TestCase):
    def test_three_chart_reconstruction_and_seam(self):
        coeff=(.38,-.21)
        for state in ((3.12,.47),(-3.10,-.72),(.4,0.),(2.8,-.2)):
            for a in range(3):
                for b in range(3):
                    via=encode(decode(encode(state,a,coeff),a,coeff),b,coeff)
                    direct=encode(state,b,coeff)
                    self.assertLess(abs(wrap(via[0]-direct[0])),1e-12)
                    self.assertLess(abs(via[1]-direct[1]),1e-12)

    def test_analytic_chart_jacobian(self):
        state=(.7,.38)
        coeff=(.38,-.21)
        eps=1e-6
        for chart in (0,1,2):
            analytic=chart_jac(state,chart,coeff)
            for j in range(2):
                hi=list(state);lo=list(state)
                hi[j]+=eps;lo[j]-=eps
                x=encode(hi,chart,coeff);y=encode(lo,chart,coeff)
                dq=wrap(x[0]-y[0])/(2*eps)
                dp=(x[1]-y[1])/(2*eps)
                self.assertAlmostEqual(dq,analytic[0][j],places=6)
                self.assertAlmostEqual(dp,analytic[1][j],places=6)

    def test_calibration_recovery(self):
        for seed in (0,2,5,7):
            true,est,den=calibrate(seed)
            self.assertGreater(den,.1)
            self.assertLess(max(abs(a-b) for a,b in zip(true,est)),.02)

    def test_gauge_cocycle(self):
        summary=gauge_check((.33,-.24))
        self.assertLess(summary["max_gluing_residual"],1e-12)
        self.assertLess(summary["max_chart_transport_residual"],1e-12)
        self.assertGreater(summary["max_frame_blind_error"],.1)

    def test_episode_replay(self):
        a=run(3,"noncommuting",epochs=40)
        b=run(3,"noncommuting",epochs=40)
        self.assertEqual(a["losses"],b["losses"])

if __name__=="__main__":
    unittest.main()

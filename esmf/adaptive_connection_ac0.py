"""G0-AC0: nonstationary, metric-compatible neutral bundle connection.

A_t = a_t X dtheta + b_t Y dz with X,Y in o(2,2), commuting with
the fixed orientation-reversing transition R but NOT with each other.
Two partial linear sensor readings calibrate one coefficient per transport.
All local chart and sensor covectors are transported covariantly.

Models have two fitted real coefficients, with RLS covariance and
fixed/variable process inflation. No biological cognition or measured power.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random
import time

from .geometry_k import identity,matmul,matvec,inverse,transpose,bilinear
from .bundle_p import ETA,commutator,maxdiff,plus,scaled
from .global_bundle_q import NeutralCylinderBundle,TWIST,TAU,det_sign


D=4
HORIZON=60
BURN_IN=12
SD=.02
X=tuple(tuple(1. if (i,j) in ((1,2),(2,1)) else 0.
              for j in range(D)) for i in range(D))
Y=tuple(tuple(1. if (i,j) in ((1,3),(3,1)) else 0.
              for j in range(D)) for i in range(D))
GENERATORS=(X,Y)
MODEL_NAMES=("adaptive","constant_forgetting","no_forgetting",
             "frozen_after_calibration","equivalent_real")
WORLDS=("static","smooth_drift","abrupt_step","sign_reversal","rapidly_variable")


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def norm2(a):
    return sum(x*x for x in a)


def boost_transport(axis,coeff,h):
    pair=((1,2),(1,3))[axis]
    i,j=pair
    s=-h*coeff
    c,u=math.cosh(s),math.sinh(s)
    rows=[list(row) for row in identity(D)]
    rows[i][i]=rows[j][j]=c
    rows[i][j]=rows[j][i]=u
    return tuple(tuple(v for v in row) for row in rows)


def derivative_transport(axis,coeff,h):
    i,j=((1,2),(1,3))[axis]
    s=-h*coeff
    c,u=math.cosh(s),math.sinh(s)
    out=[[0.]*D for _ in range(D)]
    out[i][i]=out[j][j]=-h*u
    out[i][j]=out[j][i]=-h*c
    return tuple(tuple(row) for row in out)


@dataclass(frozen=True)
class PartialProbe:
    axis:int
    h:float
    theta0:float
    theta1:float
    z0:float
    z1:float
    chart0:int
    chart1:int
    source_local:tuple
    sensors_local:tuple # two dual vectors with local target coordinates


def make_probe(rng,axis,bundle,cross_seam=False):
    if cross_seam and axis != 0:
        raise ValueError("the cylinder seam can only be crossed by a theta probe")
    theta_offset=rng.uniform(.095,.13)
    h=rng.uniform(.05,.085)
    theta0=(TAU-.5*h if cross_seam else math.pi-theta_offset)
    theta1=theta0+h if axis==0 else theta0
    z0=rng.uniform(-.65,.65)
    z1=z0+h if axis==1 else z0
    chart0,chart1=((0,1) if cross_seam else (rng.randrange(3),rng.randrange(3)))
    B0,_,_=bundle.frame(chart0,theta0,z0)
    B1,_,_=bundle.frame(chart1,theta1,z1)
    v=tuple(rng.gauss(0,.8) for i in range(D))
    source=matvec(B0,v)
    invBtT=transpose(inverse(B1))
    sensors=[]
    for k in range(2):
        covector=tuple(rng.gauss(0,.65) for i in range(D))
        sensors.append(matvec(invBtT,covector))
    return PartialProbe(axis,h,theta0,theta1,z0,z1,chart0,chart1,
                        source,tuple(sensors))


def predict(probe,coeff,bundle,gradient=False):
    B0,_,_=bundle.frame(probe.chart0,probe.theta0,probe.z0)
    B1,_,_=bundle.frame(probe.chart1,probe.theta1,probe.z1)
    vref=matvec(inverse(B0),probe.source_local)
    U=boost_transport(probe.axis,coeff,probe.h)
    crossed_seam=(probe.axis==0 and
                  math.floor(probe.theta0/TAU)!=math.floor(probe.theta1/TAU))
    if crossed_seam:
        U=matmul(TWIST,U)
    field=matvec(B1,matvec(U,vref))
    values=tuple(dot(c,field) for c in probe.sensors_local)
    if not gradient:
        return values
    dU=derivative_transport(probe.axis,coeff,probe.h)
    if crossed_seam:
        dU=matmul(TWIST,dU)
    derivative=matvec(B1,matvec(dU,vref))
    grads=tuple(dot(c,derivative) for c in probe.sensors_local)
    return values,grads


def source_coefficients(seed,t,world):
    if world=="static":
        return (.37+.03*math.sin(seed),-.32+.035*math.cos(seed))
    if world=="smooth_drift":
        phase=seed*.37
        return (.40+.18*math.sin(.15*t+phase),
                -.33+.16*math.cos(.12*t+.6*phase))
    if world=="abrupt_step":
        return ((.31,-.24) if t<30 else (.77,-.69))
    if world=="sign_reversal":
        return ((.58,-.48) if t<30 else (-.55,.49))
    if world=="rapidly_variable":
        rng=random.Random(seed*10217+t*3701+911)
        return (rng.uniform(-.7,.7),rng.uniform(-.7,.7))
    raise ValueError(world)


def metric_for_chart(bundle,chart,theta,z):
    return bundle.metric(chart,theta,z)


def curvature(coeffs):
    return scaled(commutator(X,Y),coeffs[0]*coeffs[1])


def math_checks(bundle=None):
    bundle=bundle or NeutralCylinderBundle()
    zero=scaled(identity(D),0.)
    assert maxdiff(plus(matmul(transpose(X),ETA),matmul(ETA,X)),zero)<1e-12
    assert maxdiff(plus(matmul(transpose(Y),ETA),matmul(ETA,Y)),zero)<1e-12
    assert maxdiff(matmul(X,TWIST),matmul(TWIST,X))<1e-12
    assert maxdiff(matmul(Y,TWIST),matmul(TWIST,Y))<1e-12
    assert maxdiff(commutator(X,Y),zero)>.5
    rng=random.Random(1732)
    max_chart=max_metric=max_gradient=max_curvature=0.
    wrong_chart_gap=0.
    for i in range(60):
        axis=i%2
        p=make_probe(rng,axis,bundle)
        coeff=.31 if axis==0 else -.29
        B0,_,_=bundle.frame(p.chart0,p.theta0,p.z0)
        B1,_,_=bundle.frame(p.chart1,p.theta1,p.z1)
        U=boost_transport(axis,coeff,p.h)
        local=matmul(matmul(B1,U),inverse(B0))
        h0=metric_for_chart(bundle,p.chart0,p.theta0,p.z0)
        h1=metric_for_chart(bundle,p.chart1,p.theta1,p.z1)
        max_metric=max(max_metric,maxdiff(matmul(matmul(transpose(local),h1),local),h0))
        y,g=predict(p,coeff,bundle,gradient=True)
        vref=matvec(inverse(B0),p.source_local)
        for j,sensor in enumerate(p.sensors_local):
            cref=matvec(transpose(B1),sensor)
            intrinsic=dot(cref,matvec(U,vref))
            max_chart=max(max_chart,abs(y[j]-intrinsic))
        eps=1e-6
        hi=predict(p,coeff+eps,bundle)
        lo=predict(p,coeff-eps,bundle)
        for j in (0,1):
            max_gradient=max(max_gradient,abs(g[j]-(hi[j]-lo[j])/(2*eps)))
        wrong_chart_gap=max(wrong_chart_gap,
                            maxdiff(local,U))
        # Genuine two-chart curvature covariance, not self-comparison.
        cur=curvature((.31,-.29))
        source_cur=matmul(matmul(B0,cur),inverse(B0))
        target_cur=matmul(matmul(B1,cur),inverse(B1))
        gauge=matmul(B1,inverse(B0))
        max_curvature=max(max_curvature,maxdiff(
            target_cur,matmul(matmul(gauge,source_cur),inverse(gauge))))
    # Winding orientation remains det-negative while connection changes.
    monodromy=[]
    for a in (.2,.8,-.6):
        U=boost_transport(0,a,2*math.pi)
        W=matmul(TWIST,U)
        monodromy.append(det_sign(W))
        assert maxdiff(matmul(matmul(transpose(W),ETA),W),ETA)<1e-10
    assert all(s==-1 for s in monodromy)
    assert max(max_chart,max_metric,max_gradient,max_curvature)<1e-7
    assert wrong_chart_gap>.01
    return {"paths":60,"max_chart_prediction_gap":max_chart,
            "max_metric_compatibility_residual":max_metric,
            "max_independent_gradient_gap":max_gradient,
            "max_wrong_chart_transport_gap":wrong_chart_gap,
            "max_curvature_gauge_residual":max_curvature,
            "winding_det_signs":monodromy,
            "curvature_nonzero":maxdiff(curvature((.31,-.29)),zero)>.01}


class RLSConnection:
    def __init__(self,policy):
        self.policy=policy
        self.coeff=[0.,0.]
        self.var=[1.,1.]
        self.events=0
        self.burst=[0,0]
        self.plasticity_triggers=0

    def update(self,probe,observations,bundle):
        axis=probe.axis
        if self.policy=="frozen_after_calibration" and self.events>=BURN_IN:
            self.events+=1
            return
        policy="adaptive" if self.policy=="equivalent_real" else self.policy
        if policy=="adaptive":
            q=.0008+(.04 if self.burst[axis]>0 else 0.)
        elif policy=="constant_forgetting":
            q=.01
        else:
            q=0.
        self.var[axis]+=q
        errors=[]
        for obs in observations:
            values,grads=predict(probe,self.coeff[axis],bundle,True)
            # The two independent sensor readings are updated sequentially.
            j=len(errors)
            residual=obs-values[j]
            grad=grads[j]
            predictive_var=SD*SD+grad*grad*self.var[axis]
            gain=self.var[axis]*grad/predictive_var
            self.coeff[axis]=max(-2.,min(2.,self.coeff[axis]+gain*residual))
            self.var[axis]=max(1e-11,self.var[axis]*SD*SD/predictive_var)
            errors.append(residual*residual/predictive_var)
        if policy=="adaptive":
            # Trigger changes future q, not this observation's just-used gain.
            if max(errors)>6.:
                self.burst[axis]=5
                self.plasticity_triggers+=1
            else:
                self.burst[axis]=max(0,self.burst[axis]-1)
        self.events+=1
        if not all(math.isfinite(v) for v in (*self.coeff,*self.var)):
            raise ArithmeticError("connection parameter instability")


def simulate(seed,world,policy):
    if policy not in MODEL_NAMES:
        raise ValueError(policy)
    rng=random.Random(seed*9337+691)
    bundle=NeutralCylinderBundle()
    learner=RLSConnection(policy)
    eval_errors=[]
    coeff_errors=[]
    curvature_errors=[]
    for t in range(HORIZON):
        true=source_coefficients(seed,t,world)
        axis=t%2
        train_probe=make_probe(rng,axis,bundle)
        target=predict(train_probe,true[axis],bundle)
        observed=tuple(v+rng.gauss(0,SD) for v in target)
        learner.update(train_probe,observed,bundle)
        eval_probe=make_probe(rng,axis,bundle)
        if t>=BURN_IN:
            truth=predict(eval_probe,true[axis],bundle)
            forecast=predict(eval_probe,learner.coeff[axis],bundle)
            eval_errors.append(sum((x-y)**2 for x,y in zip(forecast,truth))/2)
            coeff_errors.append(sum(abs(learner.coeff[k]-true[k]) for k in (0,1))/2)
            curvature_errors.append(maxdiff(curvature(learner.coeff),curvature(true)))
    return {"seed":seed,"world":world,"policy":policy,
            "heldout_mse":sum(eval_errors)/len(eval_errors),
            "mean_coeff_abs_error":sum(coeff_errors)/len(coeff_errors),
            "mean_curvature_entry_error":sum(curvature_errors)/len(curvature_errors),
            "final_coeff":learner.coeff[:],
            "surprise_triggers":learner.plasticity_triggers,
            "learned_state_dimension":2,
            "algorithmic_covariance_dimension":2}


def summary(rows):
    return {"world":rows[0]["world"],"policy":rows[0]["policy"],
            "episodes":len(rows),
            "mean_heldout_mse":sum(r["heldout_mse"] for r in rows)/len(rows),
            "mean_coeff_abs_error":sum(r["mean_coeff_abs_error"] for r in rows)/len(rows),
            "mean_curvature_entry_error":sum(r["mean_curvature_entry_error"] for r in rows)/len(rows),
            "mean_plasticity_triggers":sum(r["surprise_triggers"] for r in rows)/len(rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=32)
    args=p.parse_args()
    if not 1<=args.seeds<=128:
        p.error("seeds 1..128")
    print("MATH",json.dumps(math_checks()))
    for world in WORLDS:
        for model in MODEL_NAMES:
            start=time.perf_counter()
            rs=[simulate(i,world,model) for i in range(args.seeds)]
            report=summary(rs)
            report["wall_seconds"]=time.perf_counter()-start
            print("MODEL",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()

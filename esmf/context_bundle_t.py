"""G0-T: learn overlap calibration, then predict interventions across local charts.

Three chart presentations of the same cylinder state have a nontrivial
position-dependent Jacobian. Simple least squares learns two chart coefficients
from paired unlabeled observations; action dynamics are fitted on short paths.

This is a gluing/representation robustness experiment, not an intrinsic
advantage over an equally calibrated ordinary real recurrent predictor.
"""
import argparse
import json
import math
import random
from .causal_bundle_s import (
    PI,wrap,sech2,make_episodes,fit,predict,physical_world,obs_loss,mm,mi,det,diff,tangent_step
)


def encode(state,chart,params):
    q,p=state
    k,l=params
    if chart==0:
        return wrap(q),p
    if chart==1:
        return wrap(q+k*math.tanh(p)),math.sinh(p)
    if chart==2:
        return wrap(q+l*math.sin(p)),2*p
    raise ValueError(chart)


def decode(obs,chart,params):
    q,t=obs
    k,l=params
    if chart==0:
        return wrap(q),t
    if chart==1:
        p=math.asinh(t)
        return wrap(q-k*math.tanh(p)),p
    if chart==2:
        p=t/2
        return wrap(q-l*math.sin(p)),p
    raise ValueError(chart)


def chart_jac(state,chart,params):
    q,p=state
    k,l=params
    if chart==0:
        return ((1.,0.),(0.,1.))
    if chart==1:
        return ((1.,k*sech2(p)),(0.,math.cosh(p)))
    if chart==2:
        return ((1.,l*math.cos(p)),(0.,2.))
    raise ValueError(chart)


def calibrate(seed,n=32):
    rng=random.Random(seed+4422)
    true=(rng.uniform(.18,.6),rng.uniform(-.55,.55))
    # independent noisy paired samples, not labeled interventions
    observations=[]
    for _ in range(n):
        q=rng.uniform(-PI+.2,PI-.2)
        p=rng.uniform(-1.1,1.1)
        basis=(q,p)
        c1=encode(basis,1,true)
        c2=encode(basis,2,true)
        o1=(wrap(c1[0]+rng.gauss(0,.004)),c1[1]+rng.gauss(0,.002))
        o2=(wrap(c2[0]+rng.gauss(0,.004)),c2[1]+rng.gauss(0,.002))
        observations.append((basis,o1,o2))
    den1=sum(math.tanh(src[1])**2 for src,_,_ in observations)
    den2=sum(math.sin(src[1])**2 for src,_,_ in observations)
    if den1<.01 or den2<.01:
        raise ArithmeticError("calibration not identifiable")
    est1=sum(math.tanh(src[1])*wrap(o1[0]-src[0])
             for src,o1,_ in observations)/den1
    est2=sum(math.sin(src[1])*wrap(o2[0]-src[0])
             for src,_,o2 in observations)/den2
    return true,(est1,est2),min(den1,den2)


def charted(episodes,rng,true_coeffs):
    return [((ci,encode(initial,ci,true_coeffs)),
             actions,
             (cf,encode(target,cf,true_coeffs)))
            for initial,actions,target in episodes
            for ci,cf in [(rng.randrange(3),rng.randrange(3))]]


def plain_episodes(charted_episodes,estimated):
    return [(decode(initial,ci,estimated),actions,decode(target,cf,estimated))
            for (ci,initial),actions,(cf,target) in charted_episodes]


def naive_episodes(charted_episodes):
    return [(initial,actions,target)
            for (ci,initial),actions,(cf,target) in charted_episodes]


def score_charted(charted_episodes,original_episodes,params,model,estimated,naive=False):
    losses=[]
    for ((ci,initial_obs),actions,(cf,target_obs)), (_,_,physical_target) in zip(charted_episodes,original_episodes):
        if naive:
            pred_obs=predict(initial_obs,actions,params,model)
            # Naive model assumes source and target chart coordinates can be
            # compared without transport. Its target is interpreted in cf.
            predicted=decode(pred_obs,cf,estimated)
        else:
            initial=decode(initial_obs,ci,estimated)
            predicted=predict(initial,actions,params,model)
        losses.append(obs_loss(predicted,physical_target))
    return sum(losses)/len(losses)


def gauge_check(coeffs):
    rng=random.Random(73411)
    max_gluing=max_frame=max_neutral=0.
    naive_gap=0.
    a,b=.64,-.49
    for _ in range(80):
        start=(rng.uniform(-PI,PI),rng.uniform(-.8,.8))
        paths=((rng.choice(("X","Y")),rng.choice((-1.,1.))*rng.uniform(.3,.9))
               for j in range(rng.randint(2,7)))
        path=tuple(paths)
        for ca in range(3):
            for cb in range(3):
                local=encode(start,ca,coeffs)
                end=encode(decode(local,ca,coeffs),cb,coeffs)
                direct=encode(start,cb,coeffs)
                max_gluing=max(max_gluing,abs(wrap(end[0]-direct[0])),abs(end[1]-direct[1]))
        F=((1.,0.),(0.,1.))
        state=start
        current_chart=rng.randrange(3)
        Flocal=chart_jac(state,current_chart,coeffs)
        for action,d in path:
            J=tangent_step(state,action,d,(a,b))
            nxt=physical_world(state,((action,d),),(a,b),"noncommuting")
            next_chart=rng.randrange(3)
            B0=chart_jac(state,current_chart,coeffs)
            B1=chart_jac(nxt,next_chart,coeffs)
            Jlocal=mm(mm(B1,J),mi(B0))
            Flocal=mm(Jlocal,Flocal)
            F=mm(J,F)
            max_frame=max(max_frame,diff(Flocal,mm(B1,F)))
            max_neutral=max(max_neutral,abs(det(Flocal)/det(B1)-det(F)))
            naive_gap=max(naive_gap,diff(Jlocal,J))
            state=nxt
            current_chart=next_chart
    assert max(max_gluing,max_frame,max_neutral)<1e-9
    assert naive_gap>1e-3
    return {"tested_paths":80,"max_gluing_residual":max_gluing,
            "max_chart_transport_residual":max_frame,
            "max_neutral_invariant_residual":max_neutral,
            "max_frame_blind_error":naive_gap}


def run(seed,regime="noncommuting",epochs=125):
    true_physical,train,long,loops=make_episodes(seed,regime)
    true_coeff,estimated,ident=calibrate(seed)
    rng=random.Random(seed+81104)
    ct=charted(train,rng,true_coeff)
    cl=charted(long,rng,true_coeff)
    cq=charted(loops,rng,true_coeff)
    learner=fit(plain_episodes(ct,estimated),"nonlinear",epochs)
    oracle=fit(plain_episodes(ct,true_coeff),"nonlinear",epochs)
    blind=fit(naive_episodes(ct),"nonlinear",epochs)
    result={"seed":seed,"regime":regime,
            "calibration_true":[round(x,6) for x in true_coeff],
            "calibration_estimated":[round(x,6) for x in estimated],
            "calibration_abs_error":max(abs(x-y) for x,y in zip(true_coeff,estimated)),
            "calibration_denominator":ident,
            "losses":{
                "calibrated":score_charted(cl,long,learner,"nonlinear",estimated),
                "oracle":score_charted(cl,long,oracle,"nonlinear",true_coeff),
                "chart_blind":score_charted(cl,long,blind,"nonlinear",estimated,True),
                "calibrated_loops":score_charted(cq,loops,learner,"nonlinear",estimated),
                "chart_blind_loops":score_charted(cq,loops,blind,"nonlinear",estimated,True)},
            "ordinary_calibrated_recurrence_ties":True}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--regimes",default="noncommuting,commuting")
    args=parser.parse_args()
    seeds=[int(v) for v in args.seeds.split(",")]
    regimes=args.regimes.split(",")
    if not 1<=len(seeds)<=32 or any(s not in ("noncommuting","commuting") for s in regimes):
        parser.error("invalid regime/seeds")
    invariant=gauge_check((.33,-.24))
    rows=[run(seed,regime) for regime in regimes for seed in seeds]
    summary={}
    for regime in regimes:
        subset=[r for r in rows if r["regime"]==regime]
        summary[regime]={k:sum(x["losses"][k] for x in subset)/len(subset)
                         for k in ("calibrated","oracle","chart_blind","calibrated_loops","chart_blind_loops")}
    print(json.dumps({"gauge_invariants":invariant,"mean_loss":summary,"rows":rows},indent=2))


if __name__=="__main__":
    main()

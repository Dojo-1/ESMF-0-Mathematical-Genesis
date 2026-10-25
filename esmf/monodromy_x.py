"""G0-X: infer Z2 neutral monodromy from noisy causal fiber traces.

A model selects IDENTITY or TRANSPOSE seam with a handful of odd-winding
observations. This is binary model selection within a favorable simulator,
not autonomous topology discovery, intelligence, or a unique bundle advantage.
"""
import argparse
import json
import math
import random
from .causal_bundle_s import PI,TAU,wrap,make_episodes,fit,tangent_step,mm,physical_world
from .associated22_u import flatten,neutral_square,H
from .orientation_seam_w import P
from .geometry_k import matvec,matmul,transpose

ALT=((0.,0.,0.,1.),(0.,1.,0.,0.),(0.,0.,1.,0.),(1.,0.,0.,0.))


def simulate(start,actions,F0,params,twist=False):
    q_lift,p=start
    q_start=q_lift
    F=F0
    for action,delta in actions:
        pre=(wrap(q_lift),p)
        J=tangent_step(pre,action,delta,params)
        F=mm(J,F)
        if action=="X":
            q_lift+=params[0]*delta*math.tanh(p)
        elif action=="Y":
            p+=params[1]*delta*math.sin(q_lift)
        else:
            raise ValueError(action)
    winding=math.floor((q_lift+PI)/TAU)-math.floor((q_start+PI)/TAU)
    physical=(wrap(q_lift),p)
    ref=physical_world(start,actions,params,"noncommuting")
    if abs(wrap(physical[0]-ref[0]))>1e-9 or abs(physical[1]-ref[1])>1e-9:
        raise AssertionError("independent physical replay mismatch")
    vector=flatten(F)
    if winding%2:
        if twist=="alternate":
            vector=matvec(ALT,vector)
        elif twist:
            vector=matvec(P,vector)
    if abs(neutral_square(vector)-neutral_square(flatten(F0)))>1e-8:
        raise AssertionError("neutral form was not preserved")
    return physical,winding,vector


def initial_frame(rng):
    while True:
        f=((rng.uniform(-1.2,1.2),rng.uniform(-1.2,1.2)),
           (rng.uniform(-1.2,1.2),rng.uniform(-1.2,1.2)))
        if abs(f[0][1]-f[1][0])>.35:
            return f


def episode(rng,params,twist,kind="odd",noise=.006):
    a,b=params
    if kind=="odd":
        sign=rng.choice((-1.,1.))
        start=(sign*(PI-rng.uniform(.015,.055)),
               sign*rng.uniform(.5,.95))
        if rng.randrange(2):
            actions=(("X",rng.uniform(.78,1.25)),("Y",rng.uniform(-.2,.2)))
        else:
            actions=(("Y",rng.uniform(-.12,.12)),("X",rng.uniform(.85,1.3)))
    elif kind=="zero":
        start=(rng.uniform(-1.,1.),rng.uniform(-.5,.5))
        actions=(("X",rng.uniform(-.22,.22)),("Y",rng.uniform(-.18,.18)))
    elif kind=="even":
        sign=rng.choice((-1.,1.))
        p=sign*rng.uniform(.67,1.)
        start=(0.,p)
        # 2 full windings plus a margin, robust to small estimation error.
        delta=(2*TAU+.55)/(a*abs(math.tanh(p)))
        actions=(("X",delta),)
    else:
        raise ValueError(kind)
    F0=initial_frame(rng)
    end,winding,clean=simulate(start,actions,F0,params,twist)
    if kind=="odd" and winding%2==0:
        raise AssertionError("odd winding probe did not cross seam")
    if kind=="zero" and winding!=0:
        raise AssertionError("zero winding probe crossed seam")
    if kind=="even" and abs(winding)!=2:
        raise AssertionError("double winding probe misgenerated")
    observed=tuple(x+rng.gauss(0.,noise) for x in clean)
    return {"start":start,"F0":F0,"actions":actions,
            "winding":winding,"observed":observed,
            "end":end,"kind":kind}


def predict_fiber(ep,params,twist):
    _,w,vector=simulate(ep["start"],ep["actions"],ep["F0"],params,twist)
    return w,vector


def loss(episodes,params,twist):
    err=0.
    for ep in episodes:
        _,pred=predict_fiber(ep,params,twist)
        err+=sum((x-y)**2 for x,y in zip(pred,ep["observed"]))/4
    return err/len(episodes)


def select(episodes,params,reject_threshold=None):
    # Fundamental identifiability guard: no odd winding => no Z2 evidence.
    if not any(abs(predict_fiber(ep,params,False)[0])%2 for ep in episodes):
        return None,{"identity":loss(episodes,params,False),
                     "transpose":loss(episodes,params,True)}
    s0=loss(episodes,params,False)
    s1=loss(episodes,params,True)
    scores={"identity":s0,"transpose":s1}
    if reject_threshold is not None and min(s0,s1)>reject_threshold:
        return "unmodeled",scores
    return s1<s0,scores


def run(seed,twist):
    rng=random.Random(seed+91291)
    true_params,initial_training,_,_=make_episodes(seed,"noncommuting")
    learned=fit(initial_training,"nonlinear",epochs=125)
    train=[episode(rng,true_params,twist,"odd") for _ in range(10)]
    observed,fitloss=select(train,learned)
    nulls=[episode(rng,true_params,twist,"zero") for _ in range(12)]
    unidentifiable,null_scores=select(nulls,learned)
    assert unidentifiable is None
    test=[episode(rng,true_params,twist,"odd") for _ in range(38)]
    test.extend(episode(rng,true_params,twist,"zero") for _ in range(18))
    doubles=[episode(rng,true_params,twist,"even",noise=.0) for _ in range(10)]
    test.extend(doubles)
    assert all(predict_fiber(e,learned,False)[0]%2==0 for e in doubles)
    return {"seed":seed,"actual_twist":twist,"selected_twist":observed,
            "calibration_loss":fitloss,
            "heldout_loss":{"selected":loss(test,learned,observed),
                            "always_identity":loss(test,learned,False),
                            "always_transpose":loss(test,learned,True)},
            "double_winding_max_odd_vs_even_gap":max(
                max(abs(x-y) for x,y in zip(predict_fiber(e,learned,False)[1],
                                           predict_fiber(e,learned,True)[1]))
                for e in doubles),
            "null_probe_identifiable":unidentifiable is not None,
            "null_probe_model_gap":abs(null_scores["identity"]-null_scores["transpose"])}


def run_alternate(seed):
    # EXPLORATORY out-of-family monitor, added after primary I/P runs.
    rng=random.Random(seed+91291)
    true_params,training,_,_=make_episodes(seed,"noncommuting")
    learned=fit(training,"nonlinear",epochs=125)
    samples=[episode(rng,true_params,"alternate","odd") for _ in range(10)]
    decision,scores=select(samples,learned,reject_threshold=.01)
    return {"seed":seed,"decision":decision,"best_candidate_loss":min(scores.values()),
            "identity_loss":scores["identity"],"transpose_loss":scores["transpose"]}


def selfcheck():
    F=((.7,-.3),(.4,1.1))
    start=(PI-.035,.7)
    a,b=.58,-.44
    _,w,x=simulate(start,(("X",1.1),),F,(a,b),True)
    assert abs(w)==1
    assert abs(neutral_square(x)-neutral_square(flatten(F)))<1e-12
    init,end=select([episode(random.Random(123),(.58,-.44),True,"zero")],(.58,-.44))
    assert init is None
    assert matmul(matmul(transpose(ALT),H),ALT)==H
    return {"twisted_single_winding":w,"nonzero_neutral_quadratic_preserved":True,
            "zero_winding_topology_unidentifiable":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="0,1,2,3,4,5,6,7")
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(",")]
    if not 1<=len(seeds)<=32:
        parser.error("seed count must be 1..32")
    print("INVARIANTS",json.dumps(selfcheck()))
    rows=[run(seed,twist) for seed in seeds for twist in (False,True)]
    for twist in (False,True):
        subset=[r for r in rows if r["actual_twist"]==twist]
        print("SUMMARY",json.dumps({"twist":twist,"seeds":len(subset),
            "correct_selections":sum(r["selected_twist"]==twist for r in subset),
            "mean_selected_loss":sum(r["heldout_loss"]["selected"] for r in subset)/len(subset),
            "mean_identity_loss":sum(r["heldout_loss"]["always_identity"] for r in subset)/len(subset),
            "mean_transpose_loss":sum(r["heldout_loss"]["always_transpose"] for r in subset)/len(subset),
            "zero_winding_unidentifiable":all(not r["null_probe_identifiable"] for r in subset),
            "even_winding_invariance":max(r["double_winding_max_odd_vs_even_gap"] for r in subset)}))
    for r in rows:
        print("SEED",json.dumps(r))


if __name__=="__main__":
    main()

"""G0-S: causal action learning plus honest neutral (2,2) associated-bundle lift.

Environment: two reversible nonlinear area-preserving shears on S1 x R.
The learned causal state predictor is an ordinary 2-parameter real recurrence.
The optional tangent/frame lift gives a 4D neutral determinant geometry but
MUST NOT be credited for prediction improvements over that same recurrence.

Standard-library only, remote-CPU reproducible. No automatic differentiation
dependency, no GPU, no semantic/cognitive claim.
"""
import argparse
import json
import math
import random
import time

PI=math.pi
TAU=2*math.pi


def wrap(q):
    return (q+PI)%TAU-PI


def sech2(z):
    c=math.cosh(z)
    return 1.0/(c*c)


def f(mode,p):
    if mode=="nonlinear":
        return math.tanh(p),sech2(p)
    if mode=="linear":
        return p,1.
    if mode=="flat":
        return 1.,0.
    raise ValueError("unknown mode")


def g(mode,q):
    if mode=="flat":
        return 1.,0.
    return math.sin(q),math.cos(q)


def predict(init,actions,params,mode,derivatives=False):
    """Exactly differentiable (away from angle-cut) two-parameter recurrence."""
    q,p=init
    a,b=params
    dqa=dqb=dpa=dpb=0.
    for action,d in actions:
        if action=="X":
            v,dv=f(mode,p)
            q=wrap(q+a*d*v)
            dqa,dqb=(dqa+d*v+a*d*dv*dpa,
                     dqb+a*d*dv*dpb)
        elif action=="Y":
            v,dv=g(mode,q)
            p=p+b*d*v
            dpa,dpb=(dpa+b*d*dv*dqa,
                     dpb+d*v+b*d*dv*dqb)
        else:
            raise ValueError("unknown action")
    if derivatives:
        return ((q,p),((dqa,dqb),(dpa,dpb)))
    return q,p


def physical_world(init,actions,params,regime):
    """Independent outcome generator; not called through learnable predictor."""
    q,p=init
    a,b=params
    for action,d in actions:
        if regime=="noncommuting":
            if action=="X":
                q=wrap(q+a*d*math.tanh(p))
            elif action=="Y":
                p=p+b*d*math.sin(q)
            else:
                raise ValueError(action)
        elif regime=="commuting":
            if action=="X":
                q=wrap(q+a*d)
            elif action=="Y":
                p=p+b*d
            else:
                raise ValueError(action)
        else:
            raise ValueError(regime)
    return q,p


def path(rng,length):
    return tuple((rng.choice(("X","Y")),rng.choice((-1.,1.))*rng.uniform(.35,.95))
                 for _ in range(length))


def make_episodes(seed,regime,train_size=140,test_size=220,loop_size=120):
    rng=random.Random(seed+11713)
    true_params=(rng.uniform(.48,.70),-rng.uniform(.38,.63))
    def one(actions,initial,noise):
        q,p=physical_world(initial,actions,true_params,regime)
        if noise:
            q=wrap(q+rng.gauss(0.,.012))
            p=p+rng.gauss(0.,.012)
        return (initial,actions,(q,p))
    train=[one(path(rng,rng.randint(1,3)),
               (rng.uniform(-1.6,1.6),rng.uniform(-.75,.75)),True)
           for _ in range(train_size)]
    long=[one(path(rng,rng.randint(4,8)),
              (rng.uniform(-PI+.12,PI-.12),rng.uniform(-.9,.9)),False)
          for _ in range(test_size)]
    loops=[]
    for _ in range(loop_size):
        h=rng.uniform(.4,.9)
        if rng.randrange(2):
            actions=(("X",h),("Y",h),("X",-h),("Y",-h))
        else:
            actions=(("Y",h),("X",h),("Y",-h),("X",-h))
        loops.append(one(actions,(rng.uniform(-2.5,2.5),rng.uniform(-.9,.9)),False))
    return true_params,train,long,loops


def obs_loss(pred,target):
    dq=wrap(pred[0]-target[0])
    dp=pred[1]-target[1]
    return (1.-math.cos(dq))+.5*dp*dp


def fit(episodes,mode,epochs=125,lr=.06):
    params=[0.,0.]
    m=[0.,0.]
    v=[0.,0.]
    for k in range(1,epochs+1):
        da=db=0.
        for initial,actions,target in episodes:
            pred,partials=predict(initial,actions,params,mode,True)
            angle_error=math.sin(wrap(pred[0]-target[0]))
            momentum_error=pred[1]-target[1]
            da+=angle_error*partials[0][0]+momentum_error*partials[1][0]
            db+=angle_error*partials[0][1]+momentum_error*partials[1][1]
        grads=(da/len(episodes),db/len(episodes))
        for j in range(2):
            m[j]=.9*m[j]+.1*grads[j]
            v[j]=.999*v[j]+.001*grads[j]**2
            mh=m[j]/(1-.9**k)
            vh=v[j]/(1-.999**k)
            params[j]-=lr*mh/(math.sqrt(vh)+1e-8)
            if not math.isfinite(params[j]) or abs(params[j])>10:
                raise ArithmeticError("parameter diverged")
    return tuple(params)


def evaluation(episodes,params,mode):
    return sum(obs_loss(predict(init,actions,params,mode),target)
               for init,actions,target in episodes)/len(episodes)


def mm(A,B):
    return ((A[0][0]*B[0][0]+A[0][1]*B[1][0],
             A[0][0]*B[0][1]+A[0][1]*B[1][1]),
            (A[1][0]*B[0][0]+A[1][1]*B[1][0],
             A[1][0]*B[0][1]+A[1][1]*B[1][1]))


def det(A):
    return A[0][0]*A[1][1]-A[0][1]*A[1][0]


def mi(A):
    d=det(A)
    if abs(d)<1e-12:
        raise ValueError("singular")
    return ((A[1][1]/d,-A[0][1]/d),(-A[1][0]/d,A[0][0]/d))


def diff(A,B):
    return max(abs(A[i][j]-B[i][j]) for i in (0,1) for j in (0,1))


def tangent_step(pre,action,d,params):
    q,p=pre
    a,b=params
    if action=="X":
        return ((1.,a*d*sech2(p)),(0.,1.))
    if action=="Y":
        return ((1.,0.),(b*d*math.cos(q),1.))
    raise ValueError(action)


def induced_tangent(init,actions,params):
    s=init
    F=((1.,0.),(0.,1.))
    for action,d in actions:
        J=tangent_step(s,action,d,params)
        F=mm(J,F)
        s=physical_world(s,((action,d),),params,"noncommuting")
    return s,F


def chart_jacobian(s,k=.31):
    # Phi(q,p)=(q+k*tanh(p),sinh(p)) on local lifts; valid diffeomorphism.
    return ((1.,k*sech2(s[1])),(0.,math.cosh(s[1])))


def bracket(q,p,a,b):
    # [Vx,Vy]=D Vy(Vx)-D Vx(Vy)
    return (-a*b*math.sin(q)*sech2(p),
            a*b*math.tanh(p)*math.cos(q))


def chart_check():
    rng=random.Random(8324)
    max_state=max_fiber=max_quad=max_det=0.
    naive_max=0.
    for _ in range(60):
        start=(rng.uniform(-PI,PI),rng.uniform(-.8,.8))
        actions=path(rng,rng.randint(1,7))
        params=(.61,-.52)
        current=start
        F=((1.,0.),(0.,1.))
        chartF=chart_jacobian(current)
        for action,d in actions:
            J=tangent_step(current,action,d,params)
            nxt=physical_world(current,((action,d),),params,"noncommuting")
            Bstart=chart_jacobian(current)
            Bend=chart_jacobian(nxt)
            Jnew=mm(mm(Bend,J),mi(Bstart))
            F=mm(J,F)
            chartF=mm(Jnew,chartF)
            expected=mm(Bend,F)
            max_fiber=max(max_fiber,diff(chartF,expected))
            max_det=max(max_det,abs(det(J)-1))
            max_quad=max(max_quad,abs(det(chartF)/det(Bend)-det(F)))
            # Chart-blind reuse of J generally disagrees with the correct Jnew.
            naive_max=max(naive_max,diff(Jnew,J))
            current=nxt
        ref,base_F=induced_tangent(start,actions,params)
        max_state=max(max_state,abs(wrap(ref[0]-current[0])),abs(ref[1]-current[1]))
        max_fiber=max(max_fiber,diff(base_F,F))
    if max(max_state,max_fiber,max_quad,max_det)>1e-9:
        raise AssertionError("broken frame equivariance")
    return {"paths":60,"max_state_gap":max_state,
            "max_fiber_chart_gap":max_fiber,
            "max_neutral_form_gap":max_quad,
            "max_jacobian_det_error":max_det,
            "nontrivial_frame_change_seen":naive_max>1e-3,
            "chart_blind_transport_gap":naive_max}


def bracket_check():
    s=(.58,.44)
    a,b=.61,-.52
    residuals=[]
    for h in (.10,.05,.025):
        actions=(("X",h),("Y",h),("X",-h),("Y",-h))
        result=physical_world(s,actions,(a,b),"noncommuting")
        actual=(wrap(result[0]-s[0]),result[1]-s[1])
        theory=bracket(*s,a,b)
        magnitude=math.hypot(*actual)
        error=math.hypot(actual[0]-h*h*theory[0],actual[1]-h*h*theory[1])
        residuals.append({"h":h,"loop_displacement":magnitude,
                          "bracket_order2_error":error})
    if not (3.6<residuals[0]["loop_displacement"]/residuals[1]["loop_displacement"]<4.4):
        raise AssertionError("loop has wrong h^2 scaling")
    if not (5.<residuals[0]["bracket_order2_error"]/residuals[1]["bracket_order2_error"]<11.):
        raise AssertionError("bracket approximation not converging cubically")
    return residuals


def selfcheck():
    assert abs(det(((2.,1.),(2.,-1.))))==4.
    # Basis change: matrix quadratic determinant diagonalizes to signature 2+2.
    F=((1.,2.),(3.,4.))
    u1=(F[0][0]+F[1][1])/2
    u2=(F[0][1]-F[1][0])/2
    v1=(F[0][0]-F[1][1])/2
    v2=(F[0][1]+F[1][0])/2
    assert abs(det(F)-(u1*u1+u2*u2-v1*v1-v2*v2))<1e-12
    return {"fiber_signature":"(2,2) from determinant",
            "chart_metamorphic":chart_check(),
            "causal_lie_bracket":bracket_check()}


def run(seed,regime,train_size=140,epochs=125):
    true_params,train,long,loops=make_episodes(seed,regime,train_size)
    scores={}
    for mode in ("flat","linear","nonlinear"):
        t=time.perf_counter()
        params=fit(train,mode,epochs)
        training_elapsed=time.perf_counter()-t
        scores[mode]={"heldout_loss":evaluation(long,params,mode),
                      "loop_loss":evaluation(loops,params,mode),
                      "learned_params":[round(t,5) for t in params],
                      "fit_wall_s":round(training_elapsed,4)}
    return {"seed":seed,"world":regime,
            "true_params":[round(v,5) for v in true_params],
            "scores":scores}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--regimes",default="noncommuting,commuting")
    parser.add_argument("--epochs",type=int,default=125)
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(",")]
    regimes=args.regimes.split(",")
    if not (1<=len(seeds)<=16 and 10<=args.epochs<=500
            and all(s in ("noncommuting","commuting") for s in regimes)):
        parser.error("invalid benchmark constraints")
    analytic=selfcheck()
    results=[run(seed,regime,epochs=args.epochs)
             for regime in regimes for seed in seeds]
    means={}
    for regime in regimes:
        subset=[r for r in results if r["world"]==regime]
        means[regime]={mode:{"heldout":sum(r["scores"][mode]["heldout_loss"] for r in subset)/len(subset),
                             "loops":sum(r["scores"][mode]["loop_loss"] for r in subset)/len(subset)}
                       for mode in ("flat","linear","nonlinear")}
    print(json.dumps({"contract":"G0-S predeclared causal intervention study",
                      "geometry":analytic,"mean_losses":means,"results":results},indent=2))


if __name__=="__main__":
    main()

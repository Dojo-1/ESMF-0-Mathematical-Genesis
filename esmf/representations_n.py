"""G0-N representation audit: intervention prediction with matched real DOF.

No neutral-form gradient descent: indefinite quadratic penalties can diverge.
All modes have 4 real readout weights, common convex logistic loss and L2
regularization. Different fixed encoders are compared to identity under paired
data. This is a representation/conditioning smoke test, NOT cognition.
"""
import argparse
import json
import math
import random
import time


def features(a, b, nuisance, mode):
    # Environmental action a changes whether cause b changes the outcome.
    # Every mode has four real coordinates and exactly four trainable weights.
    x=(1.,float(a),float(b),float(a*b))
    if mode=="euclidean":
        return x
    if mode=="neutral22":
        # O(2,2) boost mixing a time-like and space-like direction.
        t=.65
        c,s=math.cosh(t),math.sinh(t)
        return (c*x[0]+s*x[2],x[1],s*x[0]+c*x[2],x[3])
    if mode=="complex2":
        # Equivalent to unitary phase rotation on two complex coordinates.
        t=.65
        c,s=math.cos(t),math.sin(t)
        return (c*x[0]-s*x[1],s*x[0]+c*x[1],
                c*x[2]-s*x[3],s*x[2]+c*x[3])
    raise ValueError(mode)


def draw(rng):
    a=rng.randrange(2);b=rng.randrange(2);nuisance=rng.randrange(2)
    # Observation alone of b does not specify outcome: action matters.
    y=int(a and b)
    return a,b,nuisance,y


def sigmoid(v):
    return 1/(1+math.exp(-max(-30.,min(30.,v))))


def fit(train,mode,lam=.002,lr=.09):
    w=[0.]*4
    for a,b,n,y in train:
        x=features(a,b,n,mode)
        p=sigmoid(sum(t*u for t,u in zip(w,x)))
        for i in range(4):
            w[i]-=lr*((p-y)*x[i]+lam*w[i])
    return w


def evaluate(samples,w,mode):
    loss=0.
    for a,b,n,y in samples:
        x=features(a,b,n,mode)
        p=sigmoid(sum(t*u for t,u in zip(w,x)))
        loss+=(p-y)**2
    return loss/len(samples)


def run(seed=0,train_size=700):
    rng=random.Random(seed+10000)
    train=[draw(rng) for _ in range(train_size)]
    val_rng=random.Random(seed+20000)
    test=[draw(val_rng) for _ in range(256)]
    result={}
    for mode in ("euclidean","neutral22","complex2"):
        begin=time.perf_counter()
        weights=fit(train,mode)
        loss=evaluate(test,weights,mode)
        elapsed=time.perf_counter()-begin
        result[mode]={"brier":round(loss,8),
                      "weight_l2":round(math.sqrt(sum(v*v for v in weights)),6),
                      "elapsed_s":round(elapsed,6)}
    return {"seed":seed,**result}


def selfcheck():
    for a,b in ((0,0),(1,1)):
        x=features(a,b,0,"euclidean")
        n=features(a,b,0,"neutral22")
        q=x[0]*x[0]+x[1]*x[1]-x[2]*x[2]-x[3]*x[3]
        qn=n[0]*n[0]+n[1]*n[1]-n[2]*n[2]-n[3]*n[3]
        assert abs(q-qn)<1e-12
        z=features(a,b,0,"complex2")
        assert abs(sum(v*v for v in z)-sum(v*v for v in x))<1e-12
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-size",type=int,default=700)
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(",")]
    if not 1<=len(seeds)<=32 or not 1<=args.train_size<=5000:
        parser.error("out of bounds")
    assert selfcheck()
    results=[run(s,args.train_size) for s in seeds]
    means={mode:round(sum(r[mode]["brier"] for r in results)/len(results),8)
           for mode in ("euclidean","neutral22","complex2")}
    print(json.dumps({"label":"G0-N matched real DOF, nonmatched conditioning",
                      "mean_brier":means,"results":results},indent=2))


if __name__=="__main__":
    main()

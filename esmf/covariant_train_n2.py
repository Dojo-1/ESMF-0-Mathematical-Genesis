"""G0-N2: gauge-covariant learning updates under fixed invertible chart changes.

The intent is an exact metamorphic test: equivalent coordinates must produce
equivalent predictions and losses. Only linear feature and readout transport.
This does not demonstrate any neutral-geometry advantage.
"""
import json
import random
from .geometry_k import matvec,transpose,inverse
from .representations_n import draw,sigmoid
from .neutral22 import boost


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def train_pair(seed=0, steps=700):
    rng=random.Random(seed+10000)
    data=[draw(rng) for _ in range(steps)]
    B=boost(.65)
    inverse_transpose=transpose(inverse(B))
    w=[0.]*4
    wp=[0.]*4
    lr=.09
    max_prediction_gap=0.
    max_weight_gap=0.
    for a,b,_,y in data:
        x=(1.,float(a),float(b),float(a*b))
        xp=matvec(B,x)
        p=sigmoid(dot(w,x))
        pp=sigmoid(dot(wp,xp))
        max_prediction_gap=max(max_prediction_gap,abs(p-pp))
        # Base Euclidean loss gradient in canonical frame.
        delta=[lr*(p-y)*q for q in x]
        transformed_delta=matvec(inverse_transpose,delta)
        w=[v-d for v,d in zip(w,delta)]
        wp=[v-d for v,d in zip(wp,transformed_delta)]
        expected=matvec(inverse_transpose,w)
        max_weight_gap=max(max_weight_gap,max(abs(v-q) for v,q in zip(wp,expected)))
    assert max_prediction_gap<1e-10
    assert max_weight_gap<1e-10
    return {"seed":seed,
            "max_prediction_gap":max_prediction_gap,
            "max_weight_transport_gap":max_weight_gap,
            "not_a_neutral_performance_claim":True}


if __name__=="__main__":
    print(json.dumps([train_pair(s) for s in range(8)],indent=2))

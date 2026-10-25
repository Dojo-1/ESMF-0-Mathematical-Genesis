"""G0-W: orientation-twisted (2,2) carrier and left/right organ exchange.

On Mat(2,R), Q(F)=det(F) has signature (2,2). Left SL2 and right SL2
factor actions commute with each other, though each factor contains
noncommuting transformations. Transposition preserves Q but reverses
orientation of the 4D fiber and swaps the factors. Thus a nonorientable
S1 monodromy can be modeled by a transpose seam, provided local dynamics
are conjugated across that seam. Algebraic topology only; no learning.
"""
import json
import math
import random
from .geometry_k import matmul,matvec,transpose,identity
from .associated22_u import H,lift_left,flatten,neutral_square,maxmat
from .causal_bundle_s import tangent_step,physical_world

P=((1.,0.,0.,0.),
   (0.,0.,1.,0.),
   (0.,1.,0.,0.),
   (0.,0.,0.,1.))


def det4(A):
    n=4
    work=[list(row) for row in A]
    out=1.
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(work[i][k]))
        if abs(work[pivot][k])<1e-13:
            return 0.
        if pivot!=k:
            work[k],work[pivot]=work[pivot],work[k]
            out=-out
        v=work[k][k]
        out*=v
        for i in range(k+1,n):
            factor=work[i][k]/v
            for j in range(k+1,n):
                work[i][j]-=factor*work[k][j]
            work[i][k]=0.
    return out


def right_factor(J):
    """R_J(vec(F)) = vec(F J^T), where J=[[a,b],[c,d]]."""
    a,b=J[0]
    c,d=J[1]
    return ((a,b,0.,0.),(c,d,0.,0.),
            (0.,0.,a,b),(0.,0.,c,d))


def maxvector(a,b):
    return max(abs(x-y) for x,y in zip(a,b))


def selfcheck():
    E=identity(4)
    assert maxmat(matmul(P,P),E)<1e-15
    twist_sign=det4(P)
    assert abs(twist_sign+1.)<1e-12
    metric_gap=maxmat(matmul(matmul(transpose(P),H),P),H)
    assert metric_gap<1e-14

    rng=random.Random(95431)
    max_factor_gap=max_seam_gap=max_quad=0.
    min_naive_gap=1e6
    max_naive_gap=0.
    for _ in range(75):
        q,p=rng.uniform(-2.8,2.8),rng.uniform(-.9,.9)
        mode=rng.choice(("X","Y"))
        J=tangent_step((q,p),mode,rng.choice((-1.,1.))*rng.uniform(.3,.95),(.62,-.51))
        L=lift_left(J)
        R=right_factor(J)
        max_factor_gap=max(max_factor_gap,maxmat(matmul(matmul(P,L),P),R))
        max_quad=max(max_quad,maxmat(matmul(matmul(transpose(L),H),L),H),
                     maxmat(matmul(matmul(transpose(R),H),R),H))
        F=((rng.uniform(-2.,2.),rng.uniform(-2.,2.)),
           (rng.uniform(-2.,2.),rng.uniform(-2.,2.)))
        v=flatten(F)
        correct=matvec(R,matvec(P,v))
        from_before=matvec(P,matvec(L,v))
        max_seam_gap=max(max_seam_gap,maxvector(correct,from_before))
        wrong=matvec(L,matvec(P,v))
        naive_gap=maxvector(wrong,correct)
        max_naive_gap=max(max_naive_gap,naive_gap)
        min_naive_gap=min(min_naive_gap,naive_gap)
        assert abs(neutral_square(correct)-neutral_square(v))<1e-10

    # Different regulatory factor actions commute: left A and right B.
    A=((1.,.42),(0.,1.))
    B=((1.,0.),(-.35,1.))
    LA=lift_left(A)
    RB=right_factor(B)
    cross_factor_commutator=maxmat(matmul(LA,RB),matmul(RB,LA))
    assert cross_factor_commutator<1e-14
    # But left-side intervention order genuinely matters.
    LB=lift_left(B)
    within_factor_order_gap=maxmat(matmul(LA,LB),matmul(LB,LA))
    assert within_factor_order_gap>.01

    # A real cylinder-crossing action changes the local representative.
    start=(math.pi-.045,.72)
    action=("X",.8)
    end=physical_world(start,(action,),(.62,-.51),"noncommuting")
    assert end[0]<0 and start[0]>0
    J=tangent_step(start,*action,(.62,-.51))
    seam_gauge_gap=maxmat(matmul(matmul(P,lift_left(J)),P),right_factor(J))
    assert seam_gauge_gap<1e-14

    assert max_naive_gap>.05
    return {"fiber":"Mat2(R) with neutral determinant (2,2)",
            "deck_twist_det":twist_sign,
            "metric_twist_residual":metric_gap,
            "factor_conjugacy_residual":max_factor_gap,
            "seam_equivariance_residual":max_seam_gap,
            "factor_neutral_metric_residual":max_quad,
            "cross_factor_commutator_residual":cross_factor_commutator,
            "within_factor_order_gap":within_factor_order_gap,
            "chart_blind_seam_gap_max":max_naive_gap,
            "physical_action_crossed_angular_seam":True,
            "crossing_operator_conjugacy_residual":seam_gauge_gap}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

"""G0-U: explicit four-real-dimensional neutral associated-fiber transport.

Mat(2,R) has determinant form of signature (2,2). The tangent Jacobian of
each reversible, area-preserving action acts on frames by LEFT multiplication,
inducing an O(2,2) action on the associated four-dimensional fiber.

Under non-volume-preserving local charts, the neutral metric MUST transform;
raw coordinate determinant is not gauge invariant. A positive auxiliary metric
tracks physical coordinate costs, not a preserved mechanical energy.
"""
import json
import math
import random
from .geometry_k import matmul,matvec,inverse,transpose,identity,bilinear
from .causal_bundle_s import mm,det,wrap,tangent_step,physical_world,induced_tangent
from .context_bundle_t import chart_jac

H=((0.,0.,0.,.5),(0.,0.,-.5,0.),(0.,-.5,0.,0.),(.5,0.,0.,0.))
I4=identity(4)


def flatten(F):
    return (F[0][0],F[0][1],F[1][0],F[1][1])


def lift_left(J):
    a,b=J[0]
    c,d=J[1]
    return ((a,0.,b,0.),(0.,a,0.,b),
            (c,0.,d,0.),(0.,c,0.,d))


def neutral_square(x):
    return bilinear(x,H,x)


def positive_square(x):
    return sum(v*v for v in x)


def chart_metric(B,positive=False):
    invL=inverse(lift_left(B))
    base=I4 if positive else H
    return matmul(matmul(transpose(invL),base),invL)


def chart_operator(B0,B1,J):
    return matmul(matmul(lift_left(B1),lift_left(J)),
                  inverse(lift_left(B0)))


def maxmat(A,B):
    return max(abs(a-b) for ra,rb in zip(A,B) for a,b in zip(ra,rb))


def signature_coordinates(F):
    a,b=F[0]
    c,d=F[1]
    return ((a+d)/2,(b-c)/2,(a-d)/2,(b+c)/2)


def selfcheck():
    rng=random.Random(721991)
    max_frame=max_metric=max_quadratic=max_resource=0.
    max_ignorant=0.
    for _ in range(70):
        current=(rng.uniform(-math.pi,math.pi),rng.uniform(-.8,.8))
        params=(.62,-.51)
        F=((1.,0.),(0.,1.))
        chart=rng.randrange(3)
        B=chart_jac(current,chart,(.37,-.29))
        xchart=matvec(lift_left(B),flatten(F))
        for _ in range(rng.randint(2,8)):
            action=rng.choice(("X","Y"))
            h=rng.choice((-1.,1.))*rng.uniform(.3,.9)
            J=tangent_step(current,action,h,params)
            later=physical_world(current,((action,h),),params,"noncommuting")
            next_chart=rng.randrange(3)
            C=chart_jac(later,next_chart,(.37,-.29))
            P=chart_operator(B,C,J)
            next_F=mm(J,F)
            xchart=matvec(P,xchart)
            desired=matvec(lift_left(C),flatten(next_F))
            max_frame=max(max_frame,max(abs(a-b) for a,b in zip(xchart,desired)))
            metric0=chart_metric(B)
            metric1=chart_metric(C)
            preserve=matmul(matmul(transpose(P),metric1),P)
            max_metric=max(max_metric,maxmat(preserve,metric0))
            canon=flatten(next_F)
            max_quadratic=max(max_quadratic,abs(bilinear(xchart,metric1,xchart)-neutral_square(canon)))
            # Coordinate-invariant positive *description cost*, NOT conserved.
            positive=chart_metric(C,positive=True)
            max_resource=max(max_resource,abs(bilinear(xchart,positive,xchart)-positive_square(canon)))
            max_ignorant=max(max_ignorant,maxmat(P,lift_left(J)))
            current,F,chart,B=later,next_F,next_chart,C
    assert max(max_frame,max_metric,max_quadratic,max_resource)<1e-10
    assert max_ignorant>.1
    # Independent black-box finite differences of the physical simulator,
    # rather than checking analytic J only against its own matrix identities.
    max_external_jacobian=0.
    epsilon=1e-6
    for _ in range(36):
        start=(rng.uniform(-math.pi,math.pi),rng.uniform(-.6,.6))
        actions=tuple((rng.choice(("X","Y")),rng.choice((-1.,1.))*rng.uniform(.25,.9))
                      for k in range(rng.randint(1,7)))
        params=(.62,-.51)
        _,analytic=induced_tangent(start,actions,params)
        for axis in (0,1):
            plus=list(start);minus=list(start)
            plus[axis]+=epsilon;minus[axis]-=epsilon
            hi=physical_world(plus,actions,params,"noncommuting")
            lo=physical_world(minus,actions,params,"noncommuting")
            dq=wrap(hi[0]-lo[0])/(2*epsilon)
            dp=(hi[1]-lo[1])/(2*epsilon)
            max_external_jacobian=max(max_external_jacobian,
                abs(dq-analytic[0][axis]),abs(dp-analytic[1][axis]))
    assert max_external_jacobian<2e-8
    null=(1.,0.,0.,0.)
    assert neutral_square(null)==0. and positive_square(null)>0.
    FF=((1.5,-.6),(.3,1.1))
    u1,u2,v1,v2=signature_coordinates(FF)
    assert abs(det(FF)-(u1*u1+u2*u2-v1*v1-v2*v2))<1e-12
    Jd=((1.,0.),(0.,.91))
    loss=maxmat(matmul(matmul(transpose(lift_left(Jd)),H),lift_left(Jd)),H)
    assert loss>.01 # loss of volume must break O(2,2) preservation
    return {"signature":"(2,2) from determinant polarization",
            "random_paths":70,
            "max_fourfiber_frame_gap":max_frame,
            "max_metric_transport_gap":max_metric,
            "max_neutral_square_gap":max_quadratic,
            "max_positive_cost_chart_gap":max_resource,
            "max_independent_jacobian_gap":max_external_jacobian,
            "chart_blind_operator_gap":max_ignorant,
            "dissipative_counterexample_gap":loss,
            "null_nonzero_positive_cost":True}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

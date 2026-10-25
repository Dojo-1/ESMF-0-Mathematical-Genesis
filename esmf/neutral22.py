"""G0-M: real neutral-signature (2,2) fiber, not a 2D plane or a finite field.

A neutral metric supports null nonzero vectors. Hence signed quadratic
form is not a loss norm; report both invariant signed defect and
positive auxiliary diagnostic whose transformation is explicitly tracked.
"""
import json
import math
from .geometry_k import matmul,matvec,transpose,inverse,bilinear,identity


ETA=((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,-1.,0.),(0.,0.,0.,-1.))


def boost(t):
    c,s=math.cosh(t),math.sinh(t)
    return ((c,0.,s,0.),(0.,c,0.,s),(s,0.,c,0.),(0.,s,0.,c))


def neutral_form(v,w):
    return bilinear(v,ETA,w)


def change_frame(v,metric,B):
    I=inverse(B)
    return matvec(B,v),matmul(matmul(transpose(I),metric),I)


def transport_change(U,Bi,Bj):
    return matmul(matmul(Bi,U),inverse(Bj))


def projection4to3():
    return ((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.))


def reconstruction3to4():
    return ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(0.,0.,0.))


def remainder(v):
    T=projection4to3()
    R=reconstruction3to4()
    returned=matvec(R,matvec(T,v))
    return tuple(a-b for a,b in zip(v,returned))


def selfcheck():
    B=boost(.63)
    residual=max(abs(matmul(matmul(transpose(B),ETA),B)[i][j]-ETA[i][j])
                 for i in range(4) for j in range(4))
    assert residual<1e-12
    x=(2.,1.,2.,-1.)
    assert abs(neutral_form(x,x))<1e-12 # nonzero null state
    y=(.5,-.6,1.2,2.3)
    v1,H1=change_frame(y,ETA,B)
    v2,H2=change_frame(x,ETA,boost(-.3))
    transported=transport_change(identity(4),B,boost(-.3))
    d=tuple(a-b for a,b in zip(y,x))
    dp=tuple(a-b for a,b in zip(v1,matvec(transported,v2)))
    assert abs(bilinear(d,ETA,d)-bilinear(dp,H1,dp))<1e-11
    r=remainder((0.,0.,0.,5.))
    assert r==(0.,0.,0.,5.)
    # Scalar indefinite 'energy' is -25, NOT a squared norm!
    assert neutral_form(r,r)==-25.
    positive_norm_sq=sum(t*t for t in r)
    return {"signature":"(2,2) real neutral", "form_preservation_residual":residual,
            "nonzero_null_vector":True,
            "signed_remainder_square":neutral_form(r,r),
            "auxiliary_positive_remainder_square":positive_norm_sq,
            "gauge_covariant_mismatch":True}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

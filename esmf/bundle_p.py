"""G0-P: minimal neutral-signature bundle atlas with nonflat connection.

Constant O(2,2) local frames, genuine cocycles, a metric-compatible connection,
and noncommuting transport around a square. All computations use stdlib only.
This is an algebraic bundle testbed, not yet a living/learning organism.
"""
import json
import math
from .geometry_k import matmul, matvec, transpose, inverse, identity

N=4
ETA=((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,-1.,0.),(0.,0.,0.,-1.))


def plus(A,B,scale=1.):
    return tuple(tuple(A[i][j]+scale*B[i][j] for j in range(N)) for i in range(N))


def scaled(A,k):
    return tuple(tuple(k*v for v in row) for row in A)


def commutator(A,B):
    return plus(matmul(A,B),matmul(B,A),-1.)


def maxdiff(A,B):
    return max(abs(A[i][j]-B[i][j]) for i in range(N) for j in range(N))


def exponential(A):
    result=identity(N)
    term=identity(N)
    for k in range(1,60):
        term=scaled(matmul(term,A),1./k)
        result=plus(result,term)
        if maxdiff(term,scaled(identity(N),0))<1e-16:
            break
    return result


def boost(i,j,t):
    # i is positive, j negative: hyperbolic rotation preserving ETA.
    A=scaled(identity(N),0.)
    A=tuple(tuple(t if (r,c) in ((i,j),(j,i)) else 0.
                  for c in range(N)) for r in range(N))
    return exponential(A)


def generators():
    X=tuple(tuple(1. if (i,j) in ((0,2),(2,0)) else 0.
                  for j in range(N)) for i in range(N))
    Y=tuple(tuple(.8 if (i,j) in ((1,2),(2,1)) else 0.
                  for j in range(N)) for i in range(N))
    return X,Y


def metric_compatible(A):
    return maxdiff(plus(matmul(transpose(A),ETA),matmul(ETA,A)),scaled(identity(N),0.))


def charts():
    return [identity(N),boost(0,2,.4),boost(1,3,-.3)]


def transition(a,b):
    # Coordinates of the same section satisfy v_b = G_ba v_a.
    C=charts()
    return matmul(C[b],inverse(C[a]))


def transport(Ax,Ay,h):
    # Positive oriented square: +x,+y,-x,-y.
    ex=exponential(scaled(Ax,-h))
    ey=exponential(scaled(Ay,-h))
    ix=exponential(scaled(Ax,h))
    iy=exponential(scaled(Ay,h))
    return matmul(matmul(matmul(iy,ix),ey),ex)


def selfcheck():
    C=charts()
    g01=transition(0,1);g12=transition(1,2);g02=transition(0,2)
    cocycle=maxdiff(matmul(g12,g01),g02)
    assert cocycle<1e-12
    broken=plus(g12,scaled(identity(N),.03))
    assert maxdiff(matmul(broken,g01),g02)>.01
    Ax,Ay=generators()
    assert metric_compatible(Ax)<1e-12 and metric_compatible(Ay)<1e-12
    F=commutator(Ax,Ay)
    G=transition(0,1)
    Gi=inverse(G)
    Ax1=matmul(matmul(G,Ax),Gi)
    Ay1=matmul(matmul(G,Ay),Gi)
    eta1=matmul(matmul(transpose(Gi),ETA),Gi)
    compatibility=maxdiff(plus(matmul(transpose(Ax1),eta1),matmul(eta1,Ax1)),scaled(identity(N),0.))
    assert compatibility<1e-12
    F1=commutator(Ax1,Ay1)
    curvature_gap=maxdiff(F1,matmul(matmul(G,F),Gi))
    assert curvature_gap<1e-12
    h=.002
    loop=transport(Ax,Ay,h)
    # ordered column-vector transports under Dv=0 yield -[Ax,Ay] h^2
    leading=plus(identity(N),F,-h*h)
    loop_error=maxdiff(loop,leading)
    reverse=transport(Ax,Ay,h/2)
    half_error=maxdiff(reverse,plus(identity(N),F,-h*h/4))
    assert half_error<loop_error/6
    transformed_loop=transport(Ax1,Ay1,h)
    loop_gauge=maxdiff(transformed_loop,matmul(matmul(G,loop),Gi))
    assert loop_gauge<1e-11
    return {"cocycle_residual":cocycle,"broken_cocycle_detected":True,
            "connection_metric_residual":compatibility,
            "curvature_gauge_residual":curvature_gap,
            "square_loop_order2_error":loop_error,
            "half_step_error":half_error,
            "holonomy_gauge_residual":loop_gauge,
            "curvature_nonzero":maxdiff(F,scaled(identity(N),0.))>0.1}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

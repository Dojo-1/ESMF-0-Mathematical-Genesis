"""G0-K: heterogeneous metric-fiber transport, local gauge, and event clocks.

Proof-of-algebra test instrument, not implemented neural cognition.
Supports real vector fibers with positive or indefinite nondegenerate
metrics, GL(n) chart changes, and Lorentz boosts on R^(2,1).
"""
import json
import math


def transpose(A):
    return tuple(tuple(A[j][i] for j in range(len(A))) for i in range(len(A[0])))


def matmul(A, B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(len(B)))
                       for j in range(len(B[0]))) for i in range(len(A)))


def matvec(A, v):
    return tuple(sum(row[j]*v[j] for j in range(len(v))) for row in A)


def identity(n):
    return tuple(tuple(float(i==j) for j in range(n)) for i in range(n))


def inverse(A):
    n=len(A)
    if not n or any(len(row)!=n for row in A):
        raise ValueError("requires square matrix")
    C=[list(map(float,row))+list(identity(n)[i]) for i,row in enumerate(A)]
    for i in range(n):
        k=max(range(i,n),key=lambda t:abs(C[t][i]))
        if abs(C[k][i])<1e-12:
            raise ValueError("singular local chart")
        C[i],C[k]=C[k],C[i]
        scale=C[i][i]
        C[i]=[v/scale for v in C[i]]
        for j in range(n):
            if j!=i:
                factor=C[j][i]
                C[j]=[x-factor*y for x,y in zip(C[j],C[i])]
    return tuple(tuple(row[n:]) for row in C)


def bilinear(v, H, w):
    Hw=matvec(H,w)
    return sum(a*b for a,b in zip(v,Hw))


def mismatch_energy(ui, uj, Uij, Hi):
    if len(ui)!=len(Hi) or len(uj)!=len(Uij[0]):
        raise ValueError("fiber dimension mismatch")
    transported=matvec(Uij,uj)
    delta=tuple(a-b for a,b in zip(ui,transported))
    return bilinear(delta,Hi,delta)


def change_frames(ui,uj,Uij,Hi,Bi,Bj):
    """State charts v'=Bv; transport U'=Bi U Bj^-1; H'=Bi^-T H Bi^-1."""
    I=inverse(Bi)
    return (matvec(Bi,ui),matvec(Bj,uj),
            matmul(matmul(Bi,Uij),inverse(Bj)),
            matmul(matmul(transpose(I),Hi),I))


def lorentz_boost(rapidity):
    c,s=math.cosh(rapidity),math.sinh(rapidity)
    return ((c,s,0.),(s,c,0.),(0.,0.,1.))


def metric_signature_21():
    return ((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def hyperboloid_point(r,phi):
    """One sheet of H^2: x^T eta x = -1, x0 > 0."""
    return (math.cosh(r),math.sinh(r)*math.cos(phi),
            math.sinh(r)*math.sin(phi))


def hyperbolic_distance(x,y):
    return math.acosh(max(1.,-bilinear(x,metric_signature_21(),y)))


def transport_holonomy(U01,U12,U20,B0,B1,B2):
    """Cycle 0 <- 1 <- 2 <- 0 transforms as H0'=B0 H0 B0^-1."""
    H=matmul(matmul(U01,U12),U20)
    T01=matmul(matmul(B0,U01),inverse(B1))
    T12=matmul(matmul(B1,U12),inverse(B2))
    T20=matmul(matmul(B2,U20),inverse(B0))
    return H,matmul(matmul(T01,T12),T20)


def max_abs_difference(A,B):
    return max(abs(x-y) for row_a,row_b in zip(A,B)
               for x,y in zip(row_a,row_b))


def event_leak(v,rest,dt,tau):
    """Exact quiescent state reconstruction; no intermediate integration."""
    if dt<0 or tau<=0:
        raise ValueError("invalid local clock")
    return rest+(v-rest)*math.exp(-dt/tau)


def selfcheck():
    eta=metric_signature_21()
    x=hyperboloid_point(.6,.3)
    y=hyperboloid_point(1.2,1.4)
    B0=lorentz_boost(.75)
    B1=lorentz_boost(-.48)
    # Lorentz boosts are isometries of one indefinite metric.
    assert max_abs_difference(matmul(matmul(transpose(B0),eta),B0),eta)<1e-12
    d=hyperbolic_distance(x,y)
    dx=hyperbolic_distance(matvec(B0,x),matvec(B0,y))
    assert abs(d-dx)<1e-12
    # A transport between arbitrary independent frames (not an isometry in
    # general) retains its bilinear mismatch when the metric co-transforms.
    U=lorentz_boost(.31)
    e=mismatch_energy(x,y,U,eta)
    moved=change_frames(x,y,U,eta,B0,B1)
    assert abs(e-mismatch_energy(*moved))<1e-11
    H,Hprime=transport_holonomy(U,lorentz_boost(.2),lorentz_boost(-.1),
                                B0,B1,lorentz_boost(.57))
    expected=matmul(matmul(B0,H),inverse(B0))
    assert max_abs_difference(Hprime,expected)<1e-11
    assert abs(event_leak(1.,0.,3.,2.)-math.exp(-1.5))<1e-14
    return {"hyperbolic_distance":round(d,8),
            "distance_chart_residual":abs(d-dx),
            "bilinear_mismatch_residual":abs(e-mismatch_energy(*moved)),
            "holonomy_conjugacy_residual":max_abs_difference(Hprime,expected),
            "quiescent_update":"exact"}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

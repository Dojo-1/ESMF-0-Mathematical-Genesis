"""Gauge-covariant 2D transport and external diagnostics.

These are coordinate identities, not a learner's privileged oracle.
A node's basis may change independently of every other node's basis.
No NumPy required.
"""
import math


def mm(a,b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in (0,1))
                       for j in (0,1)) for i in (0,1))


def mv(a,v):
    return tuple(sum(a[i][j]*v[j] for j in (0,1)) for i in (0,1))


def transpose(a):
    return ((a[0][0],a[1][0]),(a[0][1],a[1][1]))


def determinant(a):
    return a[0][0]*a[1][1]-a[0][1]*a[1][0]


def inverse(a):
    d=determinant(a)
    if abs(d)<1e-12:
        raise ValueError("singular local basis")
    return ((a[1][1]/d,-a[0][1]/d),(-a[1][0]/d,a[0][0]/d))


def rotation(theta):
    c,s=math.cos(theta),math.sin(theta)
    return ((c,-s),(s,c))


def difference(v,w):
    return tuple(x-y for x,y in zip(v,w))


def quad(v,h):
    hv=mv(h,v)
    return sum(a*b for a,b in zip(v,hv))


def edge_energy(ui,uj,uij,hi):
    delta=difference(ui,mv(uij,uj))
    return quad(delta,hi)


def change_chart(ui,uj,uij,hi,bi,bj):
    """v' = Bv, U' = Bi U Bj^-1, H' = Bi^-T H Bi^-1."""
    inv_i,inv_j=inverse(bi),inverse(bj)
    return (mv(bi,ui),mv(bj,uj),mm(mm(bi,uij),inv_j),
            mm(mm(transpose(inv_i),hi),inv_i))


def gram(rows):
    return tuple(tuple(sum(row[i]*row[j] for row in rows)
                       for j in (0,1)) for i in (0,1))


def gram_basis_change(g,b):
    """Observation matrix transforms O' = O B^-1, G'=B^-T G B^-1."""
    ib=inverse(b)
    return mm(mm(transpose(ib),g),ib)


def selfcheck():
    ui=(0.7,-0.4);uj=(1.1,0.2)
    uij=rotation(0.4)
    hi=((2.0,0.3),(0.3,1.5))
    e=edge_energy(ui,uj,uij,hi)
    for bi,bj in ((rotation(.7),rotation(-1.2)),
                  (((1.2,.4),(.1,.9)),((.8,-.2),(.3,1.1)))):
        transformed=change_chart(ui,uj,uij,hi,bi,bj)
        assert abs(edge_energy(*transformed)-e)<1e-10
    rows=((1.,2.),(-2.,1.),(3.,4.))
    g=gram(rows)
    for b in (rotation(.77),((1.3,.2),(.4,.9))):
        gb=gram_basis_change(g,b)
        expected=determinant(g)/(determinant(b)**2)
        assert abs(determinant(gb)-expected)<1e-9
    return {"energy":round(e,6),"gram_det":round(determinant(g),6),
            "coordinate_invariance":True,
            "determinant_covariance":True}


if __name__=="__main__":
    print(selfcheck())

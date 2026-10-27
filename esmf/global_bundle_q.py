"""G0-Q: a nontrivial real (2,2) vector bundle over a cylinder.

M = S^1 x R, fiber R^4 with neutral metric. The quotient gluing across
theta + 2*pi uses R=diag(-1,1,1,1), so det monodromy is -1 and the bundle
is non-orientable. Three local charts have nonconstant GL(4) trivializations.
A globally descending o(2,2) connection has nonzero curvature.

Local coordinate convention: v_b = g_ba v_a, nabla = d + A,
A_b = g_ba A_a g_ba^{-1} - d(g_ba) g_ba^{-1}.
This is a finite testbed, not a universal bundle framework or a learner.
"""
import json
import math
from dataclasses import dataclass
from .geometry_k import matmul,matvec,transpose,inverse,identity,bilinear
from .bundle_p import ETA,plus,scaled,commutator,exponential,maxdiff,transport

D=4
TAU=2*math.pi
TWIST=((-1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.))


@dataclass(frozen=True)
class ChartTransition:
    source: int
    target: int
    matrix: tuple

    def __post_init__(self):
        if self.source not in range(3) or self.target not in range(3):
            raise ValueError("unknown chart")
        if (len(self.matrix) != D or
                any(len(row) != D for row in self.matrix)):
            raise ValueError("transition must be a 4x4 map")
        inverse(self.matrix)

    def then(self, following):
        if self.target != following.source:
            raise ValueError("transition charts do not compose")
        return ChartTransition(self.source, following.target,
                               matmul(following.matrix, self.matrix))

    def apply(self, vector):
        if len(vector) != D:
            raise ValueError("wrong fiber dimension")
        return matvec(self.matrix, vector)


def diag(v):
    return tuple(tuple(v[i] if i==j else 0. for j in range(len(v)))
                 for i in range(len(v)))


def generator(i,j,amount):
    A=[[0.]*D for _ in range(D)]
    A[i][j]=amount
    A[j][i]=amount
    return tuple(tuple(row) for row in A)


A_THETA=generator(1,2,.22)
A_Z=generator(1,3,-.27)


def matzero():
    return scaled(identity(D),0.)


def det_sign(M):
    V=[list(row) for row in M]
    sign=1
    for k in range(D):
        pivot=max(range(k,D),key=lambda r:abs(V[r][k]))
        if abs(V[pivot][k])<1e-12:
            return 0
        if pivot!=k:
            V[k],V[pivot]=V[pivot],V[k]
            sign=-sign
        p=V[k][k]
        if p<0:
            sign=-sign
        for i in range(k+1,D):
            factor=V[i][k]/p
            for j in range(k+1,D):
                V[i][j]-=factor*V[k][j]
            V[i][k]=0.
    return sign


class NeutralCylinderBundle:
    """Two overlapping circle charts plus a triple-overlap test chart."""

    def __init__(self,overlap=.35):
        if not (0<overlap<.5):
            raise ValueError("require 0<overlap<.5 radians")
        self.overlap=overlap
        self.coefficients=(
            ((.13,-.10,.07,.04),(.03,.01,-.04,.02),"sin"),
            ((-.06,.15,.09,-.11),(-.02,.03,.02,-.01),"cos"),
            ((.18,.08,-.09,.05),(.01,-.03,.05,.02),"sin2")
        )

    def lift(self,chart,theta):
        theta=theta%TAU
        e=self.overlap
        if chart==0:
            if theta<math.pi+e:
                return theta
            if theta>TAU-e:
                return theta-TAU
        elif chart==1:
            if theta>math.pi-e:
                return theta
            if theta<e:
                return theta+TAU
        elif chart==2 and math.pi-e/2<theta<math.pi+e/2:
            return theta
        raise ValueError("point outside open chart")

    def phi(self,chart,t,z):
        a,b,kind=self.coefficients[chart]
        if kind=="sin":
            f,df=math.sin(t),math.cos(t)
        elif kind=="cos":
            f,df=math.cos(t),-math.sin(t)
        else:
            f,df=math.sin(2*t),2*math.cos(2*t)
        return ([a[i]*f+b[i]*z for i in range(D)],
                [a[i]*df for i in range(D)],list(b))

    def frame(self,chart,theta,z):
        t=self.lift(chart,theta)
        p,dp_t,dp_z=self.phi(chart,t,z)
        return diag([math.exp(v) for v in p]),diag(dp_t),diag(dp_z)

    def seam(self,a,b,theta):
        ta=self.lift(a,theta);tb=self.lift(b,theta)
        k=round((tb-ta)/TAU)
        if abs((tb-ta)-k*TAU)>1e-9:
            raise AssertionError("lift mismatch")
        return TWIST if k%2 else identity(D)

    def transition(self,a,b,theta,z):
        Ba,_,_=self.frame(a,theta,z)
        Bb,_,_=self.frame(b,theta,z)
        return matmul(matmul(Bb,self.seam(a,b,theta)),inverse(Ba))

    def typed_transition(self,a,b,theta,z):
        return ChartTransition(a,b,self.transition(a,b,theta,z))

    def transition_derivative(self,a,b,theta,z,axis):
        g=self.transition(a,b,theta,z)
        _,at,az=self.frame(a,theta,z)
        _,bt,bz=self.frame(b,theta,z)
        Da,Db=(at,bt) if axis==0 else (az,bz)
        return plus(matmul(Db,g),matmul(g,Da),-1.)

    def metric(self,chart,theta,z,positive=False):
        B,_,_=self.frame(chart,theta,z)
        I=inverse(B)
        h=identity(D) if positive else ETA
        return matmul(matmul(transpose(I),h),I)

    def connection(self,chart,theta,z,axis):
        B,Dt,Dz=self.frame(chart,theta,z)
        A=A_THETA if axis==0 else A_Z
        Df=Dt if axis==0 else Dz
        return plus(matmul(matmul(B,A),inverse(B)),Df,-1.)

    def metric_derivative(self,chart,theta,z,axis):
        h=self.metric(chart,theta,z)
        _,Dt,Dz=self.frame(chart,theta,z)
        Df=Dt if axis==0 else Dz
        return scaled(plus(matmul(transpose(Df),h),matmul(h,Df)),-1.)

    def curvature(self,chart,theta,z):
        B,_,_=self.frame(chart,theta,z)
        return matmul(matmul(B,commutator(A_THETA,A_Z)),inverse(B))

    def numerical_curvature(self,chart,theta,z,eps=1e-5):
        # F_tz = partial_t A_z - partial_z A_t + [A_t,A_z]
        dt=scaled(plus(self.connection(chart,theta+eps,z,1),
                       self.connection(chart,theta-eps,z,1),-1),1/(2*eps))
        dz=scaled(plus(self.connection(chart,theta,z+eps,0),
                       self.connection(chart,theta,z-eps,0),-1),1/(2*eps))
        return plus(plus(dt,dz,-1),
                    commutator(self.connection(chart,theta,z,0),
                               self.connection(chart,theta,z,1)))

    def local_section(self,chart,theta,z):
        t=self.lift(chart,theta)
        v=(math.sin(t/2),1.+.2*math.cos(t),.15*math.sin(t),.1*math.sin(z))
        B,_,_=self.frame(chart,theta,z)
        return matvec(B,v)

    def local_section_derivative(self,chart,theta,z,axis):
        t=self.lift(chart,theta)
        B,Dt,Dz=self.frame(chart,theta,z)
        base=(math.sin(t/2),1.+.2*math.cos(t),.15*math.sin(t),.1*math.sin(z))
        vprime=((.5*math.cos(t/2),-.2*math.sin(t),.15*math.cos(t),0.)
                if axis==0 else (0.,0.,0.,.1*math.cos(z)))
        Da=Dt if axis==0 else Dz
        return tuple(a+b for a,b in
                     zip(matvec(Da,matvec(B,base)),matvec(B,vprime)))

    def covariant_derivative_section(self,chart,theta,z,axis):
        ds=self.local_section_derivative(chart,theta,z,axis)
        As=self.connection(chart,theta,z,axis)
        return tuple(a+b for a,b in zip(ds,matvec(As,self.local_section(chart,theta,z))))

    def chart_transport_theta(self,chart,theta_a,theta_b,z):
        # Angles are continuous lifts *within the chart*; supply modulo base.
        ta=self.lift(chart,theta_a);tb=self.lift(chart,theta_b)
        if abs((tb-ta)-(theta_b-theta_a))>1e-9:
            raise ValueError("transport segment crosses local chart cut")
        B0,_,_=self.frame(chart,theta_a,z)
        B1,_,_=self.frame(chart,theta_b,z)
        return matmul(matmul(B1,exponential(scaled(A_THETA,-(tb-ta)))),inverse(B0))

    def winding_holonomy(self,theta=.1,z=.2):
        # chart 0: theta -> pi; chart 1: pi -> theta+2pi; return chart 0.
        if not (0<theta<self.overlap):
            raise ValueError("basepoint must be in seam overlap near 0")
        T0=self.chart_transport_theta(0,theta,math.pi,z)
        g01=self.transition(0,1,math.pi,z)
        T1=self.chart_transport_theta(1,math.pi,theta+TAU,z)
        g10=self.transition(1,0,theta,z)
        return matmul(matmul(matmul(g10,T1),g01),T0)

    def analytic_winding(self,theta=.1,z=.2):
        B,_,_=self.frame(0,theta,z)
        W=matmul(TWIST,exponential(scaled(A_THETA,-TAU)))
        return matmul(matmul(B,W),inverse(B))

    def selfcheck(self):
        z=.23
        p=math.pi+.09
        seam=.12
        out={}
        # Triple-overlap cocycle including variable frames.
        g01=self.transition(0,1,p,z)
        g12=self.transition(1,2,p,z)
        g02=self.transition(0,2,p,z)
        out["triple_cocycle"]=maxdiff(matmul(g12,g01),g02)
        typed=self.typed_transition(0,1,p,z).then(
            self.typed_transition(1,2,p,z))
        out["typed_triple_cocycle"]=maxdiff(typed.matrix,g02)
        broken=plus(g12,scaled(identity(D),.04))
        out["broken_cocycle_gap"]=maxdiff(matmul(broken,g01),g02)
        out["twist_det_sign"]=det_sign(self.transition(0,1,seam,z))
        out["ordinary_overlap_det_sign"]=det_sign(g01)
        assert out["twist_det_sign"]==-1
        assert out["ordinary_overlap_det_sign"]==1
        assert out["triple_cocycle"]<1e-12
        assert out["broken_cocycle_gap"]>.03

        max_connection=max_curvature=max_metric=max_section=max_cov_section=0.
        max_bad_gauge=0.
        max_numeric_curvature=0.
        min_naive_gap=1e6
        for theta in (p,seam):
            for a,b in ((0,1),) if theta==seam else ((0,1),(1,2),(0,2)):
                g=self.transition(a,b,theta,z)
                gi=inverse(g)
                sa=self.local_section(a,theta,z)
                sb=self.local_section(b,theta,z)
                max_section=max(max_section,max(abs(v-w) for v,w in
                                    zip(sb,matvec(g,sa))))
                for axis in (0,1):
                    A0=self.connection(a,theta,z,axis)
                    A1=self.connection(b,theta,z,axis)
                    dg=self.transition_derivative(a,b,theta,z,axis)
                    expected=plus(matmul(matmul(g,A0),gi),matmul(dg,gi),-1.)
                    max_connection=max(max_connection,maxdiff(A1,expected))
                    wrong=matmul(matmul(g,A0),gi)
                    max_bad_gauge=max(max_bad_gauge,maxdiff(A1,wrong))
                    h=self.metric(a,theta,z)
                    dh=self.metric_derivative(a,theta,z,axis)
                    compat=plus(dh,plus(matmul(transpose(A0),h),matmul(h,A0)),-1.)
                    max_metric=max(max_metric,maxdiff(compat,matzero()))
                    c0=self.covariant_derivative_section(a,theta,z,axis)
                    c1=self.covariant_derivative_section(b,theta,z,axis)
                    max_cov_section=max(max_cov_section,max(abs(v-w) for v,w in
                                            zip(c1,matvec(g,c0))))
                F0=self.curvature(a,theta,z)
                F1=self.curvature(b,theta,z)
                max_curvature=max(max_curvature,maxdiff(F1,matmul(matmul(g,F0),gi)))
            for a in ((0,1) if theta==seam else (0,1,2)):
                FN=self.numerical_curvature(a,theta,z)
                FF=self.curvature(a,theta,z)
                max_numeric_curvature=max(max_numeric_curvature,maxdiff(FN,FF))
                naive=commutator(self.connection(a,theta,z,0),
                                 self.connection(a,theta,z,1))
                min_naive_gap=min(min_naive_gap,maxdiff(naive,FF))
        out.update(connection_gauge=max_connection,curvature_gauge=max_curvature,
                   metric_compatibility=max_metric,section_gluing=max_section,
                   covariant_section_gluing=max_cov_section,
                   numeric_curvature=max_numeric_curvature,
                   naive_curvature_gap=min_naive_gap,
                   omitted_gauge_derivative_gap=max_bad_gauge)
        assert max(max_connection,max_curvature,max_metric,max_section,max_cov_section)<1e-10
        assert out["typed_triple_cocycle"]<1e-12
        assert max_bad_gauge>1e-3
        assert max_numeric_curvature<1e-8
        assert min_naive_gap>1e-3

        # A constant Euclidean base metric has zero Levi-Civita symbols and
        # curvature. An arbitrary predictor is not Levi-Civita unless it is
        # torsion-free and metric-compatible.
        flat_gamma=[[[0. for _ in range(2)] for _ in range(2)]
                    for _ in range(2)]
        predicted_gamma=[[[0. for _ in range(2)] for _ in range(2)]
                         for _ in range(2)]
        predicted_gamma[0][0][0]=.2

        def metric_compatibility_gap(gamma):
            return max(abs(sum(
                gamma[m][k][i]*(1. if m == j else 0.) +
                gamma[m][k][j]*(1. if i == m else 0.)
                for m in range(2)))
                for k in range(2) for i in range(2) for j in range(2))

        flat_compatibility=metric_compatibility_gap(flat_gamma)
        predicted_compatibility=metric_compatibility_gap(predicted_gamma)
        torsion=max(abs(predicted_gamma[m][i][j]-predicted_gamma[m][j][i])
                    for m in range(2) for i in range(2) for j in range(2))
        flat_lc_curvature=max(abs(sum(
            flat_gamma[rho][mu][lam]*flat_gamma[lam][nu][sigma] -
            flat_gamma[rho][nu][lam]*flat_gamma[lam][mu][sigma]
            for lam in range(2)))
            for rho in range(2) for sigma in range(2)
            for mu in range(2) for nu in range(2))
        out["flat_levi_civita_curvature"]=flat_lc_curvature
        out["flat_levi_civita_compatibility"]=flat_compatibility
        out["incompatible_christoffel_gap"]=predicted_compatibility
        out["predicted_christoffel_torsion"]=torsion
        assert flat_lc_curvature==0.
        assert flat_compatibility==0.
        assert torsion==0.
        assert predicted_compatibility>.1

        W=self.winding_holonomy(seam,z)
        expected=self.analytic_winding(seam,z)
        out["winding_transport_gap"]=maxdiff(W,expected)
        out["winding_det_sign"]=det_sign(W)
        assert out["winding_transport_gap"]<1e-10
        assert out["winding_det_sign"]==-1

        # Positive storage cost and signed neutral form live in different
        # gauge-covariant metrics.
        u=self.local_section(0,seam,z)
        us=self.local_section(1,seam,z)
        n0=bilinear(u,self.metric(0,seam,z),u)
        n1=bilinear(us,self.metric(1,seam,z),us)
        p0=bilinear(u,self.metric(0,seam,z,True),u)
        p1=bilinear(us,self.metric(1,seam,z,True),us)
        out["signed_section_gap"]=abs(n0-n1)
        out["positive_section_gap"]=abs(p0-p1)
        assert out["signed_section_gap"]<1e-10 and out["positive_section_gap"]<1e-10
        assert p0>=0.
        return out


def matzero():
    return scaled(identity(D),0.)


if __name__=="__main__":
    print(json.dumps(NeutralCylinderBundle().selfcheck(),indent=2))

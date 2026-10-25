"""G0-L: heterogeneous Worldlet transport with typed, observable remainder.

No claim of semantic adequacy: this is the linear-algebraic carrier for it.
T_ij can be rectangular. Independent GL charts change coordinates, never
underlying round-trip defect; norm requires a correspondingly changed metric.
"""
import json
import math
from .geometry_k import matvec, matmul, inverse, transpose, bilinear, identity


def sub(a,b):
    if len(a)!=len(b):
        raise ValueError("dimension mismatch")
    return tuple(x-y for x,y in zip(a,b))


class Worldlet:
    def __init__(self,dimension,metric=None,clock=1.0,kind="real"):
        if dimension<1 or clock<=0:
            raise ValueError("invalid worldlet")
        self.dimension=dimension
        self.metric=metric if metric is not None else identity(dimension)
        if len(self.metric)!=dimension or any(len(row)!=dimension for row in self.metric):
            raise ValueError("metric shape mismatch")
        self.clock=clock
        self.kind=kind


class Correspondence:
    def __init__(self,source,target,forward,backward):
        self.source,self.target=source,target
        self.forward=forward
        self.backward=backward
        if len(forward)!=target.dimension or any(len(row)!=source.dimension for row in forward):
            raise ValueError("forward shape mismatch")
        if len(backward)!=source.dimension or any(len(row)!=target.dimension for row in backward):
            raise ValueError("backward shape mismatch")

    def travel(self,v):
        if len(v)!=self.source.dimension:
            raise ValueError("source mismatch")
        return matvec(self.forward,v)

    def remainder(self,v):
        return sub(v,matvec(self.backward,self.travel(v)))

    def roundtrip_magnitude(self,v):
        r=self.remainder(v)
        value=bilinear(r,self.source.metric,r)
        if value < -1e-10:
            raise ValueError("indefinite metric is not a norm")
        return math.sqrt(max(0.,value))

    def reframe(self,source_chart,target_chart):
        si=inverse(source_chart)
        ti=inverse(target_chart)
        src_metric=matmul(matmul(transpose(si),self.source.metric),si)
        dst_metric=matmul(matmul(transpose(ti),self.target.metric),ti)
        src=Worldlet(self.source.dimension,src_metric,self.source.clock,self.source.kind)
        dst=Worldlet(self.target.dimension,dst_metric,self.target.clock,self.target.kind)
        f=matmul(matmul(target_chart,self.forward),si)
        b=matmul(matmul(source_chart,self.backward),ti)
        return Correspondence(src,dst,f,b)


def selfcheck():
    a=Worldlet(3,clock=.1,kind="vector")
    b=Worldlet(2,clock=2.,kind="response")
    f=((1.,0.,0.),(0.,1.,0.))
    g=((1.,0.),(0.,1.),(0.,0.))
    c=Correspondence(a,b,f,g)
    x=(2.,-1.,3.)
    err=c.remainder(x)
    assert err==(0.,0.,3.)
    assert abs(c.roundtrip_magnitude(x)-3.)<1e-12
    Bs=((2.,.1,0.),(0.,1.,0.),(0.,0.,.5))
    Bt=((1.,.3),(0.,1.))
    changed=c.reframe(Bs,Bt)
    x2=matvec(Bs,x)
    assert max(abs(v-w) for v,w in zip(changed.remainder(x2),matvec(Bs,err)))<1e-12
    assert abs(c.roundtrip_magnitude(x)-changed.roundtrip_magnitude(x2))<1e-12
    # Impossible to reconstruct every vector when target dimension < source
    # dimension: here kernel contains (0,0,1).
    assert c.travel((0.,0.,1.))==(0.,0.)
    return {"source_dimension":3,"target_dimension":2,
            "untransmitted_component":err[2],
            "norm_of_remainder":c.roundtrip_magnitude(x),
            "frame_covariance":True,
            "lossless_for_all_source_states":False}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

"""G0-R: rank-stratified Worldlet sheaf-like restriction data.

NOT a variable-rank vector bundle: independent vector spaces live on strata.
Linear restriction maps are typed; a descent check tests equality of
boundary observations and records invisible kernels (unrecoverable states).
"""
import json
from dataclasses import dataclass
from .geometry_k import matvec

@dataclass(frozen=True)
class Stratum:
    name: str
    rank: int
    def __post_init__(self):
        if self.rank<1:
            raise ValueError("rank must be positive")


@dataclass(frozen=True)
class Restriction:
    source: Stratum
    target: Stratum
    matrix: tuple
    def __post_init__(self):
        if len(self.matrix)!=self.target.rank or any(len(row)!=self.source.rank for row in self.matrix):
            raise ValueError("restriction incompatible with rank")
    def apply(self,state):
        if len(state)!=self.source.rank:
            raise ValueError("wrong source rank")
        return matvec(self.matrix,state)


def gluing_check(left,right,link_left,link_right):
    """Agreement of restricted values; does not assert unique existence of lift."""
    if link_left.target!=link_right.target:
        raise ValueError("different overlap spaces")
    a=link_left.apply(left)
    b=link_right.apply(right)
    remainder=tuple(x-y for x,y in zip(a,b))
    return {"compatible":all(abs(v)<1e-12 for v in remainder),
            "boundary_remainder":remainder}


def selfcheck():
    soma=Stratum("soma",4)
    process=Stratum("neurite",3)
    interface=Stratum("shared boundary",2)
    left=Restriction(soma,interface,((1.,0.,0.,0.),(0.,1.,0.,0.)))
    right=Restriction(process,interface,((1.,0.,0.),(0.,1.,0.)))
    ok=gluing_check((2.,-1.,4.,7.),(2.,-1.,9.),left,right)
    failed=gluing_check((2.,-1.,4.,7.),(2.,-2.,9.),left,right)
    invisible=left.apply((0.,0.,1.,-3.))
    assert ok["compatible"] and not failed["compatible"]
    assert invisible==(0.,0.)
    return {"compatible_overlap":ok,"incompatible_overlap":failed,
            "nonzero_invisible_state":True,
            "not_a_rank_varying_vector_bundle":True}


if __name__=="__main__":
    print(json.dumps(selfcheck(),indent=2))

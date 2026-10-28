"""G0-AA: evidence-gated temporal Worldlet splitting via exact finite mixtures.

A finite candidate ecology:
  SHARED(T) with one latent D;
  SPLIT(T,K) with one unknown boundary K and two independent latent Ds;
  DIFFUSE with independent D_t per observation.
All inference uses a closed-form Gaussian conjugate filter and normed mixture.
No biological growth, stack descent, or learned topology is asserted.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random

from .active_epistemics_y import (
    Belief, PRIOR_VAR, ENERGY_CAP, ENERGY_FLOOR,
    PROBE_BY_NAME,action_cost,sigma
)
from .provenance_z import ProvenanceBelief,SCHEDULE


@dataclass
class LocalWorldletHypothesis:
    mode: str
    twist: int | None
    cut: int | None
    weight: float
    mean: float=0.
    variance: float=PRIOR_VAR
    def reset(self):
        self.mean=0.
        self.variance=PRIOR_VAR


class TemporalEcology:
    """A finite weighted family, not a globally unified flat latent vector."""

    def __init__(self,horizon=len(SCHEDULE)):
        if horizon<2:
            raise ValueError("at least two observations needed for split")
        self.horizon=horizon
        self.events=0
        self.local_worlds=[
            LocalWorldletHypothesis("shared",t,None,.45/2)
            for t in (0,1)]
        self.local_worlds +=[
            LocalWorldletHypothesis("split",t,k,.35/(2*(horizon-1)))
            for k in range(1,horizon) for t in (0,1)]
        self.local_worlds.append(LocalWorldletHypothesis("diffuse",None,None,.20))
        if abs(sum(w.weight for w in self.local_worlds)-1)>1e-12:
            raise AssertionError("invalid prior")

    def update(self,parity,y,sd):
        if self.events>=self.horizon:
            raise ValueError("no additional observations in finite protocol")
        if sd<=0 or not math.isfinite(y):
            raise ValueError("invalid sensor")
        logs=[]
        predictions=[]
        for model in self.local_worlds:
            if model.mode=="split" and self.events==model.cut:
                model.reset() # transition BEFORE reading event indexed by cut
            if model.mode=="diffuse":
                mu=0.;variance=PRIOR_VAR+sd*sd;sign=0.
            else:
                sign=-1. if (model.twist==1 and parity%2) else 1.
                mu=sign*model.mean
                variance=model.variance+sd*sd
            predictions.append((mu,variance,sign))
            logs.append(math.log(max(1e-300,model.weight))
                        -.5*math.log(variance)-.5*(y-mu)**2/variance)
        mx=max(logs)
        raw=[math.exp(v-mx) for v in logs]
        normalizer=sum(raw)
        if normalizer<=0 or not math.isfinite(normalizer):
            raise ArithmeticError("invalid Bayes evidence")
        for model,(mu,variance,sign),raw_weight in zip(self.local_worlds,predictions,raw):
            model.weight=raw_weight/normalizer
            if model.mode!="diffuse":
                gain=model.variance*sign/variance
                model.mean+=gain*(y-mu)
                model.variance=model.variance*sd*sd/variance
                if not 0<model.variance<=PRIOR_VAR:
                    raise ArithmeticError("invalid posterior variance")
        self.events+=1
        assert abs(sum(w.weight for w in self.local_worlds)-1)<1e-12
        return self

    def probability(self,mode,twist=None,cut=None):
        return sum(m.weight for m in self.local_worlds
                   if m.mode==mode and
                   (twist is None or m.twist==twist) and
                   (cut is None or m.cut==cut))

    @property
    def p_shared(self):
        return self.probability("shared")

    @property
    def p_split(self):
        return self.probability("split")

    @property
    def p_diffuse(self):
        return self.probability("diffuse")

    @property
    def p_meaningful_orientation(self):
        return self.p_shared+self.p_split

    @property
    def p_transpose_conditional(self):
        denom=max(1e-300,self.p_meaningful_orientation)
        return (self.probability("shared",twist=1)+
                self.probability("split",twist=1))/denom

    def split_distribution(self):
        total=max(1e-300,self.p_split)
        return {k:self.probability("split",cut=k)/total
                for k in range(1,self.horizon)}

    def proposed_repair(self):
        """Returns two *typed temporal descriptors*, not physical new neurons."""
        if self.p_split<.5:
            return None
        dist=self.split_distribution()
        k,p=max(dist.items(),key=lambda item:item[1])
        if p<.30:
            return None
        return {
            "evidence_mode":"finite-single-changepoint",
            "boundary_after_event":k,
            "confidence_in_segmented_family":self.p_split,
            "conditional_boundary_probability":p,
            "worldlets":[
                {"type":"temporal-context","span":[0,k],"local_prior":"Gaussian D_pre"},
                {"type":"temporal-context","span":[k,self.horizon],"local_prior":"Gaussian D_post"}
            ],
            "remainder_type":"[R]_id"
        }

    def decision(self):
        if self.p_meaningful_orientation<.70:
            return None
        p=self.p_transpose_conditional
        if max(p,1-p)<.95:
            return None
        return int(p>=.5)

    def provenance_verdict(self):
        by={"shared":self.p_shared,"split":self.p_split,"diffuse":self.p_diffuse}
        key=max(by,key=by.get)
        return key if by[key]>=.5 else "unresolved"


def observation_stream(seed,world):
    """Uses G0-Z's existing real sensor equations/seed ledger for parity."""
    if world not in ("shared_identity","shared_transpose",
                     "change_identity","change_transpose","independent"):
        raise ValueError(world)
    d0=random.Random(seed*1009+1729).gauss(0,math.sqrt(PRIOR_VAR))
    d1=random.Random(seed*1231+891).gauss(0,math.sqrt(PRIOR_VAR))
    twist=world in ("shared_transpose","change_transpose")
    reserve=ENERGY_CAP
    values=[]
    for index,name in enumerate(SCHEDULE):
        probe=PROBE_BY_NAME[name]
        if world=="independent":
            D=random.Random(seed*4337+(index+1)*1523+367).gauss(0,math.sqrt(PRIOR_VAR))
        elif world.startswith("change") and index>=3:
            D=d1
        else:
            D=d0
        sd=sigma(probe,reserve)
        noise=random.Random(seed*104729+(index+1)*65537+
                            list(PROBE_BY_NAME).index(name)*997+103).gauss(0,sd)
        y=(-D if twist and probe.parity else D)+noise
        reserve-=action_cost(probe)
        if reserve<ENERGY_FLOOR-1e-12:
            raise AssertionError("overspent")
        values.append((probe.parity,y,sd))
    return values,ENERGY_CAP-reserve


def episode(seed,world):
    observations,spent=observation_stream(seed,world)
    naive=Belief()
    previous=ProvenanceBelief()
    revised=TemporalEcology(horizon=len(observations))
    for parity,y,sd in observations:
        naive.update(parity,y,sd)
        previous.update(parity,y,sd)
        revised.update(parity,y,sd)
    naive_dec=int(naive.w[1]>=.5) if naive.confidence()>=.95 else None
    prev_dec=previous.decision
    revised_dec=revised.decision()
    true_T=(1 if world.endswith("transpose") else 0) if world!="independent" else None
    repair=revised.proposed_repair()
    return {
        "seed":seed,"world":world,"true_twist":true_T,
        "plain_orientation_p":naive.w[1],
        "prior_model_orientation_p":previous.orientation_given_shared,
        "naive_orientation_decision":naive_dec,
        "prior_orientation_decision":prev_dec,
        "revised_orientation_decision":revised_dec,
        "p_shared":revised.p_shared,"p_split":revised.p_split,
        "p_diffuse":revised.p_diffuse,
        "orientation_given_continuity":revised.p_transpose_conditional,
        "provenance_verdict":revised.provenance_verdict(),
        "repair":repair,
        "repair_proposed":repair is not None,
        "repair_boundary":repair["boundary_after_event"] if repair else None,
        "known_true_cut":3 if world.startswith("change") else None,
        "energy":spent,
        "naive_gluing_claim":naive_dec is not None,
        "prior_gluing_claim":prev_dec is not None,
        "revised_gluing_claim":revised_dec is not None
    }


def summary(rows):
    n=len(rows)
    world=rows[0]["world"]
    known=world!="independent"
    expected_T=rows[0]["true_twist"]
    change=world.startswith("change")
    return {
        "world":world,"episodes":n,
        "naive_declaration_rate":sum(r["naive_gluing_claim"] for r in rows)/n,
        "previous_declaration_rate":sum(r["prior_gluing_claim"] for r in rows)/n,
        "revised_declaration_rate":sum(r["revised_gluing_claim"] for r in rows)/n,
        "revised_false_declaration_rate":(
            sum(r["revised_gluing_claim"] and r["revised_orientation_decision"]!=expected_T for r in rows)/n
            if known else None),
        "unsupported_declaration_rate":(
            sum(r["revised_gluing_claim"] for r in rows)/n if not known else None),
        "mean_p_shared":sum(r["p_shared"] for r in rows)/n,
        "mean_p_split":sum(r["p_split"] for r in rows)/n,
        "mean_p_diffuse":sum(r["p_diffuse"] for r in rows)/n,
        "split_proposal_rate":sum(r["repair_proposed"] for r in rows)/n,
        "split_exact_rate":(
            sum(r["repair_boundary"]==3 for r in rows)/n if change else None),
        "false_split_rate":(
            sum(r["repair_proposed"] for r in rows)/n if not change else None),
        "provenance_verdicts":{m:sum(r["provenance_verdict"]==m for r in rows)
                              for m in ("shared","split","diffuse","unresolved")},
        "mean_resource":sum(r["energy"] for r in rows)/n
    }


def selfcheck():
    b=TemporalEcology()
    baseline=[h.weight for h in b.local_worlds]
    b.update(0,1.07,.33)
    assert max(abs(x.weight-v) for x,v in zip(b.local_worlds,baseline))<1e-12
    b=TemporalEcology()
    for v in (1.0,1.08,1.15,1.01,1.1,1.07):
        b.update(0,v,.24)
    assert abs(b.p_transpose_conditional-.5)<1e-11
    assert abs(sum(h.weight for h in b.local_worlds)-1)<1e-12
    # Independent example of full 6-observation model.
    for world in ("shared_identity","shared_transpose","change_identity",
                  "change_transpose","independent"):
        assert episode(0,world)["energy"]<=ENERGY_CAP-ENERGY_FLOOR+1e-12
    return {"n_component_hypotheses":len(TemporalEcology().local_worlds),
            "one_observation_nonidentifying":True,
            "one_parity_orientation_nonidentifying":True,
            "positive_conjugate_variances":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=128)
    args=parser.parse_args()
    if not 1<=args.seeds<=256:
        parser.error("seed count 1..256")
    print("SELF",json.dumps(selfcheck(),sort_keys=True))
    for world in ("shared_identity","shared_transpose","change_identity",
                  "change_transpose","independent"):
        rows=[episode(i,world) for i in range(args.seeds)]
        print("WORLD",json.dumps(summary(rows),sort_keys=True))


if __name__=="__main__":
    main()

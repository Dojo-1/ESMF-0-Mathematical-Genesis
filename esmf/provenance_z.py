"""G0-Z: a provenance/continuity hypothesis prevents unwarranted fiber gluing.

Compare exact online H_I, H_P (shared latent D) with H_independent (fresh D
per observation), under the SAME noisy scalar sensor stream and finite energy.
No real cortex, visceral neural dynamics, or nontrivial energy measurement.
"""
import argparse
import json
import math
import random

from .active_epistemics_y import (
    Belief, ENERGY_CAP,ENERGY_FLOOR,PRIOR_VAR,
    PROBE_BY_NAME,action_cost,sigma
)


SCHEDULE=("even_focus","odd_focus","even_quiet",
          "odd_quiet","even_quiet","odd_quiet")


class ProvenanceBelief:
    def __init__(self):
        self.w=[.4,.4,.2] # shared identity, shared transpose, independent
        self.mean=[0.,0.]
        self.var=[PRIOR_VAR,PRIOR_VAR]
        self.events=0

    def update(self,parity,y,sd):
        if not math.isfinite(y) or sd<=0:
            raise ValueError("bad provenance observation")
        values=[]
        for t in range(2):
            sign=-1. if (t==1 and parity) else 1.
            predicted=sign*self.mean[t]
            variance=self.var[t]+sd*sd
            values.append((predicted,variance,sign))
        # Newly drawn latent D_k under independent-hypothesis model.
        values.append((0.,PRIOR_VAR+sd*sd,0.))
        scores=[math.log(max(1e-300,self.w[t]))
                -.5*math.log(v)-.5*(y-m)**2/v
                for t,(m,v,_) in enumerate(values)]
        biggest=max(scores)
        raw=[math.exp(z-biggest) for z in scores]
        total=sum(raw)
        oldmean=self.mean[:]
        oldvar=self.var[:]
        self.w=[r/total for r in raw]
        for t in range(2):
            predicted,var,sign=values[t]
            kalman=oldvar[t]*sign/var
            self.mean[t]=oldmean[t]+kalman*(y-predicted)
            self.var[t]=oldvar[t]*sd*sd/var
            if self.var[t]<=0 or not math.isfinite(self.mean[t]):
                raise ArithmeticError("invalid posterior for latent D")
        self.events+=1
        if not abs(sum(self.w)-1)<1e-12:
            raise ArithmeticError("evidence weights not normalized")
        return self

    @property
    def shared(self):
        return self.w[0]+self.w[1]

    @property
    def independent(self):
        return self.w[2]

    @property
    def orientation_given_shared(self):
        return self.w[1]/max(1e-300,self.shared)

    @property
    def quarantine(self):
        return self.independent>=.5

    @property
    def decision(self):
        if self.shared<.70:
            return None
        p=self.orientation_given_shared
        if max(p,1-p)<.95:
            return None
        return int(p>=.5)


def episode(seed,world,drift_start=3):
    """No hidden state is provided to either inference filter."""
    if world not in ("shared_identity","shared_transpose","independent","change_point"):
        raise ValueError(world)
    baseline_rng=random.Random(seed*1009+1729)
    d0=baseline_rng.gauss(0,math.sqrt(PRIOR_VAR))
    d1=random.Random(seed*1231+891).gauss(0,math.sqrt(PRIOR_VAR))
    twist=world=="shared_transpose"
    energy=ENERGY_CAP
    plain=Belief()
    provenance=ProvenanceBelief()
    observations=[]
    for index,name in enumerate(SCHEDULE):
        probe=PROBE_BY_NAME[name]
        if energy-action_cost(probe)<ENERGY_FLOOR-1e-12:
            raise AssertionError("schedule exceeds energy reserve")
        if world=="independent":
            d=random.Random(seed*4337+(index+1)*1523+367).gauss(0,math.sqrt(PRIOR_VAR))
        elif world=="change_point" and index>=drift_start:
            d=d1
        else:
            d=d0
        sd=sigma(probe,energy)
        sign=-1 if twist and probe.parity else 1
        noise=random.Random(seed*104729+(index+1)*65537+
                            list(PROBE_BY_NAME).index(name)*997+103).gauss(0,sd)
        y=sign*d+noise
        plain.update(probe.parity,y,sd)
        provenance.update(probe.parity,y,sd)
        observations.append((probe.parity,y,sd))
        energy-=action_cost(probe)
    if energy<ENERGY_FLOOR-1e-12:
        raise AssertionError("resource underflow")
    plain_decision=(
        int(plain.w[1]>=.5) if max(plain.w)>=.95 else None)
    return {"seed":seed,"world":world,
            "world_has_persistent_D":world.startswith("shared"),
            "plain_orientation_decision":plain_decision,
            "audited_orientation_decision":provenance.decision,
            "plain_orientation_p":plain.w[1],
            "audited_orientation_conditional_p":provenance.orientation_given_shared,
            "posterior_shared":provenance.shared,
            "posterior_independent":provenance.independent,
            "quarantined":provenance.quarantine,
            "plain_confident":plain_decision is not None,
            "audited_confident":provenance.decision is not None,
            "plain_correct":plain_decision==int(twist) if plain_decision is not None and world.startswith("shared") else False,
            "audited_correct":provenance.decision==int(twist) if provenance.decision is not None and world.startswith("shared") else False,
            "simulated_resource_spent":ENERGY_CAP-energy,
            "observations":len(observations)}


def summary(rows):
    n=len(rows)
    shared=all(r["world_has_persistent_D"] for r in rows)
    return {"episodes":n,
            "naive_confident_rate":sum(r["plain_confident"] for r in rows)/n,
            "provenance_confident_rate":sum(r["audited_confident"] for r in rows)/n,
            "naive_confident_false_rate":(
                sum(r["plain_confident"] and not r["plain_correct"] for r in rows)/n
                if shared else None),
            "provenance_confident_false_rate":(
                sum(r["audited_confident"] and not r["audited_correct"] for r in rows)/n
                if shared else None),
            "naive_false_gluing_rate":sum(r["plain_confident"] for r in rows)/n if not shared else None,
            "provenance_false_gluing_rate":sum(r["audited_confident"] for r in rows)/n if not shared else None,
            "quarantine_rate":sum(r["quarantined"] for r in rows)/n,
            "mean_posterior_independent":sum(r["posterior_independent"] for r in rows)/n,
            "mean_simulated_resource":sum(r["simulated_resource_spent"] for r in rows)/n}


def selfcheck():
    # At one observation, all hypotheses have identical marginal law.
    b=ProvenanceBelief()
    before=b.w[:]
    b.update(0,1.07,.34)
    assert max(abs(x-y) for x,y in zip(before,b.w))<1e-12
    # Even-only readings can testify about continuity, not orientation.
    b=ProvenanceBelief()
    for y in (1.12,1.08,1.14,1.15):
        b.update(0,y,.24)
    assert b.shared>.8 and abs(b.orientation_given_shared-.5)<1e-12
    # Independent world has no stable T; evidence over a sequence can
    # quarantine a model built on false state identity.
    b=ProvenanceBelief()
    for y in (1.8,-1.7,1.4,-1.9,1.6):
        b.update(0,y,.22)
    assert b.quarantine
    # Resource schedules physically permitted for all seed/world scenarios.
    example=episode(0,"shared_identity")
    assert example["observations"]==len(SCHEDULE)
    return {"one_observation_cannot_identify_continuity":True,
            "even_only_can_inform_continuity_not_orientation":True,
            "broken_continuity_can_trigger_quarantine":True,
            "schedule_within_energy_budget":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=128)
    args=parser.parse_args()
    if not 1<=args.seeds<=256:
        parser.error("1..256")
    print("SELF",json.dumps(selfcheck()))
    for world in ("shared_identity","shared_transpose","independent","change_point"):
        rows=[episode(seed,world) for seed in range(args.seeds)]
        print("WORLD",json.dumps({"world":world,**summary(rows)},sort_keys=True))


if __name__=="__main__":
    main()

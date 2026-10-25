"""G0-AB: online plasticity across fast sensing, mesoscopic gating and slow context.

The physical latent source D_t is the shear coordinate of a neutral-signature
Mat(2,R) fiber F=[[1,D],[0,1]], det F=1. This is a ONE-dimensional orbit
inside the 4D carrier, not full learned four-dimensional cognition.

An interacting multiple-model Gaussian filter tracks two hidden orientation
classes and three latent continuity modes. Its moment matching is approximate
[R]_transport. A learned slow reset hazard changes future transition laws.
No biological/energetic equivalence or exclusive geometric benefit is claimed.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random
import time

from .active_epistemics_y import PROBE_BY_NAME,action_cost,sigma,entropy
from .associated22_u import H,neutral_square,flatten
from .orientation_seam_w import P
from .geometry_k import matvec


ENERGY_CAP=8.
ENERGY_FLOOR=.25
SUPPLY=.16
HORIZON=12
MODES=("stable","drift","reset")
PRIOR_MODE=(.70,.20,.10)
RHO=(1.,.96,0.)
QVAR=(.0008,.045,1.)
SIGNS=(1.,-1.)
SENSOR_CONFIDENCE=.95


def clamp(value,lower,upper):
    return min(upper,max(lower,value))


def logsum(values):
    big=max(values)
    return big+math.log(sum(math.exp(v-big) for v in values))


def fiber(d):
    return ((1.,float(d)),(0.,1.))


def fiber_transport(d0,d1):
    return ((1.,d1-d0),(0.,1.))


def neutral_invariant(d,twist,parity):
    f=flatten(fiber(d))
    represented=matvec(P,f) if (twist and parity) else f
    return neutral_square(represented)


def transition_rows(h):
    h=clamp(h,.005,.30)
    return (
        (1.-.07-h,.07,h),
        (.06,1.-.06-min(.45,1.5*h),min(.45,1.5*h)),
        (.62,.28,.10)
    )


@dataclass
class LocalHypothesis:
    weight:float
    mean:float=0.
    var:float=1.


class AdaptiveContinuityFilter:
    def __init__(self,plastic=True):
        self.states=[[LocalHypothesis(.4*PRIOR_MODE[m]) for m in range(3)]
                     for t in range(2)]
        # A separate whole-episode no-continuity model. It has no T;
        # conditioning on it to classify orientation would be invalid.
        self.diffuse_weight=.20
        self.hazard=.045
        self.plastic=plastic
        self.fast_clock=0
        self.slow_clock=0
        self.surprise=1.
        self.last_reset=0.
        self.moment_match_remainder=0.
        self.hazard_history=[self.hazard]

    @property
    def p_persistent(self):
        return sum(x.weight for row in self.states for x in row)

    @property
    def probability_twist(self):
        # CONDITIONAL on a persistent source, never a marginal claim when
        # independent-event replacement dominates.
        return sum(h.weight for h in self.states[1])/max(1e-300,self.p_persistent)

    @property
    def p_reset(self):
        return sum(self.states[t][2].weight for t in range(2))/max(1e-300,self.p_persistent)

    def predictive_step(self,observed=None,parity=0,sd=.5):
        if sd<=0:
            raise ValueError("nonpositive sensor sd")
        priors=[]
        for t in range(2):
            if self.fast_clock==0:
                priors.append([(x.weight,x.mean,x.var) for x in self.states[t]])
                continue
            trans=transition_rows(self.hazard)
            weights=[sum(self.states[t][i].weight*trans[i][j]
                         for i in range(3)) for j in range(3)]
            mixed=[]
            for j in range(3):
                weightsj=[self.states[t][i].weight*trans[i][j]
                          for i in range(3)]
                total=weights[j]
                if total<=0:
                    raise ArithmeticError("lost all modes")
                mean=sum(weightsj[i]*self.states[t][i].mean
                         for i in range(3))/total
                var=sum(weightsj[i]*(self.states[t][i].var+
                                     (self.states[t][i].mean-mean)**2)
                        for i in range(3))/total
                # Positive spread produced by Gaussian moment closure;
                # retained only as a typed diagnostic, NOT a signed energy.
                self.moment_match_remainder+=var-sum(
                    weightsj[i]*self.states[t][i].var for i in range(3))/total
                mixed.append((total,RHO[j]*mean,RHO[j]*RHO[j]*var+QVAR[j]))
            priors.append(mixed)
        diagnostics=[]
        post=[]
        logweights=[]
        prior_total=sum(x[0] for row in priors for x in row)+self.diffuse_weight
        if abs(prior_total-1.)>1e-11:
            raise ArithmeticError("prior mixture not normalized")
        if observed is not None and not math.isfinite(observed):
            raise ValueError("nonfinite observation")
        for t in range(2):
            newrow=[]
            for j,(w,mu,var) in enumerate(priors[t]):
                if var<=0 or not math.isfinite(var):
                    raise ArithmeticError("nonpositive predicted variance")
                if observed is None:
                    newrow.append(LocalHypothesis(w,mu,var))
                    logweights.append(math.log(max(1e-300,w)))
                    continue
                s=-1. if t and parity else 1.
                pred=s*mu
                variance=var+sd*sd
                residual=observed-pred
                logweights.append(math.log(max(1e-300,w))-.5*(
                    math.log(2*math.pi*variance)+residual*residual/variance))
                gain=var*s/variance
                new_mu=mu+gain*residual
                new_var=var*sd*sd/variance
                newrow.append(LocalHypothesis(w,new_mu,new_var))
                diagnostics.append(w*residual*residual/variance)
            post.append(newrow)
        # Hypothesis outside the global orientation fiber: D_t is freshly
        # drawn each event, so Y_t has no information about T.
        if observed is None:
            logweights.append(math.log(max(1e-300,self.diffuse_weight)))
        else:
            vv=1.+sd*sd
            logweights.append(math.log(max(1e-300,self.diffuse_weight))-
                              .5*(math.log(2*math.pi*vv)+observed*observed/vv))
        norm=logsum(logweights)
        if observed is not None:
            self.surprise=.85*self.surprise+.15*sum(diagnostics)
        k=0
        for row in post:
            for state in row:
                state.weight=math.exp(logweights[k]-norm)
                if not 0<state.var<1e8 or not math.isfinite(state.mean):
                    raise ArithmeticError("invalid Kalman component")
                k+=1
        self.states=post
        self.diffuse_weight=math.exp(logweights[-1]-norm)
        if abs(sum(x.weight for row in post for x in row)+self.diffuse_weight-1.)>1e-10:
            raise ArithmeticError("posterior mixture not normalized")
        self.fast_clock+=1
        self.last_reset=self.p_reset
        if self.plastic and self.fast_clock%4==0 and observed is not None:
            # Slow metaplastic feedback; only affects FUTURE transitions.
            # This is a heuristic, not a conjugate update of a true hazard.
            target=.012+.28*self.p_reset
            self.hazard=clamp(.7*self.hazard+.3*target,.005,.30)
            self.slow_clock+=1
            self.hazard_history.append(self.hazard)
        return self.probability_twist

    def label(self):
        p=self.probability_twist
        # Declare only when the JOINT event (persistent source AND this
        # orientation) is supported. Two separate weak gates formerly
        # permitted as little as .70*.95=.665 joint credibility.
        support=self.p_persistent*max(p,1-p)
        return int(p>=.5) if support>=SENSOR_CONFIDENCE else None

    def recontextualize(self,tick):
        """A typed structural event redirects future mode mass, preserving P(T)."""
        before=self.probability_twist
        for t in range(2):
            old=self.states[t]
            total=sum(s.weight for s in old)
            # Use model selection's existing reset component, but make the
            # next prediction more receptive to a new local context.
            boost=.20*total
            old[0].weight*=.80
            old[1].weight*=.80
            old[2].weight=total-old[0].weight-old[1].weight
        if abs(self.probability_twist-before)>1e-10:
            raise AssertionError("context repair changed orientation belief")
        self.hazard=clamp(max(self.hazard,.10),.005,.30)
        return {"boundary_after_tick":tick,
                "worldlet_parent":"context-before",
                "worldlet_child":f"context-after-{tick}",
                "typed_remainder":"[R]_id",
                "orientation_mass_preserved":True}


@dataclass
class ResourceLedger:
    reserve:float=ENERGY_CAP
    delivered:float=0.
    spent:float=0.
    overflow:float=0.

    def replenish(self):
        new=self.reserve+SUPPLY
        accepted=min(ENERGY_CAP,new)
        self.delivered+=SUPPLY
        self.overflow+=new-accepted
        self.reserve=accepted
        self.verify()

    def charge(self,amount):
        if amount<0 or self.reserve-amount<ENERGY_FLOOR-1e-10:
            return False
        self.reserve-=amount
        self.spent+=amount
        self.verify()
        return True

    def verify(self):
        if not ENERGY_FLOOR-1e-10<=self.reserve<=ENERGY_CAP+1e-10:
            raise AssertionError("invalid reserve")
        if abs(ENERGY_CAP+self.delivered-self.spent-self.overflow-self.reserve)>1e-9:
            raise AssertionError("resource balance failed")


class AutonomicGate:
    def __init__(self,mode):
        self.mode=mode
        self.focus=False
        self.last_surprise=1.
        self.meso_clock=0
        self.switches=0

    def update(self,tick,belief,ledger):
        if tick%2:
            return
        self.meso_clock+=1
        if self.mode.startswith("adaptive"):
            urgent=(belief.surprise>1.16 or belief.p_reset>.25)
            relax=(belief.surprise<.92 and belief.p_reset<.20)
            old=self.focus
            if urgent and ledger.reserve>1.55:
                self.focus=True
            elif relax or ledger.reserve<1.05:
                self.focus=False
            if old!=self.focus:
                self.switches+=1

    def choose(self,tick,belief,ledger,rng):
        if self.mode=="random":
            parity=rng.randrange(2)
            preference="focus" if rng.randrange(2) else "quiet"
        else:
            parity=tick%2
            if self.mode=="fixed_focus":
                preference="focus"
            elif self.mode=="fixed_quiet":
                preference="quiet"
            else:
                preference="focus" if self.focus else "quiet"
        names=(f"{'odd' if parity else 'even'}_{preference}",
               f"{'odd' if parity else 'even'}_quiet")
        for name in names:
            probe=PROBE_BY_NAME[name]
            if ledger.reserve-action_cost(probe)>=ENERGY_FLOOR-1e-10:
                return probe
        return None


POLICIES=("adaptive","adaptive_no_slow","fixed_focus",
          "fixed_quiet","random","equivalent_real")
WORLDS=("stable","drift","jumps","diffuse","switch")


def latent_path(seed,world):
    if world not in WORLDS:
        raise ValueError(world)
    rng=random.Random(seed*17311+15059)
    d=rng.gauss(0,1)
    points=[]
    for tick in range(HORIZON):
        if tick:
            if world=="diffuse" or (world=="jumps" and tick in (5,9)):
                d=rng.gauss(0,1)
            elif world=="drift" or (world=="switch" and tick>=6):
                d=.96*d+rng.gauss(0,math.sqrt(.045))
            else:
                d+=rng.gauss(0,math.sqrt(.0008))
        points.append(d)
    return points


def simulate(seed,twist,world,policy):
    if policy not in POLICIES or twist not in (0,1):
        raise ValueError("unknown simulation choice")
    physical=latent_path(seed,world)
    belief=AdaptiveContinuityFilter(plastic=(policy not in
                                             ("adaptive_no_slow",)))
    ledger=ResourceLedger()
    gate=AutonomicGate("adaptive" if policy=="equivalent_real" else policy)
    env_rng=random.Random(seed*107+8421)
    history=[]
    buds=[]
    last_bud=-100
    for tick,d in enumerate(physical):
        ledger.replenish()
        gate.update(tick,belief,ledger)
        probe=gate.choose(tick,belief,ledger,env_rng)
        obs=None
        measured_sd=.5
        parity=tick%2
        if probe is not None:
            # Preserve G0-Y's reserve-dependent sensory law, but use the
            # present resource level before paying for the observation.
            measured_sd=sigma(probe,min(5.5,ledger.reserve))
            parity=probe.parity
            if not ledger.charge(action_cost(probe)):
                raise AssertionError("gate chose unaffordable probe")
            noise_rng=random.Random(seed*104729+(tick+1)*65537+
                                   list(PROBE_BY_NAME).index(probe.name)*997+103)
            obs=(-d if twist and parity else d)+noise_rng.gauss(0,measured_sd)
        else:
            # Even quiescent organs have maintenance cost; charge if viable.
            ledger.charge(.025)
        belief.predictive_step(obs,parity,measured_sd)
        # Amendment after the first pilot: jump evidence is often a
        # sharp, isolated event. A two-consecutive-observation rule missed
        # all true resets; use a one-event posterior reset gate with a
        # higher threshold, source-continuity floor and refractory period.
        # Never present this posthoc repair as a preregistered result.
        if (policy in ("adaptive","equivalent_real") and
              obs is not None and tick>=2 and
              belief.p_persistent>=.85 and belief.p_reset>.33
              and tick-last_bud>=3):
            buds.append(belief.recontextualize(tick))
            last_bud=tick
        history.append({"tick":tick,"probe":probe.name if probe else "sleep",
                        "reserve":round(ledger.reserve,5),
                        "p_twist":round(belief.probability_twist,5),
                        "p_reset":round(belief.p_reset,5),
                        "hazard":round(belief.hazard,5)})
    p=belief.probability_twist
    label=belief.label()
    ledger.verify()
    assert all(abs(neutral_invariant(d,twist,0)-1)<1e-12 for d in physical)
    assert all(abs(neutral_invariant(d,twist,1)-1)<1e-12 for d in physical)
    return {"world":world,"seed":seed,"twist":twist,"policy":policy,
            "label":label,"correct":label==twist if label is not None else False,
            "confidence":max(p,1-p),
            "brier":(p-twist)**2,
            "logloss":-math.log(max(1e-300,p if twist else 1-p)),
            "energy":ledger.spent,"external_supply":ledger.delivered,
            "overflow":ledger.overflow,"reserve":ledger.reserve,
            "probes":sum(1 for x in history if x["probe"]!="sleep"),
            "focus":sum(1 for x in history if "focus" in x["probe"]),
            "hazard_initial":.045,"hazard_final":belief.hazard,
            "p_independent":belief.diffuse_weight,
            "hazard_updates":belief.slow_clock,
            "gate_switches":gate.switches,
            "bud_count":len(buds),"buds":buds,
            "moment_match_dispersion":belief.moment_match_remainder,
            "trace":history}


def summary(rows):
    n=len(rows)
    covered=sum(x["label"] is not None for x in rows)
    world=rows[0]["world"]
    is_diffuse=world=="diffuse"
    return {
        "world":world,"policy":rows[0]["policy"],"episodes":n,
        "coverage":covered/n,
        "selective_accuracy":(sum(x["correct"] for x in rows)/covered if covered
                              and not is_diffuse else None),
        "false_orientation_declarations":covered if is_diffuse else
                                           sum(x["label"] is not None and not x["correct"] for x in rows),
        "mean_brier":sum(x["brier"] for x in rows)/n if not is_diffuse else None,
        "mean_abs_p_half":sum(abs(x["confidence"]-.5) for x in rows)/n,
        "mean_energy":sum(x["energy"] for x in rows)/n,
        "mean_probes":sum(x["probes"] for x in rows)/n,
        "mean_focus":sum(x["focus"] for x in rows)/n,
        "mean_final_hazard":sum(x["hazard_final"] for x in rows)/n,
        "mean_p_independent":sum(x["p_independent"] for x in rows)/n,
        "mean_buds":sum(x["bud_count"] for x in rows)/n,
        "timely_jump_event_detection":(
            sum(any(abs(b["boundary_after_tick"]-target)<=1 for b in x["buds"])
                for x in rows for target in (5,9))/(2*n)
            if world=="jumps" else None),
        "mean_gate_switches":sum(x["gate_switches"] for x in rows)/n
    }


def selfcheck():
    for x in (.0,-3.,2.7):
        assert abs(neutral_invariant(x,0,0)-1)<1e-12
        assert abs(neutral_invariant(x,1,1)-1)<1e-12
    f=AdaptiveContinuityFilter()
    for tick in range(12):
        f.predictive_step(None)
        assert abs(f.probability_twist-.5)<1e-12
        assert all(s.var>0 and math.isfinite(s.var) for row in f.states for s in row)
    r=simulate(0,0,"drift","adaptive")
    assert r["hazard_updates"]==3
    assert r["probes"]<=HORIZON
    r2=simulate(0,0,"drift","equivalent_real")
    assert {k:v for k,v in r.items() if k!="policy"}=={
        k:v for k,v in r2.items() if k!="policy"}
    return {"neutral_fiber_det_one":True,
            "no_observation_stays_nonidentifying":True,
            "three_slow_plasticity_opportunities":True,
            "identical_real_matched_control":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=64)
    parser.add_argument("--worlds",default=",".join(WORLDS))
    args=parser.parse_args()
    worlds=tuple(args.worlds.split(","))
    if not 1<=args.seeds<=256 or any(w not in WORLDS for w in worlds):
        parser.error("invalid seeds or worlds")
    print("SELF",json.dumps(selfcheck()))
    for world in worlds:
        for policy in POLICIES:
            start=time.perf_counter()
            rows=[simulate(s,t,world,policy)
                  for s in range(args.seeds) for t in (0,1)]
            result=summary(rows)
            result["wall_s"]=time.perf_counter()-start
            print("MODEL",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

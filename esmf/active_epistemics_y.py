"""G0-Y: resource-limited active sensing of a hidden neutral-bundle contrast.

The hidden orientation T∈{+1,-1} changes the sign of D under odd windings.
D~N(0,1) is shared and *unobserved* across each episode. A scalar sensor
returns Y=s(T,parity)*D+Gaussian noise. A two-component exact Gaussian
filter tracks T and D jointly. Neither parity alone identifies T.

This is a reduced, mathematically traceable partial-observation SURROGATE,
not a biological body, autonomous cognition, or measured electrical power.
"""
from dataclasses import dataclass
import argparse
import json
import math
import random
from statistics import NormalDist


PRIOR_VAR=1.0
ENERGY_CAP=5.5
ENERGY_FLOOR=.22
MAINTENANCE=.025
CONFIDENCE=.95
MAX_EVENTS=9
GAUSS_Z=tuple(NormalDist().inv_cdf((j+.5)/13) for j in range(13))


@dataclass(frozen=True)
class Probe:
    name: str
    parity: int
    mode: str
    energy: float
    baseline_noise: float


PROBES=(
    Probe("even_quiet",0,"quiet",.28,.52),
    Probe("even_focus",0,"focus",.72,.17),
    Probe("odd_quiet",1,"quiet",1.02,.52),
    Probe("odd_focus",1,"focus",1.75,.17)
)
PROBE_BY_NAME={p.name:p for p in PROBES}


def entropy(p):
    if p<=0. or p>=1.:
        return 0.
    return -p*math.log(p)-(1-p)*math.log1p(-p)


def logistic(z):
    if z>=0:
        q=math.exp(-min(z,700))
        return 1/(1+q)
    q=math.exp(max(z,-700))
    return q/(1+q)


def sigma(probe,reserve):
    """Known sensory/physiological law; not a learned biological fact."""
    factor=1+.55*(1-min(ENERGY_CAP,max(0.,reserve))/ENERGY_CAP)
    return probe.baseline_noise*factor


def action_cost(probe):
    return probe.energy+MAINTENANCE


class Belief:
    """Mixture: T∈{identity,transpose}; D|T ~ N(m_T,v_T)."""

    def __init__(self):
        self.w=[.5,.5]
        self.mean=[0.,0.]
        self.var=[PRIOR_VAR,PRIOR_VAR]
        self.observed_even=0
        self.observed_odd=0

    def clone(self):
        a=Belief()
        a.w=self.w[:]
        a.mean=self.mean[:]
        a.var=self.var[:]
        a.observed_even=self.observed_even
        a.observed_odd=self.observed_odd
        return a

    def predictive(self,t,parity,sd):
        sign=-1. if (t==1 and parity%2) else 1.
        return sign*self.mean[t],self.var[t]+sd*sd,sign

    def update(self,parity,y,sd):
        if sd<=0 or not math.isfinite(y):
            raise ValueError("invalid observation")
        old=self.clone()
        logs=[]
        for t in (0,1):
            mu,var,_=old.predictive(t,parity,sd)
            logs.append(math.log(max(1e-300,old.w[t]))
                        -.5*math.log(var)-.5*(y-mu)**2/var)
        mx=max(logs)
        weights=[math.exp(q-mx) for q in logs]
        total=sum(weights)
        self.w=[q/total for q in weights]
        for t in (0,1):
            mu,var,sign=old.predictive(t,parity,sd)
            gain=old.var[t]*sign/var
            self.mean[t]=old.mean[t]+gain*(y-mu)
            self.var[t]=old.var[t]*sd*sd/var
            if not math.isfinite(self.mean[t]) or not 0<=self.var[t]<=PRIOR_VAR:
                raise ArithmeticError("invalid Gaussian posterior")
        if parity%2:
            self.observed_odd+=1
        else:
            self.observed_even+=1
        if not 0<=self.w[0]<=1 or not abs(sum(self.w)-1)<1e-12:
            raise ArithmeticError("posterior normalization failed")
        return self

    def orientation_entropy(self):
        return entropy(self.w[1])

    def confidence(self):
        return max(self.w)


def expected_information(belief,parity,sd):
    before=belief.orientation_entropy()
    if before<1e-13:
        return 0.
    avg=0.
    for t in (0,1):
        mu,var,_=belief.predictive(t,parity,sd)
        spread=math.sqrt(var)
        weight=belief.w[t]/len(GAUSS_Z)
        for z in GAUSS_Z:
            next_b=belief.clone()
            next_b.update(parity,mu+spread*z,sd)
            avg+=weight*next_b.orientation_entropy()
    return max(0.,before-avg)


def expected_pair_gain(belief,first,second,reserve):
    """Two-step conditional information; the second gain depends on first Y."""
    r1=reserve-action_cost(first)
    r2=r1-action_cost(second)
    if r2<ENERGY_FLOOR-1e-12:
        return -1.
    sd1=sigma(first,reserve)
    sd2=sigma(second,r1)
    initial=belief.orientation_entropy()
    terminal=0.
    for t in (0,1):
        mu,var,_=belief.predictive(t,first.parity,sd1)
        spread=math.sqrt(var)
        weight=belief.w[t]/len(GAUSS_Z)
        for z in GAUSS_Z:
            post=belief.clone().update(first.parity,mu+spread*z,sd1)
            second_ig=expected_information(post,second.parity,sd2)
            terminal+=weight*(post.orientation_entropy()-second_ig)
    return max(0.,initial-terminal)


def feasible(probe,reserve):
    return reserve-action_cost(probe)>=ENERGY_FLOOR-1e-12


def choose_action(policy,belief,reserve,history,rng):
    available=[p for p in PROBES if feasible(p,reserve)]
    if not available:
        return None
    if policy=="greedy_stop":
        choices=[(expected_information(belief,p.parity,sigma(p,reserve))/action_cost(p),p)
                 for p in available]
        ratio,best=max(choices,key=lambda x:x[0])
        return best if ratio>1e-7 else None
    if policy in ("anticipatory","matched_real"):
        choices=[(expected_information(belief,p.parity,sigma(p,reserve))/action_cost(p),p)
                 for p in available]
        ratio,best=max(choices,key=lambda x:x[0])
        if ratio>1e-7:
            return best
        # The myopic estimator sees zero information at inception. Expect
        # complementary observations jointly, not each one in isolation.
        scores=[]
        for first in available:
            for second in available:
                if first.parity==second.parity:
                    continue
                gain=expected_pair_gain(belief,first,second,reserve)
                if gain>0:
                    scores.append((gain/(action_cost(first)+action_cost(second)),
                                   -action_cost(first),first.name,first))
        if not scores:
            return None
        return max(scores,key=lambda x:(x[0],x[1],x[2]))[-1]
    if policy=="alternating_quiet":
        return PROBE_BY_NAME["even_quiet" if len(history)%2==0 else "odd_quiet"] if feasible(
            PROBE_BY_NAME["even_quiet" if len(history)%2==0 else "odd_quiet"],reserve) else None
    if policy=="alternating_focus":
        p=PROBE_BY_NAME["even_focus" if len(history)%2==0 else "odd_focus"]
        return p if feasible(p,reserve) else None
    if policy=="odd_only":
        return PROBE_BY_NAME["odd_quiet"] if feasible(PROBE_BY_NAME["odd_quiet"],reserve) else None
    if policy=="even_only":
        return PROBE_BY_NAME["even_quiet"] if feasible(PROBE_BY_NAME["even_quiet"],reserve) else None
    if policy=="random":
        return rng.choice(available)
    raise ValueError("unknown policy: "+policy)


def simulate(seed,twist,policy,max_events=MAX_EVENTS):
    """T is hidden from agent and only supplied to the environment/evaluator."""
    if not 0<=seed<1000000 or twist not in (0,1):
        raise ValueError("invalid episode")
    contrast_rng=random.Random(seed*1009+1729)
    latent_D=contrast_rng.gauss(0,math.sqrt(PRIOR_VAR))
    belief=Belief()
    energy=ENERGY_CAP
    history=[]
    oracle_random=random.Random(seed*491+73991)  # same per hidden-class pair
    while len(history)<max_events and belief.confidence()<CONFIDENCE:
        probe=choose_action(policy,belief,energy,history,oracle_random)
        if probe is None:
            break
        if not feasible(probe,energy):
            raise AssertionError("energy gate broken")
        sd=sigma(probe,energy)
        sign=-1 if (twist and probe.parity) else 1
        # Stable common noise for shared seed/event number/probe type, across
        # comparison policies. Different action choices are counterfactuals.
        event_seed=(seed*104729+(len(history)+1)*65537+
                    list(PROBE_BY_NAME).index(probe.name)*997+103)
        sensor_rng=random.Random(event_seed)
        y=sign*latent_D+sensor_rng.gauss(0,sd)
        belief.update(probe.parity,y,sd)
        energy-=action_cost(probe)
        history.append(probe.name)
    confident=belief.confidence()>=CONFIDENCE
    decision=int(belief.w[1]>=.5) if confident else None
    if energy<ENERGY_FLOOR-1e-9:
        raise AssertionError("overspent budget")
    return {"seed":seed,"actual_twist":twist,"policy":policy,
            "posterior_twist":belief.w[1],
            "decision":decision,"confident":confident,
            "correct":bool(decision==twist) if confident else False,
            "spent":ENERGY_CAP-energy,
            "reserve":energy,
            "n_observations":len(history),
            "even":belief.observed_even,"odd":belief.observed_odd,
            "history":history,
            "entropy_gain_nats":math.log(2)-belief.orientation_entropy(),
            "brier":(belief.w[1]-twist)**2,
            "log_loss":-math.log(max(1e-300,belief.w[twist]))}


def closed_form_pair(z_even,z_odd,sd_even,sd_odd,prior_var=PRIOR_VAR):
    determinant=prior_var*(sd_even**2+sd_odd**2)+sd_even**2*sd_odd**2
    if determinant<=0:
        raise ValueError("singular observation covariance")
    # Log P(identity)/P(transpose)
    logratio=2*prior_var*z_even*z_odd/determinant
    return logistic(-logratio)


def numerical_selfcheck():
    for z_e,z_o in ((.7,.9),(-.6,.8),(.9,-.7),(0.,1.)):
        se,so=.32,.56
        b=Belief()
        b.update(0,z_e,se).update(1,z_o,so)
        analytic=closed_form_pair(z_e,z_o,se,so)
        assert abs(b.w[1]-analytic)<1e-12
        b2=Belief()
        b2.update(1,z_o,so).update(0,z_e,se)
        assert abs(b2.w[1]-analytic)<1e-12
    all_even=Belief()
    all_odd=Belief()
    for i in range(50):
        all_even.update(0,math.sin(i),.4)
        all_odd.update(1,math.sin(i),.4)
    assert abs(all_even.w[1]-.5)<1e-12 and abs(all_odd.w[1]-.5)<1e-12
    prior=Belief()
    ig_even=expected_information(prior,0,.3)
    ig_odd=expected_information(prior,1,.3)
    mixed=expected_pair_gain(prior,PROBE_BY_NAME["even_quiet"],
                             PROBE_BY_NAME["odd_quiet"],ENERGY_CAP)
    same=expected_pair_gain(prior,PROBE_BY_NAME["even_quiet"],
                            PROBE_BY_NAME["even_quiet"],ENERGY_CAP)
    assert ig_even<1e-12 and ig_odd<1e-12
    assert mixed>.1 and same<1e-10
    assert simulate(0,0,"anticipatory")["history"]==simulate(0,0,"matched_real")["history"]
    return {"closed_form_posterior_gap":"<1e-12","single_parity_50_observations_nonidentifying":True,
            "first_step_even_info":ig_even,"first_step_odd_info":ig_odd,
            "two_step_mixed_info_nats":mixed,"two_step_same_parity_info_nats":same,
            "matched_real_controller_ties":True}


POLICIES=("anticipatory","matched_real","greedy_stop","alternating_quiet",
          "alternating_focus","random","odd_only","even_only")


def summary(rows):
    return {
        "episodes":len(rows),
        "coverage":sum(r["confident"] for r in rows)/len(rows),
        "selective_accuracy":(
            sum(r["correct"] for r in rows)/sum(r["confident"] for r in rows)
            if any(r["confident"] for r in rows) else None),
        "correct_per_episode":sum(r["correct"] for r in rows)/len(rows),
        "false_confident":sum(r["confident"] and not r["correct"] for r in rows),
        "mean_energy":sum(r["spent"] for r in rows)/len(rows),
        "mean_observations":sum(r["n_observations"] for r in rows)/len(rows),
        "mean_brier":sum(r["brier"] for r in rows)/len(rows),
        "mean_log_loss":sum(r["log_loss"] for r in rows)/len(rows),
        "mean_information_nats":sum(r["entropy_gain_nats"] for r in rows)/len(rows),
        "mean_even":sum(r["even"] for r in rows)/len(rows),
        "mean_odd":sum(r["odd"] for r in rows)/len(rows)
    }


def run(seeds=32):
    result={}
    for policy in POLICIES:
        rows=[simulate(seed,twist,policy)
              for seed in range(seeds) for twist in (0,1)]
        result[policy]=summary(rows)
    # Same data and algorithm; geometric relabeling may not claim a benefit.
    assert result["anticipatory"]==result["matched_real"]
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.seeds<=128:
        parser.error("seed count must be 1-128")
    print("SELF",json.dumps(numerical_selfcheck(),sort_keys=True))
    for name,values in run(args.seeds).items():
        print("POLICY",json.dumps({"name":name,**values},sort_keys=True))


if __name__=="__main__":
    main()

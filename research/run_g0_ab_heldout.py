"""Reproduce G0-AB held-out tables and paired cluster bootstrap.

Run from repository root:
    python -m research.run_g0_ab_heldout --start 128 --count 64

Paired units are SEED clusters, each containing both simulated orientations.
The experimental model was corrected using seeds 0..15; 128..191 were
registered as held-out in research/G0_AB1_HELDOUT_AMENDMENT.md.
"""
import argparse
import json
import random
import time

from esmf.adaptive_ecology_ab import WORLDS,POLICIES,simulate,summary


def interval(values,rng,resamples=2000):
    n=len(values)
    sample_means=[]
    for _ in range(resamples):
        sample_means.append(
            sum(values[rng.randrange(n)] for j in range(n))/n)
    sample_means.sort()
    return sample_means[int(.025*resamples)],sample_means[int(.975*resamples)-1]


def evaluate(start=128,count=64,output_jsonl=None):
    if start<0 or count<1 or count>256:
        raise ValueError("start>=0 and 1<=count<=256")
    data={}
    summaries=[]
    lines=[]
    for world in WORLDS:
        for policy in POLICIES:
            begin=time.perf_counter()
            rows=[simulate(s,t,world,policy)
                  for s in range(start,start+count) for t in (0,1)]
            data[world,policy]=rows
            if output_jsonl:
                lines.extend(json.dumps(v,sort_keys=True) for v in rows)
            stats=summary(rows)
            stats["wall_s"]=time.perf_counter()-begin
            if policy!="equivalent_real":
                summaries.append(stats)
        # Plain real implementation is a required exact counterexample to
        # claims that the neutral-geometry notation alone gives performance.
        for a,b in zip(data[world,"adaptive"],data[world,"equivalent_real"]):
            aa={k:v for k,v in a.items() if k!="policy"}
            bb={k:v for k,v in b.items() if k!="policy"}
            if aa!=bb:
                raise AssertionError("geometric/real replay mismatch")
    if output_jsonl:
        with open(output_jsonl,"w",encoding="utf-8") as out:
            out.write("\n".join(lines)+"\n")
    rng=random.Random(51913)
    contrasts=[]
    for world in WORLDS:
        for policy in ("fixed_focus","fixed_quiet","adaptive_no_slow"):
            fields=("energy","bud_count") if world=="diffuse" else (
                "energy","bud_count","brier","correct")
            scores={}
            for field in fields:
                observations=[]
                for i in range(count):
                    row_a=data[world,"adaptive"][2*i:2*i+2]
                    row_b=data[world,policy][2*i:2*i+2]
                    observations.append(
                        sum(a[field]-b[field] for a,b in zip(row_a,row_b))/2)
                scores[field]={
                    "mean":sum(observations)/len(observations),
                    "seed_cluster_bootstrap95":interval(observations,rng)
                }
            contrasts.append({"world":world,"comparison":policy,
                              "adaptive_minus_comparison":scores})
    return summaries,contrasts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=int,default=128)
    p.add_argument("--count",type=int,default=64)
    p.add_argument("--output-jsonl",default=None)
    opts=p.parse_args()
    summaries,contrasts=evaluate(opts.start,opts.count,opts.output_jsonl)
    for row in summaries:
        print("MODEL",json.dumps(row,sort_keys=True))
    for row in contrasts:
        print("PAIRED",json.dumps(row,sort_keys=True))
    print("REPLAY",json.dumps({"start":opts.start,"count":opts.count,
                              "equivalent_real":True,"worlds":list(WORLDS)}))


if __name__=="__main__":
    main()

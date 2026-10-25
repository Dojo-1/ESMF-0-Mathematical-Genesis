"""Frozen held-out replay for G0-AC0 connection adaptation.
From root:
 python -m research.run_g0_ac0_heldout --start 100 --count 32
"""
import argparse
import json
import random
import time

from esmf.adaptive_connection_ac0 import WORLDS,MODEL_NAMES,simulate,summary,math_checks


def paired_ci(values,rng):
    means=[]
    n=len(values)
    for _ in range(2000):
        means.append(sum(values[rng.randrange(n)] for i in range(n))/n)
    means.sort()
    return (means[49],means[1949])


def evaluate(start=100,count=32,write_jsonl=None):
    if start<0 or not 1<=count<=128:
        raise ValueError("invalid range")
    math_result=math_checks()
    allrows={}
    statements=[]
    perseed=[]
    for world in WORLDS:
        for policy in MODEL_NAMES:
            began=time.perf_counter()
            rows=[simulate(seed,world,policy)
                  for seed in range(start,start+count)]
            stats=summary(rows)
            stats["wall_seconds"]=time.perf_counter()-began
            allrows[world,policy]=rows
            if write_jsonl is not None:
                perseed.extend(json.dumps(r,sort_keys=True) for r in rows)
            if policy!="equivalent_real":
                statements.append({"kind":"MODEL",**stats})
        for a,b in zip(allrows[world,"adaptive"],
                       allrows[world,"equivalent_real"]):
            if {k:v for k,v in a.items() if k!="policy"} != {
                k:v for k,v in b.items() if k!="policy"}:
                raise AssertionError("equivalent-real invariant broken")
    rng=random.Random(73159)
    for world in WORLDS:
        adaptive=allrows[world,"adaptive"]
        for comparator in ("constant_forgetting","no_forgetting",
                           "frozen_after_calibration"):
            controls=allrows[world,comparator]
            effects={}
            for metric in ("heldout_mse","mean_coeff_abs_error"):
                diffs=[x[metric]-y[metric]
                       for x,y in zip(adaptive,controls)]
                effects[metric]={"mean_adaptive_minus_"+comparator:
                                  sum(diffs)/len(diffs),
                                 "bootstrap95":paired_ci(diffs,rng)}
            statements.append({"kind":"PAIRED","world":world,
                               "comparator":comparator,"effects":effects})
    if write_jsonl is not None:
        with open(write_jsonl,"w",encoding="utf-8") as out:
            out.write("\n".join(perseed)+"\n")
    return math_result,statements


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=int,default=100)
    p.add_argument("--count",type=int,default=32)
    p.add_argument("--output-jsonl",default=None)
    args=p.parse_args()
    math_result,rows=evaluate(args.start,args.count,args.output_jsonl)
    print("GEOMETRY",json.dumps(math_result,sort_keys=True))
    for row in rows:
        print(row.pop("kind"),json.dumps(row,sort_keys=True))
    print("CONTROL",json.dumps({"real_baseline_ties":True,
                                 "start":args.start,"count":args.count}))


if __name__=="__main__":
    main()

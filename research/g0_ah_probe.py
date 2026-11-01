"""G0-AH D0: preregistered synthetic behavior, not a neurobiological result.

Four contexts x four question modes x two-bit cue conjunctions. All predictions
are committed before testing; all arms share public encounter observations.
Original G0-AG1 heldout seeds are untouched. Same resource usage ONLY for the
AH with/without glial consolidation pair, NOT versus the ordinary estimator.
"""
import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

from esmf.compartmental_ah import CompartmentalNeuriteEcology

CONTEXTS=range(4)
QUESTIONS=range(4)
CUES=((0,0),(0,1),(1,0),(1,1))
DEVELOPMENT=tuple(range(8300,8314))
SEED_HOLDOUT=tuple(range(18300,18428))
TRAIN_PER=6
TEST_PER=32


def evaluator_probability(c,q,cue):
    """Only the synthetic evaluator knows this, never a predictor."""
    match=(cue[0]^cue[1])==((c+q)%2)
    if (c+q)%4==3:
        return .55 if match else .45
    return .85 if match else .15


class OrdinaryCounts:
    def __init__(self, cue_conditioned=True):
        self.conditional=cue_conditioned
        self.data={}

    def key(self,c,q,cue):
        return (c,q,cue) if self.conditional else (c,q)

    def predict(self,c,q,cue):
        same,different=self.data.get(self.key(c,q,cue),(0,0))
        return (different+1)/(same+different+2)

    def observe(self,c,q,cue,changed):
        key=self.key(c,q,cue)
        same,diff=self.data.get(key,(0,0))
        self.data[key]=(min(6,same+int(not changed)),
                        min(6,diff+int(changed)))

    @property
    def slots(self):
        return len(CONTEXTS)*len(QUESTIONS)*(len(CUES) if self.conditional else 1)*2


def _schedule(seed):
    rng=random.Random(seed)
    tasks=[(c,q,cue) for c in CONTEXTS for q in QUESTIONS
           for cue in CUES for _ in range(TRAIN_PER)]
    rng.shuffle(tasks)
    return rng,tasks


def brier(p,y):
    return (p-int(y))**2


def one(seed):
    if type(seed) is not int or seed not in DEVELOPMENT:
        raise ValueError("D0 only: undeclared development seed")
    start=time.process_time()
    rng,tasks=_schedule(seed)
    model=CompartmentalNeuriteEcology()
    full=model.seed()
    no_glia=model.seed()
    conventional=OrdinaryCounts(True)
    pooled=OrdinaryCounts(False)
    observed=[]
    train_pred=[]
    for n,(c,q,cue) in enumerate(tasks,1):
        # Growth uses the publicly seen cue, never the observation label.
        if not any(b.cue==cue for b in full.cells[model.cell_id(c,q)].branches):
            full=model.grow(full,c,q,cue)
            no_glia=model.grow(no_glia,c,q,cue)
        pred=(model.predict(full,c,q,cue),
              model.predict(no_glia,c,q,cue),
              conventional.predict(c,q,cue),pooled.predict(c,q,cue))
        p=evaluator_probability(c,q,cue)
        outcome=rng.random()<p
        train_pred.append((c,q,cue,pred))
        observed.append((c,q,cue,outcome))
        full=model.observe(full,c,q,cue,outcome)
        no_glia=model.observe(no_glia,c,q,cue,outcome)
        conventional.observe(c,q,cue,outcome)
        pooled.observe(c,q,cue,outcome)
        if n%128==0:
            for territory in range(len(full.glia)):
                full=model.glial_pulse(full,territory)
                no_glia=model.glial_pulse(no_glia,territory)
            full=model.slow_step(full,True)
            no_glia=model.slow_step(no_glia,False)
    assert model.ledger(full)==model.ledger(no_glia)==(0,0)
    # Identical realized external events and identical proxy consumption.
    assert full.waste==no_glia.waste and sum(len(x.branches) for x in full.cells)==64
    predictors={
        'ah_glial': lambda c,q,x:model.predict(full,c,q,x),
        'ah_no_consolidation':lambda c,q,x:model.predict(no_glia,c,q,x),
        'ordinary_cue_table':conventional.predict,
        'question_pooled':pooled.predict,
    }
    committed={name:{(c,q,cue):func(c,q,cue) for c in CONTEXTS
                    for q in QUESTIONS for cue in CUES}
               for name,func in predictors.items()}
    # The evaluator draws future labels only after all predictions are fixed.
    test_rng=random.Random(seed+1000003)
    future={(c,q,cue):tuple(test_rng.random()<evaluator_probability(c,q,cue)
                for _ in range(TEST_PER))
             for c in CONTEXTS for q in QUESTIONS for cue in CUES}
    scores={name:statistics.mean(
                brier(preds[key],outcome)
                for key,ys in future.items() for outcome in ys)
            for name,preds in committed.items()}
    mode_scores={name:{f'{c}:{q}':statistics.mean(
                       brier(preds[(c,q,cue)],y)
                       for cue in CUES for y in future[c,q,cue])
                      for c in CONTEXTS for q in QUESTIONS}
                 for name,preds in committed.items()}
    result={
       'seed':seed,'scores':scores,'mode_scores':mode_scores,
       'all_predictors_committed_before_outcomes':True,
       'training_events':[{'context':c,'question':q,'cue':list(cue),'changed':y}
                          for c,q,cue,y in observed],
       'training_preobserved_predictions':[{'context':c,'question':q,'cue':list(cue),
                                    'probs':list(p)} for c,q,cue,p in train_pred],
       'committed_predictions':{name:{f'{c}:{q}:{cue[0]}{cue[1]}':v
                                 for (c,q,cue),v in preds.items()}
                                 for name,preds in committed.items()},
       'future_outcomes':{f'{c}:{q}:{cue[0]}{cue[1]}':list(ys)
                          for (c,q,cue),ys in future.items()},
       'full_diagnostics':model.diagnostics(full),
       'no_glia_diagnostics':model.diagnostics(no_glia),
       'glial_resource_parity':(full.waste==no_glia.waste and
                                 full.energy_in==no_glia.energy_in and
                                 full.material_in==no_glia.material_in),
       'ordinary_counter_slots':conventional.slots,
       'question_pooled_counter_slots':pooled.slots,
       'AH_count_slots':len(full.cells)*4*4,
       'process_seconds':time.process_time()-start,
    }
    return result


def descriptive(values):
    xs=list(values)
    mean=statistics.mean(xs)
    err=1.96*statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else 0
    return {'mean':mean,'normal_approx_95':[mean-err,mean+err],
            'n':len(xs),'confirmatory':False}


def run(seeds=DEVELOPMENT):
    if tuple(seeds)!=DEVELOPMENT:
        raise ValueError('Do not silently change registered development seeds')
    rows=[one(seed) for seed in seeds]
    arms=list(rows[0]['scores'])
    summary={arm:descriptive(r['scores'][arm] for r in rows) for arm in arms}
    paired={f'{arm}_minus_ordinary':descriptive(
            r['scores'][arm]-r['scores']['ordinary_cue_table'] for r in rows)
            for arm in ('ah_glial','ah_no_consolidation','question_pooled')}
    return {'status':'M: bounded synthetic development only',
            'source_sha':os.getenv('GITHUB_SHA'),'seeds':list(seeds),
            'cells':16,'branch_cap':4,'territories':4,
            'training_per_seed':len(CONTEXTS)*len(QUESTIONS)*len(CUES)*TRAIN_PER,
            'future_per_seed':len(CONTEXTS)*len(QUESTIONS)*len(CUES)*TEST_PER,
            'means':summary,'paired':paired,'raw':rows,
            'limits':['No real neuron, astrocyte, or energy unit',
             'AH + no-glia resources matched, tabular runtime/memory not matched',
             'No causal benefit claimed from quaternion arithmetic',
             'No synaptic inhibitory signals used in the offline task',
             'No unseen contexts and no endogenous question discovery',
             'G0-AG1 heldout seeds untouched'],
            'measured_coverage':len(rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('G0_AH_D0_SUMMARY='+json.dumps({k:v for k,v in report.items() if k!='raw'},
                                      sort_keys=True))

if __name__=='__main__':
    main()

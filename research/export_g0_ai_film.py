"""Export *observed* G0-AI morphogenesis and regulatory activity as a film.

The model selects contact opportunities stochastically using local histories;
none of the displayed contacts or unit roles is manually staged. The synthetic
world, supervision, and readout are separate from local developmental rules.
The site consumes source-bound snapshots; it never runs the Python model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from esmf.telos_substrate_ai import TelosSubstrate
from research.g0_ai_probe import BLOCK, STEPS, brier, generate

SCHEMA = 'esmf-g0-ai-telos-film/v1'
SEED = 93101  # already a D0 development seed, not held out
MODES = ('adaptive', 'fixed', 'shuffled')
STRIDE = 12


def canonical_bytes(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def fingerprint(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def snapshot(model, *, cue=None, label=None, p=None, phase=None,
             rolling_loss=None, new_events=(), new_remainders=()):
    s=model.s
    if s.ledger() != (0, 0) or len(s.units) != 20:
        raise ValueError('invalid model cannot be published as a trace')
    units=[{'xyz':[round(a,6) for a in u.xyz],
            'activity':round(u.activity,6),
            'regulome':round(u.regulome,6),
            'rate':round(u.rate,6),
            'rule_mix':[round(z,6) for z in u.rule_mix],
            'gain':round(u.gain,6),
            'atp':u.atp,'material':u.material}
           for u in s.units]
    contacts=[{'source':a,'target':b,'weight':round(e.weight,6),
               'age':e.age, 'fitness':round(e.fitness,6)}
              for (a,b),e in sorted(s.contacts.items())]
    if any(u['atp'] < 0 or u['material'] < 0 for u in units):
        raise ValueError('negative synthetic resource')
    return {'tick':s.tick,'posthoc_world_phase':phase,
            'observed_cue':list(cue) if cue is not None else None,
            'observed_label':label,'prelabel_prediction':p,
            'cumulative_brier_after_warmup':rolling_loss,
            'units':units,'contacts':contacts,
            'growth':s.growth_events,'pruning':s.pruning_events,
            'energy_spent':s.waste,
            'energy_residual':s.ledger()[0],
            'material_residual':s.ledger()[1],
            'recent_events':list(new_events),
            'recent_remainders':[{'kind':r.kind,'source':r.source,
                                  'target':r.target,'value':round(r.value,6),
                                  'tick':r.tick} for r in new_remainders]}


def export(source_commit):
    if not isinstance(source_commit,str) or not source_commit or len(source_commit)>80:
        raise ValueError('invalid provenance')
    models={name:TelosSubstrate(seed=SEED+197,mode=name) for name in MODES}
    streams={name:[] for name in MODES}
    cursors={name:(0,0) for name in MODES}
    sums={name:0. for name in MODES}
    counts={name:0 for name in MODES}
    for name,model in models.items():
        snap=snapshot(model)
        streams[name].append({'index':0,'state':snap,'state_sha256':fingerprint(snap)})
    for t,cue,label,phase,_pworld in generate(SEED):
        pred={name:model.predict(cue) for name,model in models.items()}
        for name,model in models.items():
            model.observe(label)
            if t>=20:
                sums[name]+=brier(pred[name],label)
                counts[name]+=1
        if ((t+1)%STRIDE==0 or t in (0,239,240,479,480,STEPS-1)):
            for name,model in models.items():
                ec,rc=cursors[name]
                snap=snapshot(model,cue=cue,label=label,p=pred[name],phase=phase,
                      rolling_loss=sums[name]/counts[name] if counts[name] else None,
                      new_events=model.s.events[ec:],
                      new_remainders=model.s.remainders[rc:])
                streams[name].append({'index':len(streams[name]),'state':snap,
                                     'state_sha256':fingerprint(snap)})
                cursors[name]=(len(model.s.events),len(model.s.remainders))
    if not all(len(x)==len(streams[MODES[0]]) for x in streams.values()):
        raise AssertionError('unpaired film snapshots')
    result={'schema':SCHEMA,'source_commit':source_commit,
       'source':'esmf/telos_substrate_ai.py',
       'exporter':'research/export_g0_ai_film.py',
       'd0_results':'research/G0_AI_D0_RESULTS.md',
       'evidence':'M: recorded synthetic developmental dynamics, not intelligence',
       'seed':SEED,'world_steps':STEPS,'snapshot_stride':STRIDE,
       'world_change_points':[BLOCK,BLOCK*2],
       'limitations':[
         'Nodes began without predefined semantic organs or biological cell classes.',
         'Local rules have a finite, human-chosen primitive repertoire; this is not unrestricted new law creation.',
         'Model growth is stochastic and endogenous, not a deliberately selected reachability path.',
         'The external world, observed labels and hidden phase are task/evaluator properties, not cell metadata.',
         'World phase is displayed only retrospectively after events; local cells never receive it.',
         'The 3D chart is a geometric contact constraint, not neural anatomy.',
         'Growth, regulatory variety and conserved resources did NOT improve the measured task.',
         'Snapshot playback interval is a UI setting and is not a physical clock or exact stochastic waiting time.'
       ],
       'runs':[{'mode':name,'title':{'adaptive':'Telos-nudged local rewrite',
                                   'fixed':'Same tissue, frozen local laws and contacts',
                                   'shuffled':'Local rewrite, shuffled contact proposal'}[name],
                'frames':streams[name]} for name in MODES]}
    result['payload_sha256']=fingerprint(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-sha',default=os.environ.get('GITHUB_SHA','unattributed-local'))
    args=parser.parse_args()
    d=export(args.source_sha)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print('G0_AI_FILM='+json.dumps({'source':d['source_commit'],'sha':d['payload_sha256'],
                                   'frames':[(x['mode'],len(x['frames'])) for x in d['runs']]}))

if __name__=='__main__':
    main()

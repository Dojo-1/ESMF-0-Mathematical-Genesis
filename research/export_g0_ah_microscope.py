"""Deterministic G0-AH microscope trace: selected legal operators, NOT samples.

No biological advantage, actual spiking, or cognition is being simulated.
The 3D chart is a numeric substrate. Territorially aggregated glia have no
biological or physical 3D location in this reference model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from esmf.compartmental_ah import CompartmentalNeuriteEcology, qerror

SCHEMA='esmf-g0-ah-compartmental-microscope/v1'


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def snapshot(model,s):
    if not model.valid(s) or model.ledger(s)!=(0,0):
        raise ValueError('invalid synthetic state cannot be presented as valid')
    cells=[]
    for c in s.cells:
        branches=[]
        for b in c.branches:
            parent=c.root_phase if b.parent==-1 else c.branches[b.parent].phase
            branches.append({'id':b.ident,'parent':b.parent,'voxel':list(b.voxel),
                'cue':list(b.cue),'fast':[b.fast_same,b.fast_changed],
                'long':[b.long_same,b.long_changed], 'eligibility':b.eligibility,
                'drive':b.drive,'phase':list(b.phase),
                'transport_defect':qerror(parent,b.phase,b.forward),
                'predicted_change':b.prediction()})
        cells.append({'id':c.ident,'context':c.context,'question':c.question,
            'soma':list(c.soma),'atp':c.atp,'material':c.material,
            'branches':branches})
    return {'cells':cells,'territories':[{'id':g.ident,'members':list(g.members),
        'support':g.support,'calcium':g.calcium} for g in s.glia],
        'synapses':[{'source':x.source,'target':x.target,'branch':x.branch,
            'kind':x.kind,'delay':x.delay} for x in s.synapses],
        'pending':[{'source':x.source,'target':x.target,'branch':x.branch,
            'amount':x.amount,'arrival':x.arrival} for x in s.pending],
        'fast_tick':s.tick,'slow_tick':s.slow_tick,'waste':s.waste,
        'remainders':[{'kind':r.kind,'cell':r.cell,'branch':r.branch,'reason':r.reason}
                      for r in s.remainders[-12:]],
        'remainder_count':len(s.remainders),
        'energy_residual':model.ledger(s)[0],
        'material_residual':model.ledger(s)[1]}


def capture(frames,model,s,event,classification):
    snap=snapshot(model,s)
    frames.append({'index':len(frames),'event':event,
        'selection':classification,'state':snap,'sha256':digest(snap)})


def focus_run(model):
    s=model.seed()
    frames=[]
    capture(frames,model,s,'initial','seed')
    actions=(
      ('paid growth · cue 10',lambda t:model.grow(t,0,0,(1,0))),
      ('witness change · cue 10',lambda t:model.observe(t,0,0,(1,0),True)),
      ('paid child growth · cue 01',lambda t:model.grow(t,0,0,(0,1))),
      ('witness unchanged · cue 01',lambda t:model.observe(t,0,0,(0,1),False)),
      ('glial calcium pulse · territory 0',lambda t:model.glial_pulse(t,0)),
      ('slow contextual consolidation',lambda t:model.slow_step(t)),
      ('second child branch · cue 00',lambda t:model.grow(t,0,0,(0,0))),
      ('grandchild branch · cue 11',lambda t:model.grow(t,0,0,(1,1))),
      ('witness change · cue 11',lambda t:model.observe(t,0,0,(1,1),True)),
      ('paid leaf withdrawal + typed remainder',lambda t:model.prune(t,0,0,3)),
      ('regrow from material · cue 11',lambda t:model.grow(t,0,0,(1,1))),
      ('bind delayed inhibitory axon · cell 1 to 0',lambda t:model.connect(t,1,0,3,'inhibit',2)),
      ('axon emits paid inhibitory packet',lambda t:model.emit(t,1)),
      ('fast clock · packet in flight',lambda t:model.fast_step(t)),
      ('fast clock · dendritic inhibition',lambda t:model.fast_step(t)),
      ('fast clock · drive dissipates',lambda t:model.fast_step(t)),
    )
    for name,action in actions:
        s=action(s)
        capture(frames,model,s,name,'deliberately_selected_operator')
    return {'id':'one-neurite','title':'Enter one neurite · growth, memory, glia, inhibition',
            'frames':frames}


def population_run(model):
    s=model.seed()
    frames=[]
    capture(frames,model,s,'initial','seed')
    for c in range(model.contexts):
        for q in range(model.questions):
            for cue in ((0,0),(0,1),(1,0),(1,1)):
                s=model.grow(s,c,q,cue)
            capture(frames,model,s,f'four paid branches · soma {model.cell_id(c,q)}',
                    'deliberately_selected_operator')
    return {'id':'population','title':'16-cell morphology · 64 paid branches',
            'frames':frames}


def build(source_sha):
    if not isinstance(source_sha,str) or not source_sha or len(source_sha)>80:
        raise ValueError('invalid source provenance')
    model=CompartmentalNeuriteEcology()
    d={'schema':SCHEMA,'source_commit':source_sha,
       'source':'esmf/compartmental_ah.py',
       'exporter':'research/export_g0_ah_microscope.py',
       'side':model.side,'cell_capacity':model.n,'branch_cap':model.branch_cap,
       'authority':'M: actual code-generated synthetic operator states only',
       'limitations':['All displayed transitions were deliberately selected, not drawn from a stochastic generator.',
          'Fixed context and question IDs are provided rather than discovered.',
          '3D voxels are numerical substrates; glial territory has no physical position.',
          'Quaternion phases are diagnostics, not a biological orientation or proved semantic representation.',
          'Proxy ATP/material and delayed packets are not electrophysiology.',
          'The original G0-AH D0 forecasting test contains no axonal inhibition.',
          'No cognitive, biological or hypercomplex advantage established.'],
       'runs':[focus_run(model),population_run(model)]}
    d['payload_sha256']=digest(d)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--source-sha',default=os.environ.get('GITHUB_SHA','unattributed-local'))
    args=p.parse_args()
    d=build(args.source_sha)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('G0_AH_MICROSCOPE='+json.dumps({'sha':d['payload_sha256'],
        'runs':[(r['id'],len(r['frames'])) for r in d['runs']]}))

if __name__=='__main__':
    main()

"""From-scratch synthetic baseline vs physical recurrent residual FM, full120h.

No ERA5 skill claim. Both A initializations use the SAME seed/archive/dimensions;
the deterministic A weights must be identical. B/C optimize different targets.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from smoke_moe import synthetic_archive
from climate_diffusion.train_manifold_moe import train_manifold_moe
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.manifold_diagnostics import diagnose_manifold
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.time_alignment import forecast_from_checkpoint
from climate_diffusion.trajectory_output import export_trajectories
from climate_diffusion.train import _sha256


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='outputs/recurrent-smoke-001')
    p.add_argument('--report-dir',default='outputs/recurrent-smoke-report-001')
    p.add_argument('--manifold-epochs',type=int,default=12)
    p.add_argument('--expert-epochs',type=int,default=5)
    p.add_argument('--joint-epochs',type=int,default=3)
    args=p.parse_args(argv)
    torch.set_num_threads(1)
    start=time.perf_counter()
    work,report=Path(args.work_dir),Path(args.report_dir)
    work.mkdir(parents=True,exist_ok=False); report.mkdir(parents=True,exist_ok=False)
    archive,_=synthetic_archive(work,count=480)
    common=dict(batch_size=4,window_stride=16,ensemble_size=3,integration_steps=2,sampled_leads=2,
                max_validation_windows=2,learning_rate=1e-3,weight_decay=1e-4,seed=7,device='cpu')
    losses=dict(delta_weight=.02,delta_member_weight=.001,trajectory_weight=.1,trajectory_edges=0,
                validation_trajectory_edges=0,wind_speed_weight=.01,wind_direction_weight=.005,
                temporal_warmup_epochs=3,log_gradient_norms=True)
    models={}; summaries={}
    for label,dynamics in [('baseline','lead_conditioned'),('recurrent','recurrent_residual')]:
        a=train_manifold_moe(archive,work/f'{label}-a.pt',stage='manifold',
            manifold_epochs=args.manifold_epochs,model_options=dict(history_steps=6,history_stride=1,
            horizon_steps=20,num_experts=3,manifold_dim=4,hidden_dim=32,context_dim=16,
            expert_latent_dim=64,gate_hidden_dim=160,forecast_dynamics=dynamics),**common)
        b=train_manifold_moe(archive,work/f'{label}-b.pt',stage='specialize',init_checkpoint=a,
                            expert_epochs=args.expert_epochs,**common,**losses)
        c=train_manifold_moe(archive,work/f'{label}-c.pt',stage='joint',init_checkpoint=b,
                            joint_epochs=args.joint_epochs,**common,**losses)
        models[label]=c
        pa,pb=[torch.load(x,weights_only=False) for x in (a,b)]
        frozen=[k for k in pa['model'] if k.startswith(('manifold.','physics.','reference_encoder.','latent_','gate.centers','gate.radius'))]
        assert all(torch.equal(pa['model'][k],pb['model'][k]) for k in frozen)
        rows=[]
        for path in (a,b,c): rows+=json.loads(path.with_suffix('.metrics.json').read_text())
        write(report/f'training-{label}.json',rows)
        assert all(r['train'].get('trajectory_edges',20)==20 for r in rows)
        summaries[label]={}
        for phase,path in [('a',a),('b',b),('c',c)]:
            payload=torch.load(path,weights_only=False)
            summaries[label][phase]={'sha256':_sha256(path),'best_epoch':payload['training']['best_epoch']}
    pa=torch.load(work/'baseline-a.pt',weights_only=False); qa=torch.load(work/'recurrent-a.pt',weights_only=False)
    assert all(torch.equal(v,qa['model'][k]) for k,v in pa['model'].items())
    # Same member count, ODE steps, seed, validation windows for every comparison.
    comparisons={}
    for name,path,mode in [('baseline',models['baseline'],None),('recurrent',models['recurrent'],None),
                           ('drift_only',models['recurrent'],'drift_only')]:
        saved=evaluate_flow_checkpoint(path,archive,report/f'validation-{name}.json',split_name='validation',
            ensemble_size=3,integration_steps=2,max_cases=2,seed=83,device='cpu',moe_mode=mode)
        comparisons[name]=json.loads(saved.read_text())
    diagnose_manifold(models['recurrent'],archive,report/'recurrent-diagnostics.json',max_cases=2,
                      members=3,integration_steps=2,split_name='validation')
    f=LatentFlowForecaster(models['recurrent'],device='cpu'); _,times,_=load_moe_archive(archive)
    origin=times[f.training_metadata['split']['validation'][0]+f.config.history_span_steps-1]
    forecast=forecast_from_checkpoint(models['recurrent'],archive,str(origin),work/'forecast-6h.npz',
        ensemble_size=3,integration_steps=2,forecast_steps=20,seed=83,device='cpu')
    shutil.copy2(forecast,report/'forecast-6h.npz')
    export_trajectories(forecast,archive,report/'members-6h',horizon_hours=120,interval_hours=6,
                       reference_label='Synthetic truth (NOT ERA5)',diagnostic_views=True)
    export_trajectories(forecast,archive,report/'members-12h',horizon_hours=120,interval_hours=12,
                       extension='gif',reference_label='Synthetic truth (NOT ERA5)')
    summary={'kind':'synthetic from-scratch A/B/C, not ERA5','same_A_weights':True,
        'B_frozen_exact':True,'all_training_trajectory_edges':20,'horizon_hours':120,
        'member_count':3,'integration_steps_tau':2,'physical_step_hours':6,'test_used':False,
        'seed':7,'evaluation_seed':83,'parameter_count':sum(p.numel() for p in f.model.parameters()),
        'checkpoints':summaries,'archive_sha256':_sha256(archive),
        'comparison_contract':'Both baseline/new have full-window probabilistic+mean/member-delta/wind losses; only forecast/FM target semantics differ. Drift-only uses recurrent C weights, not independently tuned.',
        'validation':{k:{'state':v['normalized_overall'],'temporal':v['temporal_overall']} for k,v in comparisons.items()},
        'elapsed_seconds':time.perf_counter()-start,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'limits':['One synthetic seed, two held-out windows, tiny grid/budget; not calibrated ERA5 skill',
                  'Persistent residual noise alone does not guarantee correct temporal law',
                  'Recurrent Euler errors may accumulate; no stability guarantee',
                  'Actual ERA5 archive/checkpoints and RunPod unavailable; not retrained here']}
    write(report/'summary.json',summary)
    from visualize_recurrent_flow import visualize
    visualize(report)
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())

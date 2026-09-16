"""Evaluate a held-out split and save ONE native member forecast for reuse."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .train_information_process import load_checkpoint,data_contract,Windows,write_json
from .physical_information import digest

def evaluate(checkpoint,archive,output,*,information=None,split='validation',members=4,tau_steps=4,
             max_cases=4,seed=83,device='cpu',forecast_output=None,drift_only=False):
    output=Path(output)
    if max_cases<0:raise ValueError('max_cases cannot be negative')
    if forecast_output and Path(forecast_output).suffix!='.npz':raise ValueError('Forecast output must end in .npz')
    if output.exists() or (forecast_output and Path(forecast_output).exists()):raise FileExistsError('Choose new report/forecast paths')
    if split not in ('validation','expert_validation','test'):raise ValueError('Evaluation must use a held-out split')
    model,p=load_checkpoint(checkpoint,device);model.eval()
    d=data_contract(archive,information,p['mode'],model.config,p)
    starts=d['split'][split][:max_cases] if max_cases else d['split'][split]
    ds=Windows(d['states'],d['times'],model.config,starts,d['mean'],d['scale'],d['schema'],information=d['information'])
    rows=[]
    with torch.no_grad():
        for index in range(len(ds)):
            batch={k:v[None].to(device) for k,v in ds[index].items()}
            truth=torch.cat((batch['origin'][:,None],batch['targets']),1)
            trace=[]
            samples,qs=model.rollout(batch['history'],batch.get('information'),members=members,tau_steps=tau_steps,
                generator=torch.Generator(device=device).manual_seed(seed+index),auxiliary=model.phase=='A',
                drift_only=drift_only,trace=trace,return_q=True)
            scores=model.scores(samples,truth,batch['dt_hours'],batch['pair_observed_mask'])
            row={k:float(v) for k,v in scores.items()}
            persistence=batch['origin'][:,None,None].expand_as(samples)
            row['persistence_rmse']=float(model.scores(persistence,truth,batch['dt_hours'],batch['pair_observed_mask'])['rmse'])
            row['origin_time']=str(np.datetime64(int(batch['origin_time_ns'][0].cpu()),'ns'))+'Z'
            row['physical_lead_hours']=(np.arange(1,21)*6).tolist()
            row['generated_routing_by_lead']=[t['routing'] for t in trace if 'routing' in t]
            row['q_per_day_rms_by_lead']={k:[float(t[k].square().mean().sqrt()) for t in trace]
                for k in ('drift_per_day','residual_per_day','final_per_day')}
            dist=(qs[:,:,1:,None,:]-model.core.gate.centers[None,None,None]).square().mean(-1).min(-1).values
            row['chart_distance_radius_mean']=float((dist/model.core.gate.radius_squared).sqrt().mean())
            if batch.get('information') is not None:
                row.update({k:float(v) for k,v in model.information_scores(qs,batch['information'],batch['information_targets'],
                    batch['dt_hours'],torch.tensor(p['information_scale'],device=device),
                    torch.tensor(p['information_tendency_scale'],device=device)).items()})
            rows.append(row)
            if index==0 and forecast_output:
                path=Path(forecast_output);path.parent.mkdir(parents=True,exist_ok=True)
                origin=np.datetime64(int(batch['origin_time_ns'][0].cpu()),'ns');leads=np.arange(1,21)*6
                physical=(samples[0,:,1:]*model.temporal.scale+model.temporal.mean).cpu().numpy()
                np.savez_compressed(path,predictions=physical,origin_time=origin,last_history_time=origin,
                    lead_hours=leads,valid_times=origin+leads.astype('timedelta64[h]'),forecast_step_hours=6,
                    schema_json=json.dumps(p['schema']),temporal_statistics_json=json.dumps(p['statistics']),
                    sampling_contract=p['sampling_contract'],checkpoint_sha256=digest(checkpoint))
    common=[k for k,v in rows[0].items() if isinstance(v,(float,int))]
    report={'format':'climate_diffusion.information_evaluation.v1','checkpoint_sha256':digest(checkpoint),
        'stage':p['stage'],'split':split,'case_count':len(rows),'members':members,'tau_steps':tau_steps,'seed':seed,
        'drift_only':drift_only,'archive_sha256':p['archive_sha256'],
        'aggregate':{k:float(np.mean([r[k] for r in rows])) for k in common},'cases':rows,
        'note':'same member endpoints; scores use physical tendency scaling; no claim of Markov state or calibrated skill'}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report);return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','archive','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--information');p.add_argument('--forecast-output');p.add_argument('--drift-only',action='store_true')
    p.add_argument('--split',choices=['validation','expert_validation','test'],default='validation')
    for k,v in dict(members=4,tau_steps=4,max_cases=4,seed=83).items():p.add_argument('--'+k.replace('_','-'),type=int,default=v)
    p.add_argument('--device',default='cpu');args=vars(p.parse_args(argv));report=evaluate(**args);print(json.dumps(report['aggregate']));return 0
if __name__=='__main__':main()

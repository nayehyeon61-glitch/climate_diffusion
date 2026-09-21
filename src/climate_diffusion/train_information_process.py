"""Separate A -> frozen-A B -> small-LR C. Versioned, no legacy reinterpretation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .information_process import InformationProcess, FORMAT, curriculum, gradient_diagnostics
from .physical_information import load_information, fit_information, digest, information_digest
from .manifold_moe import ManifoldMoEConfig
from .moe_data import load_moe_archive, field_grid, build_moe_split, validate_moe_split
from .temporal_supervision import TemporalWindowDataset, NonSingletonBatchSampler, fit_temporal_statistics, area_weights
from .moe import ensemble_scores

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class Windows(TemporalWindowDataset):
    def __init__(self,*args,information=None,**kwargs):
        super().__init__(*args,**kwargs);self.information=information
    def __getitem__(self,index):
        row=super().__getitem__(index)
        if self.information is not None:
            origin=self.starts[index]+self.config.history_span_steps-1
            row['information']=torch.from_numpy(self.information[origin].copy())
            row['information_targets']=torch.from_numpy(self.information[origin+1:origin+self.config.horizon_steps+1].copy())
        return row

def load_checkpoint(path,device='cpu'):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('format')!=FORMAT:raise ValueError('Not a separate-A information checkpoint; use the legacy trainer for old formats')
    manifest=Path(path).with_suffix('.manifest.json')
    if not manifest.exists() or json.loads(manifest.read_text())['checkpoint_sha256']!=digest(path):
        raise ValueError('Checkpoint manifest/hash mismatch')
    model=InformationProcess(ManifoldMoEConfig(**p['config']),p['schema'],p['mean'],p['scale'],p['statistics'],p['information_metadata'])
    model.load_state_dict(p['model']);model.set_phase(p['stage']);model.to(device)
    return model,p

def data_contract(archive,information_path,mode,config,parent=None):
    states,times,schema=load_moe_archive(archive)
    if [v['name'] for v in schema['variables']]!=['msl','t2m','u10','v10'] or schema['forecast_step_hours']!=6:
        raise ValueError('This profile requires canonical msl/t2m/u10/v10 and exact6h')
    info=meta=None
    if mode=='enriched':
        if not information_path:raise ValueError('Enriched mode requires --information; missing fields cannot be fabricated')
        info,meta=load_information(information_path,archive,times,schema)
    elif information_path:raise ValueError('Surface mode must not receive enriched information')
    if parent:
        if parent['archive_sha256']!=digest(archive) or parent['information_sha256']!=(information_digest(information_path) if information_path else None):
            raise ValueError('Parent archive/information hash changed')
        if hasattr(info,'pin'):info.pin(parent.get('information_shards'))
        split=parent['split'];mean=np.asarray(parent['mean']);scale=np.asarray(parent['scale'])
        stats=parent['statistics'];im=parent['information_mean'];isc=parent['information_scale'];its=parent['information_tendency_scale']
    else:
        count=len(states)-config.history_span_steps-config.horizon_steps+1
        split=build_moe_split(count,config.horizon_steps)
        end=split['train'][-1]+config.history_span_steps+config.horizon_steps
        mean=states[:end].mean(0);scale=states[:end].std(0);scale=np.where(scale>1e-6,scale,1).astype(np.float32)
        stats=fit_temporal_statistics(states,times,schema,scale,end)
        im=isc=its=None
        if hasattr(info,'statistics'):
            im,isc,its=info.statistics(end)
        elif info is not None:
            im,isc=fit_information(info,meta,end,schema)
            sh=meta['shape'];d=np.diff(info[:end],axis=0).reshape(-1,*sh)/6
            w=area_weights(schema)
            avg=(d*w).sum((-2,-1)).mean(0)
            var=((d-avg[None,:,None,None])**2*w).sum((-2,-1)).mean(0)
            channel=np.maximum(np.sqrt(var),np.maximum(isc.reshape(sh).mean((-2,-1))*1e-3/6,1e-8))
            its=np.broadcast_to(channel[:,None,None],sh).copy().reshape(-1).astype(np.float32)
    validate_moe_split(split,config.horizon_steps,len(states)-config.history_span_steps-config.horizon_steps+1)
    end=split['train'][-1]+config.history_span_steps+config.horizon_steps
    normalized=(info.normalized(im,isc) if hasattr(info,'normalized') else
                None if info is None else ((info-np.asarray(im))/np.asarray(isc)).astype(np.float32))
    data=dict(states=states,times=times,schema=schema,split=split,mean=mean,scale=scale,statistics=stats,
        information_metadata=meta,information_mean=im,information_scale=isc,information_tendency_scale=its,
        information=normalized,train_end=end)
    return data

def batch_loss(model,batch,args,epoch,streams):
    info=batch.get('information');truth=torch.cat((batch['origin'][:,None],batch['targets']),1)
    aux=model.phase=='A'
    generated,qs=model.rollout(batch['history'],info,members=args.members,tau_steps=args.tau_steps,
        generator=streams['ensemble'],auxiliary=aux,return_q=True)
    metrics=model.scores(generated,truth,batch['dt_hours'],batch['pair_observed_mask'])
    teacher=model.teacher_loss(batch,info,streams['fm']);metrics.update(teacher)
    if model.phase=='A':
        metrics.update(model.geometry_losses(batch,info))
        phase,weights=curriculum(epoch,args.curriculum_interval)
        if args.profile!='process':
            phase=2 if args.profile=='dynamics' else 1
            _,weights=curriculum(3 if phase==2 else 1,2)
            if args.profile=='information':weights.update(ae_delta=.05,decoded_drift=.05,latent_dynamics=.1,information_geometry=.02)
            for key in ('fm','state_crps','transition_crps','loss_delta','loss_trajectory','info_distribution'):weights[key]=0.
        if args.loss_weights:
            # Overrides set plateau strength, not the activation epoch.
            for key,value in json.loads(args.loss_weights).items():
                if weights[key]>0:weights[key]=float(value)
        if info is not None:
            metrics.update(model.information_scores(qs,info,batch['information_targets'],batch['dt_hours'],
                torch.as_tensor(args.info_scale,device=truth.device),torch.as_tensor(args.info_tendency_scale,device=truth.device)))
        metrics['curriculum_phase']=generated.new_tensor(phase)
        total=generated.sum()*0
        for key,weight in weights.items():
            if key in metrics:metrics['weighted_'+key]=metrics[key]*weight;total=total+metrics['weighted_'+key]
        # Fixed validation selection below is independent of curriculum weights.
    else:
        ramp=min(1.,epoch/(5 if model.phase=='B' else 3))
        weights={'loss_delta':.02,'loss_trajectory':.1,'loss_delta_member':args.b_member_weight if model.phase=='B' else 0.,
                 'loss_wind_speed':.01,'loss_wind_direction':.005}
        metrics['weighted_specialization']=teacher['loss']
        total=metrics['weighted_specialization']
        for k,w in weights.items():metrics['weighted_'+k]=metrics[k]*w*ramp;total=total+metrics['weighted_'+k]
        if model.phase=='C':
            metrics.update(model.geometry_losses(batch,info))
            samples=generated[:,:,1:].permute(0,2,1,3).reshape(-1,args.members,model.config.state_dim)
            marginal=ensemble_scores(samples,truth[:,1:].reshape(-1,model.config.state_dim))
            metrics.update(marginal)
            metrics['anchor']=(model.encode(batch['origin'],info)-model.encode(batch['origin'],info,True).detach()).square().mean()
            pi=metrics['reconstruction']+.1*metrics['physics']+.05*metrics['invariant']+.1*metrics['metric']+.1*metrics['latent_dynamics']
            metrics.update(pi=pi,weighted_marginal_energy=.5*marginal['energy'],
                weighted_marginal_crps=.5*marginal['crps'],weighted_pi=.5*pi,
                weighted_anchor=metrics['anchor'])
            # Keep the existing C expression/order: this is logging, not Loss V2.
            total=total+.5*(marginal['energy']+marginal['crps'])+.5*pi+metrics['anchor']
    metrics['loss']=total
    metrics['selection']=metrics['state_crps']+metrics['transition_crps']+.1*metrics['loss_trajectory']+.1*metrics['mean_state']
    if aux:metrics['selection']=metrics['selection']+metrics['reconstruction']+.05*(metrics['ae_delta']+metrics['decoded_drift'])
    for k,v in metrics.items():
        if not bool(torch.isfinite(v)):raise FloatingPointError(f'Nonfinite {k}')
    return metrics

def train(args):
    output=Path(args.output)
    if output.suffix!='.pt':raise ValueError('Checkpoint output must end in .pt')
    if any(output.with_suffix(s).exists() for s in ('.pt','.metrics.json','.metadata.json','.manifest.json')):raise FileExistsError('Choose a new output checkpoint')
    if args.batch_size<2 or args.members<2 or args.epochs<1:raise ValueError('Require batch>=2, members>=2, epochs>=1')
    if min(args.tau_steps,args.window_stride,args.curriculum_interval)<1 or min(args.max_windows,args.patience)<0:
        raise ValueError('Invalid sampling/curriculum counts')
    if args.max_windows==1:
        raise ValueError('max_windows must be 0 (all) or >=2 for the manifold metric')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.weight_decay) or args.weight_decay<0:
        raise ValueError('Invalid optimizer parameters')
    if not math.isfinite(args.b_member_weight) or args.b_member_weight<0:raise ValueError('Invalid member weight')
    if args.a_quality_max is not None and (not math.isfinite(args.a_quality_max) or args.a_quality_max<=0):
        raise ValueError('Quality threshold must be finite and positive')
    if args.loss_weights:
        overrides=json.loads(args.loss_weights)
        if args.stage!='A' or not isinstance(overrides,dict) or set(overrides)-set(curriculum(999)[1]):
            raise ValueError('Loss overrides are named A curriculum weights only')
        if any(not math.isfinite(float(v)) or float(v)<0 for v in overrides.values()):raise ValueError('Invalid A loss weight')
    if args.stage=='A' and args.profile=='process' and args.epochs<5*args.curriculum_interval+1:
        raise ValueError('Process A must reach phase6 before best/seal; require >=5*interval+1 epochs')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=args.device;parent=None
    if args.stage!='A':
        if not args.init:raise ValueError('B/C requires --init from previous stage')
        model,parent=load_checkpoint(args.init,device)
        if not bool(model.core.manifold_ready):raise ValueError('Parent A must have a sealed manifold')
        if parent['stage']!=('A' if args.stage=='B' else 'B'):raise ValueError('Only A -> B -> C transitions; no same-stage optimizer resume')
        config=model.config
        if parent['mode']!=args.mode:raise ValueError('Parent conditioning mode mismatch')
    else:
        if args.init:raise ValueError('A initializes from scratch; legacy/init mixing is not supported')
        states,_,schema=load_moe_archive(args.archive)
        config=ManifoldMoEConfig(state_dim=states.shape[1],grid=field_grid(schema),horizon_steps=20,step_hours=6,
            history_steps=args.history_steps,history_stride=args.history_stride,manifold_dim=args.manifold_dim,
            hidden_dim=args.hidden_dim,context_dim=args.context_dim,num_experts=args.experts,
            expert_latent_dim=args.expert_latent_dim,gate_hidden_dim=args.gate_hidden_dim,forecast_dynamics='recurrent_residual')
    d=data_contract(args.archive,args.information,args.mode,config,parent)
    if parent is None:
        model=InformationProcess(config,d['schema'],d['mean'],d['scale'],d['statistics'],d['information_metadata']).to(device)
        model.core.physics.fit(torch.as_tensor((d['states'][:d['train_end']]-d['mean'])/d['scale'],device=device))
    model.set_phase(args.stage)
    args.info_scale=d['information_scale'];args.info_tendency_scale=d['information_tendency_scale']
    params_a=[p for name,p in model.named_parameters() if p.requires_grad and (name.startswith('core.manifold.') or name.startswith('information.'))]
    ids={id(p) for p in params_a};params_b=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    lr=args.learning_rate*(.1 if args.stage=='C' else 1.)
    groups=[{'params':ps,'lr':rate,'name':name} for ps,rate,name in
        ((params_a,lr*(.1 if args.stage=='C' else 1.),'representation'),(params_b,lr,'process')) if ps]
    opt=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items()
            if args.stage=='B' and not n.startswith(('core.experts.','core.gate.correction.','core.history_encoder.'))}
    def loader(name,shuffle):
        starts=d['split'][name][::args.window_stride]
        if args.max_windows:starts=starts[:args.max_windows]
        if len(starts)<2:
            raise ValueError(f'{name}: fewer than two selected windows; reduce window_stride or increase max_windows')
        ds=Windows(d['states'],d['times'],config,starts,d['mean'],d['scale'],d['schema'],information=d['information'])
        batches=NonSingletonBatchSampler(len(ds),args.batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(args.seed))
        return DataLoader(ds,batch_sampler=batches)
    train_name='calibration' if args.stage=='C' else 'train';val_name='validation' if args.stage=='C' else 'expert_validation'
    loaders=[loader(train_name,True),loader(val_name,False)]
    rows=[];best=float('inf');best_state=None;best_epoch=0;start=time.perf_counter()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,args.epochs+1):
        record={'epoch':epoch}
        for is_train,dl in zip((True,False),loaders):
            model.train(is_train);sums={};count=0
            streams={k:torch.Generator(device=device).manual_seed(args.seed+(epoch*1000 if is_train else 900000)+offset)
                     for k,offset in (('fm',11),('ensemble',29))}
            with torch.set_grad_enabled(is_train):
                for index,batch in enumerate(dl):
                    batch={k:v.to(device) for k,v in batch.items()}
                    values=batch_loss(model,batch,args,epoch,streams)
                    if is_train:
                        if args.gradient_audit and index==0:
                            modules={'encoder':model.core.manifold.encoder,'decoder':model.core.manifold.decoder,
                                     'drift':model.core.manifold.latent_drift,'a_sampler':model.a_sampler,
                                     'information':model.information,'experts':model.core.experts,'gate':model.core.gate,
                                     'history':model.core.history_encoder}
                            selected={k:values[k] for k in ('reconstruction','ae_delta','decoded_drift','static_l2','information_geometry',
                                'fm','state_crps','transition_crps','loss_trajectory') if k in values}
                            selected.update({k:v for k,v in values.items() if k.startswith('weighted_')})
                            record['gradient_first_batch']=gradient_diagnostics(selected,{k:list(m.parameters()) for k,m in modules.items() if m is not None})
                        opt.zero_grad(set_to_none=True);values['loss'].backward()
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                    b=len(batch['origin']);count+=b
                    for k,v in values.items():sums[k]=sums.get(k,0.)+float(v.detach())*b
            record['train' if is_train else 'validation']={k:v/count for k,v in sums.items()}
        eligible=args.stage!='A' or args.profile!='process' or epoch>=5*args.curriculum_interval+1
        record['eligible_for_best']=eligible
        if eligible and record['validation']['selection']<best:
            best=record['validation']['selection'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        record.update(runtime_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
        rows.append(record);print(json.dumps({'stage':args.stage,'epoch':epoch,'selection':record['validation']['selection']},allow_nan=False),flush=True)
        if args.patience and best_state is not None and epoch-best_epoch>=args.patience:break
    model.load_state_dict(best_state)
    if args.stage=='A':
        chosen=rows[best_epoch-1]['validation']
        if args.a_quality_max is not None and max(chosen['ae_delta'],chosen['decoded_drift'])>args.a_quality_max:
            raise ValueError('Best A failed user-specified dynamics quality gate; not sealed')
        x=torch.as_tensor((d['states'][:d['train_end']]-d['mean'])/d['scale'],device=device)
        c=None if d['information'] is None else torch.as_tensor(d['information'][:d['train_end']],device=device)
        model.seal(x,c)
    if args.stage=='B':
        for n,v in frozen.items():
            if not torch.equal(v,model.state_dict()[n].cpu()):raise AssertionError('Frozen A changed: '+n)
    output.parent.mkdir(parents=True,exist_ok=True)
    persisted={k:v for k,v in d.items() if k not in ('states','times','information')}
    for k,v in persisted.items():
        if isinstance(v,np.ndarray):persisted[k]=v.tolist()
    options={k:v for k,v in vars(args).items() if not k.startswith('info_')}
    payload={**persisted,'format':FORMAT,'stage':args.stage,'mode':args.mode,'config':asdict(config),'model':model.state_dict(),
        'options':options,'best_epoch':best_epoch,'best_selection':best,'archive_sha256':digest(args.archive),
        'information_sha256':information_digest(args.information) if args.information else None,
        'information_shards':d['information'].provenance() if hasattr(d['information'],'provenance') else None,'parent_sha256':digest(args.init) if args.init else None,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_file_sha256':{str(f.relative_to(Path(__file__).parent)):digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'optimizer_groups':[{'name':g['name'],'lr':g['lr']} for g in groups],
        'resume':'best weights only; optimizer/RNG not stored; stage A->B->C only',
        'sampling_contract':'origin-fixed information/history; persistent independent member noise; raw-z auxiliary A or projected MoE B/C; full20step'}
    torch.save(payload,output);write_json(output.with_suffix('.metrics.json'),rows)
    write_json(output.with_suffix('.metadata.json'),{k:v for k,v in payload.items() if k!='model'})
    write_json(output.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(output),'stage':args.stage})
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--information');p.add_argument('--init');p.add_argument('--stage',choices=['A','B','C'],required=True)
    p.add_argument('--mode',choices=['surface','enriched'],default='enriched')
    p.add_argument('--profile',choices=['baseline','dynamics','information','process'],default='process')
    for k,v in dict(epochs=12,batch_size=2,members=4,tau_steps=4,history_steps=6,history_stride=4,
                    manifold_dim=16,hidden_dim=128,context_dim=64,experts=4,expert_latent_dim=64,
                    gate_hidden_dim=160,window_stride=4,max_windows=0,seed=7,curriculum_interval=2,patience=0).items():
        p.add_argument('--'+k.replace('_','-'),type=int,default=v)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--b-member-weight',type=float,default=.001)
    p.add_argument('--loss-weights',help='A-only JSON plateau coefficients; preserves six-phase activation schedule')
    p.add_argument('--a-quality-max',type=float);p.add_argument('--gradient-audit',action='store_true')
    p.add_argument('--device',default='cpu');a=p.parse_args(argv);print(train(a));return 0
if __name__=='__main__':main()

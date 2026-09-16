"""Separated A information learner and frozen-A full-state residual MoE.

A auxiliary FM is a small raw-z/day sampler, not the B experts. B replaces it
with existing projected full-state MoE fields; A representation stays frozen.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from .manifold_moe import ManifoldMoE
from .moe import mlp
from .recurrent_flow import drift_per_day, physical_step
from .temporal_supervision import TemporalObjective

FORMAT='climate_diffusion.separate_a_information_process.v1'

def fair_crps(x,y,weight=None):
    if x.shape[1]<2 or x.shape[:1]+x.shape[2:]!=y.shape: raise ValueError('CRPS requires [B,M>=2,...] and matching truth')
    m=x.shape[1];ordered=x.sort(dim=1).values
    ranks=torch.arange(1,m+1,device=x.device,dtype=x.dtype)*2-m-1
    score=(x-y[:,None]).abs().mean(1)-(ordered*ranks.reshape(1,m,*([1]*(x.ndim-2)))).sum(1)/(m*(m-1))
    if weight is None: return score.mean()
    return (score*weight).sum()/weight.expand_as(score).sum()

class InformationProcess(nn.Module):
    def __init__(self,config,schema,mean,scale,statistics,info_metadata=None):
        super().__init__()
        self.core=ManifoldMoE(config,schema,mean,scale)
        self.temporal=TemporalObjective(schema,mean,scale,statistics)
        self.info_metadata=info_metadata
        f=math.prod(info_metadata['shape']) if info_metadata else 0
        r,h,c=config.manifold_dim,config.hidden_dim,config.context_dim
        self.information=mlp(f,h,r) if f else None
        self.info_head=mlp(r,h,f) if f else None
        self.reference_information=copy.deepcopy(self.information)
        # Raw-z coordinates make A's own sampler invariant to the later q seal.
        self.a_context=mlp(r,h,c)
        self.a_sampler=mlp(2*r+c+2,h,r)
        self.phase='A';self.set_phase('A')

    @property
    def config(self): return self.core.config

    def set_phase(self,phase):
        if phase not in ('A','B','C'): raise ValueError('Expected separate phase A, B or C')
        self.phase=phase;self.requires_grad_(False)
        self.core.set_stage({'A':'manifold','B':'specialize','C':'joint'}[phase])
        if phase in ('A','C') and self.information is not None: self.information.requires_grad_(True)
        if phase=='A':
            self.a_sampler.requires_grad_(True);self.a_context.requires_grad_(True)
            if self.info_head is not None:self.info_head.requires_grad_(True)
        if self.reference_information is not None:self.reference_information.requires_grad_(False)

    def raw_encode(self,x,information=None,reference=False):
        if reference:
            z=self.core.reference_encoder(self.core.manifold.spatial_dct(x))
            encoder=self.reference_information
        else:z=self.core.manifold.encode(x);encoder=self.information
        if encoder is not None:
            if information is None or information.shape[:-1]!=x.shape[:-1]:
                raise ValueError('Enriched checkpoint requires matching origin information; never substitute zero/missing fields')
            z=z+encoder(information)
        elif information is not None:raise ValueError('Surface-only model does not accept enriched information')
        return z

    def encode(self,x,information=None,reference=False):
        return (self.raw_encode(x,information,reference)-self.core.latent_mean)/self.core.latent_scale

    def context(self,history,information,auxiliary=False):
        ci=None if information is None else information[:,None].expand(-1,history.shape[1],-1)
        raw=self.raw_encode(history,ci)
        if auxiliary:return self.a_context(raw.mean(1))
        q=(raw-self.core.latent_mean)/self.core.latent_scale
        return self.core.history_encoder(self.core.temporal_dct(q,dim=1).flatten(1))

    @torch.no_grad()
    def seal(self,states,information=None):
        if bool(self.core.manifold_ready):raise ValueError('Seal only once, after best A; retrain B after A changes')
        raw=torch.cat([self.raw_encode(states[i:i+256],None if information is None else information[i:i+256])
                       for i in range(0,len(states),256)])
        self.core.latent_mean.copy_(raw.mean(0));self.core.latent_scale.copy_(raw.std(0,unbiased=False).clamp_min(.05))
        self.core.gate.fit_centers((raw-self.core.latent_mean)/self.core.latent_scale)
        self.core.reference_encoder.load_state_dict(self.core.manifold.encoder.state_dict())
        if self.information is not None:self.reference_information.load_state_dict(self.information.state_dict())
        self.core.manifold_ready.fill_(True)

    def auxiliary_field(self,r,z,context,tau,hours):
        return self.a_sampler(torch.cat((r,z,context,tau[:,None],hours[:,None]/self.config.horizon_hours),-1))

    def rollout(self,history,information,*,members=4,tau_steps=4,steps=None,generator=None,noise=None,
                auxiliary=False,drift_only=False,trace=None,return_q=False):
        steps=self.config.horizon_steps if steps is None else steps
        if not 1<=steps<=self.config.horizon_steps or members<2 or tau_steps<1:raise ValueError('Invalid physical horizon/member/tau contract')
        b=len(history);r=self.config.manifold_dim
        noise=torch.randn(b,members,r,device=history.device,generator=generator) if noise is None else noise
        if noise.shape!=(b,members,r):raise ValueError('Noise must be [B,M,r]')
        q0=self.encode(history[:,-1],information);q=q0[:,None].expand(-1,members,-1).reshape(b*members,r)
        context=self.context(history,information,auxiliary).repeat_interleave(members,0)
        origin=history[:,-1];offset=origin-self.core.decode(q0)
        paths=[origin[:,None].expand(-1,members,-1)];qs=[q.reshape(b,members,r)]
        for j in range(steps):
            prev=q;hours=q.new_full((len(q),),j*self.config.step_hours)
            if auxiliary:
                z=q*self.core.latent_scale+self.core.latent_mean
                residual=noise.reshape(b*members,r)*self.config.residual_noise_std
                for k in range(tau_steps):
                    tau=q.new_full((len(q),),k/tau_steps)
                    v=self.auxiliary_field(residual,z,context,tau,hours)
                    v2=self.auxiliary_field(residual+v/(2*tau_steps),z,context,tau+.5/tau_steps,hours)
                    residual=residual+v2/tau_steps
                residual=residual/self.core.latent_scale
                if drift_only:residual=torch.zeros_like(residual)
                drift=drift_per_day(self.core,q)
                q=q+self.config.step_hours/24*(drift+residual)
                decomposition={'drift_per_day':drift,'residual_per_day':residual,'final_per_day':drift+residual}
            else:
                q,decomposition=physical_step(self.core,q,context,hours,noise.reshape(b*members,r),
                    dt_hours=self.config.step_hours,integration_steps=tau_steps,mode='drift_only' if drift_only else 'local')
            if not bool(torch.isfinite(q).all()):raise FloatingPointError('Nonfinite recurrent q')
            paths.append(self.core.decode(q).reshape(b,members,-1)+offset[:,None]);qs.append(q.reshape(b,members,r))
            if trace is not None:
                row={'input':prev.detach().clone(),'output':q.detach().clone(),
                    **{k:v.detach().clone() for k,v in decomposition.items() if k.endswith('per_day')}}
                field=decomposition.get('transport')
                if field is not None:
                    gate=field['router'].detach();raw=field['raw_candidates'].detach();projected=field['candidates'].detach()
                    norm=lambda v:(v.square()*self.core.physics.metric_weights).sum(-1).sqrt()
                    rn,pn=norm(raw),norm(projected);directions=torch.nn.functional.normalize(projected,dim=-1)
                    off=~torch.eye(self.config.num_experts,dtype=torch.bool,device=q.device)
                    row['routing']={'gate_mean':gate.mean(0).cpu().tolist(),
                        'entropy_nats':float(-(gate*gate.clamp_min(1e-12).log()).sum(-1).mean()),
                        'candidate_cosine':float((directions@directions.transpose(1,2))[:,off].mean()),
                        'raw_transport_norm':float(rn.mean()),'projected_transport_norm':float(pn.mean()),
                        'projection_ratio':float((pn[rn>1e-8]/rn[rn>1e-8]).mean()) if bool((rn>1e-8).any()) else None,
                        'unit':'weighted normalized-state/day per tau; NOT physical tendency; final tau evaluation only'}
                trace.append(row)
        result=torch.stack(paths,2)
        return (result,torch.stack(qs,2)) if return_q else result

    def teacher_loss(self,batch,information,generator):
        """One randomly selected actual 6h pair per window; labels never condition rollout."""
        history=batch['history'];target=torch.cat((batch['origin'][:,None],batch['targets']),1)
        b=len(history);p=torch.randint(self.config.horizon_steps,(b,),device=history.device,generator=generator)
        rows=torch.arange(b,device=history.device);x,y=target[rows,p],target[rows,p+1]
        dt=batch['dt_hours'][rows,p,None]/24
        if self.phase=='A':
            z=self.raw_encode(x,information)
            with torch.no_grad():label=(self.raw_encode(y,information)-self.raw_encode(x,information))/dt-self.core.manifold.latent_drift(self.raw_encode(x,information))
            source=torch.randn(z.shape,device=z.device,generator=generator)*self.config.residual_noise_std
            tau=torch.rand(b,device=z.device,generator=generator)
            pred=self.auxiliary_field((1-tau[:,None])*source+tau[:,None]*label,z,
                self.context(history,information,True),tau,p.to(z)*self.config.step_hours)
            return {'fm':(pred-(label-source)).square().mean()}
        with torch.no_grad():
            q=self.encode(x,information)
            label=(self.encode(y,information)-q)/dt-drift_per_day(self.core,q)
        source=torch.randn(q.shape,device=q.device,generator=generator)*self.config.residual_noise_std
        tau=torch.rand(b,device=q.device,generator=generator)
        return self.core.specialization_loss((1-tau[:,None])*source+tau[:,None]*label,label-source,tau,
            self.context(history,information),p.to(q)/self.config.horizon_steps,physical_q=q)

    def geometry_losses(self,batch,information):
        x=batch['origin'];y=batch['targets'][:,0];dt=batch['dt_hours'][:,0,None]
        z=self.raw_encode(x,information);zy=self.raw_encode(y,information)
        rx=self.core.manifold.decode(z);ry=self.core.manifold.decode(zy)
        next_z=z+dt/24*self.core.manifold.latent_drift(z);dr=self.core.manifold.decode(next_z)
        t=self.temporal;true=(y-x)*t.scale/dt/t.tendency_scale
        ae=(ry-rx)*t.scale/dt/t.tendency_scale;dv=(dr-rx)*t.scale/dt/t.tendency_scale
        weighted=lambda e:(e.square()*t.metric).sum(-1).mean()
        values=self.core.physics.reconstruction_losses(rx,x)
        values.update(ae_delta=weighted(ae-true),decoded_drift=weighted(dv-true),
            forecast_anchor=weighted(dr-rx+x-y),latent_dynamics=(next_z-zy.detach()).square().mean(),
            latent_variance=z.var(0,unbiased=False).mean())
        if len(x)<2:raise ValueError('A/C geometry requires batch >=2')
        values['metric']=((z-z.roll(1,0)).square().mean(-1).clamp_min(1e-12).sqrt()
            -self.core.physics.pair_distance(x,x.roll(1,0)).detach()).square().mean()
        values.update(t.state_metrics(rx,x,'reconstruction'))
        values.update(t.state_metrics(ae,true,'ae_tendency'));values.update(t.state_metrics(dv,true,'drift_tendency'))
        if information is not None:
            fitted=self.info_head(z);sh=self.info_metadata['shape'];cells=sh[1]*sh[2]
            mask=torch.tensor([v['kind']=='static' for v in self.info_metadata['variables']],device=x.device).repeat_interleave(cells)
            w=t.area.flatten().repeat(sh[0]);error=(fitted-information.detach()).square()*w
            values['static_l2']=error[:,mask].sum(-1).mean()/max(1,int(mask.sum())//cells)
            values['info_reconstruction']=error[:,~mask].sum(-1).mean()/max(1,int((~mask).sum())//cells)
            # Paired fixed-target geometry; not marginal MMD of two trainable encoders.
            target_dist=((information-information.roll(1,0)).square()*w).sum(-1).div(sh[0]).sqrt().detach()
            latent_dist=(z-z.roll(1,0)).square().mean(-1).clamp_min(1e-12).sqrt()
            values['information_geometry']=(latent_dist-target_dist).square().mean()
        return values

    def scores(self,generated,truth,dt,mask):
        t=self.temporal;v=t(generated,truth,dt,mask)
        ds=generated.diff(dim=2)*t.scale/dt[:,None,:,None]/t.tendency_scale
        dy=truth.diff(dim=1)*t.scale/dt[:,:,None]/t.tendency_scale
        v['state_crps']=fair_crps(generated[:,:,1:],truth[:,1:],t.metric)
        v['transition_crps']=fair_crps(ds,dy,t.metric)
        v['mean_state']=((generated[:,:,1:].mean(1)-truth[:,1:]).square()*t.metric).sum(-1).mean()
        v['rmse']=v['mean_state'].sqrt();v['spread']=(generated[:,:,1:].var(1,unbiased=False)*t.metric).sum(-1).mean().sqrt()
        lo,hi=generated[:,:,1:].quantile(.1,dim=1),generated[:,:,1:].quantile(.9,dim=1)
        v['coverage80']=(((truth[:,1:]>=lo)&(truth[:,1:]<=hi)).to(generated)*t.metric).sum(-1).mean()
        cells=self.config.grid[1]*self.config.grid[2]
        for i,name in enumerate(t.names):
            sl=slice(i*cells,(i+1)*cells)
            v['state_crps_'+name]=fair_crps(generated[:,:,1:,sl],truth[:,1:,sl],t.area.flatten())
            v['transition_crps_'+name]=fair_crps(ds[:,:,:,sl],dy[:,:,sl],t.area.flatten())
        return v

    def information_scores(self,qs,origin_info,future_info,dt,info_scale,info_tendency_scale):
        sh=self.info_metadata['shape'];cells=sh[1]*sh[2]
        z=qs*self.core.latent_scale+self.core.latent_mean
        decoded=self.info_head(z)
        pred=decoded-decoded[:,:,:1]+origin_info[:,None,None]
        truth=torch.cat((origin_info[:,None],future_info),1)
        out={}
        for i,var in enumerate(self.info_metadata['variables']):
            if var['kind']=='static':continue
            sl=slice(i*cells,(i+1)*cells)
            out['info_crps_'+var['name']]=fair_crps(pred[:,:,1:,sl],truth[:,1:,sl],self.temporal.area.flatten())
            dp=pred[:,:,:,sl].diff(dim=2)*info_scale[sl]/dt[:,None,:,None]/info_tendency_scale[sl]
            dy=truth[:,:,sl].diff(dim=1)*info_scale[sl]/dt[:,:,None]/info_tendency_scale[sl]
            out['info_transition_'+var['name']]=fair_crps(dp,dy,self.temporal.area.flatten())
        out['info_distribution']=torch.stack(list(out.values())).mean()
        return out

def curriculum(epoch,interval=2):
    if interval<1:raise ValueError('Curriculum interval must be positive')
    phase=min(6,1+(epoch-1)//interval)
    return phase,{'reconstruction':1.,'forecast_anchor':.1,'physics':.1,'invariant':.05,'metric':.1,
        'latent_dynamics':.1 if phase>=2 else 0.,'ae_delta':.05 if phase>=2 else 0.,
        'decoded_drift':.05 if phase>=2 else 0.,'static_l2':.05,'info_reconstruction':.05,
        'fm':1. if phase>=3 else 0.,'state_crps':.25 if phase>=3 else 0.,
        'information_geometry':.02 if phase>=4 else 0.,'info_distribution':.1 if phase>=4 else 0.,
        'transition_crps':.25 if phase>=5 else 0.,'loss_delta':.02 if phase>=5 else 0.,
        'loss_trajectory':.1 if phase>=6 else 0.}

def gradient_diagnostics(losses,groups):
    """Reusable parameter lists, explicit zero for frozen/unused groups."""
    groups={k:[p for p in v if p.requires_grad] for k,v in groups.items()};out={};vectors={}
    for name,loss in losses.items():
        if not loss.requires_grad:continue
        for group,params in groups.items():
            if not params:continue
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vec=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,grads)])
            out[f'gradient/{name}/{group}']=float(vec.detach().norm());vectors[name,group]=vec.detach()
    names=list(losses)
    for group in groups:
        for a,b in zip(names,names[1:]):
            if (a,group) in vectors and (b,group) in vectors:
                x,y=vectors[a,group],vectors[b,group];den=x.norm()*y.norm()
                out[f'cosine/{a}:{b}/{group}']=float(x@y/den) if den>1e-12 else 0.
    return out

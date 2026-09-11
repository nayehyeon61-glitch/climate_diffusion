"""Physical recurrence, units, joint gradients and explicit checkpoint semantics."""
from dataclasses import replace
import types

import pytest
import torch

from test_manifold_moe import model as old_model, archive
from climate_diffusion.recurrent_flow import drift_per_day, residual_target, physical_step
from climate_diffusion.train_manifold_moe import train_manifold_moe, _pairs
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.manifold_moe import RECURRENT_FORMAT
from climate_diffusion.temporal_supervision import TemporalObjective, fit_temporal_statistics
from climate_diffusion.moe_data import load_moe_archive


def model():
    m = old_model()
    m.config = replace(m.config, forecast_dynamics="recurrent_residual",horizon_steps=20)
    m.set_stage("specialize")
    return m


def test_recurrent_coupling_prefix_and_no_member_contamination():
    m = model()
    history = torch.randn(1,3,32)
    context = m.encode_history(history)
    noise = torch.randn(1,3,3)
    leads = torch.arange(21)[None]
    trace = []
    with torch.no_grad():
        full = m.sample_trajectory(context,leads,ensemble_size=3,integration_steps=1,
                                  origin=history[:,-1],initial=noise,trace=trace)
        prefix = m.sample_trajectory(context,leads[:,:3],ensemble_size=3,integration_steps=1,
                                    origin=history[:,-1],initial=noise)
        one = m.sample_trajectory(context,leads,ensemble_size=1,integration_steps=1,
                                 origin=history[:,-1],initial=noise[:,:1])
    torch.testing.assert_close(full[:,:,:3],prefix,rtol=0,atol=0)
    torch.testing.assert_close(full[:,:1],one,atol=2e-5,rtol=2e-5)
    for a,b in zip(trace,trace[1:]):
        torch.testing.assert_close(a["q_output"],b["q_input"],atol=0,rtol=0)
    assert not torch.equal(full[:,0],full[:,1])
    torch.testing.assert_close(full[:,:,0],history[:,-1,None].expand(-1,3,-1))
    with pytest.raises(ValueError,match="observed origin"):
        m.sample_trajectory(context,leads[:,1:],ensemble_size=3,integration_steps=1)


def test_day_hour_scaling_drift_only_and_zero_drift_residual():
    m = model()
    q = torch.randn(2,3); h = torch.randn(2,4); hours = torch.zeros(2); noise = torch.randn(2,3)
    z = q*m.latent_scale+m.latent_mean
    torch.testing.assert_close(drift_per_day(m,q),m.manifold.latent_drift(z)/m.latent_scale)
    q6,a = physical_step(m,q,h,hours,noise,dt_hours=6,integration_steps=1,mode="drift_only")
    q12,_ = physical_step(m,q,h,hours,noise,dt_hours=12,integration_steps=1,mode="drift_only")
    torch.testing.assert_close(q12-q,2*(q6-q))
    assert torch.count_nonzero(a['residual_per_day']) == 0
    qres,a = physical_step(m,q,h,hours,noise,dt_hours=6,integration_steps=1,mode="residual_only")
    torch.testing.assert_close(qres,q+.25*a['residual_per_day'])
    assert torch.count_nonzero(a['drift_per_day']) == 0
    for p in m.manifold.latent_drift.parameters(): p.data.zero_()
    qzero,_ = physical_step(m,q,h,hours,noise,dt_hours=6,integration_steps=1)
    torch.testing.assert_close(qzero,qres)
    for dt in (0,-1,float('nan')):
        with pytest.raises(ValueError,match="dt_hours"):
            physical_step(m,q,h,hours,noise,dt_hours=dt,integration_steps=1)


def test_constant_physical_velocity_20_steps_analytic_stub():
    # An integrator contract, NOT a claim of learned synthetic skill.
    m = model()
    m.manifold.latent_drift = torch.nn.Linear(3,3)
    with torch.no_grad():
        m.manifold.latent_drift.weight.zero_(); m.manifold.latent_drift.bias.copy_(m.latent_scale)
    m.decode = types.MethodType(lambda self,q: q.repeat(1,11)[:,:32],m)
    history = torch.randn(1,3,32)
    values = m.sample_trajectory(m.encode_history(history),torch.arange(21)[None],ensemble_size=2,
        integration_steps=1,origin=history[:,-1],mode="drift_only")
    torch.testing.assert_close(values.diff(dim=2),torch.full_like(values[:,:,1:],.25),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('stage',['specialize','joint'])
def test_full120h_gradient_and_frozen_b(stage):
    m = model(); m.set_stage(stage)
    before = {k:v.clone() for k,v in m.manifold.state_dict().items()}
    history = torch.randn(2,3,32)
    # Exercise actual nonlinear decode + projection + residual tau solves, all 20 physical steps.
    values = m.sample_trajectory(m.encode_history(history),torch.arange(21)[None].expand(2,-1),
        ensemble_size=2,integration_steps=1,origin=history[:,-1])
    loss = values[:,:,1:].square().mean() + values.diff(dim=2).square().mean()
    loss.backward()
    for module in (m.experts,m.gate,m.history_encoder):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    if stage=='specialize': assert all(p.grad is None for p in m.manifold.parameters())
    else:
        for module in (m.manifold.encoder,m.manifold.decoder,m.manifold.latent_drift):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4).step()
    if stage=='specialize':
        for k,v in before.items(): torch.testing.assert_close(v,m.manifold.state_dict()[k],atol=0,rtol=0)


def test_residual_fm_target_units_detach_and_shared_source():
    m=model(); m.set_stage('joint')
    batch={'history':torch.randn(2,3,32),'origin':torch.randn(2,32),
           'targets':torch.randn(2,20,32),'dt_hours':torch.full((2,20),6.)}
    r,v,tau,h,lead,target,steps,q=_pairs(m,batch,torch.Generator().manual_seed(3),2,
                                       return_steps=True,return_physical=True)
    assert not q.requires_grad
    trajectory=torch.cat((batch['origin'][:,None],batch['targets']),1)
    rows=torch.arange(2)[:,None]
    prev=trajectory[rows,steps-1].reshape(-1,32)
    expected_q,expected=residual_target(m,prev,target,torch.full((4,),6.))
    torch.testing.assert_close(q,expected_q)
    source=(r-tau[:,None]*expected)/(1-tau[:,None])
    torch.testing.assert_close(source.reshape(2,2,3)[:,0],source.reshape(2,2,3)[:,1],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(v,expected-source,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(lead,(steps.flatten()-1).float()/20)
    m.specialization_loss(r,v,tau,h,lead,physical_q=q)['loss'].backward()
    assert any(p.grad is not None for p in m.history_encoder.parameters())
    with pytest.raises(ValueError,match="physical_q"):
        m.field(r,tau,h,lead)


def test_member_delta_variance_identity(archive):
    path,_=archive
    states,times,schema=load_moe_archive(path)
    stats=fit_temporal_statistics(states,times,schema,states.std(0),50)
    loss=TemporalObjective(schema,states.mean(0),states.std(0),stats)
    samples=torch.randn(2,3,4,32,requires_grad=True); truth=torch.randn(2,4,32)
    metrics=loss(samples,truth,torch.full((2,3),6.))
    torch.testing.assert_close(metrics['loss_delta_member'],metrics['loss_delta']+metrics['delta_member_variance_penalty'])
    metrics['loss_delta_member'].backward(); assert samples.grad.abs().sum()>0


def test_recurrent_checkpoint_and_old_format_rejection(archive,tmp_path):
    path,_=archive
    output=train_manifold_moe(path,tmp_path/'recurrent.pt',manifold_epochs=1,expert_epochs=1,joint_epochs=1,
        model_options={'forecast_dynamics':'recurrent_residual','manifold_dim':3,'history_steps':3,
        'history_stride':1,'horizon_steps':3,'num_experts':2,'hidden_dim':12,'context_dim':4,
        'gate_hidden_dim':8,'expert_latent_dim':6},batch_size=4,window_stride=4,ensemble_size=2,
        integration_steps=1,max_validation_windows=2,delta_member_weight=.001,delta_weight=.01,
        trajectory_weight=.1,trajectory_edges=0,device='cpu')
    p=torch.load(output,weights_only=False)
    assert p['format']==RECURRENT_FORMAT and p['training']['loss_options']['delta_member_weight']==.001
    f=LatentFlowForecaster(output,device='cpu')
    states,_,_=load_moe_archive(path); history=f.select_history(states)
    assert f.forecast(history,months=3,ensemble_size=2,integration_steps=1).shape==(2,3,32)
    # Mutate the actual archive's future observations, not a disconnected local
    # tensor. Forecast at a fixed held-out origin must remain exactly identical.
    import numpy as np
    from climate_diffusion.time_alignment import forecast_from_checkpoint, align_forecast, temporal_diagnostics
    from climate_diffusion.trajectory_output import select_interval
    _,times,_=load_moe_archive(path)
    i=f.training_metadata['split']['validation'][0]+f.config.history_span_steps-1
    first=forecast_from_checkpoint(output,path,str(times[i]),tmp_path/'first.npz',
        ensemble_size=2,integration_steps=1,forecast_steps=3,seed=91,device='cpu')
    with np.load(path,allow_pickle=False) as source: copied={k:source[k].copy() for k in source.files}
    copied['states'][i+1:]+=123.
    changed=tmp_path/'future-changed.npz'; np.savez_compressed(changed,**copied)
    import shutil
    shutil.copyfile(path.with_suffix('.schema.json'),changed.with_suffix('.schema.json'))
    second=forecast_from_checkpoint(output,changed,str(times[i]),tmp_path/'second.npz',
        ensemble_size=2,integration_steps=1,forecast_steps=3,seed=91,device='cpu')
    with np.load(first) as a, np.load(second) as b:
        np.testing.assert_array_equal(a['predictions'],b['predictions'])
    aligned=align_forecast(first,path)
    assert 'physical_recurrent' in temporal_diagnostics(aligned)['sampling_contract']
    twelve=select_interval(aligned,12,12)
    np.testing.assert_array_equal(twelve.members,aligned.members[:,1:2])
    np.testing.assert_array_equal(twelve.valid_times,aligned.origin_time+np.array([12],dtype='timedelta64[h]'))
    from climate_diffusion.manifold_diagnostics import diagnose_manifold
    import json
    audit=diagnose_manifold(output,path,tmp_path/'audit.json',max_cases=2,members=2,
                           integration_steps=1,split_name='validation')
    assert len(json.loads(audit.read_text())['physical_trace'])==3
    p['format']='climate_diffusion.manifold_moe.v1'
    invalid=tmp_path/'wrong.pt'; torch.save(p,invalid)
    with pytest.raises(ValueError,match='format/dynamics mismatch'): LatentFlowForecaster(invalid)


def test_all_experts_share_physical_state_and_residual_source():
    m=model(); seen=[[],[]]
    hooks=[e.register_forward_pre_hook(lambda module,args,k=k: seen[k].append(tuple(a.detach().clone() for a in args)))
           for k,e in enumerate(m.experts)]
    h=torch.randn(1,3,32)
    m.forecast(h,ensemble_size=2,integration_steps=1,lead_indices=[0,1],generator=torch.Generator().manual_seed(7))
    for handle in hooks: handle.remove()
    assert len(seen[0])==4  # 2 tau evaluations × 2 physical steps
    for a,b in zip(*seen):
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=0,rtol=0)
    # First tau source is identical each physical step, but physical conditioning changes.
    r=m.config.manifold_dim
    torch.testing.assert_close(seen[0][0][1][:,:r],seen[0][2][1][:,:r],atol=0,rtol=0)
    assert not torch.equal(seen[0][0][0],seen[0][2][0])

"""New physical-time contracts: differentiable, stochastic, causal and explicit."""
from dataclasses import replace
import json
import types

import numpy as np
import pytest
import torch

from test_manifold_moe import model, archive
from climate_diffusion.temporal_supervision import (TemporalWindowDataset, TemporalObjective,
    NonSingletonBatchSampler, fit_temporal_statistics, fair_energy, wind_features, select_block)
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.train_manifold_moe import train_manifold_moe
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.time_alignment import AlignedForecast, _validate_leads, align_forecast
from climate_diffusion.trajectory_output import select_interval, member_diagnostics, export_trajectories


def objective(m):
    raw = torch.randn(50,32).numpy()*m.physics.scale.numpy()+m.physics.mean.numpy()
    times=np.datetime64("2000-01-01")+np.arange(50)*np.timedelta64(6,"h")
    coords={"lat":[-45.,45.],"lon":[0.,90.,180.,270.]}
    schema={"state_dim":32,"forecast_step_hours":6,"variables":[
        {"name":n,"dims":["lat","lon"],"shape":[2,4],"slice":[k*8,(k+1)*8],"coords":coords}
        for k,n in enumerate(("msl","t2m","u10","v10"))]}
    stats=fit_temporal_statistics(raw,times,schema,m.physics.scale.numpy(),30)
    return TemporalObjective(schema,m.physics.mean,m.physics.scale,stats),raw,times,schema


def test_window_120h_causal_mask_statistics_and_units():
    m=model(); loss,raw,times,schema=objective(m)
    c=replace(m.config,horizon_steps=20)
    ds=TemporalWindowDataset(raw,times,c,[0,1],loss.mean.numpy(),loss.scale.numpy(),schema)
    batch=ds[1]
    assert batch["trajectory_raw"].shape==(21,32)
    assert batch["delta_raw"].shape==(20,32)
    assert (batch["valid_time_ns"][-1]-batch["origin_time_ns"]).item()==120*3600*10**9
    torch.testing.assert_close(batch["delta_raw"],batch["trajectory_raw"].diff(dim=0))
    torch.testing.assert_close(batch["tendency_raw"],batch["delta_raw"]/6)
    changed=raw.copy(); changed[4:]+=10
    ds2=TemporalWindowDataset(changed,times,c,[0],loss.mean.numpy(),loss.scale.numpy(),schema)
    torch.testing.assert_close(ds[0]["history"],ds2[0]["history"])
    changed=raw.copy(); changed[30:]*=99
    stats=fit_temporal_statistics(changed,times,schema,loss.scale.numpy(),30)
    assert stats==loss.stats
    assert stats==fit_temporal_statistics(changed,times,schema,loss.scale.numpy(),30,
                                          {n:1 for n in ("msl","t2m","u10","v10")})
    mask=np.ones_like(raw); mask[3,0]=0
    with pytest.raises(ValueError,match="Missing endpoint"):
        TemporalWindowDataset(raw,times,c,[0],loss.mean.numpy(),loss.scale.numpy(),schema,mask)
    with pytest.raises(ValueError,match="physical dt"):
        TemporalWindowDataset(raw,times+np.arange(len(times))*np.timedelta64(1,"h"),c,[0],loss.mean.numpy(),loss.scale.numpy(),schema)
    schema["variables"][0]["attrs"]={"units":"hPa"}
    with pytest.raises(ValueError,match="units"):
        fit_temporal_statistics(raw,times,schema,loss.scale.numpy(),30)


def test_common_sampler_training_inference_prefix_and_noise():
    m=model(); m.set_stage("specialize"); history=torch.randn(2,3,32)
    context=m.encode_history(history)
    original=m.integrate; seen=[]
    def record(self,initial,context,lead,**kw):
        seen.append((initial.detach().clone(),lead.detach().clone()))
        return original(initial,context,lead,**kw)
    m.integrate=types.MethodType(record,m)
    steps=torch.tensor([[1,2,3],[1,2,3]])
    a=m.sample_trajectory(context,steps,ensemble_size=3,integration_steps=1,generator=torch.Generator().manual_seed(10))
    assert len(seen)==3
    for s in seen: torch.testing.assert_close(s[0],seen[0][0])
    assert not torch.equal(seen[0][0][0],seen[0][0][1])
    b=m.forecast(history,ensemble_size=3,integration_steps=1,generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    prefix=m.forecast(history,ensemble_size=3,integration_steps=1,lead_indices=[0],generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(a[:,:,:1],prefix,atol=0,rtol=0)


@pytest.mark.parametrize("stage",["specialize","joint"])
def test_new_loss_backpropagates_and_freeze(stage):
    m=model(); loss,_,_,_=objective(m); m.set_stage(stage)
    before={k:v.clone() for k,v in m.manifold.state_dict().items()}
    history=torch.randn(2,3,32); origin=history[:,-1]
    samples=m.sample_trajectory(m.encode_history(history),torch.tensor([[0,1,2],[1,2,3]]),
                                ensemble_size=3,integration_steps=1,origin=origin)
    target=torch.randn(2,3,32)
    metrics=loss(samples,target,torch.full((2,2),6.))
    total=metrics["loss_trajectory"]+0.05*metrics["loss_delta"]+0.01*metrics["loss_wind_speed"]+0.01*metrics["loss_wind_direction"]
    total.backward()
    for module in (m.experts,m.gate,m.history_encoder):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    if stage=="specialize": assert all(p.grad is None for p in m.manifold.parameters())
    else: assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.manifold.parameters())
    torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-4).step()
    if stage=="specialize":
        for k,v in before.items(): torch.testing.assert_close(v,m.manifold.state_dict()[k],atol=0,rtol=0)


def test_scores_variable_gradients_dt_scaling_and_permutation():
    m=model(); loss,_,_,_=objective(m)
    samples=torch.randn(2,4,3,32,requires_grad=True); target=torch.randn(2,3,32)
    metrics=loss(samples,target,torch.full((2,2),6.))
    twice=loss(samples,target,torch.full((2,2),12.))
    torch.testing.assert_close(metrics["loss_delta"],4*twice["loss_delta"])
    perm=loss(samples[:,[3,1,2,0]],target,torch.full((2,2),6.))
    torch.testing.assert_close(metrics["loss_trajectory"],perm["loss_trajectory"])
    shuffled=samples.detach().clone(); shuffled[:,:,1]=shuffled[:,[1,2,3,0],1]
    assert not torch.isclose(metrics["loss_trajectory"],loss(shuffled,target,torch.full((2,2),6.))["loss_trajectory"])
    metrics["loss_delta"].backward()
    assert all(samples.grad[...,k*8:(k+1)*8].abs().sum()>0 for k in range(4))
    with pytest.raises(ValueError,match="dt"):
        loss(samples,target,torch.zeros(2,2))
    with pytest.raises(ValueError,match="Energy"):
        fair_energy(torch.zeros(1,1,3),torch.zeros(1,3))


def test_wind_wrap_and_calm_finite_gradient():
    angle=torch.deg2rad(torch.tensor([359.,1.]))
    _,direction=wind_features(angle.cos(),angle.sin(),1e-6)
    assert ((direction[0]-direction[1])**2).sum()<0.002
    u=torch.tensor([0.],requires_grad=True); v=torch.tensor([0.],requires_grad=True)
    speed,direction=wind_features(u,v,1e-3)
    (speed.sum()+direction.sum()).backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
    assert torch.equal(direction,torch.zeros_like(direction))


def test_singleton_is_not_silently_zero_metric():
    batches=list(NonSingletonBatchSampler(5,2))
    assert batches==[[0,1],[2,3,4]]
    with pytest.raises(ValueError,match="two"):
        list(NonSingletonBatchSampler(1,2))
    with pytest.raises(ValueError,match="two"):
        model().manifold_loss(torch.zeros(1,32),torch.zeros(1,32))


def test_old_720h_checkpoint_prefix_preserves_condition():
    m=model(); m.config=replace(m.config,horizon_steps=120)
    m.set_stage("specialize")
    seen=[]
    def fast(self,initial,context,lead,**kwargs):
        seen.append(lead.clone())
        return initial+lead[:,None]
    m.integrate=types.MethodType(fast,m)
    history=torch.randn(1,3,32)
    all_steps=m.forecast(history,ensemble_size=2,integration_steps=1,generator=torch.Generator().manual_seed(5))
    prefix=m.forecast(history,ensemble_size=2,integration_steps=1,lead_indices=list(range(20)),
                      generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(all_steps[:,:,:20],prefix,atol=0,rtol=0)
    torch.testing.assert_close(seen[-1],torch.full((2,),20/120))


def test_from_scratch_enabled_losses_checkpoint_and_validation(archive,tmp_path):
    from climate_diffusion.evaluation import evaluate_flow_checkpoint
    from climate_diffusion.manifold_diagnostics import diagnose_manifold
    path,_=archive
    checkpoint=train_manifold_moe(path,tmp_path/"enabled.pt",manifold_epochs=1,expert_epochs=1,
        joint_epochs=1,model_options={"history_steps":3,"history_stride":1,"horizon_steps":3,
        "manifold_dim":3,"hidden_dim":12,"context_dim":4,"gate_hidden_dim":8,
        "expert_latent_dim":6,"num_experts":2},batch_size=2,window_stride=4,
        ensemble_size=2,integration_steps=1,max_validation_windows=2,device="cpu",
        delta_weight=.02,trajectory_weight=.1,wind_speed_weight=.01,wind_direction_weight=.005,
        trajectory_edges=0,log_gradient_norms=True)
    rows=json.loads(checkpoint.with_suffix(".metrics.json").read_text())
    for row in rows[1:]:
        assert row["train"]["trajectory_edges"]==3
        assert row["train"]["loss_delta"]>0
        assert all(row["train"]["temporal_output_grad_rms_"+n]>0 for n in ("msl","t2m","u10","v10"))
    a=torch.load(tmp_path/"enabled.manifold.pt",weights_only=False)
    b=torch.load(tmp_path/"enabled.specialize.pt",weights_only=False)
    for k,v in a["model"].items():
        if k.startswith("manifold."): torch.testing.assert_close(v,b["model"][k],atol=0,rtol=0)
    c=torch.load(checkpoint,weights_only=False)
    assert c["training"]["loss_options"]["delta_weight"]==.02
    assert c["training"]["temporal_statistics"]==a["training"]["temporal_statistics"]
    f=LatentFlowForecaster(checkpoint,device="cpu")
    assert f.model.config.horizon_steps==3
    result=evaluate_flow_checkpoint(checkpoint,path,tmp_path/"val.json",split_name="validation",
        ensemble_size=2,integration_steps=1,max_cases=1,device="cpu")
    assert json.loads(result.read_text())["test_windows"]==[]
    result=diagnose_manifold(checkpoint,path,tmp_path/"geometry.json",split_name="validation",
                             max_cases=2,members=2,integration_steps=1)
    audit=json.loads(result.read_text())
    assert audit["evaluation_split"]=="validation" and "test_pca" not in audit
    with pytest.raises(FileExistsError):
        train_manifold_moe(path,checkpoint)


def test_twelve_hour_selection_rates_and_all_member_render(archive,tmp_path):
    pytest.importorskip("matplotlib")
    path,_=archive; raw,times,schema=load_moe_archive(path)
    origin=times[5]; leads=np.arange(1,21)*6
    pred=np.stack([raw[6:26],raw[6:26]+0.1])
    forecast=tmp_path/"forecast.npz"
    np.savez_compressed(forecast,predictions=pred,origin_time=origin,lead_hours=leads,
                        valid_times=origin+leads.astype("timedelta64[h]"),forecast_step_hours=6)
    aligned=align_forecast(forecast,path)
    selected=select_interval(aligned,120,12)
    np.testing.assert_equal(selected.members,pred[:,1::2])
    diagnostics=member_diagnostics(aligned,0)
    assert diagnostics["by_variable"]["u10"]["tendency_rmse"]==[0.]*20
    with pytest.raises(ValueError,match="integer"):
        _validate_leads(np.array([6.5]),1)
    out=export_trajectories(forecast,path,tmp_path/"render",horizon_hours=12,interval_hours=6,
                           extension="gif",reference_label="Synthetic truth")
    assert len(list(out.glob("member-*.gif")))==2
    assert json.loads((out/"member-001.json").read_text())["member_id"]==1
    with np.load(out/"trajectory.npz") as result:
        assert int(result["forecast_step_hours"])==6
        np.testing.assert_equal(result["predictions"],pred[:,:2])

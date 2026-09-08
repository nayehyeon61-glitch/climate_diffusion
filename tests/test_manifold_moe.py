"""Geometry, local responsibility, phase lifecycle and backward compatibility."""
import json
import types

import numpy as np
import pytest
import torch
import xarray as xr
import pandas as pd

from climate_diffusion.manifold_moe import ManifoldMoE, ManifoldMoEConfig, tangent_lift
from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.train_manifold_moe import train_manifold_moe
from climate_diffusion.inference import LatentFlowForecaster, main as forecast_main
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


def model():
    torch.manual_seed(2)
    coords = {"lat": [-45., 45.], "lon": [0., 90., 180., 270.]}
    schema = {"state_dim": 32, "field_dim": 32, "integrated_feature_names": [],
              "variables": [{"name": n, "dims": ["lat", "lon"], "shape": [2, 4],
                             "slice": [8*k, 8*(k+1)], "coords": coords, "attrs": {}}
                            for k, n in enumerate(("msl", "t2m", "u10", "v10"))]}
    c = ManifoldMoEConfig(state_dim=32, grid=(4,2,4), manifold_dim=3, history_steps=3,
                          history_stride=1, horizon_steps=3, num_experts=2, hidden_dim=12,
                          context_dim=4, gate_hidden_dim=8, expert_latent_dim=6)
    mean = torch.tensor([101000.]*8 + [285.]*8 + [0.]*16)
    scale = torch.tensor([1000.]*8 + [5.]*8 + [10.]*16)
    m = ManifoldMoE(c, schema, mean, scale)
    states = torch.randn(30,32)
    m.physics.fit(states)
    m.seal_manifold(states)
    return m


def test_weighted_tangent_projection_and_pullback():
    torch.manual_seed(1)
    j = torch.randn(3,7,2,dtype=torch.float64)
    raw = torch.randn(3,4,7,dtype=torch.float64)
    weights = torch.arange(1,8,dtype=torch.float64)
    intrinsic, projected, metric = tangent_lift(j, raw, weights, 0)
    torch.testing.assert_close(projected, intrinsic @ j.transpose(1,2))
    torch.testing.assert_close(j.transpose(1,2) @ ((raw-projected)*weights).transpose(1,2),
                               torch.zeros(3,2,4,dtype=torch.float64), atol=1e-12, rtol=0)
    _, twice, _ = tangent_lift(j,projected,weights,0)
    torch.testing.assert_close(projected,twice)
    assert (torch.linalg.eigvalsh(metric)>0).all()
    j[:,:,1] = j[:,:,0]
    lifted,_,_ = tangent_lift(j,raw,weights,1e-3)
    assert torch.isfinite(lifted).all()


def test_decoder_jacobian_and_gradient():
    m = model().double()
    m.set_stage("joint")
    q = torch.randn(2,3,dtype=torch.float64)
    direction = torch.randn_like(q)
    j = m.jacobian(q)
    eps = 1e-5
    central = (m.decode(q+eps*direction)-m.decode(q-eps*direction))/(2*eps)
    torch.testing.assert_close((j@direction[...,None]).squeeze(-1),central,atol=1e-8,rtol=1e-5)
    j.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.manifold.decoder.parameters())
    with torch.no_grad():
        torch.testing.assert_close(m.jacobian(q),j)
    with torch.inference_mode():
        torch.testing.assert_close(m.jacobian(q.clone()),j)


def test_local_prior_cannot_be_erased_by_gate_correction():
    m = model()
    with torch.no_grad():
        m.gate.centers.copy_(torch.tensor([[-2.,0.,0.],[2.,0.,0.]]))
        m.gate.radius_squared.fill_(1)
        m.gate.correction[-1].bias.copy_(torch.tensor([1000.,-1000.]))
    condition = torch.zeros(2,m.config.context_dim+2*m.config.time_embedding_dim)
    log_pi,prior = m.gate(m.gate.centers.clone(),condition)
    assert log_pi.argmax(1).tolist() == [0,1]
    assert prior.argmax(1).tolist() == [0,1]
    torch.testing.assert_close(log_pi.exp().sum(-1),torch.ones(2))
    with pytest.raises(ValueError,match="centers collapsed"):
        m.gate.fit_centers(torch.zeros(10,3))


def test_frozen_manifold_and_different_expert_gradients():
    m = model()
    m.set_stage("specialize")
    q = m.gate.centers.clone()
    context = m.encode_history(torch.randn(2,3,32))
    loss = m.specialization_loss(q,torch.ones_like(q),torch.ones(2)*0.5,context,torch.ones(2))["loss"]
    before = {k:v.clone() for k,v in m.manifold.state_dict().items()}
    loss.backward()
    assert all(p.grad is None for p in m.manifold.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.gate.parameters())
    gradients = [torch.cat([p.grad.flatten() for p in e.parameters()]) for e in m.experts]
    assert all(g.abs().sum()>0 for g in gradients)
    assert not torch.allclose(*gradients)
    torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3).step()
    for k,v in before.items():
        torch.testing.assert_close(m.manifold.state_dict()[k],v,atol=0,rtol=0)


def test_same_member_inputs_and_one_intrinsic_ode():
    m = model()
    m.set_stage("specialize")
    seen = [[],[]]
    handles = [e.register_forward_pre_hook(lambda module,args,k=k: seen[k].append(args[0].clone()))
               for k,e in enumerate(m.experts)]
    result = m.forecast(torch.randn(1,3,32),ensemble_size=3,integration_steps=1,lead_indices=[0],
                        generator=torch.Generator().manual_seed(10))
    for handle in handles:
        handle.remove()
    assert len(seen[0]) == 2
    for a,b in zip(*seen):
        torch.testing.assert_close(a,b)
    assert result.shape == (1,3,1,32)
    assert not torch.equal(result[:,0],result[:,1])
    def linear_field(self,q,tau,context,lead,*,mode=None):
        return {"velocity":(0.1*q+0.9*q)/2}
    m.field = types.MethodType(linear_field,m)
    q = torch.randn(2,3)
    out = m.integrate(q,torch.zeros(2,4),torch.ones(2),integration_steps=2)
    torch.testing.assert_close(out,q*(1+0.5*0.5+0.5*(0.5*0.5)**2)**2)


def test_physics_reconstruction_does_not_impose_incompressibility():
    m = model()
    states = torch.randn(3,32,requires_grad=True)
    for value in m.physics.reconstruction_losses(states,states).values():
        torch.testing.assert_close(value,torch.zeros_like(value))
    features,_ = m.physics.raw_features(states)
    assert features[:,0].abs().sum()>0
    features.square().mean().backward()
    assert torch.isfinite(states.grad).all()
    features,_ = m.physics.raw_features(torch.zeros(1,32))
    torch.testing.assert_close(features,torch.zeros_like(features),atol=1e-8,rtol=0)


@pytest.fixture
def archive(tmp_path):
    values = np.random.default_rng(4).normal(size=(90,4,2,4)).astype(np.float32)
    values = values*np.array([1000,5,10,10])[None,:,None,None]+np.array([101000,285,0,0])[None,:,None,None]
    fields = xr.Dataset({n:(("time","lat","lon"),values[:,k].astype(np.float32))
                         for k,n in enumerate(("msl","t2m","u10","v10"))},
                        coords={"time":pd.date_range("2000-01-01",periods=90,freq="6h"),
                                "lat":[-45.,45.],"lon":[0.,90.,180.,270.]})
    raw = tmp_path/"source.nc"
    fields.to_netcdf(raw,engine="scipy")
    path,_ = prepare_fixed_step_archive(raw,tmp_path/"fields.npz",step_hours=6,
                                        target_lat_points=2,target_lon_points=4)
    return path,fields


def test_abc_checkpoint_inference_evaluation(archive,tmp_path):
    path,fields = archive
    checkpoint = train_manifold_moe(path,tmp_path/"model.pt",manifold_epochs=1,expert_epochs=1,
                                    joint_epochs=1,model_options={"manifold_dim":3,"history_steps":3,
                                    "history_stride":1,"horizon_steps":3,"num_experts":2,
                                    "hidden_dim":12,"context_dim":4,"gate_hidden_dim":8,"expert_latent_dim":6},
                                    batch_size=4,window_stride=4,ensemble_size=2,integration_steps=1,
                                    max_validation_windows=2,device="cpu")
    a = torch.load(tmp_path/"model.manifold.pt",weights_only=False)
    b = torch.load(tmp_path/"model.specialize.pt",weights_only=False)
    c = torch.load(checkpoint,weights_only=False)
    for key,value in a["model"].items():
        if key.startswith(("manifold.","physics.","reference_encoder.","latent_","gate.centers","gate.radius")):
            torch.testing.assert_close(value,b["model"][key],atol=0,rtol=0)
    assert any(not torch.equal(v,c["model"][k]) for k,v in b["model"].items() if k.startswith("manifold.encoder."))
    torch.testing.assert_close(b["model"]["latent_scale"],c["model"]["latent_scale"],atol=0,rtol=0)
    assert c["training"]["train_split"] == "calibration"
    states,_,_ = load_moe_archive(path)
    end = c["training"]["normalization_span"][1]
    np.testing.assert_allclose(c["state_mean"],states[:end].mean(0))
    f = LatentFlowForecaster(checkpoint,device="cpu")
    assert f.is_manifold and f.model.stage == "joint"
    history = f.select_history(states)
    prediction = f.forecast(history,months=3,ensemble_size=3,integration_steps=1)
    assert prediction.shape == (3,3,32) and np.isfinite(prediction).all()
    np.testing.assert_equal(prediction[:,:1],f.forecast(history,months=1,ensemble_size=3,integration_steps=1))
    assert not np.allclose(prediction,f.forecast(history,months=3,ensemble_size=3,integration_steps=1,moe_mode="uniform"))
    with pytest.raises(ValueError,match="Stage A"):
        LatentFlowForecaster(tmp_path/"model.manifold.pt").forecast(history)
    with pytest.raises(ValueError,match="Manifold mode"):
        f.forecast(history,moe_mode="meta")
    forecast_main(["--checkpoint",str(checkpoint),"--archive",str(path),"--ensemble-size","2",
                   "--integration-steps","1","--device","cpu","--output",str(tmp_path/"forecast.npz")])
    out = FlowMatchingWeatherRunner(checkpoint,integration_steps=1,device="cpu").rollout(
        fields.isel(time=slice(-3,None)),horizon_hours=18)
    assert out.msl.shape == (3,2,4)
    report = evaluate_flow_checkpoint(checkpoint,path,tmp_path/"evaluation.json",ensemble_size=3,
                                      integration_steps=1,max_cases=2,device="cpu")
    result = json.loads(report.read_text())
    assert result["format"] == "climate_diffusion.manifold_moe_evaluation.v1"
    assert sum(result["rank_histogram_counts"]) == 2*3*32
    assert 0<=result["normalized_overall"]["coverage_80"]<=1
    standalone = train_manifold_moe(path,tmp_path/"b-again.pt",stage="specialize",
                                    init_checkpoint=tmp_path/"model.manifold.pt",expert_epochs=1,
                                    batch_size=4,window_stride=8,ensemble_size=2,integration_steps=1,
                                    max_validation_windows=2,device="cpu")
    assert LatentFlowForecaster(standalone).model.stage == "specialize"
    joint = train_manifold_moe(path,tmp_path/"c-again.pt",stage="joint",
                               init_checkpoint=standalone,joint_epochs=1,
                               batch_size=4,window_stride=8,ensemble_size=2,integration_steps=1,
                               max_validation_windows=2,device="cpu")
    assert LatentFlowForecaster(joint).model.stage == "joint"
    from climate_diffusion.manifold_diagnostics import diagnose_manifold
    for stage, saved in (("manifold",tmp_path/"model.manifold.pt"),("joint",joint)):
        diagnostic = diagnose_manifold(saved,path,tmp_path/f"diagnostic-{stage}.json",
                                        max_cases=2,members=2,integration_steps=1)
        values = json.loads(diagnostic.read_text())
        assert values["stage"] == stage and len(values["test_pca"]) == 2
        assert ("generated_audit" in values) == (stage == "joint")
    with pytest.raises(ValueError,match="requires a specialize-stage"):
        train_manifold_moe(path,tmp_path/"invalid.pt",stage="joint",init_checkpoint=tmp_path/"model.manifold.pt")
    manifest = checkpoint.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata["checkpoint_sha256"] = "0"*64
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="checksum mismatch"):
        LatentFlowForecaster(checkpoint)

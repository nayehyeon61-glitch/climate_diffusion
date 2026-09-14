import math
import pytest
import torch

from test_manifold_moe import archive

from climate_diffusion.joint_objective import (
    fair_crps, fair_energy, normalized_tendencies, profile,
    trajectory_scores, weighted_v2,
)


def brute_crps(x,y):
    m=x.shape[1]
    return ((x-y[:,None]).abs().mean(1)
            -.5*(x[:,:,None]-x[:,None,:]).abs().sum((1,2))/(m*(m-1))).mean()


def test_fair_crps_matches_bruteforce_and_permutation():
    torch.manual_seed(2)
    x=torch.randn(2,5,3,7,requires_grad=True)
    y=torch.randn(2,3,7)
    actual=fair_crps(x,y)
    assert torch.allclose(actual,brute_crps(x,y),atol=1e-6)
    assert torch.allclose(actual,fair_crps(x[:,torch.tensor([4,1,3,0,2])],y))
    actual.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_fair_crps_two_member_truth_bracket_is_zero():
    truth=torch.tensor([[0.]])
    members=torch.tensor([[[-1.],[1.]]],requires_grad=True)
    score=fair_crps(members,truth)
    assert score.item()==pytest.approx(0.,abs=1e-7)
    score.backward()
    assert members.grad is not None


def test_fair_scores_fail_fast_for_one_member():
    with pytest.raises(ValueError,match="M >= 2"):
        fair_crps(torch.zeros(1,1,2),torch.zeros(1,2))
    with pytest.raises(ValueError,match="M >= 2"):
        fair_energy(torch.zeros(1,1,2),torch.zeros(1,2))


def test_transition_uses_same_member_and_actual_dt():
    samples=torch.tensor([[[[0.],[2.],[6.]],[[0.],[-2.],[-6.]]]])
    truth=torch.tensor([[[0.],[1.],[3.]]])
    pred,obs=normalized_tendencies(samples,truth,torch.tensor([[2.,4.]]),torch.ones(1))
    assert torch.equal(pred,torch.tensor([[[[1.],[1.]],[[-1.],[-1.]]]]))
    assert torch.equal(obs,torch.tensor([[[.5],[.5]]]))


def test_full_trajectory_scores_backward_reaches_first_and_last_future():
    torch.manual_seed(4)
    x=torch.randn(2,4,21,6,requires_grad=True)
    y=torch.randn(2,21,6)
    values=trajectory_scores(x,y,torch.full((2,20),6.),torch.ones(6),torch.ones(6))
    for key in ("state_crps","transition_crps","trajectory_energy","mean_state","mean_tendency"):
        grad=torch.autograd.grad(values[key],x,retain_graph=True)[0]
        assert grad[:,:,1].abs().sum()>0
        assert grad[:,:,-1].abs().sum()>0


def test_profiles_disable_member_mse_and_are_finite():
    values={"fm":torch.tensor(2.),"expert_fm":torch.tensor(1.),
            "transition_crps":torch.tensor(.5)}
    total,weighted=weighted_v2(values,profile("v2_minimal"))
    assert total.item()==pytest.approx(3.125)
    assert "delta_member" not in weighted
    with pytest.raises(ValueError):
        profile("unknown")


def test_joint_ab_checkpoint_roundtrip_and_all_modules_update(archive,tmp_path):
    from climate_diffusion.train_manifold_moe import train_manifold_moe
    from climate_diffusion.inference import LatentFlowForecaster
    from climate_diffusion.joint_objective import JOINT_CHECKPOINT_FORMAT
    archive_path,_=archive
    options={"forecast_dynamics":"recurrent_residual","manifold_dim":3,
             "history_steps":3,"history_stride":1,"horizon_steps":3,
             "num_experts":2,"hidden_dim":12,"context_dim":4,
             "gate_hidden_dim":8,"expert_latent_dim":6}
    a=train_manifold_moe(archive_path,tmp_path/"a.pt",stage="manifold",
        model_options=options,manifold_epochs=1,batch_size=4,window_stride=8,
        ensemble_size=2,integration_steps=1,max_validation_windows=2,device="cpu",
        ae_delta_weight=.01,finite_step_drift_weight=.01)
    before=torch.load(a,weights_only=False)["model"]
    ab=train_manifold_moe(archive_path,tmp_path/"ab.pt",stage="joint_ab",
        init_checkpoint=a,joint_ab_epochs=1,batch_size=4,window_stride=8,
        ensemble_size=2,integration_steps=1,max_validation_windows=2,device="cpu",
        trajectory_edges=0,loss_profile="v2_minimal")
    payload=torch.load(ab,weights_only=False)
    assert payload["format"]==JOINT_CHECKPOINT_FORMAT
    assert payload["training"]["stage"]=="joint_ab"
    loaded=LatentFlowForecaster(ab,device="cpu")
    assert loaded.model.stage=="joint_ab"
    groups=("manifold.encoder","manifold.decoder","manifold.latent_drift",
            "experts","gate.correction","history_encoder")
    for prefix in groups:
        assert any(not torch.equal(value,before[name]) for name,value in payload["model"].items()
                   if name.startswith(prefix) and name in before)

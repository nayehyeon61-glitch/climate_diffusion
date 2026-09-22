"""A integration gates: physical input contract, useful gradients, and phase isolation."""
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_hybrid_pinn import synthetic_pinn_information
from smoke_moe import synthetic_archive
from climate_diffusion.hybrid_pinn import HybridPINNConfig
from climate_diffusion.information_process import FORMAT, InformationProcess
from climate_diffusion.manifold_moe import ManifoldMoEConfig
from climate_diffusion.moe_data import field_grid, load_moe_archive
from climate_diffusion.physical_information import digest
from climate_diffusion.train_information_process import (
    Windows, batch_loss, data_contract, load_checkpoint, write_json,
)


@pytest.fixture
def pinn_prepared(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    archive, _ = synthetic_archive(tmp_path, count=320)
    information = synthetic_pinn_information(archive, tmp_path)
    states, _, schema = load_moe_archive(archive)
    config = ManifoldMoEConfig(
        state_dim=states.shape[1], grid=field_grid(schema), horizon_steps=20,
        step_hours=6, history_steps=6, history_stride=1, manifold_dim=4,
        hidden_dim=24, context_dim=8, num_experts=2, expert_latent_dim=8,
        gate_hidden_dim=12, forecast_dynamics="recurrent_residual",
    )
    data = data_contract(archive, information, "enriched", config)
    model = InformationProcess(
        config, schema, data["mean"], data["scale"], data["statistics"],
        data["information_metadata"],
        pinn_config=HybridPINNConfig(warmup_epochs=1, ramp_epochs=2, weight=0.1),
        information_mean=data["information_mean"],
        information_scale=data["information_scale"],
    )
    normalized = torch.tensor((states[:data["train_end"]] - data["mean"]) / data["scale"])
    model.core.physics.fit(normalized)
    windows = Windows(
        states, data["times"], config, [0, 1], data["mean"], data["scale"],
        schema, information=data["information"],
    )
    batch = {key: torch.stack([windows[0][key], windows[1][key]]) for key in windows[0]}
    return model, batch, data, archive


def test_pinn_rejects_old_information_missing_colocated_fields(pinn_prepared, tmp_path):
    model, _, _, archive = pinn_prepared
    old = data_contract(archive, tmp_path / "information.npz", "enriched", model.config)
    with pytest.raises(ValueError, match="(?i)(missing|requires|required)"):
        InformationProcess(
            model.config, old["schema"], old["mean"], old["scale"], old["statistics"],
            old["information_metadata"], pinn_config=HybridPINNConfig(),
            information_mean=old["information_mean"], information_scale=old["information_scale"],
        )


def test_pinn_loss_reaches_A_encoder_decoder_and_physical_drift(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    values = model.pinn_losses(batch)
    assert all(torch.isfinite(value).all() for value in values.values())
    assert values["pinn_total"] > 0
    values["pinn_total"].backward()
    groups = {
        "encoder": model.core.manifold.encoder,
        "surface_decoder": model.core.manifold.decoder,
        "physical_drift": model.core.manifold.latent_drift,
        "information_encoder": model.information,
        "pressure_decoder": model.info_head,
        "closure": model.pinn,
    }
    for name, module in groups.items():
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert sum(gradient.abs().sum() for gradient in gradients) > 0, name
    assert all(parameter.grad is None for parameter in model.core.experts.parameters())
    assert all(parameter.grad is None for parameter in model.a_sampler.parameters())


def test_pinn_warmup_updates_closure_only_then_restores_A(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    model.set_pinn_warmup(True)
    assert all(name.startswith("pinn.") for name, parameter in model.named_parameters() if parameter.requires_grad)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.pinn_losses(batch, warmup=True)["pinn_total"].backward()
    optimizer.step()
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in before.items() if not name.startswith("pinn."))
    assert any(not torch.equal(value, model.state_dict()[name])
               for name, value in before.items() if name.startswith("pinn."))
    model.set_pinn_warmup(False)
    assert all(p.requires_grad for p in model.core.manifold.encoder.parameters())
    assert all(p.requires_grad for p in model.core.manifold.decoder.parameters())
    assert all(p.requires_grad for p in model.core.manifold.latent_drift.parameters())


def test_decoded_tendency_can_fit_without_closure_absorbing_errors(pinn_prepared):
    """Small-batch optimization gate, not a weather skill/generalization claim."""
    model, batch, _, _ = pinn_prepared
    model.pinn.requires_grad_(False)
    frozen = {name: value.clone() for name, value in model.pinn.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    initial = model.pinn_losses(batch)
    keys = ("pinn_tendency", "pinn_surface_tendency")
    initial = {key: float(initial[key].detach()) for key in keys}
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        values = model.pinn_losses(batch)
        (values["pinn_tendency"] + values["pinn_surface_tendency"]).backward()
        optimizer.step()
    final = model.pinn_losses(batch)
    for key in keys:
        assert float(final[key].detach()) < 0.95 * initial[key], key
    assert all(torch.equal(value, model.pinn.state_dict()[name]) for name, value in frozen.items())


@pytest.mark.parametrize("epoch,weight", [(1, 0.1), (2, 0.05), (7, 0.1)])
def test_pinn_warmup_and_ramped_joint_loss_are_fully_accounted(pinn_prepared, epoch, weight):
    model, batch, data, _ = pinn_prepared
    model.set_pinn_warmup(epoch == 1)
    args = SimpleNamespace(
        members=2, tau_steps=1, curriculum_interval=1, profile="process", loss_weights=None,
        info_scale=data["information_scale"], info_tendency_scale=data["information_tendency_scale"],
        b_member_weight=0.001,
    )
    streams = {name: torch.Generator().manual_seed(seed) for name, seed in (("fm", 11), ("ensemble", 29))}
    values = batch_loss(model, batch, args, epoch, streams)
    weighted_sum = sum(value for key, value in values.items() if key.startswith("weighted_"))
    assert torch.allclose(values["loss"], weighted_sum, rtol=1e-6, atol=1e-6)
    assert float(values["pinn_weight"]) == pytest.approx(weight)
    assert torch.allclose(values["weighted_pinn"], weight * values["pinn_total"])
    assert float(values["pinn_warmup"]) == float(epoch == 1)
    if epoch == 1:
        assert "weighted_reconstruction" not in values
    else:
        assert values["weighted_reconstruction"] > 0
        assert int(values["curriculum_phase"]) == (1 if epoch == 2 else 6)


@pytest.mark.parametrize("phase", ["B", "C"])
def test_later_phases_freeze_pinn_and_reject_A_physics_objective(pinn_prepared, phase):
    model, batch, data, _ = pinn_prepared
    model.seal(
        torch.tensor((data["states"][:data["train_end"]] - data["mean"]) / data["scale"]),
        torch.tensor(data["information"][:data["train_end"]]),
    )
    model.set_phase(phase)
    assert not any(p.requires_grad for p in model.pinn.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="A-only"):
        model.pinn_losses(batch)
    prediction = model.rollout(batch["history"], batch["information"], members=2, tau_steps=1, steps=2)
    prediction.square().mean().backward()
    assert all(p.grad is None for p in model.pinn.parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in before.items() if name.startswith("pinn."))
    if phase == "B":
        trainable = ("core.experts.", "core.gate.correction.", "core.history_encoder.")
        assert all(torch.equal(value, model.state_dict()[name])
                   for name, value in before.items() if not name.startswith(trainable))


def test_pinn_checkpoint_roundtrip_preserves_loss_and_prediction(pinn_prepared, tmp_path):
    model, batch, data, _ = pinn_prepared
    payload = {
        "format": FORMAT, "config": asdict(model.config), "schema": data["schema"],
        "mean": data["mean"], "scale": data["scale"], "statistics": data["statistics"],
        "information_metadata": data["information_metadata"],
        "information_mean": data["information_mean"], "information_scale": data["information_scale"],
        "pinn_config": asdict(model.pinn.config), "model": model.state_dict(), "stage": "A",
    }
    path = tmp_path / "hybrid-pinn.pt"
    torch.save(payload, path)
    write_json(path.with_suffix(".manifest.json"), {"checkpoint_sha256": digest(path)})
    restored, loaded = load_checkpoint(path)
    assert loaded["pinn_config"] == payload["pinn_config"]
    assert asdict(restored.pinn.config) == asdict(model.pinn.config)
    noise = torch.randn(2, 2, model.config.manifold_dim)
    for key, value in model.pinn_losses(batch).items():
        assert torch.equal(value, restored.pinn_losses(batch)[key]), key
    args = (batch["history"], batch["information"])
    assert torch.equal(
        model.rollout(*args, auxiliary=True, noise=noise, members=2, tau_steps=1, steps=2),
        restored.rollout(*args, auxiliary=True, noise=noise, members=2, tau_steps=1, steps=2),
    )

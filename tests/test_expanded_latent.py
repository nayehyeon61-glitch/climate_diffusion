"""Exercise the 64-coordinate A / 512-feature B model, not forecast skill."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

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
    Windows, data_contract, load_checkpoint, write_json,
)


def _model(config, data):
    return InformationProcess(
        config, data["schema"], data["mean"], data["scale"], data["statistics"],
        data["information_metadata"], pinn_config=HybridPINNConfig(),
        information_mean=data["information_mean"],
        information_scale=data["information_scale"],
    )


@pytest.fixture
def expanded(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    archive, _ = synthetic_archive(tmp_path, count=320)
    information = synthetic_pinn_information(archive, tmp_path)
    states, _, schema = load_moe_archive(archive)
    config = ManifoldMoEConfig(
        state_dim=states.shape[1], grid=field_grid(schema), horizon_steps=20,
        step_hours=6, history_steps=6, history_stride=1, manifold_dim=64,
        expert_latent_dim=512, hidden_dim=512, context_dim=64, num_experts=2,
        forecast_dynamics="recurrent_residual",
    )
    data = data_contract(archive, information, "enriched", config)
    model = _model(config, data)
    states = torch.tensor((data["states"][:data["train_end"]] - data["mean"]) / data["scale"])
    model.core.physics.fit(states)
    windows = Windows(
        data["states"], data["times"], config, [0, 1], data["mean"], data["scale"],
        schema, information=data["information"],
    )
    batch = {key: torch.stack([windows[0][key], windows[1][key]]) for key in windows[0]}
    try:
        yield model, batch, data, states
    finally:
        torch.set_num_threads(previous_threads)


def _assert_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_expanded_A_information_PINN_and_sampler_share_64_coordinates(expanded):
    model, batch, _, _ = expanded
    assert model.config.state_dim > 64  # Exercise compression, not an invalid tiny grid.
    surface = model.core.manifold.encode(batch["origin"])
    information = model.information(batch["information"])
    code = model.raw_encode(batch["origin"], batch["information"])
    assert surface.shape == information.shape == code.shape == (2, 64)
    torch.testing.assert_close(code, surface + information)
    assert model.info_head(code).shape == batch["information"].shape

    noise = torch.randn(2, 2, 64)
    path, coordinates = model.rollout(
        batch["history"], batch["information"], members=2, tau_steps=1, steps=2,
        auxiliary=True, noise=noise, return_q=True,
    )
    assert coordinates.shape == (2, 2, 3, 64)
    assert path.shape == (2, 2, 3, model.config.state_dim)
    torch.testing.assert_close(path[:, :, 0], batch["origin"][:, None].expand(-1, 2, -1))
    assert not torch.allclose(path[:, 0, 1:], path[:, 1, 1:])
    with pytest.raises(ValueError, match="Noise must"):
        model.rollout(batch["history"], batch["information"], members=2,
                      auxiliary=True, steps=1, noise=torch.randn(2, 2, 16))

    physics = model.pinn_losses(batch)
    fm = model.teacher_loss(batch, batch["information"], torch.Generator().manual_seed(8))["fm"]
    loss = physics["pinn_total"] + fm + path[:, :, 1:].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.core.manifold.encoder, model.core.manifold.decoder,
                   model.core.manifold.latent_drift, model.information,
                   model.info_head, model.pinn, model.a_sampler):
        _assert_gradients(module)
    assert all(parameter.grad is None for parameter in model.core.experts.parameters())


@pytest.mark.parametrize("phase", ["B", "C"])
def test_expanded_B_projection_and_C_backward_preserve_phase_contract(expanded, phase):
    model, batch, data, states = expanded
    model.seal(states, torch.tensor(data["information"][:data["train_end"]]))
    model.set_phase(phase)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    q = model.encode(batch["origin"], batch["information"])
    context = model.context(batch["history"], batch["information"])
    residual, tau, lead = torch.randn_like(q), torch.full((2,), 0.5), torch.zeros(2)
    seen = []

    def record_expert_code(module, inputs, output):
        seen.append((inputs[0].shape, output.shape))

    handle = model.core.experts[0].encoder.register_forward_hook(record_expert_code)
    try:
        field = model.core.field(residual, tau, context, lead, physical_q=q)
        geometry = model.core.prepare_field_geometry(q)
        cached = model.core.field(residual, tau, context, lead, physical_q=q, geometry=geometry)
    finally:
        handle.remove()
    condition_width = 64 + model.config.context_dim + 2 * model.config.time_embedding_dim
    assert seen == [((2, model.config.state_dim + condition_width), (2, 512))] * 2
    assert field["jacobian"].shape == (2, model.config.state_dim, 64)
    assert field["metric"].shape == (2, 64, 64)
    assert field["intrinsic_candidates"].shape == (2, 2, 64)
    assert field["raw_candidates"].shape == field["candidates"].shape == (2, 2, model.config.state_dim)
    torch.testing.assert_close(
        field["candidates"], field["intrinsic_candidates"] @ field["jacobian"].transpose(1, 2),
    )
    torch.testing.assert_close(field["velocity"], cached["velocity"], atol=2e-5, rtol=2e-4)

    path = model.rollout(batch["history"], batch["information"], members=2, tau_steps=1,
                         steps=2, noise=torch.randn(2, 2, 64))
    loss = path[:, :, 1:].square().mean() + field["velocity"].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.core.experts, model.core.gate, model.core.history_encoder):
        _assert_gradients(module)
    if phase == "C":
        for module in (model.core.manifold.encoder, model.core.manifold.decoder,
                       model.core.manifold.latent_drift, model.information):
            _assert_gradients(module)
    else:
        assert all(parameter.grad is None for parameter in model.core.manifold.parameters())
        assert all(parameter.grad is None for parameter in model.information.parameters())
    assert all(parameter.grad is None for parameter in model.pinn.parameters())
    assert all(parameter.grad is None for parameter in model.a_sampler.parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
    parameters = dict(model.named_parameters())
    assert all(torch.equal(before, parameters[name]) for name, before in frozen.items())


@pytest.mark.parametrize("legacy", [False, True], ids=["expanded64", "explicit_legacy16"])
def test_checkpoint_config_preserves_dimensions_and_forecasts(expanded, tmp_path, legacy):
    model, batch, data, _ = expanded
    if legacy:
        model = _model(replace(model.config, manifold_dim=16, expert_latent_dim=64, hidden_dim=128), data)
    payload = {
        "format": FORMAT, "config": asdict(model.config), "schema": data["schema"],
        "mean": data["mean"], "scale": data["scale"], "statistics": data["statistics"],
        "information_metadata": data["information_metadata"],
        "information_mean": data["information_mean"], "information_scale": data["information_scale"],
        "pinn_config": asdict(model.pinn.config), "model": model.state_dict(), "stage": "A",
    }
    path = tmp_path / "dimension-checkpoint.pt"
    torch.save(payload, path)
    write_json(path.with_suffix(".manifest.json"), {"checkpoint_sha256": digest(path)})
    restored, loaded = load_checkpoint(path)
    assert asdict(restored.config) == loaded["config"] == asdict(model.config)
    noise = torch.randn(2, 2, model.config.manifold_dim)
    expected = model.rollout(batch["history"], batch["information"], members=2,
                             auxiliary=True, noise=noise, tau_steps=1, steps=2)
    actual = restored.rollout(batch["history"], batch["information"], members=2,
                              auxiliary=True, noise=noise, tau_steps=1, steps=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

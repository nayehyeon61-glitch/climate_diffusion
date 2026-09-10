"""Regression tests for full-state, member-wise FLOW fusion, not endpoint fusion."""
import copy
import json
import types
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from scipy.fft import dct

from climate_diffusion.moe import FlowMatchingMoE, MoEConfig, OrthoDCT, FieldDCT, fair_energy_score
from climate_diffusion.moe_data import build_moe_split, load_moe_archive, validate_moe_split, align_moe_grid
from climate_diffusion.data import vectorize_dataset
from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.train_moe import train_moe
from climate_diffusion.train_moe import _pairs as training_pairs
from climate_diffusion.inference import LatentFlowForecaster, main as forecast_main
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


def config(**kwargs):
    return MoEConfig(state_dim=8, grid=(2, 2, 2), history_steps=3, history_stride=2,
                     horizon_steps=3, num_experts=3, hidden_dim=16, context_dim=8,
                     expert_latent_dim=4, meta_latent_dim=10, **kwargs)


@pytest.mark.parametrize("size", [1, 3, 8])
def test_dct_matches_scipy_roundtrip_and_gradient(size):
    transform = OrthoDCT(size)
    values = torch.randn(2, size, 4, dtype=torch.float64, requires_grad=True)
    spectral = transform(values, dim=1)
    np.testing.assert_allclose(spectral.detach(), dct(values.detach(), axis=1, norm="ortho"), atol=1e-12)
    recovered = transform(spectral, inverse=True, dim=1)
    torch.testing.assert_close(recovered, values)
    recovered.square().sum().backward()
    torch.testing.assert_close(values.grad, 2 * values)
    spatial = FieldDCT((2, 2, 3))
    x = torch.randn(2, 3, 12, dtype=torch.float64)
    torch.testing.assert_close(spatial(spatial(x), inverse=True), x)


def test_config_contract_and_simplexes():
    c = MoEConfig(state_dim=8, grid=(2, 2, 2))
    assert c.expert_latent_dim == 64 and c.meta_latent_dim == 160
    assert MoEConfig(**json.loads(json.dumps(asdict(c)))) == c
    model = FlowMatchingMoE(config())
    history = torch.randn(2, 3, 8)
    context = model.encode_history(history)
    for mode in ("experts", "meta", "uniform"):
        result = model.field(history[:, -1], torch.rand(2), context, torch.ones(2), mode=mode)
        for key in ("router", "alpha"):
            torch.testing.assert_close(result[key].sum(-1), torch.ones(2))
            assert (result[key] >= 0).all()
        assert result["candidates"].shape == (2, 3, 8)
        if mode == "meta":
            torch.testing.assert_close(result["velocity"], result["candidates"].mean(1))
    with pytest.raises(ValueError):
        MoEConfig(state_dim=8, grid=(2, 2, 3))


def test_shared_state_and_fusion_precedes_integration():
    model = FlowMatchingMoE(config())
    observed = [[] for _ in model.experts]
    factors = [0.1, 0.6, 1.1]
    for index, expert in enumerate(model.experts):
        def forward(self, state, condition, *, reconstruct=False, index=index):
            observed[index].append(state.detach().clone())
            return factors[index] * state, None
        expert.forward = types.MethodType(forward, expert)
    initial = torch.randn(2, 8)
    context = torch.randn(2, 8)
    prediction = model.integrate(initial, context, torch.ones(2), integration_steps=2, mode="uniform")
    for expert_inputs in observed[1:]:
        for a, b in zip(observed[0], expert_inputs):
            torch.testing.assert_close(a, b)
    a, step = np.mean(factors), 0.5
    expected = initial * (1 + step * a + 0.5 * (step * a) ** 2) ** 2
    torch.testing.assert_close(prediction, expected)
    endpoint_average = initial * np.mean([(1 + step * f + 0.5 * (step * f) ** 2) ** 2 for f in factors])
    assert not torch.allclose(prediction, endpoint_average, atol=1e-3)


def test_member_noise_and_reproducibility():
    model = FlowMatchingMoE(config()).eval()
    def zero_field(self, state, *args, **kwargs):
        return {"velocity": torch.zeros_like(state)}
    model.field = types.MethodType(zero_field, model)
    history = torch.randn(2, 3, 8)
    def predict(seed, leads):
        return model.forecast(history, ensemble_size=4, integration_steps=2,
                              lead_indices=leads, generator=torch.Generator().manual_seed(seed))
    full = predict(7, [0, 1, 2])
    assert full.shape == (2, 4, 3, 8)
    torch.testing.assert_close(full[:, :, 0], full[:, :, 2])
    torch.testing.assert_close(full[:, :, 1:2], predict(7, [1]))
    assert not torch.equal(full[:, 0], full[:, 1])
    assert not torch.equal(full, predict(8, [0, 1, 2]))
    with pytest.raises(ValueError, match="lead_indices"):
        predict(7, [3])


def test_training_adjacent_leads_reuse_member_source_noise():
    m = FlowMatchingMoE(config()).eval()
    batch = {"history": torch.randn(2, 3, 8), "targets": torch.randn(2, 3, 8)}
    _, velocity, _, _, lead, target = training_pairs(
        m, batch, torch.Generator().manual_seed(3), 3)
    source = (target - velocity).reshape(2, 3, 8)
    torch.testing.assert_close(source[:, :1].expand_as(source), source)
    torch.testing.assert_close(lead.reshape(2, 3).diff(dim=1), torch.full((2, 2), 1 / 3))


def test_frozen_experts_and_meta_gradient_finite_difference():
    torch.manual_seed(2)
    model = FlowMatchingMoE(config()).double()
    history = torch.randn(2, 3, 8, dtype=torch.float64)
    x = torch.randn(2, 8, dtype=torch.float64)
    tau = torch.rand(2, dtype=torch.float64)
    lead = torch.ones(2, dtype=torch.float64)
    loss = model.warmup_loss(x, torch.randn_like(x), tau, model.encode_history(history), lead)["loss"]
    loss.backward()
    assert all(any(p.grad is not None and p.grad.norm() > 0 for p in e.parameters()) for e in model.experts)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.router.parameters())
    model.zero_grad(set_to_none=True)
    model.set_stage("meta")
    model.train()
    assert not model.history_encoder.training and not model.experts.training
    frozen = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("meta.")}
    assert not any(p.requires_grad for p in model.experts.parameters())
    with torch.no_grad():
        model.meta.residual.weight.normal_(std=0.02)
    context = model.encode_history(history)
    def objective():
        return model.integrate(x, context, lead, integration_steps=3).square().mean()
    value = objective()
    value.backward()
    parameter = model.meta.residual.bias
    analytic = float(parameter.grad[0])
    epsilon = 1e-5
    with torch.no_grad():
        parameter[0] += epsilon
        plus = float(objective())
        parameter[0] -= 2 * epsilon
        minus = float(objective())
        parameter[0] += epsilon
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-4, abs=1e-6)
    torch.optim.Adam(model.meta.parameters(), lr=1e-3).step()
    for k, v in frozen.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    assert all(p.grad is None for p in model.experts.parameters())


def test_energy_score_and_member_permutation():
    target = torch.randn(2, 8)
    samples = target[:, None].repeat(1, 4, 1).requires_grad_()
    loss = fair_energy_score(samples, target)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(samples.grad).all()
    noisy = samples.detach() + torch.randn_like(samples)
    torch.testing.assert_close(fair_energy_score(noisy, target), fair_energy_score(noisy[:, [2, 1, 3, 0]], target))


@pytest.fixture
def archive(tmp_path):
    values = np.random.default_rng(4).normal(size=(80, 4, 2, 4)).astype(np.float32)
    times = pd.date_range("2000-01-01", periods=80, freq="6h")
    fields = xr.Dataset({name: (("time", "lat", "lon"), values[:, i])
                         for i, name in enumerate(("msl", "t2m", "u10", "v10"))},
                        coords={"time": times, "lat": [-45., 45.], "lon": [0., 90., 180., 270.]})
    path = tmp_path / "fields.nc"
    fields.to_netcdf(path, engine="scipy")
    output, _ = prepare_fixed_step_archive(path, tmp_path / "states.npz", step_hours=6,
                                          target_lat_points=2, target_lon_points=4)
    return output, fields


def test_missing_masks_fail_fast(archive):
    path, _ = archive
    load_moe_archive(path)
    with np.load(path) as data:
        payload = dict(data)
    payload["observed_mask"][2, 1] = 0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="fully observed"):
        load_moe_archive(path)
    del payload["observed_mask"]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="observed_mask"):
        load_moe_archive(path)


def test_five_way_temporal_contract():
    split = build_moe_split(100, 5)
    assert split["purge_windows"] == 4
    validate_moe_split(split, 5, 100)
    bad = copy.deepcopy(split)
    bad["calibration"][0] = bad["expert_validation"][-1] + 1
    with pytest.raises(ValueError, match="overlapping"):
        validate_moe_split(bad, 5, 100)


def test_two_stage_end_to_end(archive, tmp_path):
    path, fields = archive
    checkpoint = train_moe(path, tmp_path / "moe.pt", expert_epochs=1, meta_epochs=1,
                            model_options={"history_steps": 3, "history_stride": 2,
                                           "horizon_steps": 3, "hidden_dim": 16,
                                           "context_dim": 8, "num_experts": 2},
                            batch_size=4, window_stride=4, ensemble_size=2, integration_steps=1,
                            expert_leads=1, meta_leads=1, device="cpu")
    warm_path = tmp_path / "moe.experts.pt"
    warm = torch.load(warm_path, weights_only=False)
    final = torch.load(checkpoint, weights_only=False)
    warm_forecaster = LatentFlowForecaster(warm_path, device="cpu")
    with pytest.raises(ValueError, match="completed meta-stage"):
        warm_forecaster.forecast(np.zeros((3, 32)), moe_mode="meta")
    for key, value in warm["model"].items():
        if not key.startswith("meta."):
            torch.testing.assert_close(value, final["model"][key], rtol=0, atol=0)
    assert final["training"]["train_split"] == "calibration"
    split = final["training"]["split"]
    assert final["training"]["normalization_span"][1] <= split["expert_validation"][0] + 5
    f = LatentFlowForecaster(checkpoint, device="cpu")
    assert f.is_moe and f.model.stage == "meta"
    assert not any(p.requires_grad for p in f.model.parameters())
    states, _, _ = load_moe_archive(path)
    normalization_end = final["training"]["normalization_span"][1]
    np.testing.assert_allclose(final["state_mean"], states[:normalization_end].mean(0))
    np.testing.assert_allclose(final["state_scale"], states[:normalization_end].std(0))
    history = f.select_history(states)
    p = f.forecast(history, months=3, ensemble_size=3, integration_steps=2)
    assert p.shape == (3, 3, 32) and np.isfinite(p).all()
    np.testing.assert_equal(p, f.forecast(history, months=3, ensemble_size=3, integration_steps=2))
    raw = f.forecast(history, months=3, ensemble_size=3, integration_steps=2, moe_mode="experts")
    assert not np.allclose(p, raw)
    out = tmp_path / "forecast.npz"
    forecast_main(["--archive", str(path), "--checkpoint", str(checkpoint), "--output", str(out),
                   "--integration-steps", "1", "--ensemble-size", "2"])
    with np.load(out) as saved:
        assert saved["predictions"].shape == (2, 3, 32)
    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=1, device="cpu")
    forecast = runner.rollout(fields.isel(time=slice(-5, None)), horizon_hours=18)
    assert forecast.msl.shape == (3, 2, 4)
    missing = fields.isel(time=slice(-5, None)).copy(deep=True)
    missing.msl.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fully observed"):
        runner.rollout(missing, horizon_hours=18)
    report = evaluate_flow_checkpoint(checkpoint, path, tmp_path / "eval.json", ensemble_size=2,
                                      integration_steps=1, max_cases=2, device="cpu")
    result = json.loads(report.read_text())
    assert len(result["test_windows"]) == 2
    assert np.isfinite(result["normalized_overall"]["energy"])
    assert result["moe_mode"] == "meta"
    standalone = train_moe(path, tmp_path / "meta-again.pt", stage="meta", init_checkpoint=warm_path,
                            expert_epochs=1, meta_epochs=1, batch_size=4, window_stride=4,
                            ensemble_size=2, integration_steps=1, meta_leads=1, device="cpu")
    assert LatentFlowForecaster(standalone).model.stage == "meta"
    with pytest.raises(ValueError, match="requires an experts-stage"):
        train_moe(path, tmp_path / "invalid.pt", stage="meta", init_checkpoint=checkpoint)
    manifest = checkpoint.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata["checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum mismatch"):
        LatentFlowForecaster(checkpoint, device="cpu")


def test_archive_contract_tampering(archive):
    path, _ = archive
    with np.load(path) as data:
        payload = dict(data)
    payload["times"][1] = payload["times"][0]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="timestamps"):
        load_moe_archive(path)


def test_diversity_does_not_reward_unbounded_spread():
    model = FlowMatchingMoE(config())
    model.set_stage("meta")
    x = torch.randn(2, 8)
    context = model.encode_history(torch.randn(2, 3, 8))
    samples = torch.randn(2, 4, 8)
    args = (x, x, torch.ones(2) * 0.5, context, torch.ones(2))
    small = model.meta_loss(*args, samples=samples, target=x)
    enormous = model.meta_loss(*args, samples=samples * 1000, target=x)
    assert enormous["diversity"] > small["diversity"]
    assert enormous["loss"] > small["loss"]


def test_adapter_pooling_equals_archive_preparation(tmp_path):
    values = np.random.default_rng(42).normal(size=(3, 8, 16)).astype(np.float32)
    fields = xr.Dataset({"msl": (("time", "lat", "lon"), values)},
                         coords={"time": pd.date_range("2000-01-01", periods=3, freq="6h"),
                                 "lat": np.linspace(-80, 80, 8), "lon": np.arange(16) * 22.5})
    raw = tmp_path / "high-resolution.nc"
    fields.to_netcdf(raw, engine="scipy")
    path, _ = prepare_fixed_step_archive(raw, tmp_path / "coarse.npz", step_hours=6,
                                          target_lat_points=4, target_lon_points=8)
    states, _, schema = load_moe_archive(path)
    prepared = align_moe_grid(fields, schema)
    vectors = vectorize_dataset(prepared, schema, require_observed=True)
    np.testing.assert_allclose(vectors, states, atol=0, rtol=0)
    # Already-pooled input is accepted without applying another downsampling.
    np.testing.assert_allclose(vectorize_dataset(align_moe_grid(prepared, schema), schema), states)
    with pytest.raises(ValueError, match="grid differs"):
        align_moe_grid(fields.assign_coords(lon=fields.lon + 1), schema)

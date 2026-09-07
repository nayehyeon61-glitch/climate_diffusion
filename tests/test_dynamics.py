"""Gradient, temporal separation and deployed dynamics checkpoint regressions."""
from dataclasses import replace
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climate_diffusion.dynamics import DynamicsModelConfig, LatentDynamicsFlow
from climate_diffusion.data import load_monthly_archive
from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.train_dynamics import train_dynamics_model
from climate_diffusion.inference import LatentFlowForecaster, main as forecast_main
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


def test_adjoint_trajectory_gradient_reaches_gru_and_matches_direct_solver():
    torch.manual_seed(4)
    config = DynamicsModelConfig(
        state_dim=4, latent_dim=3, hidden_dim=8, history_steps=2,
        history_stride=2, horizon_steps=3, autoencoder_blocks=0,
        latent_normalization=False, dynamics_solver="dopri5",
        dynamics_rtol=1e-7, dynamics_atol=1e-8,
    )
    adjoint = LatentDynamicsFlow(config).double().eval()
    direct = LatentDynamicsFlow(replace(config, dynamics_adjoint=False)).double().eval()
    direct.load_state_dict(adjoint.state_dict())
    history = torch.randn(2, 2, 4, dtype=torch.float64)
    gradients = []
    for model in (adjoint, direct):
        trajectory, _ = model.rollout(history, history[:, -1])
        # Only trajectory loss: the flow head must not hide the missing ODE path.
        trajectory[:, 1:].square().mean().backward()
        gradient = model.history_encoder.encoder.weight_ih_l0.grad
        assert gradient is not None and gradient.norm() > 1e-8
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-4, atol=2e-6)


def test_rollout_uses_one_latent_scale_and_eval_does_not_update_it():
    model = LatentDynamicsFlow(DynamicsModelConfig(
        state_dim=4, latent_dim=3, hidden_dim=8, history_steps=2,
        horizon_steps=2, autoencoder_blocks=0, dynamics_adjoint=False,
    ))
    history = torch.randn(2, 2, 4)
    _, condition = model.rollout(history, history[:, -1])
    expected = model.history_encoder(model.encode_latent(history))
    torch.testing.assert_close(condition, expected)
    assert model.latent_scale_count == 1
    model.eval()
    scale = model.latent_scale.clone()
    model.forecast(history, history[:, -1], integration_steps=1)
    assert model.latent_scale_count == 1
    torch.testing.assert_close(model.latent_scale, scale)
    with pytest.raises(ValueError, match="lead_indices"):
        model.forecast(history, history[:, -1], lead_indices=[2])
    with pytest.raises(ValueError, match="positive"):
        model.forecast(history, history[:, -1], ensemble_size=0)


@pytest.fixture
def trained_dynamics(tmp_path):
    times = pd.date_range("2018-01-01", periods=40, freq="6h")
    fields = xr.Dataset(
        {"msl": (("time", "lat", "lon"),
                 (1000 + np.random.default_rng(2).normal(size=(40, 1, 2))).astype(np.float32))},
        coords={"time": times, "lat": [30.0], "lon": [120.0, 140.0]},
    )
    path = tmp_path / "fields.nc"
    fields.to_netcdf(path, engine="scipy")
    archive, _ = prepare_fixed_step_archive(
        path, tmp_path / "fixed.npz", step_hours=6,
        target_lat_points=1, target_lon_points=2,
    )
    checkpoint = train_dynamics_model(
        archive, tmp_path / "dynamics.pt", history_steps=3, history_stride=2,
        horizon_steps=3, latent_dim=3, hidden_dim=8, autoencoder_hidden_dim=8,
        autoencoder_blocks=0, epochs=1, batch_size=4, window_stride=2,
        purge_windows=0, ensemble_size=2, ensemble_weight=0.1, ensemble_steps=1,
    )
    return fields, archive, checkpoint


def test_dynamics_checkpoint_forecast_adapter_evaluation_and_split(trained_dynamics, tmp_path):
    fields, archive, checkpoint = trained_dynamics
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    split = payload["training"]["split"]
    span, horizon = 5, 3
    assert split["purge_windows"] == horizon - 1
    for left, right in (("train", "validation"), ("validation", "test")):
        last_target = max(split[left]) + span + horizon - 1
        next_first_target = min(split[right]) + span
        assert last_target < next_first_target
    assert payload["training"]["normalization_span"][1] <= min(split["validation"]) + span

    states, times, schema = load_monthly_archive(archive)
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert not any(p.requires_grad for p in forecaster.model.parameters())
    history = forecaster.select_history(states)
    np.testing.assert_equal(history, states[[-5, -3, -1]])
    kwargs = dict(months=3, ensemble_size=2, integration_steps=2, seed=9)
    samples = forecaster.forecast(history, **kwargs)
    assert samples.shape == (2, 3, 2) and np.isfinite(samples).all()
    np.testing.assert_equal(samples, forecaster.forecast(history, **kwargs))
    with pytest.raises(ValueError, match="horizon"):
        forecaster.forecast(history, months=4)
    with pytest.raises(ValueError, match="consecutive"):
        forecaster.select_history(states[-3:])
    wrong_schema = copy.deepcopy(schema)
    wrong_schema["variables"][0]["name"] = "wrong"
    with pytest.raises(ValueError, match="schema mismatch"):
        forecaster.validate_archive(wrong_schema, times)
    bad_times = times.copy().astype("datetime64[m]")
    bad_times[1] += np.timedelta64(30, "m")
    with pytest.raises(ValueError, match="timestamps"):
        forecaster.validate_archive(schema, bad_times)

    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=2, device="cpu")
    forecast = runner.rollout(fields.isel(time=slice(-5, None)), horizon_hours=18)
    assert forecast.msl.shape == (3, 1, 2)
    np.testing.assert_equal(forecast.time.values,
                            times[-1] + np.arange(1, 4) * np.timedelta64(6, "h"))
    output = tmp_path / "forecast.npz"
    forecast_main(["--checkpoint", str(checkpoint), "--archive", str(archive),
                   "--integration-steps", "1", "--output", str(output)])
    with np.load(output) as result:
        assert result["predictions"].shape == (1, 3, 2)
        np.testing.assert_equal(result["lead_hours"], [6, 12, 18])
    metrics_path = evaluate_flow_checkpoint(
        checkpoint, archive, tmp_path / "eval.json",
        ensemble_size=2, integration_steps=2, device="cpu",
    )
    metrics = json.loads(metrics_path.read_text())
    assert len(metrics["by_lead_normalized"]) == 3
    assert np.isfinite(metrics["normalized_overall"]["crps"])
    assert np.isfinite(metrics["by_variable_raw_units"]["msl"]["rmse"])

    # Legacy artifacts remain loadable, but contaminated split scores are refused.
    payload["training"]["split"]["validation"][0] = max(split["train"]) + 1
    old = tmp_path / "unsafe.pt"
    torch.save(payload, old)
    with pytest.raises(ValueError, match="overlapping future targets"):
        evaluate_flow_checkpoint(old, archive, tmp_path / "unsafe.json", device="cpu")

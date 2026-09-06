import json

import numpy as np
import pandas as pd
import torch
import xarray as xr

from climate_diffusion.data import prepare_monthly_archive
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.train import train_flow_model
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


def test_one_epoch_checkpoint_runs_as_monthly_weather_replacement(tmp_path):
    times = pd.date_range("2018-01-01", periods=10, freq="MS")
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                np.linspace(990, 1020, 20, dtype=np.float32).reshape(10, 1, 2),
            )
        },
        coords={"time": times, "lat": [30.0], "lon": [120.0, 140.0]},
    )
    fields_path = tmp_path / "fields.nc"
    fields.to_netcdf(fields_path, engine="scipy")
    archive, _ = prepare_monthly_archive(
        fields_path,
        tmp_path / "monthly.npz",
        target_lat_points=1,
        target_lon_points=2,
    )
    checkpoint = train_flow_model(
        archive,
        tmp_path / "flow.pt",
        history_months=3,
        latent_dim=2,
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        test_fraction=0.15,
    )
    manifest = json.loads(checkpoint.with_suffix(".manifest.json").read_text())
    assert manifest["checkpoint_sha256"]
    assert manifest["split"]["test"]
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert all(not parameter.requires_grad for parameter in forecaster.model.parameters())
    runner = FlowMatchingWeatherRunner(
        checkpoint, integration_steps=2, device="cpu"
    )
    forecast = runner.rollout(fields.isel(time=slice(-3, None)), horizon_hours=720)
    assert forecast.sizes["time"] == 1
    assert forecast["msl"].shape == (1, 1, 2)
    assert forecast.attrs["forecast_backend"] == "flow_matching"
    assert forecast.attrs["forecast_checkpoint_kind"] == "flow_matching"
    assert forecast.attrs["weather_next_replacement"] is True

    metrics_path = evaluate_flow_checkpoint(
        checkpoint,
        archive,
        tmp_path / "evaluation.json",
        ensemble_size=2,
        integration_steps=2,
        device="cpu",
    )
    metrics = json.loads(metrics_path.read_text())
    assert metrics["test_windows"] == manifest["split"]["test"]
    assert np.isfinite(metrics["normalized_overall"]["crps"])


def test_recency_weighting_and_expanded_autoencoder_are_recorded(tmp_path):
    times = pd.date_range("2015-01-01", periods=40, freq="MS")
    rng = np.random.default_rng(0)
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                (1000 + rng.normal(size=(40, 1, 2))).astype(np.float32),
            )
        },
        coords={"time": times, "lat": [30.0], "lon": [120.0, 140.0]},
    )
    fields_path = tmp_path / "fields.nc"
    fields.to_netcdf(fields_path, engine="scipy")
    archive, _ = prepare_monthly_archive(
        fields_path,
        tmp_path / "monthly.npz",
        target_lat_points=1,
        target_lon_points=2,
    )
    checkpoint = train_flow_model(
        archive,
        tmp_path / "flow.pt",
        history_months=3,
        latent_dim=4,
        hidden_dim=8,
        autoencoder_hidden_dim=16,
        autoencoder_blocks=2,
        autoencoder_dropout=0.1,
        epochs=1,
        batch_size=4,
        validation_fraction=0.25,
        test_fraction=0.15,
        recency_halflife=5.0,
        normalization_states=12,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["model_config"]["autoencoder_blocks"] == 2
    assert payload["model_config"]["autoencoder_hidden_dim"] == 16
    assert payload["training"]["recency_halflife"] == 5.0
    start, end = payload["training"]["normalization_span"]
    assert end - start == 12

    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text())
    assert metadata["autoencoder_parameter_count"] < metadata["parameter_count"]

    # A frozen checkpoint of the expanded model still loads and samples.
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert forecaster.config.autoencoder_blocks == 2
    prediction = forecaster.forecast(
        np.zeros((3, forecaster.config.state_dim), dtype=np.float32),
        integration_steps=2,
    )
    assert np.isfinite(prediction).all()

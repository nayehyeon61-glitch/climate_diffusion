import json

import numpy as np
import pandas as pd
import xarray as xr
import torch
import pytest

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import prepare_monthly_archive
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.train import train_flow_model
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner
from climate_diffusion.model import MonthlyLatentFlow


def test_one_epoch_checkpoint_runs_as_monthly_weather_replacement(tmp_path):
    times = pd.date_range("2018-01-01", periods=18, freq="MS")
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                    np.linspace(990, 1020, 36, dtype=np.float32).reshape(18, 1, 2),
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

    schema_path = archive.with_suffix(".schema.json")
    schema = json.loads(schema_path.read_text())
    schema["target_lon_points"] = 999
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="contract does not match checkpoint"):
        evaluate_flow_checkpoint(
            checkpoint,
            archive,
            tmp_path / "mismatched-evaluation.json",
            ensemble_size=1,
            integration_steps=1,
            device="cpu",
        )


def test_v2_vector_checkpoint_remains_loadable(tmp_path):
    config = FlowModelConfig(state_dim=4, history_months=2, latent_dim=2, hidden_dim=8)
    model = MonthlyLatentFlow(config)
    checkpoint = tmp_path / "legacy-v2.pt"
    torch.save(
        {
            "format": "climate_diffusion.monthly_latent_flow.v2",
            "model": model.state_dict(),
            "model_config": {
                "state_dim": 4,
                "history_months": 2,
                "latent_dim": 2,
                "hidden_dim": 8,
                "time_embedding_dim": 32,
            },
            "state_mean": torch.zeros(4),
            "state_scale": torch.ones(4),
            "schema": {"format": "climate_diffusion.monthly_state.v1"},
        },
        checkpoint,
    )
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert forecaster.config.backend == "vector_mlp"
    assert all(not parameter.requires_grad for parameter in forecaster.model.parameters())


def test_clean_spatial_archive_trains_reloads_and_evaluates(tmp_path):
    times = pd.date_range("2015-01-01", periods=18, freq="MS")
    values = np.linspace(-1.0, 1.0, 18 * 4 * 8, dtype=np.float32).reshape(
        18, 4, 8
    )
    fields = xr.Dataset(
        {"msl": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": np.linspace(-90.0, 90.0, 4),
            "lon": np.arange(8) * 45.0,
        },
    )
    source = tmp_path / "spatial.nc"
    fields.to_netcdf(source, engine="scipy")
    archive, _ = prepare_monthly_archive(
        source,
        tmp_path / "spatial-archive",
        layout="spatial",
        target_lat_points=4,
        target_lon_points=8,
    )
    checkpoint = train_flow_model(
        archive,
        tmp_path / "spatial.pt",
        history_months=3,
        latent_dim=2,
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        test_fraction=0.15,
        model_backend="spatial_conv",
        spatial_base_channels=2,
        spatial_latent_channels=2,
        spatial_downsample_levels=1,
        operator_modes_lat=1,
        operator_modes_lon=2,
        patch_height=4,
        patch_width=8,
        tile_overlap=1,
    )
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    prediction = forecaster.forecast(values[-3:, None], integration_steps=1)
    assert prediction.shape == (1, 1, 1, 4, 8)
    assert np.isfinite(prediction).all()
    metrics = evaluate_flow_checkpoint(
        checkpoint,
        archive,
        tmp_path / "spatial-evaluation.json",
        ensemble_size=1,
        integration_steps=1,
        device="cpu",
    )
    assert np.isfinite(json.loads(metrics.read_text())["normalized_overall"]["rmse"])

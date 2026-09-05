import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import (
    MonthlyWindowDataset,
    load_monthly_archive,
    prepare_monthly_archive,
)
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.train import (
    forecast_skill,
    patch_latitude_weights,
    train_flow_model,
)


def _seasonal_archive(tmp_path, months=36, height=8, width=12):
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(0)
    season = np.sin(2 * np.pi * np.arange(months) / 12)[:, None, None]
    values = (season + 0.2 * rng.normal(size=(months, height, width))).astype(np.float32)
    dataset = xr.Dataset(
        {"msl": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": np.linspace(-88.0, 88.0, height),
            "lon": np.arange(width) * (360.0 / width),
        },
    )
    source = tmp_path / "fields.nc"
    dataset.to_netcdf(source, engine="scipy")
    archive, _ = prepare_monthly_archive(
        source,
        tmp_path / "archive",
        layout="spatial",
        target_lat_points=height,
        target_lon_points=width,
    )
    return archive


def test_patch_latitude_weights_track_the_centre_crop():
    schema = {"layout": "spatial", "coords": {"lat": [80.0, 60.0, 0.0, -60.0, -80.0]}}
    full = patch_latitude_weights(schema, None)
    assert full.shape == (1, 5, 1)
    # A 3-row centre crop drops the outermost row on each side.
    cropped = patch_latitude_weights(schema, (3, 4))
    assert cropped.shape == (1, 3, 1)
    np.testing.assert_allclose(cropped.reshape(-1), full.reshape(-1)[1:4])
    assert patch_latitude_weights({"layout": "vector"}, None) is None


def test_forecast_skill_reports_model_against_persistence(tmp_path):
    archive = _seasonal_archive(tmp_path)
    states, times, schema = load_monthly_archive(archive)
    dataset = MonthlyWindowDataset(
        states,
        3,
        1,
        mean=states.mean(axis=0, keepdims=True).squeeze(0),
        scale=np.ones_like(states[0]),
    )
    config = FlowModelConfig(
        state_dim=int(schema["state_dim"]),
        history_months=3,
        backend="spatial_conv",
        spatial_channels=1,
        grid_height=8,
        grid_width=12,
        spatial_latent_channels=2,
        spatial_base_channels=2,
        spatial_downsample_levels=1,
    )
    model = MonthlyLatentFlow(config)
    metrics = forecast_skill(
        model,
        dataset,
        torch.device("cpu"),
        windows=3,
        ensemble_size=2,
        integration_steps=2,
        seed=0,
        weights=patch_latitude_weights(schema, None),
    )
    assert metrics["windows"] == 3.0
    assert metrics["forecast_rmse"] > 0
    assert metrics["persistence_rmse"] > 0
    assert metrics["skill_vs_persistence"] == pytest.approx(
        1.0 - metrics["forecast_rmse"] / metrics["persistence_rmse"]
    )


def test_forecast_skill_is_reproducible_for_a_fixed_seed(tmp_path):
    archive = _seasonal_archive(tmp_path)
    states, _, schema = load_monthly_archive(archive)
    dataset = MonthlyWindowDataset(
        states, 3, 1, mean=states.mean(axis=0), scale=np.ones_like(states[0])
    )
    model = MonthlyLatentFlow(
        FlowModelConfig(
            state_dim=int(schema["state_dim"]),
            history_months=3,
            backend="spatial_conv",
            spatial_channels=1,
            grid_height=8,
            grid_width=12,
            spatial_latent_channels=2,
            spatial_base_channels=2,
            spatial_downsample_levels=1,
        )
    )

    def run(seed):
        return forecast_skill(
            model,
            dataset,
            torch.device("cpu"),
            windows=2,
            ensemble_size=2,
            integration_steps=2,
            seed=seed,
        )["forecast_rmse"]

    assert run(0) == run(0)
    assert run(0) != run(5)


def test_forecast_skill_leaves_the_model_in_training_mode(tmp_path):
    archive = _seasonal_archive(tmp_path)
    states, _, schema = load_monthly_archive(archive)
    dataset = MonthlyWindowDataset(
        states, 3, 1, mean=states.mean(axis=0), scale=np.ones_like(states[0])
    )
    model = MonthlyLatentFlow(
        FlowModelConfig(
            state_dim=int(schema["state_dim"]),
            history_months=3,
            backend="spatial_conv",
            spatial_channels=1,
            grid_height=8,
            grid_width=12,
            spatial_latent_channels=2,
            spatial_base_channels=2,
            spatial_downsample_levels=1,
        )
    )
    model.train()
    forecast_skill(
        model,
        dataset,
        torch.device("cpu"),
        windows=1,
        ensemble_size=1,
        integration_steps=1,
        seed=0,
    )
    assert model.training


def _train(tmp_path, archive, **kwargs):
    return train_flow_model(
        archive,
        tmp_path / f"model-{kwargs.get('skill_every_epochs', 1)}.pt",
        history_months=3,
        epochs=2,
        batch_size=2,
        validation_fraction=0.2,
        test_fraction=0.2,
        purge_windows=1,
        model_backend="spatial_conv",
        spatial_base_channels=2,
        spatial_latent_channels=2,
        spatial_downsample_levels=1,
        patch_height=8,
        patch_width=12,
        tile_overlap=1,
        seed=7,
        skill_windows=2,
        skill_ensemble_size=2,
        skill_integration_steps=2,
        **kwargs,
    )


def test_training_records_forecast_skill_and_selects_on_it(tmp_path):
    archive = _seasonal_archive(tmp_path)
    checkpoint = _train(tmp_path, archive, skill_every_epochs=1)
    history = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
    assert len(history) == 2
    for record in history:
        skill = record["forecast_skill"]
        assert skill["forecast_rmse"] > 0
        assert skill["persistence_rmse"] > 0
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training"]["selection_metric"] == "forecast_rmse"
    # best_validation_loss now holds whatever the selection metric measured.
    chosen = min(record["forecast_skill"]["forecast_rmse"] for record in history)
    assert payload["training"]["best_validation_loss"] == pytest.approx(chosen)


def test_disabling_the_skill_metric_keeps_validation_loss_selection(tmp_path):
    archive = _seasonal_archive(tmp_path)
    checkpoint = _train(tmp_path, archive, skill_every_epochs=0)
    history = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
    assert all("forecast_skill" not in record for record in history)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training"]["selection_metric"] == "loss"
    chosen = min(record["validation"]["loss"] for record in history)
    assert payload["training"]["best_validation_loss"] == pytest.approx(chosen)

import json

import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.fixed_step_train import train_runpod_fixed_step_flow
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


def _train_fixed(archive, root, *, history_steps=3, batch_size=2):
    return train_runpod_fixed_step_flow(
        archive,
        root,
        history_steps=history_steps,
        latent_dim=2,
        hidden_dim=8,
        epochs=1,
        batch_size=batch_size,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_windows=1,
    )


def test_fixed_step_360h_checkpoint_rolls_out_exact_day15_endpoint(tmp_path):
    times = pd.date_range("2018-01-01", periods=12, freq="360h")
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                np.linspace(990, 1025, 24, dtype=np.float32).reshape(12, 1, 2),
            )
        },
        coords={"time": times, "lat": [30.0], "lon": [120.0, 140.0]},
    )
    fields_path = tmp_path / "fields.nc"
    fields.to_netcdf(fields_path, engine="scipy")
    archive, schema_path = prepare_fixed_step_archive(
        fields_path,
        tmp_path / "fixed360.npz",
        step_hours=360,
        target_lat_points=1,
        target_lon_points=2,
    )
    schema = json.loads(schema_path.read_text())
    assert schema["forecast_step_hours"] == 360
    assert schema["aggregation"] == "exact_fixed_step_snapshot"

    checkpoint = _train_fixed(archive, tmp_path / "flow360")
    manifest = json.loads((checkpoint.parent / "best.manifest.json").read_text())
    assert manifest["forecast_step_hours"] == 360

    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert forecaster.forecast_step_hours == 360
    assert all(not parameter.requires_grad for parameter in forecaster.model.parameters())

    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=2, device="cpu")
    initial = fields.isel(time=slice(-3, None))
    init_time = pd.Timestamp(initial.time.values[-1])
    forecast = runner.rollout(initial, horizon_hours=360)
    assert forecast.sizes["time"] == 1
    assert pd.Timestamp(forecast.time.values[0]) - init_time == pd.Timedelta(hours=360)
    assert forecast.attrs["forecast_step_hours"] == 360
    assert forecast.attrs["forecast_horizon_hours"] == 360
    assert forecast.attrs["parameters_frozen"] is True


def test_24h_flow_checkpoint_produces_full_0_to_360h_trajectory(tmp_path):
    # A production run may use 6h for WeatherNext parity. The tiny 24h test keeps
    # CI fast while verifying the same multi-step trajectory contract.
    times = pd.date_range("2018-01-01", periods=30, freq="24h")
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                np.linspace(990, 1020, 60, dtype=np.float32).reshape(30, 1, 2),
            )
        },
        coords={"time": times, "lat": [25.0], "lon": [125.0, 135.0]},
    )
    path = tmp_path / "daily.nc"
    fields.to_netcdf(path, engine="scipy")
    archive, _ = prepare_fixed_step_archive(
        path, tmp_path / "daily.npz", step_hours=24,
        target_lat_points=1, target_lon_points=2,
    )
    checkpoint = _train_fixed(archive, tmp_path / "flow24", batch_size=4)
    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=1, device="cpu")
    initial = fields.isel(time=slice(-3, None))
    init_time = pd.Timestamp(initial.time.values[-1])
    forecast = runner.rollout(initial, horizon_hours=360)

    assert forecast.sizes["time"] == 15
    leads = np.asarray([
        (pd.Timestamp(value) - init_time).total_seconds() / 3600.0
        for value in forecast.time.values
    ])
    assert np.array_equal(leads, np.arange(24, 361, 24, dtype=float))
    assert forecast.attrs["forecast_step_hours"] == 24
    assert forecast.attrs["forecast_horizon_hours"] == 360


def test_360h_checkpoint_rejects_incompatible_fraction(tmp_path):
    times = pd.date_range("2018-01-01", periods=12, freq="360h")
    fields = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.ones((12, 1, 1), dtype=np.float32))},
        coords={"time": times, "lat": [30.0], "lon": [130.0]},
    )
    fields_path = tmp_path / "fields.nc"
    fields.to_netcdf(fields_path, engine="scipy")
    archive, _ = prepare_fixed_step_archive(
        fields_path, tmp_path / "fixed.npz", step_hours=360,
        target_lat_points=1, target_lon_points=1,
    )
    checkpoint = _train_fixed(archive, tmp_path / "flow")
    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=2, device="cpu")
    try:
        runner.rollout(fields.isel(time=slice(-3, None)), horizon_hours=540)
    except ValueError as exc:
        assert "360-hour multiples" in str(exc)
    else:
        raise AssertionError("Expected an incompatible 540h horizon to be rejected")

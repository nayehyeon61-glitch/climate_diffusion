import json

import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.train import train_flow_model
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


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

    checkpoint = train_flow_model(
        archive,
        tmp_path / "flow360.pt",
        history_months=3,
        latent_dim=2,
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        validation_fraction=0.2,
        test_fraction=0.1,
        purge_windows=1,
    )
    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text())
    assert metadata["forecast_step_hours"] == 360
    assert metadata["checkpoint_kind"] == "fixed_step_latent_flow_matching"

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


def test_360h_checkpoint_rejects_720_incompatible_fraction(tmp_path):
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
    checkpoint = train_flow_model(
        archive, tmp_path / "flow.pt", history_months=3, latent_dim=2,
        hidden_dim=8, epochs=1, batch_size=2,
        validation_fraction=0.2, test_fraction=0.1, purge_windows=1,
    )
    runner = FlowMatchingWeatherRunner(checkpoint, integration_steps=2, device="cpu")
    try:
        runner.rollout(fields.isel(time=slice(-3, None)), horizon_hours=540)
    except ValueError as exc:
        assert "360-hour multiples" in str(exc)
    else:
        raise AssertionError("Expected an incompatible 540h horizon to be rejected")

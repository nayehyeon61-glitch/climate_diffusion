import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.data import prepare_monthly_archive
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
    )
    runner = FlowMatchingWeatherRunner(
        checkpoint, integration_steps=2, device="cpu"
    )
    forecast = runner.rollout(fields.isel(time=slice(-3, None)), horizon_hours=720)
    assert forecast.sizes["time"] == 1
    assert forecast["msl"].shape == (1, 1, 2)
    assert forecast.attrs["forecast_backend"] == "flow_matching"
    assert forecast.attrs["weather_next_replacement"] is True

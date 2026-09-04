import json

import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.era5 import open_era5_dataset, prepare_era5_archive
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.train import train_flow_model


def _write_split_era5(directory, periods=30):
    times = pd.date_range("2018-01-01", periods=periods, freq="MS")
    latitude = [90.0, 30.0, -30.0, -90.0]
    longitude = [-180.0, -90.0, 0.0, 90.0]
    surface = np.linspace(98_000, 103_000, periods * 4 * 4, dtype=np.float32).reshape(
        periods, 4, 4
    )
    pressure = np.linspace(240, 290, periods * 2 * 4 * 4, dtype=np.float32).reshape(
        periods, 2, 4, 4
    )
    xr.Dataset(
        {"msl": (("valid_time", "latitude", "longitude"), surface)},
        coords={"valid_time": times, "latitude": latitude, "longitude": longitude},
    ).to_netcdf(directory / "surface.nc", engine="scipy")
    xr.Dataset(
        {
            "t": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                pressure,
            )
        },
        coords={
            "valid_time": times,
            "pressure_level": [500, 850],
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(directory / "pressure.nc", engine="scipy")


def test_split_era5_is_canonicalised_without_changing_archive_contract(tmp_path):
    _write_split_era5(tmp_path)
    with open_era5_dataset(tmp_path) as (dataset, files):
        assert len(files) == 2
        assert set(dataset.data_vars) == {"msl", "t"}
        assert dataset.t.dims == ("time", "level", "lat", "lon")
        assert dataset.lon.values.tolist() == [0.0, 90.0, 180.0, 270.0]
        assert dataset.lat.values.tolist() == [90.0, 30.0, -30.0, -90.0]

    archive, schema_path = prepare_era5_archive(
        tmp_path,
        tmp_path / "archive",
        variables=("msl", "t"),
        target_lat_points=4,
        target_lon_points=4,
    )
    schema = json.loads(schema_path.read_text())
    assert schema["source_metadata"]["adapter"] == "era5.v2"
    assert schema["layout"] == "spatial"
    assert schema["spatial_channels"] == 3
    assert schema["grid_shape"] == [4, 4]
    assert archive == tmp_path / "archive"


def test_era5_archive_smoke_trains_and_evaluates_real_data_path(tmp_path):
    source = tmp_path / "era5"
    source.mkdir()
    _write_split_era5(source)
    archive, _ = prepare_era5_archive(
        source,
        tmp_path / "archive",
        variables=("msl", "t"),
        target_lat_points=4,
        target_lon_points=4,
    )
    checkpoint = train_flow_model(
        archive,
        tmp_path / "era5-smoke.pt",
        history_months=3,
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        test_fraction=0.2,
        purge_windows=0,
        model_backend="spatial_conv",
        spatial_base_channels=2,
        spatial_latent_channels=2,
        spatial_downsample_levels=1,
        operator_modes_lat=1,
        operator_modes_lon=1,
        patch_height=4,
        patch_width=4,
        tile_overlap=1,
    )
    metrics_path = evaluate_flow_checkpoint(
        checkpoint,
        archive,
        tmp_path / "era5-evaluation.json",
        ensemble_size=1,
        integration_steps=1,
        device="cpu",
    )
    metrics = json.loads(metrics_path.read_text())
    assert set(metrics["by_variable_raw_units"]) == {"msl", "t"}
    assert np.isfinite(metrics["normalized_overall"]["rmse"])

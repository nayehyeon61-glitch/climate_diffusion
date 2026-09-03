import json

import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.hres import open_hres_dataset, prepare_hres_archive


def _write_hres(path, *, periods=18):
    init = pd.date_range("2020-01-01", periods=periods, freq="MS")
    step = np.asarray([0, 6], dtype=np.int32)
    values = np.arange(periods * 2 * 5 * 8, dtype=np.float32).reshape(periods, 2, 5, 8)
    dataset = xr.Dataset(
        {"msl": (("forecast_reference_time", "step", "latitude", "longitude"), values)},
        coords={
            "forecast_reference_time": init,
            "step": ("step", step, {"units": "hours"}),
            "latitude": np.linspace(-90.0, 90.0, 5),
            "longitude": np.arange(8) * 45.0,
        },
    )
    dataset.to_netcdf(path, engine="scipy")


def test_hres_selects_one_lead_and_regrids_to_archive_contract(tmp_path):
    source = tmp_path / "hres.nc"
    _write_hres(source)
    with open_hres_dataset(
        source, lead_hours=6, target_lat_points=5, target_lon_points=8
    ) as (dataset, _):
        assert "step" not in dataset.dims
        assert dataset.msl.dims == ("time", "lat", "lon")
        assert pd.Timestamp(dataset.time.values[0]) == pd.Timestamp("2020-01-01 06:00")
        assert dataset.lat.values.tolist() == [90.0, 45.0, 0.0, -45.0, -90.0]

    archive, schema_path = prepare_hres_archive(
        source,
        tmp_path / "archive",
        variables=("msl",),
        lead_hours=6,
        target_lat_points=5,
        target_lon_points=8,
    )
    schema = json.loads(schema_path.read_text())
    assert archive == tmp_path / "archive"
    assert schema["source_metadata"]["adapter"] == "hres.v1"
    assert schema["source_metadata"]["selected_lead_hours"] == 6
    assert schema["grid_shape"] == [5, 8]

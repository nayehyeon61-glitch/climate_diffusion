import json

import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.data import (
    aggregate_monthly_fields,
    load_monthly_archive,
    prepare_monthly_archive,
    reconstruct_dataset,
    vectorize_dataset,
)


def test_high_frequency_monthly_mean_is_labelled_when_it_becomes_available():
    times = pd.date_range("2020-01-01", "2020-02-01", freq="6h")
    fields = xr.Dataset(
        {"msl": (("time",), np.arange(len(times), dtype=np.float32))},
        coords={"time": times},
    )
    monthly, aggregation = aggregate_monthly_fields(fields)
    assert aggregation == "calendar_month_mean_available_next_month"
    assert list(pd.to_datetime(monthly.time.values)) == [pd.Timestamp("2020-02-01")]


def test_main_system_data_is_combined_with_monthly_fields(tmp_path):
    times = pd.date_range("2020-01-01", periods=8, freq="MS")
    fields = xr.Dataset(
        {
            "msl": (
                ("time", "lat", "lon"),
                np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 2, 3),
            )
        },
        coords={"time": times, "lat": [-30.0, 30.0], "lon": [0.0, 120.0, 240.0]},
    )
    fields_path = tmp_path / "fields.nc"
    fields.to_netcdf(fields_path)
    integrated = pd.DataFrame(
        {
            "time": times,
            "typhoon_pressure_hpa": np.linspace(1000, 970, 8),
            "high_pressure_hpa": np.linspace(1015, 1025, 8),
        }
    )
    integrated_path = tmp_path / "integrated.csv"
    integrated.to_csv(integrated_path, index=False)

    archive, schema_path = prepare_monthly_archive(
        fields_path,
        tmp_path / "monthly.npz",
        integrated=integrated_path,
    )
    states, archive_times, schema = load_monthly_archive(archive)
    assert states.shape == (8, 8)
    assert schema["field_dim"] == 6
    assert len(schema["integrated_feature_names"]) == 2
    assert json.loads(schema_path.read_text())["state_dim"] == 8
    assert archive_times[0] == np.datetime64("2020-01-01")

    vectors = vectorize_dataset(
        fields,
        schema,
        integrated_defaults=states.mean(axis=0),
    )
    assert vectors.shape == states.shape
    rebuilt = reconstruct_dataset(states[-1], schema, pd.Timestamp("2020-09-01"))
    assert rebuilt["msl"].shape == (1, 2, 3)

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.time_alignment import (
    align_forecast,
    load_saved_forecast,
    render_animation,
    temporal_diagnostics,
)


@pytest.fixture
def moving_archive(tmp_path):
    times = pd.date_range("2018-11-16T00:00:00", periods=10, freq="6h")
    lat = np.linspace(25, 45, 4)
    lon = np.linspace(110, 150, 6)
    xx, yy = np.meshgrid(lon, lat)
    fields = []
    for index in range(len(times)):
        center = 116 + 2 * index
        blob = np.exp(-((xx - center) ** 2 + (yy - 35) ** 2) / 80)
        fields.append(np.stack((101000 - 500 * blob, 280 + 8 * blob,
                                np.full_like(blob, 2 + index), np.full_like(blob, -1))))
    values = np.asarray(fields, dtype=np.float32)
    source = xr.Dataset({name: (("time", "lat", "lon"), values[:, i])
                         for i, name in enumerate(("msl", "t2m", "u10", "v10"))},
                        coords={"time": times, "lat": lat, "lon": lon})
    raw = tmp_path / "moving.nc"
    source.to_netcdf(raw, engine="scipy")
    archive, _ = prepare_fixed_step_archive(raw, tmp_path / "moving.npz", step_hours=6,
                                            target_lat_points=4, target_lon_points=6)
    return archive


def _forecast(archive, path, *, leads=(6, 12, 18), slow=False):
    with np.load(archive) as saved:
        states, times = saved["states"], saved["times"]
    origin_index = 2
    truth = states[origin_index + 1:origin_index + 4]
    if slow:
        prediction = np.stack((truth[0], truth[0], truth[0]))
    else:
        prediction = truth.copy()
    members = np.stack((prediction - 0.1, prediction + 0.1))
    origin = times[origin_index]
    leads = np.asarray(leads)
    np.savez_compressed(path, predictions=members, origin_time=origin, lead_hours=leads,
                        valid_times=origin + leads.astype("timedelta64[h]"))


def test_exact_valid_time_and_tendency_diagnostic(moving_archive, tmp_path):
    forecast = tmp_path / "forecast.npz"
    _forecast(moving_archive, forecast, slow=True)
    aligned = align_forecast(forecast, moving_archive)
    np.testing.assert_array_equal(aligned.valid_times,
                                  np.asarray(["2018-11-16T18:00:00", "2018-11-17T00:00:00",
                                              "2018-11-17T06:00:00"], dtype="datetime64[ns]"))
    report = temporal_diagnostics(aligned)
    assert report["alignment"] == "exact_utc_valid_time_no_shift"
    assert report["median_tendency_amplitude_ratio_prediction_over_truth"] < 1
    assert report["by_lead"][1]["state_tendency_rms_per_hour"] == 0
    assert report["by_lead"][1]["uv_vector_rmse_mps"] > 0


def test_rejects_inconsistent_or_missing_times(moving_archive, tmp_path):
    forecast = tmp_path / "forecast.npz"
    _forecast(moving_archive, forecast, leads=(6, 12, 19))
    with np.load(forecast) as saved:
        payload = dict(saved)
    payload["valid_times"][-1] -= np.timedelta64(1, "h")
    np.savez_compressed(forecast, **payload)
    with pytest.raises(ValueError, match="valid_times disagree"):
        load_saved_forecast(forecast)
    _forecast(moving_archive, forecast, leads=(6, 12, 24))
    # The explicit times are valid, but the +24h truth is deliberately skipped.
    with np.load(forecast) as saved:
        payload = dict(saved)
    payload["valid_times"] = payload["origin_time"] + payload["lead_hours"].astype("timedelta64[h]")
    np.savez_compressed(forecast, **payload)
    aligned = align_forecast(forecast, moving_archive)
    assert aligned.lead_hours[-1] == 24


def test_render_gif_with_common_contract(moving_archive, tmp_path):
    pytest.importorskip("matplotlib")
    forecast = tmp_path / "forecast.npz"
    _forecast(moving_archive, forecast)
    output = render_animation(align_forecast(forecast, moving_archive), tmp_path / "comparison.gif",
                              fps=2)
    assert output.stat().st_size > 1000

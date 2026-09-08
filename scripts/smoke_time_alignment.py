"""Create a deterministic moving-field audit without claiming ERA5 training."""
from pathlib import Path

import json
import numpy as np
import pandas as pd
import xarray as xr

from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.time_alignment import main as diagnose_main


def main():
    output = Path("outputs/time-alignment-synthetic")
    output.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2018-11-16T06:00:00", periods=34, freq="24h")
    lat, lon = np.linspace(20, 50, 12), np.linspace(105, 155, 20)
    xx, yy = np.meshgrid(lon, lat)
    values = []
    for step in range(len(times)):
        center_x, center_y = 112 + 0.9 * step, 31 + 0.25 * step
        blob = np.exp(-((xx - center_x) ** 2 / 90 + (yy - center_y) ** 2 / 45))
        values.append(np.stack((101200 - 800 * blob, 279 + 10 * blob,
                                3 + 7 * blob, -1 + 4 * blob)))
    values = np.asarray(values, dtype=np.float32)
    dataset = xr.Dataset({name: (("time", "lat", "lon"), values[:, index])
                          for index, name in enumerate(("msl", "t2m", "u10", "v10"))},
                         coords={"time": times, "lat": lat, "lon": lon})
    source = output / "moving-fields.nc"
    dataset.to_netcdf(source, engine="scipy")
    archive, _ = prepare_fixed_step_archive(source, output / "moving-fields.npz",
                                            step_hours=24, target_lat_points=len(lat),
                                            target_lon_points=len(lon))
    with np.load(archive) as saved:
        states = saved["states"]
    origin = np.datetime64(times[0].to_datetime64(), "ns")
    leads = np.arange(1, 31, dtype=np.int64) * 24
    truth = states[1:31]
    # Deliberately under-evolving forecast: 0.55 of the truth anomaly. Members
    # carry distinct coherent scenarios; the mean remains slower than truth.
    mean = states[0] + 0.55 * (truth - states[0])
    pattern = np.sin(np.linspace(0, 2 * np.pi, states.shape[1], endpoint=False)).astype(np.float32)
    members = np.stack([mean + amplitude * pattern for amplitude in (-0.3, -0.1, 0.1, 0.3)])
    forecast = output / "slow-ensemble.npz"
    np.savez_compressed(forecast, predictions=members, origin_time=origin,
                        last_history_time=origin, lead_hours=leads,
                        valid_times=origin + leads.astype("timedelta64[h]"),
                        forecast_step_hours=np.asarray(24))
    diagnose_main(["--forecast", str(forecast), "--archive", str(archive),
                   "--output", str(output / "comparison.mp4"),
                   "--report", str(output / "diagnostics.json"), "--fps", "2.5"])
    report = json.loads((output / "diagnostics.json").read_text())
    assert report["alignment"] == "exact_utc_valid_time_no_shift"
    assert 0.45 < report["median_tendency_amplitude_ratio_prediction_over_truth"] < 0.65
    print(output)


if __name__ == "__main__":
    main()

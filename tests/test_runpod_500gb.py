from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from climate_diffusion.data import load_monthly_archive
from climate_diffusion.era5 import _select_pressure_levels
from climate_diffusion.era5_streaming import (
    _month_availability_time,
    finalize_era5_streaming_archive,
)
from climate_diffusion.runpod_train import _atomic_torch_save, _restore_rng, _rng_state


def test_pressure_level_subset_leaves_surface_fields_untouched():
    dataset = xr.Dataset(
        {
            "z": (("time", "level", "lat", "lon"), np.zeros((1, 6, 2, 3), dtype=np.float32)),
            "msl": (("time", "lat", "lon"), np.ones((1, 2, 3), dtype=np.float32)),
        },
        coords={
            "time": [np.datetime64("2020-01-01")],
            "level": [1000, 925, 850, 700, 500, 200],
            "lat": [1.0, 0.0],
            "lon": [0.0, 1.0, 2.0],
        },
    )
    selected = _select_pressure_levels(dataset, (1000, 850, 700, 500, 200))
    assert selected.z.shape[1] == 5
    assert selected.level.values.tolist() == [1000, 850, 700, 500, 200]
    assert selected.msl.shape == (1, 2, 3)


def test_complete_six_hour_month_maps_to_next_month_availability():
    times = pd.date_range("2020-01-01", "2020-02-01", freq="6h", inclusive="left")
    assert _month_availability_time(times, 6) == pd.Timestamp("2020-02-01")


def test_incomplete_month_is_rejected():
    times = pd.date_range("2020-01-01", "2020-01-31 12:00", freq="6h")
    try:
        _month_availability_time(times, 6)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete raw month should fail")


def _write_month_shard(path: Path, raw_month: str, availability: str, value: float):
    levels = [1000, 850, 700, 500, 200]
    height, width = 2, 4
    variables = [
        {
            "name": "msl",
            "dims": ["lat", "lon"],
            "non_spatial_dims": [],
            "non_spatial_shape": [],
            "shape": [height, width],
            "channel_slice": [0, 1],
            "coords": {"lat": [1.0, 0.0], "lon": [0.0, 90.0, 180.0, 270.0]},
            "attrs": {},
        },
        {
            "name": "z",
            "dims": ["level", "lat", "lon"],
            "non_spatial_dims": ["level"],
            "non_spatial_shape": [5],
            "shape": [5, height, width],
            "channel_slice": [1, 6],
            "coords": {
                "level": levels,
                "lat": [1.0, 0.0],
                "lon": [0.0, 90.0, 180.0, 270.0],
            },
            "attrs": {},
        },
    ]
    schema = {
        "format": "climate_diffusion.era5_month_shard.v1",
        "availability_time": availability,
        "raw_calendar_month": raw_month,
        "raw_cadence_hours": 6,
        "variables": variables,
        "channel_names": ["field:msl:0", *[f"field:z:{index}" for index in range(5)]],
        "spatial_channels": 6,
        "grid_shape": [height, width],
        "coords": {"lat": [1.0, 0.0], "lon": [0.0, 90.0, 180.0, 270.0]},
        "pressure_levels_hpa": levels,
        "source_files": [f"{raw_month}.nc"],
        "target_lat_points": height,
        "target_lon_points": width,
    }
    state = np.full((6, height, width), value, dtype=np.float32)
    mask = np.ones_like(state, dtype=bool)
    np.savez(
        path,
        state=state,
        observed_mask=mask,
        observed_fraction=np.ones(6, dtype=np.float32),
        availability_time=np.asarray(np.datetime64(availability, "ns")),
        schema_json=np.asarray(json.dumps(schema)),
    )


def test_month_shards_finalize_to_standard_memmap_archive(tmp_path):
    staging = tmp_path / "monthly"
    staging.mkdir()
    _write_month_shard(staging / "2020-01.npz", "2020-01", "2020-02-01", 1.0)
    _write_month_shard(staging / "2020-02.npz", "2020-02", "2020-03-01", 2.0)
    output = tmp_path / "archive"
    finalize_era5_streaming_archive(staging, output)
    states, times, schema = load_monthly_archive(output)
    assert states.shape == (2, 6, 2, 4)
    assert np.allclose(states[0], 1.0)
    assert np.allclose(states[1], 2.0)
    expected_times = np.asarray(["2020-02-01", "2020-03-01"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(np.asarray(times, dtype="datetime64[ns]"), expected_times)
    assert schema["source_metadata"]["pressure_levels_hpa"] == [1000, 850, 700, 500, 200]


def test_atomic_checkpoint_and_rng_restore(tmp_path):
    generator = torch.Generator().manual_seed(123)
    torch.manual_seed(456)
    np.random.seed(789)
    state = _rng_state(generator)
    expected_torch = torch.rand(3)
    expected_numpy = np.random.rand(3)
    expected_loader = torch.rand(3, generator=generator)
    _restore_rng(state, generator)
    assert torch.allclose(torch.rand(3), expected_torch)
    assert np.allclose(np.random.rand(3), expected_numpy)
    assert torch.allclose(torch.rand(3, generator=generator), expected_loader)

    path = tmp_path / "latest.pt"
    _atomic_torch_save({"epoch": 3, "tensor": torch.tensor([1.0])}, path)
    payload = torch.load(path, weights_only=False)
    assert payload["epoch"] == 3
    assert payload["tensor"].item() == 1.0

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_diffusion.data import (
    MonthlyWindowDataset,
    load_monthly_archive,
    load_observation_mask,
    load_observed_fraction,
    prepare_monthly_archive,
)


def _archive(tmp_path, values, times, height, width):
    dataset = xr.Dataset(
        {"msl": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": np.linspace(-80.0, 80.0, height),
            "lon": np.arange(width) * (360.0 / width),
        },
    )
    source = tmp_path / "fields.nc"
    dataset.to_netcdf(source, engine="scipy")
    return prepare_monthly_archive(
        source,
        tmp_path / "archive",
        layout="spatial",
        target_lat_points=height,
        target_lon_points=width,
    )[0]


def _partly_missing(months=14, height=4, width=6):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(months, height, width)).astype(np.float32)
    # Punch holes of differing size so windows land on both sides of a threshold.
    values[3, 0, :] = np.nan
    values[7, :2, :3] = np.nan
    values[11, 0, 0] = np.nan
    return values


def test_observed_fraction_matches_the_mask_it_summarises(tmp_path):
    months, height, width = 14, 4, 6
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    archive = _archive(tmp_path, _partly_missing(months, height, width), times, height, width)
    states, _, schema = load_monthly_archive(archive)
    fraction = load_observed_fraction(archive, states, schema)
    mask = load_observation_mask(archive, states, schema)
    assert fraction.shape == (months, int(schema["spatial_channels"]))
    np.testing.assert_allclose(
        np.asarray(fraction), np.asarray(mask).mean(axis=(2, 3)), rtol=1e-6
    )


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 0.95])
def test_recorded_fractions_select_the_same_windows_as_a_mask_scan(tmp_path, threshold):
    """The fast path must be a pure speedup, not a different filter."""
    months, height, width = 14, 4, 6
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    archive = _archive(tmp_path, _partly_missing(months, height, width), times, height, width)
    states, _, schema = load_monthly_archive(archive)
    mask = load_observation_mask(archive, states, schema)
    fraction = load_observed_fraction(archive, states, schema)

    scanned = MonthlyWindowDataset(
        states, 3, 1, observation_mask=mask, min_observed_fraction=threshold
    )
    recorded = MonthlyWindowDataset(
        states,
        3,
        1,
        observation_mask=mask,
        min_observed_fraction=threshold,
        observed_fraction=fraction,
    )
    assert recorded.indices == scanned.indices


def test_recorded_fractions_do_not_read_the_mask(tmp_path):
    months, height, width = 14, 4, 6
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    archive = _archive(tmp_path, _partly_missing(months, height, width), times, height, width)
    states, _, schema = load_monthly_archive(archive)
    fraction = load_observed_fraction(archive, states, schema)

    class ExplodingMask:
        shape = states.shape

        def __getitem__(self, item):
            raise AssertionError("window filtering must not touch the mask memmap")

    MonthlyWindowDataset(
        states,
        3,
        1,
        observation_mask=ExplodingMask(),
        min_observed_fraction=0.5,
        observed_fraction=fraction,
    )


def test_archive_records_that_it_was_scanned_for_infinities(tmp_path):
    months, height, width = 6, 4, 6
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(1)
    values = rng.normal(size=(months, height, width)).astype(np.float32)
    archive = _archive(tmp_path, values, times, height, width)
    schema = json.loads((archive / "schema.json").read_text())
    assert schema["inf_checked"] is True

    # A legacy archive carries no such record, so loading still scans it.
    del schema["inf_checked"]
    (archive / "schema.json").write_text(json.dumps(schema))
    states = np.load(archive / "states.npy", mmap_mode="r+")
    states[0, 0, 0, 0] = np.inf
    states.flush()
    with pytest.raises(ValueError, match="infinity"):
        load_monthly_archive(archive)

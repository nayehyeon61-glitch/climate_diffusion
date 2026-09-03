"""HRES analysis/forecast adapter for the existing spatial archive contract."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .era5 import _normalise_global_grid, resolve_era5_files


HRES_COORDINATE_ALIASES = {
    "datetime": "time",
    "latitude": "lat",
    "longitude": "lon",
    "pressure_level": "level",
    "isobaricInhPa": "level",
    "forecast_period": "step",
}


def _select_forecast_lead(dataset: xr.Dataset, lead_hours: int, source: Path) -> xr.Dataset:
    if lead_hours < 0:
        raise ValueError("HRES lead_hours cannot be negative")
    if "step" not in dataset.dims and "step" not in dataset.coords:
        if lead_hours:
            raise ValueError(f"HRES input {source} has no step coordinate for lead={lead_hours}h")
        return dataset
    step = dataset.step
    values = np.asarray(step.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        hours = values.astype("timedelta64[s]").astype(np.int64) / 3600.0
    else:
        units = str(step.attrs.get("units", "hours")).lower()
        if "hour" not in units:
            raise ValueError(f"HRES step coordinate in {source} must use hours or timedelta")
        hours = values.astype(float)
    matches = np.flatnonzero(np.isclose(hours, float(lead_hours)))
    if matches.size != 1:
        available = [float(value) for value in np.ravel(hours)[:16]]
        raise ValueError(
            f"HRES input {source} does not contain exactly one {lead_hours}h step; "
            f"available={available}"
        )
    return dataset.isel(step=int(matches[0]), drop=True)


def _canonicalise_hres(dataset: xr.Dataset, source: Path, lead_hours: int) -> xr.Dataset:
    rename = {
        old: new
        for old, new in HRES_COORDINATE_ALIASES.items()
        if (old in dataset.coords or old in dataset.dims)
        and new not in dataset.coords
        and new not in dataset.dims
    }
    dataset = dataset.rename(rename)
    dataset = _select_forecast_lead(dataset, lead_hours, source)

    if "valid_time" in dataset.coords:
        valid = dataset.valid_time
        if valid.ndim != 1:
            raise ValueError(f"HRES valid_time in {source} must be 1-D after lead selection")
        dimension = valid.dims[0]
        dataset = dataset.assign_coords({dimension: valid.values}).drop_vars("valid_time")
        if dimension != "time":
            dataset = dataset.rename({dimension: "time"})
    elif "time" not in dataset.coords and "forecast_reference_time" in dataset.coords:
        reference = dataset.forecast_reference_time
        if reference.ndim != 1:
            raise ValueError(f"HRES forecast_reference_time in {source} must be 1-D")
        dimension = reference.dims[0]
        valid = pd.to_datetime(reference.values) + pd.to_timedelta(lead_hours, unit="h")
        dataset = dataset.assign_coords({dimension: valid})
        if dimension != "time":
            dataset = dataset.rename({dimension: "time"})
    if "time" not in dataset.coords:
        raise ValueError(f"HRES input {source} has no usable valid/init time coordinate")

    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if times.hasnans or times.has_duplicates:
        raise ValueError(f"HRES input {source} has invalid or duplicate selected valid times")
    return _normalise_global_grid(dataset.sortby("time"))


def _target_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if min(height, width) < 2:
        raise ValueError("HRES target grid dimensions must be at least two")
    latitude = np.linspace(90.0, -90.0, height, dtype=np.float64)
    longitude = np.arange(width, dtype=np.float64) * (360.0 / width)
    return latitude, longitude


def _regrid_hres(dataset: xr.Dataset, height: int, width: int) -> xr.Dataset:
    target_lat, target_lon = _target_grid(height, width)
    current_lat = np.asarray(dataset.lat.values, dtype=np.float64)
    current_lon = np.asarray(dataset.lon.values, dtype=np.float64)
    if np.array_equal(current_lat, target_lat) and np.array_equal(current_lon, target_lon):
        return dataset
    if current_lat.max() < 89.9 or current_lat.min() > -89.9:
        raise ValueError("HRES input is not global in latitude; cannot create global archive")
    spacing = 360.0 / width
    if current_lon.size < width or np.max(np.diff(np.sort(current_lon))) > spacing * 1.5:
        raise ValueError("HRES input does not cover the requested global longitude grid")
    # xarray interpolation requires a monotonic source; restore the established
    # north-to-south archive order after interpolation.
    return (
        dataset.sortby("lat")
        .interp(lat=target_lat[::-1], lon=target_lon, method="linear")
        .sortby("lat", ascending=False)
    )


@contextmanager
def open_hres_dataset(
    source: str | Path | Sequence[str | Path],
    *,
    lead_hours: int = 0,
    target_lat_points: int = 721,
    target_lon_points: int = 1440,
) -> Iterator[tuple[xr.Dataset, list[Path]]]:
    """Open split HRES files at one lead and expose a canonical 0.25-degree view."""
    files = resolve_era5_files(source)
    opened: list[xr.Dataset] = []
    try:
        for path in files:
            raw = (
                xr.open_zarr(path, decode_timedelta=True)
                if path.suffix == ".zarr"
                else xr.open_dataset(path, decode_timedelta=True)
            )
            opened.append(_canonicalise_hres(raw, path, lead_hours))
        try:
            combined = xr.combine_by_coords(opened, combine_attrs="drop_conflicts")
        except ValueError as error:
            raise ValueError(
                "HRES files cannot be combined after lead selection; check grids, "
                "overlapping valid times, levels, and duplicate variables"
            ) from error
        yield _regrid_hres(combined, target_lat_points, target_lon_points), files
    finally:
        for dataset in opened:
            dataset.close()


def prepare_hres_archive(
    source: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    integrated: str | Path | None = None,
    variables: tuple[str, ...] | None = None,
    lead_hours: int = 0,
    target_lat_points: int = 721,
    target_lon_points: int = 1440,
) -> tuple[Path, Path]:
    from .data import prepare_monthly_archive

    with open_hres_dataset(
        source,
        lead_hours=lead_hours,
        target_lat_points=target_lat_points,
        target_lon_points=target_lon_points,
    ) as (dataset, files):
        return prepare_monthly_archive(
            dataset,
            output,
            integrated=integrated,
            variables=variables,
            target_lat_points=target_lat_points,
            target_lon_points=target_lon_points,
            layout="spatial",
            source_label=str(source),
            source_metadata={
                "adapter": "hres.v1",
                "files": [str(path) for path in files],
                "selected_lead_hours": lead_hours,
                "regridding": "linear_to_regular_global_grid",
                "canonical_coordinates": {"latitude": "descending", "longitude": "[0,360)"},
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a spatial monthly archive from HRES")
    parser.add_argument("--source", required=True, nargs="+", help="Files, directory, or glob")
    parser.add_argument("--integrated")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--lead-hours", type=int, default=0)
    parser.add_argument("--target-lat-points", type=int, default=721)
    parser.add_argument("--target-lon-points", type=int, default=1440)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    archive, schema = prepare_hres_archive(
        args.source,
        args.output,
        integrated=args.integrated,
        variables=None if not args.variables else tuple(args.variables),
        lead_hours=args.lead_hours,
        target_lat_points=args.target_lat_points,
        target_lon_points=args.target_lon_points,
    )
    print(f"archive={archive}")
    print(f"schema={schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

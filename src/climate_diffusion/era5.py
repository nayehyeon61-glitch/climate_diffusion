"""ERA5 input adapter that feeds the existing monthly archive contract."""

from __future__ import annotations

import argparse
import glob
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr


ERA5_COORDINATE_ALIASES = {
    "valid_time": "time",
    "datetime": "time",
    "latitude": "lat",
    "longitude": "lon",
    "pressure_level": "level",
    "isobaricInhPa": "level",
}


def resolve_era5_files(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve one file, a directory, a glob, or an explicit file sequence."""
    values = [source] if isinstance(source, (str, Path)) else list(source)
    files: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir() and path.suffix != ".zarr":
            files.extend(sorted(path.rglob("*.nc")))
            files.extend(sorted(path.rglob("*.nc4")))
            files.extend(sorted(path.rglob("*.netcdf")))
        elif any(character in str(value) for character in "*?["):
            files.extend(Path(item) for item in sorted(glob.glob(str(value))))
        else:
            files.append(path)
    unique = list(dict.fromkeys(item.resolve() for item in files))
    if not unique:
        raise FileNotFoundError(f"No ERA5 NetCDF/Zarr inputs found under {source!r}")
    missing = [str(item) for item in unique if not item.exists()]
    if missing:
        raise FileNotFoundError(f"ERA5 input does not exist: {missing[0]}")
    return unique


def _canonicalise(dataset: xr.Dataset, source: Path) -> xr.Dataset:
    rename = {
        old: new
        for old, new in ERA5_COORDINATE_ALIASES.items()
        if (old in dataset.coords or old in dataset.dims)
        and new not in dataset.coords
        and new not in dataset.dims
    }
    dataset = dataset.rename(rename)
    if "time" not in dataset.coords:
        raise ValueError(f"ERA5 input {source} has no time coordinate")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if times.hasnans or times.has_duplicates:
        raise ValueError(f"ERA5 input {source} has invalid or duplicate timestamps")
    if "expver" in dataset.dims:
        collapsed = {}
        for name, array in dataset.data_vars.items():
            if "expver" not in array.dims:
                collapsed[name] = array
                continue
            result = array.isel(expver=0, drop=True)
            for index in range(1, array.sizes["expver"]):
                result = result.combine_first(array.isel(expver=index, drop=True))
            collapsed[name] = result
        dataset = xr.Dataset(collapsed, attrs=dataset.attrs)
    return dataset.sortby("time")


def _normalise_global_grid(dataset: xr.Dataset) -> xr.Dataset:
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("ERA5 spatial input requires latitude and longitude coordinates")
    lat = np.asarray(dataset.lat.values, dtype=np.float64)
    lon = np.asarray(dataset.lon.values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or min(lat.size, lon.size) < 1:
        raise ValueError("ERA5 latitude/longitude coordinates must be non-empty 1-D axes")
    if np.unique(lat).size != lat.size or np.unique(lon).size != lon.size:
        raise ValueError("ERA5 latitude/longitude coordinates must be unique")
    canonical_lon = np.mod(lon, 360.0)
    if np.unique(canonical_lon).size != canonical_lon.size:
        raise ValueError("ERA5 longitude aliases collapse to duplicate [0, 360) values")
    dataset = dataset.assign_coords(lon=canonical_lon).sortby("lon")
    return dataset.sortby("lat", ascending=False)


def _select_pressure_levels(
    dataset: xr.Dataset,
    pressure_levels: tuple[int, ...] | None,
) -> xr.Dataset:
    """Keep only requested pressure levels while leaving surface fields untouched."""
    if pressure_levels is None:
        return dataset
    if "level" not in dataset.coords:
        raise ValueError(
            "--pressure-levels was supplied but the ERA5 source has no pressure-level coordinate"
        )
    requested = np.asarray(pressure_levels, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0 or np.any(requested <= 0):
        raise ValueError("pressure_levels must be a non-empty sequence of positive hPa values")
    if np.unique(requested).size != requested.size:
        raise ValueError("pressure_levels contains duplicates")
    available = np.asarray(dataset.level.values)
    missing = [int(value) for value in requested if value not in available]
    if missing:
        raise ValueError(
            f"ERA5 source is missing requested pressure levels {missing}; "
            f"available examples={available[:16].tolist()}"
        )
    return dataset.sel(level=requested)


@contextmanager
def open_era5_dataset(
    source: str | Path | Sequence[str | Path],
) -> Iterator[tuple[xr.Dataset, list[Path]]]:
    """Open split ERA5 files as one canonical, lazily evaluated dataset."""
    files = resolve_era5_files(source)
    opened: list[xr.Dataset] = []
    try:
        for path in files:
            raw = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
            opened.append(_canonicalise(raw, path))
        try:
            combined = xr.combine_by_coords(opened, combine_attrs="drop_conflicts")
        except ValueError as error:
            raise ValueError(
                "ERA5 files cannot be combined by coordinates; check overlapping times, "
                "grid coordinates, levels, and duplicate variable files"
            ) from error
        combined = _normalise_global_grid(combined)
        if not combined.data_vars:
            raise ValueError("ERA5 source contains no data variables")
        yield combined, files
    finally:
        for dataset in opened:
            dataset.close()


def prepare_era5_archive(
    source: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    integrated: str | Path | None = None,
    variables: tuple[str, ...] | None = None,
    pressure_levels: tuple[int, ...] | None = None,
    target_lat_points: int = 721,
    target_lon_points: int = 1440,
) -> tuple[Path, Path]:
    """Adapt ERA5 and delegate all archive semantics to prepare_monthly_archive."""
    from .data import prepare_monthly_archive

    with open_era5_dataset(source) as (dataset, files):
        dataset = _select_pressure_levels(dataset, pressure_levels)
        metadata = {
            "adapter": "era5.v2",
            "files": [str(path) for path in files],
            "canonical_coordinates": {"latitude": "descending", "longitude": "[0,360)"},
            "pressure_levels_hpa": None if pressure_levels is None else list(pressure_levels),
        }
        return prepare_monthly_archive(
            dataset,
            output,
            integrated=integrated,
            variables=variables,
            target_lat_points=target_lat_points,
            target_lon_points=target_lon_points,
            layout="spatial",
            source_label=str(source),
            source_metadata=metadata,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the existing spatial monthly archive from split ERA5 files"
    )
    parser.add_argument("--source", required=True, nargs="+", help="Files, directory, or glob")
    parser.add_argument("--integrated", help="Optional integrated Parquet/CSV")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument(
        "--pressure-levels",
        nargs="+",
        type=int,
        help="Pressure levels in hPa, e.g. 1000 850 700 500 200",
    )
    parser.add_argument("--target-lat-points", type=int, default=721)
    parser.add_argument("--target-lon-points", type=int, default=1440)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    archive, schema = prepare_era5_archive(
        args.source,
        args.output,
        integrated=args.integrated,
        variables=None if not args.variables else tuple(args.variables),
        pressure_levels=None if not args.pressure_levels else tuple(args.pressure_levels),
        target_lat_points=args.target_lat_points,
        target_lon_points=args.target_lon_points,
    )
    print(f"archive={archive}")
    print(f"schema={schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

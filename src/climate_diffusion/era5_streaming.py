"""Month-by-month ERA5 preprocessing for storage-bounded RunPod jobs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _coarsen_global_fields, _json_values, validate_monthly_times
from .era5 import _select_pressure_levels, open_era5_dataset
from .validation import require_no_inf_numpy


def _month_availability_time(times: pd.DatetimeIndex, cadence_hours: int) -> pd.Timestamp:
    if len(times) < 2:
        raise ValueError("A raw ERA5 month requires at least two timestamps")
    times = times.sort_values()
    periods = times.to_period("M")
    if len(periods.unique()) != 1:
        raise ValueError("Streaming ERA5 append accepts exactly one calendar month at a time")
    expected = pd.Timedelta(hours=cadence_hours)
    deltas = pd.Series(times[1:].values - times[:-1].values)
    if not (deltas == expected).all():
        raise ValueError(f"ERA5 month is not on an exact {cadence_hours}h cadence")
    start = periods[0].start_time
    next_month = periods[0].end_time.normalize() + pd.offsets.MonthBegin(1)
    if times[0] != start:
        raise ValueError(f"ERA5 month starts at {times[0]}, expected {start}")
    if times[-1] + expected != next_month:
        raise ValueError(
            f"ERA5 month is incomplete: last={times[-1]}, next expected={next_month}"
        )
    return pd.Timestamp(next_month)


def append_era5_month(
    source,
    staging_dir: str | Path,
    *,
    variables: tuple[str, ...],
    pressure_levels: tuple[int, ...] = (1000, 850, 700, 500, 200),
    cadence_hours: int = 6,
    target_lat_points: int = 721,
    target_lon_points: int = 1440,
) -> Path:
    """Aggregate one complete raw ERA5 calendar month into a restart-safe shard."""
    staging = Path(staging_dir).expanduser().resolve()
    staging.mkdir(parents=True, exist_ok=True)
    with open_era5_dataset(source) as (dataset, files):
        dataset = _select_pressure_levels(dataset, pressure_levels)
        missing = sorted(set(variables).difference(dataset.data_vars))
        if missing:
            raise ValueError(f"ERA5 month is missing requested variables: {missing}")
        dataset = dataset[list(variables)].sortby("time")
        raw_times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
        availability = _month_availability_time(raw_times, cadence_hours)
        monthly = dataset.mean(dim="time", skipna=True, keep_attrs=True)
        monthly = _coarsen_global_fields(
            monthly, target_lat_points, target_lon_points
        ).load()

        height = int(monthly.sizes["lat"])
        width = int(monthly.sizes["lon"])
        blocks = []
        masks = []
        variable_schema = []
        channel_names = []
        channel_offset = 0
        for name in variables:
            array = monthly[name]
            non_spatial = tuple(dim for dim in array.dims if dim not in {"lat", "lon"})
            dims = (*non_spatial, "lat", "lon")
            values = np.asarray(array.transpose(*dims).values, dtype=np.float32).reshape(-1, height, width)
            require_no_inf_numpy(values, f"ERA5 streaming month variable {name!r}")
            finite = np.isfinite(values)
            if (~finite.any(axis=(1, 2))).any():
                raise ValueError(f"ERA5 month variable {name!r} has a fully missing channel")
            channels = values.shape[0]
            variable_schema.append(
                {
                    "name": name,
                    "dims": list(dims),
                    "non_spatial_dims": list(non_spatial),
                    "non_spatial_shape": [int(array.sizes[dim]) for dim in non_spatial],
                    "shape": [int(array.sizes[dim]) for dim in non_spatial] + [height, width],
                    "channel_slice": [channel_offset, channel_offset + channels],
                    "coords": {
                        dim: _json_values(np.asarray(array.coords[dim].values))
                        for dim in dims
                        if dim in array.coords and array.coords[dim].dims == (dim,)
                    },
                    "attrs": {key: str(value) for key, value in array.attrs.items()},
                }
            )
            channel_names.extend(f"field:{name}:{index}" for index in range(channels))
            channel_offset += channels
            blocks.append(values)
            masks.append(finite)
        state = np.concatenate(blocks, axis=0).astype(np.float32)
        mask = np.concatenate(masks, axis=0).astype(np.bool_)
        observed_fraction = mask.mean(axis=(1, 2)).astype(np.float32)
        shard_schema = {
            "format": "climate_diffusion.era5_month_shard.v1",
            "availability_time": str(availability),
            "raw_calendar_month": str(raw_times[0].to_period("M")),
            "raw_cadence_hours": cadence_hours,
            "variables": variable_schema,
            "channel_names": channel_names,
            "spatial_channels": int(state.shape[0]),
            "grid_shape": [height, width],
            "coords": {
                "lat": _json_values(np.asarray(monthly.lat.values)),
                "lon": _json_values(np.asarray(monthly.lon.values)),
            },
            "pressure_levels_hpa": list(pressure_levels),
            "source_files": [str(path) for path in files],
            "target_lat_points": target_lat_points,
            "target_lon_points": target_lon_points,
        }

    shard = staging / f"{raw_times[0].strftime('%Y-%m')}.npz"
    if shard.exists():
        raise FileExistsError(f"Monthly ERA5 shard already exists: {shard}")
    np.savez(
        shard,
        state=state,
        observed_mask=mask,
        observed_fraction=observed_fraction,
        availability_time=np.asarray(availability.to_datetime64(), dtype="datetime64[ns]"),
        schema_json=np.asarray(json.dumps(shard_schema, ensure_ascii=False)),
    )
    return shard


def _load_shard_schema(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["schema_json"].item()))


def finalize_era5_streaming_archive(
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    delete_shards: bool = False,
) -> Path:
    """Convert ordered monthly shards into the standard spatial memmap archive."""
    staging = Path(staging_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    shards = sorted(staging.glob("????-??.npz"))
    if len(shards) < 2:
        raise ValueError("Need at least two monthly ERA5 shards before finalization")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output archive directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    schemas = [_load_shard_schema(path) for path in shards]
    first = schemas[0]
    contract_keys = (
        "variables",
        "channel_names",
        "spatial_channels",
        "grid_shape",
        "coords",
        "pressure_levels_hpa",
        "target_lat_points",
        "target_lon_points",
    )
    for path, schema in zip(shards[1:], schemas[1:]):
        if any(schema[key] != first[key] for key in contract_keys):
            raise ValueError(f"Monthly ERA5 shard contract changed at {path}")
    times = np.asarray(
        [np.datetime64(schema["availability_time"], "ns") for schema in schemas],
        dtype="datetime64[ns]",
    )
    validate_monthly_times(times)
    channels = int(first["spatial_channels"])
    height, width = map(int, first["grid_shape"])
    states = np.lib.format.open_memmap(
        output / "states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(shards), channels, height, width),
    )
    masks = np.lib.format.open_memmap(
        output / "observed_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(shards), channels, height, width),
    )
    fractions = np.empty((len(shards), channels), dtype=np.float32)
    for index, path in enumerate(shards):
        with np.load(path, allow_pickle=False) as archive:
            state = archive["state"]
            mask = archive["observed_mask"]
            if state.shape != (channels, height, width) or mask.shape != state.shape:
                raise ValueError(f"Monthly ERA5 shard shape mismatch: {path}")
            states[index] = state
            masks[index] = mask
            fractions[index] = archive["observed_fraction"]
        states.flush()
        masks.flush()
    np.save(output / "times.npy", times)
    np.save(output / "observed_fraction.npy", fractions)
    np.save(output / "auxiliary.npy", np.empty((len(shards), 0), dtype=np.float32))
    schema = {
        "format": "climate_diffusion.monthly_spatial.v3",
        "layout": "spatial",
        "storage": "npy_memmap_directory",
        "source_fields": "month-by-month ERA5 streaming shards",
        "source_metadata": {
            "adapter": "era5.streaming.v1",
            "pressure_levels_hpa": first["pressure_levels_hpa"],
            "raw_cadence_hours": first["raw_cadence_hours"],
            "monthly_shards": len(shards),
        },
        "source_integrated": None,
        "state_dim": int(channels * height * width),
        "spatial_channels": channels,
        "grid_shape": [height, width],
        "channel_names": first["channel_names"],
        "auxiliary_feature_names": [],
        "auxiliary_dim": 0,
        "variables": first["variables"],
        "coords": first["coords"],
        "monthly_frequency": "calendar_month",
        "state_time_semantics": "availability_time",
        "aggregation": "calendar_month_mean_available_next_month",
        "missing_value_policy": "preserve_nan_impute_from_train_only",
        "infinity_policy": "fail_fast",
        "target_lat_points": first["target_lat_points"],
        "target_lon_points": first["target_lon_points"],
    }
    (output / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if delete_shards:
        for path in shards:
            path.unlink()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Storage-bounded ERA5 month streaming")
    sub = parser.add_subparsers(dest="command", required=True)
    append = sub.add_parser("append", help="Convert one complete raw ERA5 month into a shard")
    append.add_argument("--source", nargs="+", required=True)
    append.add_argument("--staging-dir", required=True)
    append.add_argument("--variables", nargs="+", required=True)
    append.add_argument("--pressure-levels", nargs="+", type=int, default=[1000, 850, 700, 500, 200])
    append.add_argument("--cadence-hours", type=int, default=6)
    append.add_argument("--target-lat-points", type=int, default=721)
    append.add_argument("--target-lon-points", type=int, default=1440)
    finalize = sub.add_parser("finalize", help="Build the standard spatial archive from month shards")
    finalize.add_argument("--staging-dir", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--delete-shards", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "append":
        result = append_era5_month(
            args.source,
            args.staging_dir,
            variables=tuple(args.variables),
            pressure_levels=tuple(args.pressure_levels),
            cadence_hours=args.cadence_hours,
            target_lat_points=args.target_lat_points,
            target_lon_points=args.target_lon_points,
        )
    else:
        result = finalize_era5_streaming_archive(
            args.staging_dir,
            args.output,
            delete_shards=args.delete_shards,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

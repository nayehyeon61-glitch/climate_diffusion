"""Prepare causal fixed-step atmospheric state archives for Flow Matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .data import (
    _coarsen_global_fields,
    _json_values,
    _normalise_coordinates,
    _open_dataset,
)


def sample_fixed_step_history(dataset: xr.Dataset, step_hours: int) -> xr.Dataset:
    """Select exact causal snapshots separated by ``step_hours`` ending at the latest state."""
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    dataset = _normalise_coordinates(dataset).sortby("time")
    if "time" not in dataset.coords:
        raise ValueError("Fixed-step Flow input requires a time coordinate")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if len(times) < 2:
        return dataset
    available = {int(value.value) for value in times}
    step = pd.Timedelta(hours=step_hours)
    selected = []
    current = times[-1]
    first = times[0]
    while current >= first:
        if int(current.value) not in available:
            raise ValueError(
                f"Input history does not contain exact {step_hours}h snapshot {current}"
            )
        selected.append(current)
        current = current - step
    selected.reverse()
    return dataset.sel(time=np.asarray(selected, dtype="datetime64[ns]"))


def _load_integrated_asof(
    path: str | Path,
    times: pd.DatetimeIndex,
) -> tuple[np.ndarray, list[str]]:
    value = str(path)
    frame = pd.read_parquet(value) if value.endswith(".parquet") else pd.read_csv(value)
    if "time" not in frame:
        raise ValueError("Integrated main-system data requires a 'time' column")
    frame["time"] = pd.to_datetime(frame["time"])
    if frame["time"].duplicated().any():
        raise ValueError(
            "Fixed-step Flow integrated input must contain at most one row per timestamp; "
            "do not mix multiple storm rows into a global atmospheric state"
        )
    numeric = [
        name for name in frame.select_dtypes(include=[np.number]).columns
        if name not in {"init_time_ns"}
    ]
    if not numeric:
        raise ValueError("Integrated data has no numeric climate/track features")
    ordered = frame.sort_values("time")[["time", *numeric]]
    query = pd.DataFrame({"time": times})
    aligned = pd.merge_asof(query, ordered, on="time", direction="backward")
    # Never use future rows or a full-series mean for imputation. Forward filling
    # after a backward as-of join is causal; leading unavailable values use zero.
    values_frame = aligned[numeric].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    values = values_frame.to_numpy(np.float32)
    return values, [f"integrated:{name}" for name in numeric]


def prepare_fixed_step_archive(
    fields: str | Path,
    output: str | Path,
    *,
    step_hours: int = 360,
    integrated: str | Path | None = None,
    variables: tuple[str, ...] | None = None,
    target_lat_points: int = 18,
    target_lon_points: int = 36,
) -> tuple[Path, Path]:
    """Create an exact-snapshot archive whose one model step is ``step_hours``.

    No temporal averaging is applied. Production 0--360h trajectory models should
    use a sub-360 step (6h for WeatherNext cadence parity, or 24h for a lighter
    experiment); a 360h step is supported as an endpoint-only ablation.
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    if min(target_lat_points, target_lon_points) < 1:
        raise ValueError("Target grid dimensions must be positive")
    with _open_dataset(fields) as source:
        dataset = _normalise_coordinates(source)
        if "time" not in dataset.coords:
            raise ValueError("Gridded climate fields require a time coordinate")
        selected_names = tuple(variables or dataset.data_vars)
        missing = sorted(set(selected_names).difference(dataset.data_vars))
        if missing:
            raise ValueError(f"Missing requested variables: {missing}")
        snapshots = sample_fixed_step_history(dataset[list(selected_names)], step_hours)
        snapshots = _coarsen_global_fields(
            snapshots, target_lat_points, target_lon_points
        ).load()

    times = pd.DatetimeIndex(pd.to_datetime(snapshots.time.values))
    if len(times) < 2:
        raise ValueError("At least two fixed-step field states are required")
    actual = np.diff(times.values).astype("timedelta64[h]").astype(np.int64)
    if not np.all(actual == step_hours):
        raise ValueError(
            f"Fixed-step archive is not uniformly {step_hours}h: {sorted(set(actual.tolist()))}"
        )

    blocks: list[np.ndarray] = []
    variable_schema: list[dict[str, Any]] = []
    offset = 0
    feature_names: list[str] = []
    for name in selected_names:
        array = snapshots[name]
        dims = tuple(dim for dim in array.dims if dim != "time")
        array = array.transpose("time", *dims)
        values = np.asarray(array.values, dtype=np.float32).reshape(len(times), -1)
        blocks.append(values)
        size = values.shape[1]
        variable_schema.append(
            {
                "name": name,
                "dims": list(dims),
                "shape": [int(array.sizes[dim]) for dim in dims],
                "slice": [offset, offset + size],
                "coords": {
                    dim: _json_values(np.asarray(array.coords[dim].values))
                    for dim in dims
                    if dim in array.coords and array.coords[dim].dims == (dim,)
                },
                "attrs": {key: str(value) for key, value in array.attrs.items()},
            }
        )
        feature_names.extend(f"field:{name}:{index}" for index in range(size))
        offset += size

    field_dim = offset
    if integrated is not None:
        integrated_values, integrated_names = _load_integrated_asof(integrated, times)
        blocks.append(integrated_values)
        feature_names.extend(integrated_names)

    states = np.concatenate(blocks, axis=1).astype(np.float32)
    observed_mask = np.isfinite(states)
    # Keep training/inference missing-value semantics aligned and leakage-free:
    # vectorize_dataset also maps non-finite physical values to zero. The mask is
    # persisted so a future mask-aware Flow encoder can consume it explicitly.
    states = np.where(observed_mask, states, 0.0).astype(np.float32)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        observed_mask=observed_mask.astype(np.float32),
        times=times.values.astype("datetime64[ns]"),
        feature_names=np.asarray(feature_names),
    )
    schema_path = output_path.with_suffix(".schema.json")
    schema = {
        "format": "climate_diffusion.fixed_step_state.v1",
        "source_fields": str(fields),
        "source_integrated": None if integrated is None else str(integrated),
        "state_dim": int(states.shape[1]),
        "field_dim": int(field_dim),
        "integrated_feature_names": feature_names[field_dim:],
        "variables": variable_schema,
        "forecast_step_hours": int(step_hours),
        "state_time_semantics": "snapshot_valid_time",
        "aggregation": "exact_fixed_step_snapshot",
        "missing_value_policy": "zero_with_observed_mask_no_future_statistics",
        "target_lat_points": target_lat_points,
        "target_lon_points": target_lon_points,
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path, schema_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exact fixed-step atmospheric states for Flow Matching"
    )
    parser.add_argument("--fields", required=True)
    parser.add_argument("--integrated")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--step-hours", type=int, default=360)
    parser.add_argument("--target-lat-points", type=int, default=18)
    parser.add_argument("--target-lon-points", type=int, default=36)
    parser.add_argument("--output", default="data/flow_states_fixed_step.npz")
    args = parser.parse_args(argv)
    archive, schema = prepare_fixed_step_archive(
        args.fields,
        args.output,
        step_hours=args.step_hours,
        integrated=args.integrated,
        variables=None if not args.variables else tuple(args.variables),
        target_lat_points=args.target_lat_points,
        target_lon_points=args.target_lon_points,
    )
    print(f"archive={archive}")
    print(f"schema={schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

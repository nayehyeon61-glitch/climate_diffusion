"""Build monthly state vectors from gridded fields plus the main-system table."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

COORDINATE_ALIASES = {
    "valid_time": "time",
    "datetime": "time",
    "latitude": "lat",
    "longitude": "lon",
}


def _open_dataset(path: str | Path) -> xr.Dataset:
    value = str(path)
    return xr.open_zarr(value) if value.endswith(".zarr") else xr.open_dataset(value)


def _normalise_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    rename = {
        old: new
        for old, new in COORDINATE_ALIASES.items()
        if (old in dataset.coords or old in dataset.dims)
        and new not in dataset.coords
        and new not in dataset.dims
    }
    return dataset.rename(rename)


def _json_values(values: np.ndarray) -> list[Any]:
    result = []
    for value in values.tolist():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        result.append(value)
    return result


def _coarsen_global_fields(
    dataset: xr.Dataset,
    target_lat_points: int,
    target_lon_points: int,
) -> xr.Dataset:
    factors = {}
    if "lat" in dataset.dims:
        factors["lat"] = max(1, int(np.ceil(dataset.sizes["lat"] / target_lat_points)))
    if "lon" in dataset.dims:
        factors["lon"] = max(1, int(np.ceil(dataset.sizes["lon"] / target_lon_points)))
    return dataset.coarsen(factors, boundary="pad").mean(skipna=True) if factors else dataset


def aggregate_monthly_fields(
    dataset: xr.Dataset,
    *,
    complete_only: bool = True,
) -> tuple[xr.Dataset, str]:
    """Return causal monthly states labelled by the time they become available."""
    dataset = dataset.sortby("time")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if len(times) < 2:
        return dataset, "preaggregated_snapshot"
    deltas = np.diff(times.values).astype("timedelta64[s]").astype(np.int64)
    median_seconds = int(np.median(deltas))
    if median_seconds >= 27 * 24 * 3600:
        return dataset, "preaggregated_snapshot"

    monthly = dataset.resample(time="MS").mean(skipna=True)
    availability = pd.DatetimeIndex(pd.to_datetime(monthly.time.values)) + pd.offsets.MonthBegin(1)
    monthly = monthly.assign_coords(time=availability.values.astype("datetime64[ns]"))
    if complete_only:
        latest_available = times[-1] + pd.to_timedelta(median_seconds, unit="s")
        monthly = monthly.sel(time=monthly.time <= latest_available.to_datetime64())
    return monthly, "calendar_month_mean_available_next_month"


def _load_integrated_monthly(
    path: str | Path,
    months: pd.DatetimeIndex,
    *,
    availability_shift: bool,
) -> tuple[np.ndarray, list[str]]:
    value = str(path)
    frame = pd.read_parquet(value) if value.endswith(".parquet") else pd.read_csv(value)
    if "time" not in frame:
        raise ValueError("Integrated main-system data requires a 'time' column")
    frame["time"] = pd.to_datetime(frame["time"])
    numeric = [
        name
        for name in frame.select_dtypes(include=[np.number]).columns
        if name not in {"init_time_ns"}
    ]
    if not numeric:
        raise ValueError("Integrated data has no numeric climate/track features")
    frame["month"] = frame["time"].dt.to_period("M").dt.to_timestamp()
    if availability_shift:
        frame["month"] = frame["month"] + pd.offsets.MonthBegin(1)
    monthly = frame.groupby("month", sort=True)[numeric].mean()
    aligned = monthly.reindex(months)
    values = aligned.to_numpy(np.float32)
    fill = np.nanmean(values, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    values = np.where(np.isfinite(values), values, fill[None, :])
    return values.astype(np.float32), [f"integrated:{name}" for name in numeric]


def prepare_monthly_archive(
    fields: str | Path,
    output: str | Path,
    *,
    integrated: str | Path | None = None,
    variables: tuple[str, ...] | None = None,
    target_lat_points: int = 18,
    target_lon_points: int = 36,
) -> tuple[Path, Path]:
    """Aggregate fields and the main-system table into one monthly state archive."""
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
        dataset = dataset[list(selected_names)].sortby("time")
        monthly, aggregation = aggregate_monthly_fields(dataset, complete_only=True)
        monthly = _coarsen_global_fields(
            monthly, target_lat_points, target_lon_points
        ).load()

    months = pd.DatetimeIndex(pd.to_datetime(monthly.time.values))
    if len(months) < 2:
        raise ValueError("At least two monthly field states are required")

    blocks = []
    variable_schema = []
    offset = 0
    feature_names = []
    for name in selected_names:
        array = monthly[name]
        dims = tuple(dim for dim in array.dims if dim != "time")
        array = array.transpose("time", *dims)
        values = np.asarray(array.values, dtype=np.float32).reshape(len(months), -1)
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
        integrated_values, integrated_names = _load_integrated_monthly(
            integrated,
            months,
            availability_shift=aggregation
            == "calendar_month_mean_available_next_month",
        )
        blocks.append(integrated_values)
        feature_names.extend(integrated_names)

    states = np.concatenate(blocks, axis=1).astype(np.float32)
    observed_mask = np.isfinite(states)
    feature_fill = np.nanmean(states, axis=0)
    feature_fill = np.where(np.isfinite(feature_fill), feature_fill, 0.0)
    states = np.where(observed_mask, states, feature_fill[None, :]).astype(np.float32)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        observed_mask=observed_mask.astype(np.float32),
        times=months.values.astype("datetime64[ns]"),
        feature_names=np.asarray(feature_names),
    )
    schema_path = output_path.with_suffix(".schema.json")
    schema = {
        "format": "climate_diffusion.monthly_state.v1",
        "source_fields": str(fields),
        "source_integrated": None if integrated is None else str(integrated),
        "state_dim": int(states.shape[1]),
        "field_dim": int(field_dim),
        "integrated_feature_names": feature_names[field_dim:],
        "variables": variable_schema,
        "monthly_frequency": "calendar_month",
        "state_time_semantics": "availability_time",
        "aggregation": aggregation,
        "target_lat_points": target_lat_points,
        "target_lon_points": target_lon_points,
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path, schema_path


def load_monthly_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        states = archive["states"].astype(np.float32)
        times = archive["times"].astype("datetime64[ns]")
    schema = json.loads(
        archive_path.with_suffix(".schema.json").read_text(encoding="utf-8")
    )
    if states.ndim != 2 or states.shape[1] != schema["state_dim"]:
        raise ValueError("Monthly archive does not match its schema")
    return states, times, schema


@dataclass(frozen=True)
class MonthlyWindow:
    history: torch.Tensor
    target: torch.Tensor


class MonthlyWindowDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        history_months: int,
        lead_months: int = 1,
        indices: list[int] | None = None,
    ):
        if min(history_months, lead_months) < 1:
            raise ValueError("history_months and lead_months must be positive")
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.history_months = history_months
        self.lead_months = lead_months
        last_start = len(states) - history_months - lead_months + 1
        available = list(range(max(0, last_start)))
        self.indices = available if indices is None else indices
        if any(index not in available for index in self.indices):
            raise ValueError("Window index is outside the available monthly states")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.indices[index]
        target_index = start + self.history_months + self.lead_months - 1
        return {
            "history": self.states[start : start + self.history_months],
            "target": self.states[target_index],
        }


def vectorize_dataset(
    dataset: xr.Dataset,
    schema: dict[str, Any],
    *,
    integrated_defaults: np.ndarray | None = None,
) -> np.ndarray:
    """Map monthly fields to the exact state layout stored in a training schema."""
    dataset = _normalise_coordinates(dataset)
    if "time" not in dataset.dims:
        dataset = dataset.expand_dims(time=[pd.Timestamp.utcnow().to_datetime64()])
    vectors = []
    for time_index in range(dataset.sizes["time"]):
        state = np.zeros(schema["state_dim"], dtype=np.float32)
        field_dim = int(schema["field_dim"])
        if integrated_defaults is not None:
            defaults = np.asarray(integrated_defaults, dtype=np.float32)
            if defaults.shape != (schema["state_dim"],):
                raise ValueError("Integrated defaults do not match the training state")
            state[field_dim:] = defaults[field_dim:]
        for variable in schema["variables"]:
            name = variable["name"]
            if name not in dataset:
                raise ValueError(f"Initial state is missing trained variable {name!r}")
            array = dataset[name].isel(time=time_index)
            interpolation = {}
            for dim, values in variable["coords"].items():
                if dim not in array.coords:
                    continue
                requested = np.asarray(values)
                current = np.asarray(array.coords[dim].values)
                if np.array_equal(current, requested):
                    continue
                if current.size < 2:
                    raise ValueError(
                        f"Cannot interpolate singleton coordinate {dim!r} to the training grid"
                    )
                interpolation[dim] = requested
            if interpolation:
                array = array.interp(interpolation)
            values = np.asarray(
                array.transpose(*variable["dims"]).values, dtype=np.float32
            ).reshape(-1)
            start, end = variable["slice"]
            if values.size != end - start:
                raise ValueError(f"Variable {name!r} shape does not match training schema")
            state[start:end] = np.nan_to_num(values)
        for offset, feature_name in enumerate(schema["integrated_feature_names"]):
            source_name = feature_name.removeprefix("integrated:")
            if source_name in dataset:
                value = dataset[source_name]
                if "time" in value.dims:
                    value = value.isel(time=time_index)
                if value.size == 1:
                    state[field_dim + offset] = float(value.values)
        vectors.append(state)
    return np.stack(vectors)


def reconstruct_dataset(
    state: np.ndarray,
    schema: dict[str, Any],
    valid_time: pd.Timestamp,
) -> xr.Dataset:
    data_vars = {}
    coords: dict[str, Any] = {"time": [valid_time.to_datetime64()]}
    for variable in schema["variables"]:
        start, end = variable["slice"]
        dims = tuple(variable["dims"])
        values = state[start:end].reshape(variable["shape"])
        for dim, coordinate in variable["coords"].items():
            coords.setdefault(dim, coordinate)
        data_vars[variable["name"]] = (
            ("time", *dims),
            values[None, ...],
            variable.get("attrs", {}),
        )
    return xr.Dataset(data_vars=data_vars, coords=coords)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine gridded fields and main-system data into monthly states"
    )
    parser.add_argument("--fields", required=True, help="NetCDF or Zarr climate fields")
    parser.add_argument("--integrated", help="Integrated main-system Parquet or CSV")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--target-lat-points", type=int, default=18)
    parser.add_argument("--target-lon-points", type=int, default=36)
    parser.add_argument("--output", default="data/monthly_climate_states.npz")
    args = parser.parse_args(argv)
    archive, schema = prepare_monthly_archive(
        args.fields,
        args.output,
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

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

from .validation import require_finite_numpy, require_no_inf_numpy

COORDINATE_ALIASES = {
    "valid_time": "time",
    "datetime": "time",
    "latitude": "lat",
    "longitude": "lon",
}


def _validate_source_times(times: pd.DatetimeIndex) -> None:
    if times.hasnans:
        raise ValueError("Climate source time coordinate contains NaT")
    duplicated = times.duplicated(keep=False)
    if duplicated.any():
        examples = [str(value) for value in times[duplicated][:3]]
        raise ValueError(f"Climate source contains duplicate timestamps: {examples}")


def validate_monthly_times(times: np.ndarray | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Require unique, strictly increasing, consecutive calendar-month labels."""
    index = pd.DatetimeIndex(pd.to_datetime(times))
    if len(index) == 0:
        raise ValueError("Monthly archive has no timestamps")
    if index.hasnans:
        raise ValueError("Monthly archive timestamps contain NaT")
    if index.has_duplicates:
        duplicates = [str(value) for value in index[index.duplicated(keep=False)][:3]]
        raise ValueError(f"Monthly archive contains duplicate timestamps: {duplicates}")
    if not index.is_monotonic_increasing:
        raise ValueError("Monthly archive timestamps must be strictly increasing")
    periods = index.to_period("M")
    if periods.duplicated().any():
        raise ValueError("Monthly archive contains more than one state for a calendar month")
    ordinal = periods.astype("int64")
    gaps = np.diff(ordinal)
    if len(gaps) and np.any(gaps != 1):
        location = int(np.flatnonzero(gaps != 1)[0])
        raise ValueError(
            "Monthly archive has a missing/non-consecutive calendar month between "
            f"{index[location]} and {index[location + 1]}"
        )
    return index


def _reject_fully_missing_months(
    values: np.ndarray, months: pd.DatetimeIndex, variable: str
) -> None:
    flattened = np.asarray(values).reshape(len(months), -1)
    missing = ~np.isfinite(flattened).any(axis=1)
    if missing.any():
        examples = [str(value) for value in months[missing][:3]]
        raise ValueError(
            f"Variable {variable!r} has fully missing calendar month(s): {examples}"
        )


def _require_no_inf_monthly(values: np.ndarray, name: str) -> None:
    """Scan month-by-month so a 0.25-degree memmap does not allocate a global mask."""
    for index in range(len(values)):
        require_no_inf_numpy(np.asarray(values[index]), f"{name} month_index={index}")


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
    source_times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    _validate_source_times(source_times)
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
    if frame["time"].isna().any():
        raise ValueError("Integrated main-system data contains invalid timestamps")
    numeric = [
        name
        for name in frame.select_dtypes(include=[np.number]).columns
        if name not in {"init_time_ns"}
    ]
    if not numeric:
        raise ValueError("Integrated data has no numeric climate/track features")
    raw_values = frame[numeric].to_numpy(np.float32)
    require_no_inf_numpy(raw_values, "integrated main-system data")
    frame["month"] = frame["time"].dt.to_period("M").dt.to_timestamp()
    if availability_shift:
        frame["month"] = frame["month"] + pd.offsets.MonthBegin(1)
    monthly = frame.groupby("month", sort=True)[numeric].mean()
    aligned = monthly.reindex(months)
    values = aligned.to_numpy(np.float32)
    require_no_inf_numpy(values, "monthly integrated main-system data")
    return values.astype(np.float32), [f"integrated:{name}" for name in numeric]


def prepare_monthly_archive(
    fields: str | Path | xr.Dataset,
    output: str | Path,
    *,
    integrated: str | Path | None = None,
    variables: tuple[str, ...] | None = None,
    target_lat_points: int = 18,
    target_lon_points: int = 36,
    layout: str = "vector",
    source_label: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Aggregate fields and the main-system table into one monthly state archive."""
    if layout not in {"vector", "spatial"}:
        raise ValueError("layout must be 'vector' or 'spatial'")
    if min(target_lat_points, target_lon_points) < 1:
        raise ValueError("Target grid dimensions must be positive")
    opener = fields if isinstance(fields, xr.Dataset) else _open_dataset(fields)
    with opener as source:
        dataset = _normalise_coordinates(source)
        if "time" not in dataset.coords:
            raise ValueError("Gridded climate fields require a time coordinate")
        selected_names = tuple(variables or dataset.data_vars)
        missing = sorted(set(selected_names).difference(dataset.data_vars))
        if missing:
            raise ValueError(f"Missing requested variables: {missing}")
        dataset = dataset[list(selected_names)]
        monthly, aggregation = aggregate_monthly_fields(dataset, complete_only=True)
        monthly = _coarsen_global_fields(monthly, target_lat_points, target_lon_points)
        if layout == "vector":
            monthly = monthly.load()

    months = pd.DatetimeIndex(pd.to_datetime(monthly.time.values))
    if len(months) < 2:
        raise ValueError("At least two monthly field states are required")
    months = validate_monthly_times(months)

    if layout == "spatial":
        return _write_spatial_archive(
            monthly,
            selected_names,
            months,
            output,
            fields=source_label or str(fields),
            integrated=integrated,
            aggregation=aggregation,
            target_lat_points=target_lat_points,
            target_lon_points=target_lon_points,
            source_metadata=source_metadata,
        )

    blocks = []
    variable_schema = []
    offset = 0
    feature_names = []
    for name in selected_names:
        array = monthly[name]
        dims = tuple(dim for dim in array.dims if dim != "time")
        array = array.transpose("time", *dims)
        values = np.asarray(array.values, dtype=np.float32).reshape(len(months), -1)
        require_no_inf_numpy(values, f"monthly field variable {name!r}")
        _reject_fully_missing_months(values, months, name)
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
    require_no_inf_numpy(states, "vector monthly archive")
    observed_mask = np.isfinite(states)

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
        "format": "climate_diffusion.monthly_state.v2",
        "layout": "vector",
        "source_fields": source_label or str(fields),
        "source_metadata": source_metadata,
        "source_integrated": None if integrated is None else str(integrated),
        "state_dim": int(states.shape[1]),
        "field_dim": int(field_dim),
        "integrated_feature_names": feature_names[field_dim:],
        "variables": variable_schema,
        "monthly_frequency": "calendar_month",
        "state_time_semantics": "availability_time",
        "aggregation": aggregation,
        "missing_value_policy": "preserve_nan_impute_from_train_only",
        "infinity_policy": "fail_fast",
        # Every value was scanned above, so loading need not rescan the archive.
        "inf_checked": True,
        "target_lat_points": target_lat_points,
        "target_lon_points": target_lon_points,
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path, schema_path


def _write_spatial_archive(
    monthly: xr.Dataset,
    selected_names: tuple[str, ...],
    months: pd.DatetimeIndex,
    output: str | Path,
    *,
    fields: str | Path,
    integrated: str | Path | None,
    aggregation: str,
    target_lat_points: int,
    target_lon_points: int,
    source_metadata: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Write a memory-mapped spatial archive without materialising every month."""
    if "lat" not in monthly.dims or "lon" not in monthly.dims:
        raise ValueError("Spatial layout requires latitude and longitude dimensions")
    height, width = int(monthly.sizes["lat"]), int(monthly.sizes["lon"])
    variable_schema: list[dict[str, Any]] = []
    channel_offset = 0
    channel_names: list[str] = []
    for name in selected_names:
        array = monthly[name]
        if "lat" not in array.dims or "lon" not in array.dims:
            raise ValueError(f"Spatial variable {name!r} must contain lat and lon dimensions")
        non_spatial = tuple(
            dim for dim in array.dims if dim not in {"time", "lat", "lon"}
        )
        non_spatial_shape = [int(array.sizes[dim]) for dim in non_spatial]
        channels = int(np.prod(non_spatial_shape, dtype=np.int64)) if non_spatial else 1
        variable_schema.append(
            {
                "name": name,
                "dims": list(non_spatial) + ["lat", "lon"],
                "non_spatial_dims": list(non_spatial),
                "non_spatial_shape": non_spatial_shape,
                "shape": non_spatial_shape + [height, width],
                "channel_slice": [channel_offset, channel_offset + channels],
                "coords": {
                    dim: _json_values(np.asarray(array.coords[dim].values))
                    for dim in (*non_spatial, "lat", "lon")
                    if dim in array.coords and array.coords[dim].dims == (dim,)
                },
                "attrs": {key: str(value) for key, value in array.attrs.items()},
            }
        )
        channel_names.extend(f"field:{name}:{index}" for index in range(channels))
        channel_offset += channels

    output_path = Path(output)
    if output_path.suffix:
        raise ValueError(
            "Spatial archives are directories; use a path such as data/monthly_climate_spatial"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        output_path / "states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(months), channel_offset, height, width),
    )
    observed_mask = np.lib.format.open_memmap(
        output_path / "observed_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(months), channel_offset, height, width),
    )
    observed_fraction = np.zeros((len(months), channel_offset), dtype=np.float32)
    for time_index in range(len(months)):
        for variable in variable_schema:
            array = monthly[variable["name"]].isel(time=time_index).transpose(
                *variable["dims"]
            )
            values = np.asarray(array.values, dtype=np.float32).reshape(-1, height, width)
            start, end = variable["channel_slice"]
            context = (
                f"spatial variable {variable['name']!r} month={months[time_index]}"
            )
            require_no_inf_numpy(values, context)
            finite = np.isfinite(values)
            missing_channels = ~finite.any(axis=(1, 2))
            if missing_channels.any():
                channels = np.flatnonzero(missing_channels).tolist()
                raise ValueError(
                    f"{context} has fully missing channel(s): {channels[:8]}"
                )
            observed_fraction[time_index, start:end] = finite.mean(axis=(1, 2))
            states[time_index, start:end] = values
            observed_mask[time_index, start:end] = finite
    states.flush()
    observed_mask.flush()
    np.save(output_path / "times.npy", months.values.astype("datetime64[ns]"))
    np.save(output_path / "observed_fraction.npy", observed_fraction)

    auxiliary_names: list[str] = []
    auxiliary_values = np.empty((len(months), 0), dtype=np.float32)
    if integrated is not None:
        auxiliary_values, auxiliary_names = _load_integrated_monthly(
            integrated,
            months,
            availability_shift=aggregation == "calendar_month_mean_available_next_month",
        )
    np.save(output_path / "auxiliary.npy", auxiliary_values.astype(np.float32))
    schema_path = output_path / "schema.json"
    schema = {
        "format": "climate_diffusion.monthly_spatial.v3",
        "layout": "spatial",
        "storage": "npy_memmap_directory",
        "source_fields": str(fields),
        "source_metadata": source_metadata,
        "source_integrated": None if integrated is None else str(integrated),
        "state_dim": int(channel_offset * height * width),
        "spatial_channels": channel_offset,
        "grid_shape": [height, width],
        "channel_names": channel_names,
        "auxiliary_feature_names": auxiliary_names,
        "auxiliary_dim": len(auxiliary_names),
        "variables": variable_schema,
        "coords": {
            "lat": _json_values(np.asarray(monthly.lat.values)),
            "lon": _json_values(np.asarray(monthly.lon.values)),
        },
        "monthly_frequency": "calendar_month",
        "state_time_semantics": "availability_time",
        "aggregation": aggregation,
        "missing_value_policy": "preserve_nan_impute_from_train_only",
        "infinity_policy": "fail_fast",
        # Every month was scanned while writing, so loading need not rescan.
        "inf_checked": True,
        "target_lat_points": target_lat_points,
        "target_lon_points": target_lon_points,
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path, schema_path


def load_monthly_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    archive_path = Path(path)
    if archive_path.is_dir():
        schema = json.loads((archive_path / "schema.json").read_text(encoding="utf-8"))
        states = np.load(archive_path / "states.npy", mmap_mode="r")
        times = np.load(archive_path / "times.npy", mmap_mode="r").astype("datetime64[ns]")
        expected = (
            schema["spatial_channels"],
            *schema["grid_shape"],
        )
        if states.ndim != 4 or states.shape[1:] != expected:
            raise ValueError("Spatial monthly archive does not match its schema")
        if len(states) != len(times):
            raise ValueError("Spatial archive state/time lengths do not match")
        validate_monthly_times(times)
        if not schema.get("inf_checked"):
            # Pre-`inf_checked` archives have no recorded guarantee, so scan.
            _require_no_inf_monthly(states, "spatial monthly archive")
        return states, times, schema
    with np.load(archive_path, allow_pickle=False) as archive:
        states = archive["states"].astype(np.float32)
        times = archive["times"].astype("datetime64[ns]")
    schema = json.loads(
        archive_path.with_suffix(".schema.json").read_text(encoding="utf-8")
    )
    if states.ndim != 2 or states.shape[1] != schema["state_dim"]:
        raise ValueError("Monthly archive does not match its schema")
    if len(states) != len(times):
        raise ValueError("Vector archive state/time lengths do not match")
    validate_monthly_times(times)
    if not schema.get("inf_checked"):
        _require_no_inf_monthly(states, "vector monthly archive")
    return states, times, schema


def load_auxiliary_states(path: str | Path, schema: dict[str, Any]) -> np.ndarray | None:
    archive_path = Path(path)
    if schema.get("layout") != "spatial" or not archive_path.is_dir():
        return None
    values = np.load(archive_path / "auxiliary.npy", mmap_mode="r")
    if values.ndim != 2 or values.shape[1] != int(schema.get("auxiliary_dim", 0)):
        raise ValueError("Spatial auxiliary archive does not match its schema")
    _require_no_inf_monthly(values, "spatial auxiliary archive")
    return values


POSITIONAL_CHANNEL_NAMES = ("sin_lat", "cos_lon", "sin_lon")


def positional_grid(schema: dict[str, Any]) -> np.ndarray | None:
    """Static ``[3, lat, lon]`` coordinate planes for a spatial archive.

    A randomly cropped patch is otherwise indistinguishable between the equator
    and a pole, though the physics is not. Longitude is encoded as a sine/cosine
    pair so the dateline carries no discontinuity.
    """
    if schema.get("layout") != "spatial":
        return None
    latitudes = np.asarray(schema["coords"]["lat"], dtype=np.float32)
    longitudes = np.asarray(schema["coords"]["lon"], dtype=np.float32)
    lat_radians = np.deg2rad(latitudes)[:, None]
    lon_radians = np.deg2rad(longitudes)[None, :]
    height, width = len(latitudes), len(longitudes)
    planes = np.empty((3, height, width), dtype=np.float32)
    planes[0] = np.broadcast_to(np.sin(lat_radians), (height, width))
    planes[1] = np.broadcast_to(np.cos(lon_radians), (height, width))
    planes[2] = np.broadcast_to(np.sin(lon_radians), (height, width))
    return planes


def load_observed_fraction(
    path: str | Path, states: np.ndarray, schema: dict[str, Any]
) -> np.ndarray | None:
    """Per-month, per-channel observed fractions recorded when the archive was built.

    Deciding whether a window is observed enough only needs these numbers. Without
    them the mask memmap is rescanned once per window, which re-reads the whole
    archive several times over before the first gradient step.
    """
    archive_path = Path(path)
    if schema.get("layout") != "spatial" or not archive_path.is_dir():
        return None
    fraction_path = archive_path / "observed_fraction.npy"
    if not fraction_path.is_file():
        return None
    values = np.load(fraction_path, mmap_mode="r")
    expected = (len(states), int(schema["spatial_channels"]))
    if values.shape != expected:
        raise ValueError("Observed-fraction array does not match the archive shape")
    return values


def load_observation_mask(
    path: str | Path, states: np.ndarray, schema: dict[str, Any]
) -> np.ndarray | None:
    """Load a persisted mask, falling back to finite values for legacy archives."""
    archive_path = Path(path)
    if archive_path.is_dir():
        mask_path = archive_path / "observed_mask.npy"
        mask = np.load(mask_path, mmap_mode="r") if mask_path.is_file() else None
    else:
        with np.load(archive_path, allow_pickle=False) as archive:
            mask = (
                archive["observed_mask"].astype(bool)
                if "observed_mask" in archive.files
                else np.isfinite(states)
            )
    if mask is not None and mask.shape != states.shape:
        raise ValueError("Observation mask does not match monthly states")
    return mask


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
        *,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        observation_mask: np.ndarray | None = None,
        times: np.ndarray | None = None,
        min_observed_fraction: float = 0.0,
        auxiliary_states: np.ndarray | None = None,
        auxiliary_mean: np.ndarray | None = None,
        auxiliary_scale: np.ndarray | None = None,
        patch_size: tuple[int, int] | None = None,
        random_crop: bool = False,
        observed_fraction: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
    ):
        if min(history_months, lead_months) < 1:
            raise ValueError("history_months and lead_months must be positive")
        self.states = states
        self.observed_fraction = observed_fraction
        self.coordinates = (
            None if coordinates is None else np.asarray(coordinates, dtype=np.float32)
        )
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)
        self.observation_mask = observation_mask
        self.times = None if times is None else np.asarray(times, dtype="datetime64[ns]")
        if self.times is not None:
            if len(self.times) != len(states):
                raise ValueError("Monthly state/time lengths do not match")
            validate_monthly_times(self.times)
        if not 0.0 <= min_observed_fraction < 1.0:
            raise ValueError("min_observed_fraction must be in [0, 1)")
        self.min_observed_fraction = min_observed_fraction
        self.auxiliary_states = auxiliary_states
        self.auxiliary_mean = auxiliary_mean
        self.auxiliary_scale = auxiliary_scale
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.history_months = history_months
        self.lead_months = lead_months
        last_start = len(states) - history_months - lead_months + 1
        available = list(range(max(0, last_start)))
        requested = available if indices is None else indices
        if any(index not in available for index in requested):
            raise ValueError("Window index is outside the available monthly states")
        self.indices = [value for value in requested if self._window_is_observed(value)]
        if indices and not self.indices:
            raise ValueError("No monthly windows satisfy the observation-mask contract")

    def _mask_slice(self, selection: Any) -> np.ndarray:
        values = np.asarray(self.states[selection])
        if self.observation_mask is None:
            return np.isfinite(values)
        return np.asarray(self.observation_mask[selection], dtype=bool)

    def _window_is_observed(self, start: int) -> bool:
        target_index = start + self.history_months + self.lead_months - 1
        if self.observed_fraction is not None:
            # Identical to the mask scan below: each stored value is already the
            # spatial mean for that month and channel, and every month covers the
            # same number of cells.
            recorded = np.concatenate(
                (
                    np.asarray(self.observed_fraction[start : start + self.history_months]),
                    np.asarray(self.observed_fraction[target_index])[None],
                ),
                axis=0,
            )
            return bool(np.all(recorded.mean(axis=0) > self.min_observed_fraction))
        history_mask = self._mask_slice(slice(start, start + self.history_months))
        target_mask = self._mask_slice(target_index)[None]
        combined = np.concatenate((history_mask, target_mask), axis=0)
        if combined.ndim == 2:
            fractions = combined.mean(axis=0)
        else:
            fractions = combined.mean(axis=(0, *range(2, combined.ndim)))
        return bool(np.all(fractions > self.min_observed_fraction))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.indices[index]
        target_index = start + self.history_months + self.lead_months - 1
        history = np.asarray(
            self.states[start : start + self.history_months], dtype=np.float32
        )
        target = np.asarray(self.states[target_index], dtype=np.float32)
        history_mask = self._mask_slice(slice(start, start + self.history_months))
        target_mask = self._mask_slice(target_index)
        coordinates = self.coordinates
        require_no_inf_numpy(history, f"history window start={start}")
        require_no_inf_numpy(target, f"target window start={start}")
        if self.mean is not None and self.scale is not None:
            history = np.where(history_mask, history, self.mean)
            target = np.where(target_mask, target, self.mean)
            history = (history - self.mean) / self.scale
            target = (target - self.mean) / self.scale
        require_finite_numpy(history, f"normalized history window start={start}")
        require_finite_numpy(target, f"normalized target window start={start}")
        if self.patch_size is not None:
            if history.ndim != 4:
                raise ValueError("patch_size is only valid for [time,channel,lat,lon] states")
            patch_height, patch_width = self.patch_size
            height, width = history.shape[-2:]
            if patch_height > height or patch_width > width:
                raise ValueError("patch_size cannot exceed the archive grid")
            if self.random_crop:
                top = int(torch.randint(height - patch_height + 1, (1,)).item())
                left = int(torch.randint(width, (1,)).item())
            else:
                top = (height - patch_height) // 2
                left = (width - patch_width) // 2
            lon_index = np.arange(left, left + patch_width) % width
            history = np.take(history[..., top : top + patch_height, :], lon_index, axis=-1)
            target = np.take(target[..., top : top + patch_height, :], lon_index, axis=-1)
            target_mask = np.take(
                target_mask[..., top : top + patch_height, :], lon_index, axis=-1
            )
            if coordinates is not None:
                # The crop is what tells the model where on the globe it is, so
                # the coordinate planes have to follow it exactly.
                coordinates = np.take(
                    coordinates[..., top : top + patch_height, :], lon_index, axis=-1
                )
        item = {
            "history": torch.as_tensor(history.copy(), dtype=torch.float32),
            "target": torch.as_tensor(target.copy(), dtype=torch.float32),
            "target_mask": torch.as_tensor(target_mask.copy(), dtype=torch.bool),
        }
        if coordinates is not None:
            item["coordinates"] = torch.as_tensor(coordinates.copy(), dtype=torch.float32)
        if self.auxiliary_states is not None:
            auxiliary = np.asarray(
                self.auxiliary_states[start : start + self.history_months],
                dtype=np.float32,
            )
            if self.auxiliary_mean is not None and self.auxiliary_scale is not None:
                require_no_inf_numpy(auxiliary, f"auxiliary history window start={start}")
                auxiliary = np.where(
                    np.isfinite(auxiliary), auxiliary, self.auxiliary_mean
                )
                auxiliary = (auxiliary - self.auxiliary_mean) / self.auxiliary_scale
            require_finite_numpy(auxiliary, f"normalized auxiliary window start={start}")
            item["history_auxiliary"] = torch.as_tensor(
                auxiliary.copy(), dtype=torch.float32
            )
        return item


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
        state = np.full(schema["state_dim"], np.nan, dtype=np.float32)
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
            require_no_inf_numpy(values, f"inference variable {name!r}")
            state[start:end] = values
        for offset, feature_name in enumerate(schema["integrated_feature_names"]):
            source_name = feature_name.removeprefix("integrated:")
            if source_name in dataset:
                value = dataset[source_name]
                if "time" in value.dims:
                    value = value.isel(time=time_index)
                if value.size == 1:
                    scalar = float(value.values)
                    if np.isinf(scalar):
                        raise ValueError(
                            f"Inference auxiliary {source_name!r} contains Inf"
                        )
                    state[field_dim + offset] = scalar
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


def spatialize_dataset(dataset: xr.Dataset, schema: dict[str, Any]) -> np.ndarray:
    """Map an xarray history to ``[time, channel, lat, lon]`` spatial states."""
    if schema.get("layout") != "spatial":
        raise ValueError("spatialize_dataset requires a spatial schema")
    dataset = _normalise_coordinates(dataset)
    if "time" not in dataset.dims:
        dataset = dataset.expand_dims(time=[pd.Timestamp.utcnow().to_datetime64()])
    requested_lat = np.asarray(schema["coords"]["lat"])
    requested_lon = np.asarray(schema["coords"]["lon"])
    result = np.zeros(
        (
            dataset.sizes["time"],
            schema["spatial_channels"],
            *schema["grid_shape"],
        ),
        dtype=np.float32,
    )
    for variable in schema["variables"]:
        name = variable["name"]
        if name not in dataset:
            raise ValueError(f"Initial state is missing trained variable {name!r}")
        array = dataset[name]
        interpolation = {}
        if not np.array_equal(np.asarray(array.lat.values), requested_lat):
            interpolation["lat"] = requested_lat
        if not np.array_equal(np.asarray(array.lon.values), requested_lon):
            interpolation["lon"] = requested_lon
        if interpolation:
            array = array.interp(interpolation)
        values = np.asarray(
            array.transpose("time", *variable["dims"]).values,
            dtype=np.float32,
        ).reshape(dataset.sizes["time"], -1, *schema["grid_shape"])
        start, end = variable["channel_slice"]
        if values.shape[1] != end - start:
            raise ValueError(f"Variable {name!r} channels do not match training schema")
        require_no_inf_numpy(values, f"spatial inference variable {name!r}")
        result[:, start:end] = values
    return result


def auxiliary_from_dataset(
    dataset: xr.Dataset,
    schema: dict[str, Any],
    defaults: np.ndarray,
) -> np.ndarray:
    names = schema.get("auxiliary_feature_names", [])
    result = np.broadcast_to(
        np.asarray(defaults, dtype=np.float32), (dataset.sizes["time"], len(names))
    ).copy()
    for index, feature_name in enumerate(names):
        source_name = feature_name.removeprefix("integrated:")
        if source_name in dataset:
            values = dataset[source_name]
            if "time" in values.dims and values.ndim == 1:
                raw = np.asarray(values.values, dtype=np.float32)
                require_no_inf_numpy(raw, f"inference auxiliary {source_name!r}")
                result[:, index] = raw
    return result


def reconstruct_spatial_dataset(
    state: np.ndarray,
    schema: dict[str, Any],
    valid_time: pd.Timestamp,
) -> xr.Dataset:
    if schema.get("layout") != "spatial":
        raise ValueError("reconstruct_spatial_dataset requires a spatial schema")
    coords: dict[str, Any] = {
        "time": [valid_time.to_datetime64()],
        "lat": schema["coords"]["lat"],
        "lon": schema["coords"]["lon"],
    }
    data_vars = {}
    for variable in schema["variables"]:
        start, end = variable["channel_slice"]
        values = np.asarray(state[start:end]).reshape(variable["shape"])
        for dim, coordinate in variable["coords"].items():
            coords.setdefault(dim, coordinate)
        data_vars[variable["name"]] = (
            ("time", *variable["dims"]),
            values[None],
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
    parser.add_argument("--layout", choices=("vector", "spatial"), default="vector")
    parser.add_argument("--output", default="data/monthly_climate_states.npz")
    args = parser.parse_args(argv)
    archive, schema = prepare_monthly_archive(
        args.fields,
        args.output,
        integrated=args.integrated,
        variables=None if not args.variables else tuple(args.variables),
        target_lat_points=args.target_lat_points,
        target_lon_points=args.target_lon_points,
        layout=args.layout,
    )
    print(f"archive={archive}")
    print(f"schema={schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

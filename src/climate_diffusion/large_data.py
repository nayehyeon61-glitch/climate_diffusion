"""Disk-bounded sharded fixed-step archives for large Flow Matching runs.

The legacy NPZ archive is convenient for small experiments but materializes the
full state matrix in memory.  This module keeps the same state/schema contract
while writing independent NPY shards that can be memory-mapped during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .data import _coarsen_global_fields, _json_values, _normalise_coordinates

GIB = 1024 ** 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


def select_source_files(
    paths: Iterable[str | Path], *, source_budget_gb: float | None = 500.0
) -> tuple[list[Path], int]:
    """Select whole local source files without exceeding a storage budget.

    Files are sorted lexicographically so date-named ERA5/HRES files remain
    reproducible.  A single Zarr directory is allowed and budget checked as one
    source.  The function never silently truncates a source file.
    """
    sources = sorted({Path(value).expanduser().resolve() for value in paths})
    if not sources:
        raise ValueError("At least one --fields path is required")
    budget = None if source_budget_gb is None else int(source_budget_gb * GIB)
    selected: list[Path] = []
    total = 0
    for path in sources:
        size = _local_size(path)
        if budget is not None and selected and total + size > budget:
            break
        if budget is not None and not selected and size > budget:
            raise ValueError(
                f"First source {path} is {size / GIB:.2f} GiB, larger than "
                f"the {source_budget_gb:.2f} GiB budget; split the source first"
            )
        selected.append(path)
        total += size
    if not selected:
        raise ValueError("No source files fit inside the requested data budget")
    return selected, total


def _open_sources(paths: list[Path], *, time_chunk: int) -> xr.Dataset:
    if len(paths) == 1 and str(paths[0]).endswith(".zarr"):
        return xr.open_zarr(str(paths[0]), chunks={"time": time_chunk})
    if len(paths) == 1:
        return xr.open_dataset(paths[0], chunks={"time": time_chunk})
    return xr.open_mfdataset(
        [str(path) for path in paths],
        combine="by_coords",
        chunks={"time": time_chunk},
        parallel=False,
    )


def _exact_fixed_times(dataset: xr.Dataset, step_hours: int) -> pd.DatetimeIndex:
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values)).sort_values().unique()
    if len(times) < 2:
        raise ValueError("At least two atmospheric snapshots are required")
    step = pd.Timedelta(hours=step_hours)
    selected = [times[0]]
    cursor = times[0] + step
    available = {int(value.value) for value in times}
    while cursor <= times[-1]:
        if int(cursor.value) not in available:
            raise ValueError(
                f"Source has a gap at {cursor}; exact {step_hours}h cadence is required"
            )
        selected.append(cursor)
        cursor += step
    return pd.DatetimeIndex(selected)


def _schema_and_state(chunk: xr.Dataset, variable_names: tuple[str, ...]):
    blocks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    variable_schema: list[dict[str, Any]] = []
    feature_names: list[str] = []
    offset = 0
    time_count = chunk.sizes["time"]
    for name in variable_names:
        array = chunk[name]
        dims = tuple(dim for dim in array.dims if dim != "time")
        array = array.transpose("time", *dims)
        values = np.asarray(array.values, dtype=np.float32).reshape(time_count, -1)
        observed = np.isfinite(values)
        blocks.append(np.where(observed, values, 0.0).astype(np.float32))
        masks.append(observed.astype(np.uint8))
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
    return (
        np.concatenate(blocks, axis=1),
        np.concatenate(masks, axis=1),
        variable_schema,
        feature_names,
    )


def prepare_sharded_fixed_step_archive(
    fields: Iterable[str | Path],
    output_dir: str | Path,
    *,
    source_budget_gb: float | None = 500.0,
    step_hours: int = 6,
    shard_steps: int = 256,
    variables: tuple[str, ...] | None = None,
    target_lat_points: int = 45,
    target_lon_points: int = 90,
    max_state_dim: int = 250_000,
) -> Path:
    """Create mmap-friendly fixed-step shards from up to ``source_budget_gb``.

    The 500 GiB profile intentionally coarsens before flattening because the
    current dense autoencoder is not the 0.25-degree operator backend.  Set a
    larger grid only after checking the resulting state dimension/parameter
    count; ``max_state_dim`` is a fail-fast guard against accidental OOM runs.
    """
    if shard_steps < 2:
        raise ValueError("shard_steps must be at least 2")
    if min(target_lat_points, target_lon_points) < 1:
        raise ValueError("Target grid dimensions must be positive")
    selected_sources, selected_bytes = select_source_files(
        fields, source_budget_gb=source_budget_gb
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")

    source = _normalise_coordinates(_open_sources(selected_sources, time_chunk=shard_steps))
    try:
        if "time" not in source.coords:
            raise ValueError("Gridded fields require a time coordinate")
        variable_names = tuple(variables or source.data_vars)
        missing = sorted(set(variable_names).difference(source.data_vars))
        if missing:
            raise ValueError(f"Missing requested variables: {missing}")
        source = source[list(variable_names)].sortby("time")
        times = _exact_fixed_times(source, step_hours)

        shards = []
        canonical_schema = None
        feature_names = None
        global_start = 0
        for shard_id, begin in enumerate(range(0, len(times), shard_steps)):
            shard_times = times[begin : begin + shard_steps]
            loaded = _coarsen_global_fields(
                source.sel(time=shard_times.values),
                target_lat_points,
                target_lon_points,
            ).load()
            states, observed, variable_schema, names = _schema_and_state(
                loaded, variable_names
            )
            if states.shape[1] > max_state_dim:
                raise ValueError(
                    f"state_dim={states.shape[1]:,} exceeds max_state_dim={max_state_dim:,}; "
                    "use a coarser test grid or the spatial/operator Flow backend"
                )
            if canonical_schema is None:
                canonical_schema = variable_schema
                feature_names = names
            elif variable_schema != canonical_schema:
                raise ValueError("Spatial/variable schema changed between time shards")

            stem = f"shard-{shard_id:05d}"
            state_path = output / f"{stem}.states.npy"
            mask_path = output / f"{stem}.mask.npy"
            time_path = output / f"{stem}.times.npy"
            np.save(state_path, states.astype(np.float32), allow_pickle=False)
            np.save(mask_path, observed.astype(np.uint8), allow_pickle=False)
            np.save(
                time_path,
                shard_times.values.astype("datetime64[ns]"),
                allow_pickle=False,
            )
            count = len(shard_times)
            shards.append(
                {
                    "id": shard_id,
                    "start": global_start,
                    "end": global_start + count,
                    "states": state_path.name,
                    "mask": mask_path.name,
                    "times": time_path.name,
                    "state_sha256": _sha256(state_path),
                    "first_time": str(shard_times[0]),
                    "last_time": str(shard_times[-1]),
                    "bytes": state_path.stat().st_size + mask_path.stat().st_size + time_path.stat().st_size,
                }
            )
            global_start += count
    finally:
        source.close()

    assert canonical_schema is not None and feature_names is not None
    state_dim = int(canonical_schema[-1]["slice"][1])
    index = {
        "format": "climate_diffusion.sharded_fixed_step.v1",
        "forecast_step_hours": int(step_hours),
        "state_time_semantics": "snapshot_valid_time",
        "missing_value_policy": "zero_with_observed_mask_no_future_statistics",
        "storage_dtype": "float32",
        "mask_dtype": "uint8",
        "total_steps": int(global_start),
        "state_dim": state_dim,
        "field_dim": state_dim,
        "integrated_feature_names": [],
        "variables": canonical_schema,
        "feature_names": feature_names,
        "target_lat_points": target_lat_points,
        "target_lon_points": target_lon_points,
        "shard_steps": shard_steps,
        "source_budget_gb": source_budget_gb,
        "selected_source_bytes": selected_bytes,
        "selected_source_gib": selected_bytes / GIB,
        "sources": [
            {"path": str(path), "bytes": _local_size(path)} for path in selected_sources
        ],
        "processed_bytes": int(sum(item["bytes"] for item in shards)),
        "shards": shards,
    }
    index_path = output / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        {
            "selected_source_gib": round(selected_bytes / GIB, 3),
            "processed_gib": round(index["processed_bytes"] / GIB, 3),
            "states": global_start,
            "state_dim": state_dim,
            "shards": len(shards),
            "step_hours": step_hours,
        }
    )
    return index_path


@dataclass(frozen=True)
class ShardLocation:
    shard_index: int
    local_index: int


class ShardedStateArchive:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        index_path = self.root / "index.json" if self.root.is_dir() else self.root
        self.root = index_path.parent
        self.index_path = index_path
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != "climate_diffusion.sharded_fixed_step.v1":
            raise ValueError("Unsupported sharded Flow archive format")
        self.shards = self.index["shards"]
        self.total_steps = int(self.index["total_steps"])
        self.state_dim = int(self.index["state_dim"])
        self.schema = {
            key: value for key, value in self.index.items()
            if key not in {"shards", "sources", "feature_names"}
        }

    @lru_cache(maxsize=8)
    def _states(self, shard_index: int):
        return np.load(
            self.root / self.shards[shard_index]["states"],
            mmap_mode="r",
            allow_pickle=False,
        )

    @lru_cache(maxsize=8)
    def _mask(self, shard_index: int):
        return np.load(
            self.root / self.shards[shard_index]["mask"],
            mmap_mode="r",
            allow_pickle=False,
        )

    def _locate(self, index: int) -> ShardLocation:
        if index < 0 or index >= self.total_steps:
            raise IndexError(index)
        # Number of shards is small enough for a linear lookup and this keeps the
        # on-disk index human-readable.  Shard sizes are typically 256-1024 steps.
        for shard_index, item in enumerate(self.shards):
            if int(item["start"]) <= index < int(item["end"]):
                return ShardLocation(shard_index, index - int(item["start"]))
        raise RuntimeError(f"Archive index does not cover state {index}")

    def read_range(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        if start < 0 or end > self.total_steps or start >= end:
            raise IndexError((start, end))
        states = []
        masks = []
        cursor = start
        while cursor < end:
            location = self._locate(cursor)
            record = self.shards[location.shard_index]
            available = min(end - cursor, int(record["end"]) - cursor)
            sl = slice(location.local_index, location.local_index + available)
            states.append(np.asarray(self._states(location.shard_index)[sl], dtype=np.float32))
            masks.append(np.asarray(self._mask(location.shard_index)[sl], dtype=np.uint8))
            cursor += available
        return np.concatenate(states, axis=0), np.concatenate(masks, axis=0)

    def time_at(self, index: int) -> np.datetime64:
        location = self._locate(index)
        record = self.shards[location.shard_index]
        times = np.load(self.root / record["times"], mmap_mode="r", allow_pickle=False)
        return times[location.local_index]

    @property
    def index_sha256(self) -> str:
        return _sha256(self.index_path)


class ShardedWindowDataset(Dataset):
    def __init__(
        self,
        archive: ShardedStateArchive,
        history_steps: int,
        lead_steps: int = 1,
        *,
        indices: list[int] | None = None,
        state_mean: np.ndarray | None = None,
        state_scale: np.ndarray | None = None,
    ):
        if min(history_steps, lead_steps) < 1:
            raise ValueError("history_steps and lead_steps must be positive")
        self.archive = archive
        self.history_steps = history_steps
        self.lead_steps = lead_steps
        count = archive.total_steps - history_steps - lead_steps + 1
        available = list(range(max(0, count)))
        self.indices = available if indices is None else list(indices)
        if any(index not in available for index in self.indices):
            raise ValueError("Window index is outside the sharded archive")
        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        self.state_scale = None if state_scale is None else np.asarray(state_scale, dtype=np.float32)
        if (self.state_mean is None) != (self.state_scale is None):
            raise ValueError("state_mean and state_scale must be supplied together")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        start = self.indices[item]
        target_index = start + self.history_steps + self.lead_steps - 1
        history, _ = self.archive.read_range(start, start + self.history_steps)
        target, _ = self.archive.read_range(target_index, target_index + 1)
        target = target[0]
        if self.state_mean is not None:
            history = (history - self.state_mean) / self.state_scale
            target = (target - self.state_mean) / self.state_scale
        return {
            "history": torch.from_numpy(np.asarray(history, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
        }


def masked_training_statistics(
    archive: ShardedStateArchive, *, end_state_exclusive: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-feature train-only normalization without materializing the archive."""
    if end_state_exclusive < 1 or end_state_exclusive > archive.total_steps:
        raise ValueError("Invalid normalization boundary")
    sums = np.zeros(archive.state_dim, dtype=np.float64)
    sums2 = np.zeros(archive.state_dim, dtype=np.float64)
    counts = np.zeros(archive.state_dim, dtype=np.int64)
    cursor = 0
    chunk = 1024
    while cursor < end_state_exclusive:
        end = min(end_state_exclusive, cursor + chunk)
        states, mask = archive.read_range(cursor, end)
        valid = mask.astype(bool)
        sums += np.where(valid, states, 0.0).sum(axis=0, dtype=np.float64)
        sums2 += np.where(valid, states * states, 0.0).sum(axis=0, dtype=np.float64)
        counts += valid.sum(axis=0, dtype=np.int64)
        cursor = end
    safe_count = np.maximum(counts, 1)
    mean = sums / safe_count
    variance = np.maximum(sums2 / safe_count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    mean = np.where(counts > 0, mean, 0.0).astype(np.float32)
    scale = np.where((counts > 1) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return mean, scale, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a disk-bounded sharded fixed-step archive for large Flow runs"
    )
    parser.add_argument("--fields", nargs="+", required=True, help="Chronological NetCDF files or one Zarr store")
    parser.add_argument("--source-budget-gb", type=float, default=500.0)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--shard-steps", type=int, default=256)
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--target-lat-points", type=int, default=45)
    parser.add_argument("--target-lon-points", type=int, default=90)
    parser.add_argument("--max-state-dim", type=int, default=250_000)
    parser.add_argument("--output-dir", default="data/flow-500gb-shards")
    args = parser.parse_args(argv)
    path = prepare_sharded_fixed_step_archive(
        args.fields,
        args.output_dir,
        source_budget_gb=args.source_budget_gb,
        step_hours=args.step_hours,
        shard_steps=args.shard_steps,
        variables=None if not args.variables else tuple(args.variables),
        target_lat_points=args.target_lat_points,
        target_lon_points=args.target_lon_points,
        max_state_dim=args.max_state_dim,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

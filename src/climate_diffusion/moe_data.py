"""Strict archive and five-way temporal contracts for two-stage MoE training."""
from __future__ import annotations

import numpy as np

from .data import load_monthly_archive, _coarsen_global_fields

SPLIT_ORDER = ("train", "expert_validation", "calibration", "validation", "test")


def field_grid(schema):
    variables = schema.get("variables", [])
    if not variables or schema.get("integrated_feature_names"):
        raise ValueError("MoE v1 requires gridded fields only, no integrated table features")
    shape = variables[0]["shape"]
    coords = variables[0]["coords"]
    offset = 0
    for variable in variables:
        if (variable["dims"] != ["lat", "lon"] or variable["shape"] != shape
                or variable["coords"] != coords or len(shape) != 2):
            raise ValueError("MoE variables must share exactly one [lat, lon] grid")
        size = int(np.prod(shape))
        if variable["slice"] != [offset, offset + size]:
            raise ValueError("MoE variable slices must be contiguous in schema order")
        offset += size
    if offset != schema["state_dim"]:
        raise ValueError("MoE field layout does not match state_dim")
    return (len(variables), *shape)


def load_moe_archive(path):
    states, times, schema = load_monthly_archive(path)
    if schema.get("format") != "climate_diffusion.fixed_step_state.v1":
        raise ValueError("MoE requires a fixed-step archive")
    field_grid(schema)
    step = int(schema.get("forecast_step_hours", 0))
    if (step < 1 or len(times) != len(states) or len(times) < 2
            or not np.all(np.diff(times) == np.timedelta64(step, "h"))):
        raise ValueError("MoE archive timestamps violate the fixed-step contract")
    if not np.isfinite(states).all():
        raise ValueError("MoE v1 rejects non-finite state values")
    with np.load(path, allow_pickle=False) as archive:
        if "observed_mask" not in archive:
            raise ValueError("MoE requires observed_mask; rebuild the archive to certify missingness")
        mask = archive["observed_mask"]
        if mask.shape != states.shape or not np.all(mask == 1):
            raise ValueError("MoE v1 requires fully observed data; missing values cannot be zero-filled labels")
    return states, times, schema


def align_moe_grid(dataset, schema):
    """Reuse archive pooling; reject mismatched grids instead of silent interpolation.

    Dataset coordinates must already be canonical (lat/lon), as returned by
    sample_fixed_step_history. The observation policy concerns pooled cells,
    matching prepare_fixed_step_archive's existing skipna pooling contract.
    """
    _, lat_size, lon_size = field_grid(schema)
    names = [variable["name"] for variable in schema["variables"]]
    missing = sorted(set(names).difference(dataset.data_vars))
    if missing:
        raise ValueError(f"Initial state is missing trained variables: {missing}")
    pooled = _coarsen_global_fields(dataset[names],
                                    int(schema.get("target_lat_points", lat_size)),
                                    int(schema.get("target_lon_points", lon_size)))
    for dim, expected in schema["variables"][0]["coords"].items():
        if dim not in pooled.coords or not np.array_equal(pooled.coords[dim].values, expected):
            raise ValueError(f"MoE pooled {dim} grid differs from training; prepare matching source/grid")
    return pooled


def build_moe_split(sample_count, horizon_steps, *, purge_windows=0):
    if purge_windows < 0 or horizon_steps < 1:
        raise ValueError("Invalid purge or horizon")
    purge = max(purge_windows, horizon_steps - 1)
    # Fixed ratios are recorded; the four embargoes are taken from train.
    heldout = [max(1, int(sample_count * f)) for f in (0.1, 0.15, 0.1, 0.1)]
    train_count = sample_count - sum(heldout) - 4 * purge
    if train_count < 2:
        raise ValueError("Archive too short for five disjoint MoE splits; add data or reduce history/horizon")
    result = {"purge_windows": purge}
    position = 0
    for name, count in zip(SPLIT_ORDER, [train_count, *heldout]):
        result[name] = list(range(position, position + count))
        position += count + purge
    validate_moe_split(result, horizon_steps, sample_count)
    return result


def validate_moe_split(split, horizon_steps, sample_count):
    for name in SPLIT_ORDER:
        indices = split.get(name, [])
        if (not indices or indices != sorted(set(indices))
                or indices[0] < 0 or indices[-1] >= sample_count):
            raise ValueError(f"Invalid MoE split: {name}")
    for left, right in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        if split[left][-1] + horizon_steps > split[right][0]:
            raise ValueError("MoE split has overlapping future targets")

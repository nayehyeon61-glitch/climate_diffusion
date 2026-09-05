"""Shared data-contract and finite-value validation helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch


def _first_index(mask: np.ndarray) -> tuple[int, ...] | None:
    locations = np.argwhere(mask)
    return None if locations.size == 0 else tuple(int(value) for value in locations[0])


def require_no_inf_numpy(values: np.ndarray, name: str) -> None:
    """Reject positive/negative infinity without rejecting imputable NaNs."""
    array = np.asarray(values)
    mask = np.isinf(array)
    if mask.any():
        positive = int(np.isposinf(array).sum())
        negative = int(np.isneginf(array).sum())
        raise ValueError(
            f"{name} contains non-finite infinity values: +Inf={positive}, "
            f"-Inf={negative}, first_index={_first_index(mask)}"
        )


def require_finite_numpy(values: np.ndarray, name: str) -> None:
    array = np.asarray(values)
    mask = ~np.isfinite(array)
    if mask.any():
        raise ValueError(
            f"{name} contains {int(mask.sum())} NaN/Inf values; "
            f"first_index={_first_index(mask)}"
        )


def require_finite_tensor(values: torch.Tensor, name: str) -> None:
    mask = ~torch.isfinite(values)
    if bool(mask.any()):
        first = tuple(int(value) for value in mask.nonzero()[0].detach().cpu().tolist())
        raise FloatingPointError(
            f"{name} contains {int(mask.sum().detach().cpu())} NaN/Inf values; "
            f"first_index={first}"
        )


def archive_contract_fingerprint(
    schema: dict[str, Any], times: np.ndarray, state_shape: tuple[int, ...]
) -> str:
    """Fingerprint the immutable shape/schema/time contract, not large state payloads."""
    payload = {
        "schema": schema,
        "times_ns": np.asarray(times, dtype="datetime64[ns]").astype(np.int64).tolist(),
        "state_shape": list(state_shape),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


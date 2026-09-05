"""Held-out monthly forecast evaluation for frozen flow checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_auxiliary_states, load_monthly_archive
from .inference import LatentFlowForecaster
from .validation import archive_contract_fingerprint, require_finite_numpy


def latitude_weights(schema: dict[str, Any]) -> np.ndarray | None:
    """Grid-cell area weights, shaped to broadcast against ``[..., C, H, W]``.

    A 0.25-degree cell at 89.875 degrees covers roughly 1/460 the area of one at
    the equator. Weighting by ``cos(latitude)`` keeps a global score from being
    dominated by the poles. Vector archives have no grid, so they stay unweighted.
    """
    if schema.get("layout") != "spatial":
        return None
    latitudes = np.asarray(schema["coords"]["lat"], dtype=np.float64)
    weights = np.cos(np.deg2rad(latitudes))
    # A pole row lands on exactly zero; keep it present but negligible.
    weights = np.clip(weights, 1e-6, None)
    return weights.reshape(1, -1, 1)


def _broadcast_weights(
    weights: np.ndarray | None, valid: np.ndarray
) -> np.ndarray:
    if weights is None:
        return np.ones_like(valid, dtype=np.float64)
    return np.broadcast_to(weights, valid.shape).astype(np.float64)


def _ensemble_crps(
    samples: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> float:
    require_finite_numpy(samples, "evaluation ensemble samples")
    valid = np.isfinite(target)
    if not valid.any():
        raise ValueError("Evaluation target contains no observed values")
    selected_samples = samples[:, valid]
    selected_target = target[valid]
    cell_weights = _broadcast_weights(weights, valid)[valid]
    total = cell_weights.sum()
    accuracy = float(
        (np.abs(selected_samples - selected_target[None, :]) * cell_weights[None, :]).sum()
        / (samples.shape[0] * total)
    )
    ordered = np.sort(selected_samples, axis=0)
    members = samples.shape[0]
    coefficients = (2 * np.arange(members) - members + 1).reshape((members, 1))
    half_pairwise = float(
        ((ordered * coefficients).sum(axis=0) * cell_weights).sum()
        / (members**2 * total)
    )
    return float(accuracy - half_pairwise)


def _error_metrics(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        raise ValueError("Metric has no jointly finite prediction/target values")
    cell_weights = _broadcast_weights(weights, valid)[valid]
    total = cell_weights.sum()
    error = prediction[valid] - target[valid]
    return {
        "mae": float((np.abs(error) * cell_weights).sum() / total),
        "rmse": float(np.sqrt((np.square(error) * cell_weights).sum() / total)),
        "bias": float((error * cell_weights).sum() / total),
    }


def weighted_rmse(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Area-weighted RMSE, shared by held-out evaluation and training selection."""
    return _error_metrics(prediction, target, weights)["rmse"]


def _anomaly_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted anomaly correlation of prediction and target about climatology."""
    valid = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(climatology)
    if not valid.any():
        raise ValueError("Anomaly correlation has no jointly finite values")
    cell_weights = _broadcast_weights(weights, valid)[valid]
    predicted_anomaly = prediction[valid] - climatology[valid]
    target_anomaly = target[valid] - climatology[valid]
    covariance = (cell_weights * predicted_anomaly * target_anomaly).sum()
    predicted_power = (cell_weights * np.square(predicted_anomaly)).sum()
    target_power = (cell_weights * np.square(target_anomaly)).sum()
    denominator = np.sqrt(predicted_power * target_power)
    if denominator <= 0.0:
        # A constant anomaly field has no correlation to report.
        return 0.0
    return float(covariance / denominator)


def seasonal_climatology(
    states: np.ndarray,
    times: np.ndarray,
    train_range: tuple[int, int] | None,
) -> dict[int, np.ndarray]:
    """Month-of-year means built from training months only.

    The whole-period mean is a weak baseline for monthly prediction because the
    seasonal cycle is the dominant signal; beating it is close to automatic.
    """
    months = np.asarray(times, dtype="datetime64[M]").astype("datetime64[M]")
    calendar = months.astype(int) % 12 + 1
    start, end = train_range if train_range else (0, len(states))
    result: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        selected = [
            index
            for index in range(start, end)
            if int(calendar[index]) == month and np.isfinite(np.asarray(states[index])).any()
        ]
        if not selected:
            continue
        stacked = np.stack([np.asarray(states[index], dtype=np.float64) for index in selected])
        with np.errstate(invalid="ignore"):
            result[month] = np.nanmean(stacked, axis=0).astype(np.float32)
    if not result:
        raise ValueError("Seasonal climatology has no usable training months")
    return result


def _calendar_month(value: np.datetime64) -> int:
    return int(np.datetime64(value, "M").astype(int) % 12 + 1)


def _skill_score(rmse: float, reference_rmse: float) -> float:
    """Fraction of the reference baseline's error removed; 0 means no better."""
    if reference_rmse <= 0.0:
        return 0.0
    return float(1.0 - rmse / reference_rmse)


def evaluate_flow_checkpoint(
    checkpoint_path: str | Path,
    archive_path: str | Path,
    output_path: str | Path,
    *,
    ensemble_size: int = 8,
    integration_steps: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> Path:
    """Evaluate only the test windows recorded in a checkpoint split manifest."""
    if min(ensemble_size, integration_steps) < 1:
        raise ValueError("ensemble_size and integration_steps must be positive")
    states, times, schema = load_monthly_archive(archive_path)
    auxiliary = load_auxiliary_states(archive_path, schema)
    forecaster = LatentFlowForecaster(checkpoint_path, device=device)
    if int(schema["state_dim"]) != forecaster.config.state_dim:
        raise ValueError("Evaluation archive does not match checkpoint state dimension")
    expected_fingerprint = forecaster.training_metadata.get(
        "archive_contract_fingerprint"
    )
    actual_fingerprint = archive_contract_fingerprint(
        schema, times, tuple(int(value) for value in states.shape)
    )
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Evaluation archive schema/grid/channel/time contract does not match checkpoint"
        )

    training = forecaster.training_metadata
    split = training.get("split", {})
    test_indices = [int(value) for value in split.get("test", [])]
    if not test_indices:
        raise ValueError(
            "Checkpoint has no held-out test split; retrain with artifact format v2"
        )
    history_months = forecaster.config.history_months
    lead_months = int(training.get("lead_months", 1))
    scale = forecaster.state_scale.detach().cpu().numpy()
    mean = forecaster.state_mean.detach().cpu().numpy()
    weights = latitude_weights(schema)
    raw_ranges = split.get("raw_month_ranges", {})
    train_range = (
        (int(raw_ranges["train"][0]), int(raw_ranges["train"][1]))
        if "train" in raw_ranges
        else None
    )
    monthly_climatology = seasonal_climatology(states, times, train_range)

    def normalized_climatology(target_index: int) -> np.ndarray:
        """Normalized month-of-year climatology, or the whole-period mean.

        A short archive may not cover every calendar month in its training
        range; falling back to the overall mean keeps the baseline defined.
        """
        value = monthly_climatology.get(_calendar_month(times[target_index]))
        if value is None:
            return np.zeros_like(np.asarray(states[target_index], dtype=np.float32))
        return ((value - mean) / scale).astype(np.float32)

    predictions, targets = [], []
    case_rows: list[dict[str, Any]] = []
    for case_number, start in enumerate(test_indices):
        target_index = start + history_months + lead_months - 1
        if target_index >= len(states):
            raise ValueError(f"Test window {start} is outside the evaluation archive")
        samples = forecaster.forecast(
            states[start : start + history_months],
            months=1,
            ensemble_size=ensemble_size,
            integration_steps=integration_steps,
            seed=seed + case_number * ensemble_size,
            history_auxiliary=(
                None
                if auxiliary is None
                else np.asarray(auxiliary[start : start + history_months])
            ),
        )[:, 0, :]
        require_finite_numpy(samples, f"evaluation forecast window={start}")
        target = states[target_index]
        ensemble_mean = samples.mean(axis=0)
        normalized_samples = (samples - mean[None, :]) / scale[None, :]
        normalized_target = np.where(
            np.isfinite(target), (target - mean) / scale, np.nan
        )
        normalized_mean = normalized_samples.mean(axis=0)
        case_metrics = _error_metrics(normalized_mean, normalized_target, weights)
        case_metrics.update(
            {
                "window_index": start,
                "target_time": str(times[target_index]),
                "crps": _ensemble_crps(normalized_samples, normalized_target, weights),
                "ensemble_spread": float(normalized_samples.std(axis=0).mean()),
            }
        )
        case_metrics["acc"] = _anomaly_correlation(
            normalized_mean, normalized_target, normalized_climatology(target_index), weights
        )
        case_rows.append(case_metrics)
        predictions.append(ensemble_mean)
        targets.append(target)

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    normalized_prediction = (prediction_array - mean[None, :]) / scale[None, :]
    normalized_target = np.where(
        np.isfinite(target_array),
        (target_array - mean[None, :]) / scale[None, :],
        np.nan,
    )
    require_finite_numpy(normalized_prediction, "normalized evaluation prediction")
    overall = _error_metrics(normalized_prediction, normalized_target, weights)
    overall["crps"] = float(np.mean([row["crps"] for row in case_rows]))
    overall["ensemble_spread"] = float(
        np.mean([row["ensemble_spread"] for row in case_rows])
    )
    persistence = np.stack(
        [states[index + history_months - 1] for index in test_indices]
    )
    normalized_persistence = (persistence - mean[None, :]) / scale[None, :]
    overall["persistence_rmse"] = _error_metrics(
        normalized_persistence,
        normalized_target,
        weights,
    )["rmse"]
    # Zero in normalized space is the whole-period training mean.
    overall["climatology_rmse"] = _error_metrics(
        np.zeros_like(normalized_target),
        normalized_target,
        weights,
    )["rmse"]
    seasonal = np.stack(
        [
            normalized_climatology(index + history_months + lead_months - 1)
            for index in test_indices
        ]
    )
    overall["seasonal_climatology_rmse"] = _error_metrics(
        seasonal, normalized_target, weights
    )["rmse"]
    overall["acc"] = _anomaly_correlation(
        normalized_prediction, normalized_target, seasonal, weights
    )
    overall["skill_vs_persistence"] = _skill_score(
        overall["rmse"], overall["persistence_rmse"]
    )
    overall["skill_vs_seasonal_climatology"] = _skill_score(
        overall["rmse"], overall["seasonal_climatology_rmse"]
    )
    overall["spread_skill_ratio"] = (
        float(overall["ensemble_spread"] / overall["rmse"]) if overall["rmse"] > 0 else 0.0
    )
    overall["latitude_weighted"] = float(weights is not None)

    by_variable = {}
    for variable in schema["variables"]:
        if schema.get("layout") == "spatial":
            start, end = variable["channel_slice"]
            selected_prediction = prediction_array[:, start:end]
            selected_target = target_array[:, start:end]
        else:
            start, end = variable["slice"]
            selected_prediction = prediction_array[:, start:end]
            selected_target = target_array[:, start:end]
        by_variable[variable["name"]] = _error_metrics(
            selected_prediction, selected_target, weights
        )

    result = {
        "format": "climate_diffusion.evaluation.v2",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": forecaster.checkpoint_sha256,
        "archive": str(Path(archive_path)),
        "test_windows": test_indices,
        "ensemble_size": ensemble_size,
        "integration_steps": integration_steps,
        "normalized_overall": overall,
        "by_variable_raw_units": by_variable,
        "by_case_normalized": case_rows,
        "archive_contract_fingerprint": actual_fingerprint,
    }
    metric_values = list(overall.values())
    for values in by_variable.values():
        metric_values.extend(values.values())
    for values in case_rows:
        metric_values.extend(
            value for key, value in values.items() if key not in {"window_index", "target_time"}
        )
    require_finite_numpy(np.asarray(metric_values, dtype=np.float64), "evaluation metrics")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen monthly flow model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--output", default="outputs/monthly-flow-evaluation.json")
    args = parser.parse_args(argv)
    path = evaluate_flow_checkpoint(
        args.checkpoint,
        args.archive,
        args.output,
        ensemble_size=args.ensemble_size,
        integration_steps=args.integration_steps,
        seed=args.seed,
        device=args.device,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _ensemble_crps(samples: np.ndarray, target: np.ndarray) -> float:
    require_finite_numpy(samples, "evaluation ensemble samples")
    valid = np.isfinite(target)
    if not valid.any():
        raise ValueError("Evaluation target contains no observed values")
    selected_samples = samples[:, valid]
    selected_target = target[valid]
    accuracy = np.abs(selected_samples - selected_target[None, :]).mean()
    ordered = np.sort(selected_samples, axis=0)
    members = samples.shape[0]
    coefficients = (2 * np.arange(members) - members + 1).reshape(
        (members, 1)
    )
    half_pairwise = float((ordered * coefficients).sum(axis=0).mean() / (members**2))
    return float(accuracy - half_pairwise)


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        raise ValueError("Metric has no jointly finite prediction/target values")
    error = prediction[valid] - target[valid]
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


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
        case_metrics = _error_metrics(normalized_mean, normalized_target)
        case_metrics.update(
            {
                "window_index": start,
                "target_time": str(times[target_index]),
                "crps": _ensemble_crps(normalized_samples, normalized_target),
                "ensemble_spread": float(normalized_samples.std(axis=0).mean()),
            }
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
    overall = _error_metrics(normalized_prediction, normalized_target)
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
    )["rmse"]
    overall["climatology_rmse"] = _error_metrics(
        np.zeros_like(normalized_target),
        normalized_target,
    )["rmse"]

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
            selected_prediction, selected_target
        )

    result = {
        "format": "climate_diffusion.evaluation.v1",
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

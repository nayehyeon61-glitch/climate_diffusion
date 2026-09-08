"""Held-out forecast evaluation for frozen monthly, dynamics and MoE checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_monthly_archive
from .inference import LatentFlowForecaster
from .moe_data import load_moe_archive, validate_moe_split
from .train import _sha256


def _ensemble_crps(samples: np.ndarray, target: np.ndarray) -> float:
    accuracy = np.abs(samples - target[None, :]).mean()
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :]).mean()
    return float(accuracy - 0.5 * pairwise)


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
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
    max_cases: int | None = None,
    moe_mode: str | None = None,
) -> Path:
    """Evaluate only test windows recorded in a checkpoint split manifest."""
    if min(ensemble_size, integration_steps) < 1:
        raise ValueError("ensemble_size and integration_steps must be positive")
    forecaster = LatentFlowForecaster(checkpoint_path, device=device)
    loader = load_moe_archive if forecaster.is_moe else load_monthly_archive
    states, times, schema = loader(archive_path)
    if schema["state_dim"] != forecaster.config.state_dim:
        raise ValueError("Evaluation archive does not match checkpoint state dimension")

    training = forecaster.training_metadata
    split = training.get("split", {})
    test_indices = [int(value) for value in split.get("test", [])]
    if not test_indices:
        raise ValueError(
            "Checkpoint has no held-out test split; retrain with artifact format v2"
        )
    history_months = forecaster.history_span_steps
    horizon = forecaster.config.horizon_steps if forecaster.is_dynamics else 1
    if forecaster.is_moe:
        validate_moe_split(split, horizon, len(states) - history_months - horizon + 1)
        if training["archive_sha256"] != _sha256(Path(archive_path)):
            raise ValueError("MoE evaluation must use the identical training archive")
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        positions = np.linspace(0, len(test_indices) - 1, min(max_cases, len(test_indices)), dtype=int)
        test_indices = [test_indices[i] for i in positions]
    if forecaster.is_dynamics:
        forecaster.validate_archive(schema, times)
        for key, actual in (("first_time", str(times[0])), ("last_time", str(times[-1]))):
            if training.get(key) != actual:
                raise ValueError("Evaluation archive time range differs from the training archive")
        if training.get("archive_state_count", len(states)) != len(states):
            raise ValueError("Evaluation archive state count differs from training")
        for left, right in (("train", "validation"), ("validation", "test")):
            a, b = split.get(left, []), split.get(right, [])
            if not a or not b or max(a) + horizon > min(b):
                raise ValueError("Checkpoint split has overlapping future targets; retrain with horizon-aware purging")
    lead_months = int(training.get("lead_months", 1))
    scale = forecaster.state_scale.detach().cpu().numpy()
    mean = forecaster.state_mean.detach().cpu().numpy()

    predictions, targets = [], []
    case_rows: list[dict[str, Any]] = []
    rank_counts = np.zeros(ensemble_size + 1, dtype=np.int64)
    rank_generator = np.random.default_rng(seed + 100000)
    for case_number, start in enumerate(test_indices):
        target_index = start + history_months + lead_months - 1
        target_end = target_index + horizon
        if start < 0 or target_end > len(states):
            raise ValueError(f"Test window {start} is outside the evaluation archive")
        samples = forecaster.forecast(
            forecaster.select_history(states[start : start + history_months]),
            months=horizon,
            ensemble_size=ensemble_size,
            integration_steps=integration_steps,
            seed=seed + case_number * ensemble_size,
            moe_mode=moe_mode,
        )
        if forecaster.is_dynamics:
            target = states[target_index:target_end]
        else:
            samples = samples[:, 0, :]
            target = states[target_index]
        ensemble_mean = samples.mean(axis=0)
        normalized_samples = (samples - mean[None, :]) / scale[None, :]
        normalized_target = (target - mean) / scale
        normalized_mean = normalized_samples.mean(axis=0)
        case_metrics = _error_metrics(normalized_mean, normalized_target)
        case_metrics.update(
            {
                "window_index": start,
                "target_time": str(times[target_index]),
                "last_target_time": str(times[target_end - 1]),
                "crps": _ensemble_crps(normalized_samples, normalized_target),
                "ensemble_spread": float(normalized_samples.std(axis=0).mean()),
            }
        )
        case_rows.append(case_metrics)
        if forecaster.is_moe:
            # Multivariate field energy score at each lead, then average leads.
            accuracy = np.linalg.norm(normalized_samples - normalized_target[None], axis=-1).mean()
            pairwise = np.linalg.norm(normalized_samples[:, None] - normalized_samples[None, :], axis=-1).mean()
            case_metrics["energy"] = float((accuracy - 0.5 * pairwise) / np.sqrt(states.shape[1]))
        if forecaster.is_manifold:
            low, high = np.quantile(normalized_samples, [0.1, 0.9], axis=0)
            case_metrics["coverage_80"] = float(((normalized_target >= low) & (normalized_target <= high)).mean())
            case_metrics["mean_variance"] = float(normalized_samples.var(axis=0).mean())
            below = (normalized_samples < normalized_target[None]).sum(axis=0)
            ties = (normalized_samples == normalized_target[None]).sum(axis=0)
            ranks = below + np.floor(rank_generator.random(ties.shape) * (ties + 1)).astype(int)
            rank_counts += np.bincount(ranks.ravel(), minlength=ensemble_size + 1)
        predictions.append(ensemble_mean)
        targets.append(target)

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    normalized_prediction = (prediction_array - mean[None, :]) / scale[None, :]
    normalized_target = (target_array - mean[None, :]) / scale[None, :]
    overall = _error_metrics(normalized_prediction, normalized_target)
    overall["crps"] = float(np.mean([row["crps"] for row in case_rows]))
    overall["ensemble_spread"] = float(
        np.mean([row["ensemble_spread"] for row in case_rows])
    )
    persistence = np.stack(
        [states[index + history_months - 1] for index in test_indices]
    )
    if forecaster.is_dynamics:
        persistence = np.broadcast_to(persistence[:, None, :], target_array.shape)
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
        start, end = variable["slice"]
        by_variable[variable["name"]] = _error_metrics(
            prediction_array[..., start:end], target_array[..., start:end]
        )

    result = {
        "format": "climate_diffusion.evaluation.v1",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": forecaster.checkpoint_sha256,
        "archive": str(Path(archive_path)),
        "test_windows": test_indices,
        "ensemble_size": ensemble_size,
        "integration_steps": integration_steps,
        "max_cases": max_cases,
        "normalized_overall": overall,
        "by_variable_raw_units": by_variable,
        "by_case_normalized": case_rows,
    }
    if forecaster.is_dynamics:
        result["format"] = "climate_diffusion.dynamics_evaluation.v1"
        result["sampling_contract"] = "per_lead_conditional_marginals"
        result["by_lead_normalized"] = [
            {"lead_hours": (lead + 1) * forecaster.forecast_step_hours,
             **_error_metrics(normalized_prediction[:, lead], normalized_target[:, lead]),
             "persistence_rmse": _error_metrics(
                 normalized_persistence[:, lead], normalized_target[:, lead])["rmse"]}
            for lead in range(horizon)
        ]
    if forecaster.is_moe:
        result["format"] = "climate_diffusion.moe_evaluation.v1"
        result["moe_mode"] = moe_mode or ("local" if forecaster.is_manifold else forecaster.model.stage)
        result["sampling_contract"] = training["sampling_contract"]
        result["normalized_overall"]["energy"] = float(np.mean([row["energy"] for row in case_rows]))
        result["probabilistic_score_estimator"] = "empirical (diagonal pairs included), not training fair estimator"
    if forecaster.is_manifold:
        result["format"] = "climate_diffusion.manifold_moe_evaluation.v1"
        result["stage"] = forecaster.model.stage
        result["rank_histogram_counts"] = rank_counts.tolist()
        result["rank_histogram_contract"] = "pooled scalar coordinates/leads; randomized ties; correlated samples"
        overall["coverage_80"] = float(np.mean([row["coverage_80"] for row in case_rows]))
        overall["rms_spread"] = float(np.sqrt(np.mean([row["mean_variance"] for row in case_rows])))
        overall["spread_skill_ratio"] = overall["rms_spread"] / overall["rmse"] if overall["rmse"] > 0 else None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen climate flow model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--moe-mode", help="MoE: experts/meta/uniform; manifold: local/uniform/expert:<index>")
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
        max_cases=args.max_cases,
        moe_mode=args.moe_mode,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

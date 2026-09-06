"""Verify the physical-time dynamics runs the way a forecast is normally judged.

The question for a trajectory model is not one aggregate number but the skill
curve: RMSE as a function of lead time, against persistence and climatology.
This also plots the loss components and the autoencoder against its linear-PCA
floor, because that is where the first runs turned out to be bottlenecked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from climate_diffusion.data import load_monthly_archive
from climate_diffusion.dynamics import (
    DynamicsModelConfig,
    LatentDynamicsFlow,
    TrajectoryWindowDataset,
)

ARCHIVE = Path("data/era5_6h_states.npz")
RUN_ROOT = Path("download/flow-matching/dynamics")
OUTPUT_ROOT = Path("outputs")

RUNS = {
    "dyn-h24": "Baseline · 144h · latent 128",
    "dyn-h120": "Baseline · 720h · latent 128",
    "dyn-h120-ens": "Baseline · 720h · latent 128 + CRPS",
    "fix-h24-l512": "Fixed · 144h · latent 512",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df",
)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 150,
    })


def available_runs() -> dict[str, str]:
    return {
        name: label
        for name, label in RUNS.items()
        if (RUN_ROOT / name / f"{name}.pt").is_file()
    }


def pca_floor(states: np.ndarray, train_end: int, latents: list[int]) -> dict[int, float]:
    """Linear reconstruction error, the bar a learned autoencoder must clear."""
    mean = states[:train_end].mean(axis=0)
    scale = states[:train_end].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = ((states[:train_end] - mean) / scale).astype(np.float32)
    rng = np.random.default_rng(0)
    sample = normalized[
        rng.choice(len(normalized), size=min(6000, len(normalized)), replace=False)
    ]
    spectrum = np.linalg.svd(sample, compute_uv=False) ** 2
    total = spectrum.sum()
    return {
        k: float(np.sqrt(max(0.0, 1.0 - spectrum[:k].sum() / total)))
        for k in latents
        if k < min(sample.shape)
    }


@torch.no_grad()
def evaluate_run(
    name: str, states: np.ndarray, *, cases: int, ensemble_size: int,
    integration_steps: int, device: torch.device,
) -> dict:
    checkpoint = RUN_ROOT / name / f"{name}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = DynamicsModelConfig(**payload["model_config"])
    model = LatentDynamicsFlow(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    mean = payload["state_mean"].numpy()
    scale = payload["state_scale"].numpy()
    normalized = ((states - mean) / scale).astype(np.float32)
    training = payload["training"]
    test = [int(value) for value in training["split"]["test"]]
    stride = max(1, len(test) // cases)
    dataset = TrajectoryWindowDataset(normalized, config, test[::stride][:cases])

    horizon = config.horizon_steps
    model_sq = np.zeros(horizon)
    persistence_sq = np.zeros(horizon)
    climatology_sq = np.zeros(horizon)
    recon_sq = 0.0
    count = 0
    batch = 8
    for begin in range(0, len(dataset), batch):
        rows = [dataset[i] for i in range(begin, min(begin + batch, len(dataset)))]
        history = torch.stack([r["history"] for r in rows]).to(device)
        origin = torch.stack([r["origin"] for r in rows]).to(device)
        targets = torch.stack([r["targets"] for r in rows]).to(device)
        prediction = model.deterministic_forecast(history, origin)
        model_sq += (prediction - targets).square().mean(dim=(0, 2)).cpu().numpy() * len(rows)
        persistence_sq += (
            origin[:, None, :] - targets
        ).square().mean(dim=(0, 2)).cpu().numpy() * len(rows)
        climatology_sq += targets.square().mean(dim=(0, 2)).cpu().numpy() * len(rows)
        flat = targets.reshape(-1, targets.shape[-1])
        recon = model.decode_latent(model.encode_latent(flat))
        recon_sq += float((recon - flat).square().mean()) * len(rows)
        count += len(rows)

    leads = (np.arange(1, horizon + 1) * config.step_hours).tolist()
    result = {
        "label": RUNS[name],
        "horizon_hours": config.horizon_hours,
        "step_hours": config.step_hours,
        "latent_dim": config.latent_dim,
        "cases": count,
        "lead_hours": leads,
        "model_rmse": np.sqrt(model_sq / count).tolist(),
        "persistence_rmse": np.sqrt(persistence_sq / count).tolist(),
        "climatology_rmse": np.sqrt(climatology_sq / count).tolist(),
        "autoencoder_reconstruction_rmse": float(np.sqrt(recon_sq / count)),
        "best_epoch": training["best_epoch"],
    }

    if ensemble_size >= 2:
        picks = sorted({0, horizon // 4, horizon // 2, horizon - 1})
        rows = [dataset[i] for i in range(min(len(dataset), 16))]
        history = torch.stack([r["history"] for r in rows]).to(device)
        origin = torch.stack([r["origin"] for r in rows]).to(device)
        targets = torch.stack([r["targets"] for r in rows]).to(device)
        samples = model.forecast(
            history, origin, ensemble_size=ensemble_size,
            integration_steps=integration_steps, lead_indices=picks,
        )
        chosen = targets[:, picks]
        result["ensemble_lead_hours"] = [(p + 1) * config.step_hours for p in picks]
        result["ensemble_mean_rmse"] = [
            float((samples.mean(dim=1)[:, i] - chosen[:, i]).square().mean().sqrt())
            for i in range(len(picks))
        ]
        result["ensemble_spread"] = [
            float(samples[:, :, i].std(dim=1).mean()) for i in range(len(picks))
        ]
    return result


def figure_skill(results: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    reference = max(results.values(), key=lambda row: row["horizon_hours"])
    axes[0].plot(reference["lead_hours"], reference["persistence_rmse"],
                 color=INK_MUTED, linestyle=(0, (4, 3)), label="persistence")
    axes[0].plot(reference["lead_hours"], reference["climatology_rmse"],
                 color=INK_MUTED, linestyle=(0, (1, 2)), label="climatology")
    for index, (name, row) in enumerate(results.items()):
        axes[0].plot(row["lead_hours"], row["model_rmse"],
                     color=SERIES[index], label=row["label"])
    axes[0].set_xlabel("lead time (hours)")
    axes[0].set_ylabel("RMSE, normalized units")
    axes[0].set_title("Forecast skill against lead time", color=INK, loc="left")
    axes[0].legend(fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower right")

    for index, (name, row) in enumerate(results.items()):
        skill = 1.0 - np.asarray(row["model_rmse"]) / np.asarray(row["persistence_rmse"])
        axes[1].plot(row["lead_hours"], 100 * skill, color=SERIES[index], label=row["label"])
    axes[1].axhline(0.0, color=INK_SECONDARY, linewidth=1.0)
    axes[1].set_xlabel("lead time (hours)")
    axes[1].set_ylabel("skill score vs persistence (%)")
    axes[1].set_title("Positive means the model beats persistence", color=INK, loc="left")
    axes[1].legend(fontsize=7.5, labelcolor=INK_SECONDARY, loc="lower right")

    figure.suptitle(
        "Latent dynamics ODE on 6-hourly ERA5: held-out verification",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def figure_training(results: dict[str, dict], output: Path) -> None:
    components = [
        ("reconstruction_mse", "Validation reconstruction MSE"),
        ("trajectory_mse", "Validation trajectory MSE (latent space)"),
        ("flow_matching_mse", "Validation flow-matching MSE"),
        ("trajectory_rmse", "Validation trajectory RMSE (state space)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (key, title) in zip(axes.flat, components):
        for index, name in enumerate(results):
            metrics = json.loads(
                (RUN_ROOT / name / f"{name}.metrics.json").read_text(encoding="utf-8")
            )
            epochs = [entry["epoch"] for entry in metrics]
            values = [entry["validation"][key] for entry in metrics]
            axis.plot(epochs, values, color=SERIES[index], label=results[name]["label"])
        axis.set_xlabel("epoch")
        axis.set_title(title, color=INK, loc="left")
    axes[0, 0].legend(fontsize=7.5, labelcolor=INK_SECONDARY)
    figure.suptitle("Dynamics training: which term is actually stuck",
                    color=INK, x=0.008, ha="left", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def figure_autoencoder(results: dict[str, dict], floor: dict[int, float],
                       output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    latents = sorted(floor)
    axis.plot(latents, [floor[k] for k in latents], color=INK_MUTED,
              marker="o", markersize=5, label="linear PCA floor")
    for index, (name, row) in enumerate(results.items()):
        axis.plot([row["latent_dim"]], [row["autoencoder_reconstruction_rmse"]],
                  marker="D", markersize=8, color=SERIES[index],
                  markeredgecolor=SURFACE, markeredgewidth=1.4,
                  linestyle="none", label=row["label"])
    axis.set_xscale("log", base=2)
    axis.set_xticks(latents, [str(k) for k in latents])
    axis.set_xlabel("latent dimension")
    axis.set_ylabel("reconstruction RMSE, normalized units")
    axis.set_title(
        "Autoencoder versus its linear floor (a point above the line is undertrained)",
        color=INK, loc="left",
    )
    axis.legend(fontsize=8, labelcolor=INK_SECONDARY)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--integration-steps", type=int, default=16)
    args = parser.parse_args(argv)
    _style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    states, times, schema = load_monthly_archive(ARCHIVE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = available_runs()
    if not runs:
        raise SystemExit("No finished dynamics checkpoints yet")
    results = {
        name: evaluate_run(
            name, states, cases=args.cases, ensemble_size=args.ensemble_size,
            integration_steps=args.integration_steps, device=device,
        )
        for name in runs
    }

    train_end = int(len(states) * 0.75)
    floor = pca_floor(states, train_end, [16, 32, 64, 128, 256, 512, 1024])

    figure_skill(results, OUTPUT_ROOT / "dynamics_skill.png")
    figure_training(results, OUTPUT_ROOT / "dynamics_training.png")
    figure_autoencoder(results, floor, OUTPUT_ROOT / "dynamics_autoencoder.png")

    summary = {
        "archive": str(ARCHIVE),
        "pca_reconstruction_floor": floor,
        "runs": results,
    }
    path = OUTPUT_ROOT / "dynamics-summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

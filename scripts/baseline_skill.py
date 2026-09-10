"""Score the monthly flow checkpoints against baselines that are actually hard.

``evaluation.py`` reports ``climatology_rmse`` as the error of the unconditional
training mean. For monthly-mean fields the seasonal cycle dominates the
variance, so that baseline is far too weak to judge skill by: a month-of-year
climatology is the honest reference, and anomaly persistence on top of it is
the standard next rung.

Everything here is computed on the same held-out windows the checkpoints
declare, with climatologies estimated only from the causal training span.
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

from climate_diffusion.data import load_monthly_archive
from climate_diffusion.inference import LatentFlowForecaster

ARCHIVE = Path("data/era5_monthly_states_full.npz")
OUTPUT_ROOT = Path("outputs")

MODELS = {
    "Monthly flow (recency, AE 1.2M)":
        "download/flow-matching/expanded-ae/recency-legacy-ae/recency-legacy-ae.pt",
    "Monthly flow (AE 8.5M, L+S)":
        "download/flow-matching/latent-fix/fix-ae-medium/fix-ae-medium.pt",
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df",
)
UNITS = {"msl": "hPa", "t2m": "K", "u10": "m/s", "v10": "m/s"}
UNIT_SCALE = {"msl": 0.01, "t2m": 1.0, "u10": 1.0, "v10": 1.0}


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


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.square(prediction - target).mean()))


def seasonal_climatology(states: np.ndarray, times: np.ndarray, train_end: int) -> np.ndarray:
    """Month-of-year mean, estimated only on the causal training span."""
    months = pd.DatetimeIndex(pd.to_datetime(times)).month.to_numpy()
    table = np.zeros((13, states.shape[1]), dtype=np.float32)
    for month in range(1, 13):
        rows = states[:train_end][months[:train_end] == month]
        table[month] = rows.mean(axis=0) if len(rows) else states[:train_end].mean(axis=0)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--integration-steps", type=int, default=32)
    args = parser.parse_args(argv)
    _style()

    states, times, schema = load_monthly_archive(ARCHIVE)
    months = pd.DatetimeIndex(pd.to_datetime(times)).month.to_numpy()

    rows: dict[str, dict[str, float]] = {}
    per_variable: dict[str, dict[str, float]] = {}
    target_indices: list[int] = []
    reference_mean = reference_scale = None

    for label, path in MODELS.items():
        forecaster = LatentFlowForecaster(path)
        training = forecaster.training_metadata
        history_months = forecaster.config.history_months
        lead = int(training.get("lead_months", 1))
        test = [int(value) for value in training["split"]["test"]]
        mean = forecaster.state_mean.detach().cpu().numpy()
        scale = forecaster.state_scale.detach().cpu().numpy()
        train_end = int(training["normalization_span"][1])

        predictions, targets, origins, target_months = [], [], [], []
        for case, start in enumerate(test):
            index = start + history_months + lead - 1
            samples = forecaster.forecast(
                states[start : start + history_months],
                months=1,
                ensemble_size=args.ensemble_size,
                integration_steps=args.integration_steps,
                seed=case * args.ensemble_size,
            )[:, 0, :]
            predictions.append(samples.mean(axis=0))
            targets.append(states[index])
            origins.append(states[start + history_months - 1])
            target_months.append(months[index])
        prediction = np.stack(predictions)
        target = np.stack(targets)
        origin = np.stack(origins)
        target_months = np.asarray(target_months)
        target_indices = [s + history_months + lead - 1 for s in test]
        reference_mean, reference_scale = mean, scale

        table = seasonal_climatology(states, times, train_end)
        seasonal = table[target_months]
        origin_months = months[[s + history_months - 1 for s in test]]
        anomaly_persistence = seasonal + (origin - table[origin_months])

        def norm(values: np.ndarray) -> np.ndarray:
            return (values - mean[None, :]) / scale[None, :]

        candidates = {
            label: prediction,
            "Persistence (previous month)": origin,
            "Unconditional climatology": np.repeat(mean[None, :], len(target), axis=0),
            "Seasonal climatology (month-of-year)": seasonal,
            "Anomaly persistence": anomaly_persistence,
        }
        normalized_target = norm(target)
        for name, values in candidates.items():
            if name in rows:
                continue
            rows[name] = {"normalized_rmse": _rmse(norm(values), normalized_target)}
            per_variable[name] = {
                variable["name"]: _rmse(
                    values[:, slice(*variable["slice"])] * UNIT_SCALE[variable["name"]],
                    target[:, slice(*variable["slice"])] * UNIT_SCALE[variable["name"]],
                )
                for variable in schema["variables"]
            }

    reference = rows["Seasonal climatology (month-of-year)"]["normalized_rmse"]
    for name, row in rows.items():
        row["skill_vs_seasonal_climatology"] = 1.0 - row["normalized_rmse"] / reference

    order = sorted(rows, key=lambda name: rows[name]["normalized_rmse"])
    baseline_names = {
        "Persistence (previous month)", "Unconditional climatology",
        "Seasonal climatology (month-of-year)", "Anomaly persistence",
    }

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    positions = np.arange(len(order))[::-1]
    colors = [
        INK_MUTED if name in baseline_names else SERIES[0] for name in order
    ]
    values = [rows[name]["normalized_rmse"] for name in order]
    axes[0].barh(positions, values, height=0.62, color=colors)
    axes[0].set_yticks(positions, order, fontsize=8)
    axes[0].set_xlim(0, max(values) * 1.25)
    axes[0].set_ylim(-0.6, len(order) - 1 + 1.3)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_title("Held-out RMSE, normalized units", color=INK, loc="left")
    axes[0].set_xlabel("lower is better")
    axes[0].axvline(reference, color=SERIES[1], linewidth=1.2, linestyle=(0, (4, 3)),
                    zorder=0, label="seasonal climatology")
    axes[0].legend(loc="upper right", fontsize=7.5, labelcolor=INK_SECONDARY)
    for position, value in zip(positions, values):
        axes[0].annotate(f"{value:.3f}", (value, position), textcoords="offset points",
                         xytext=(5, 0), va="center", fontsize=8, color=INK_SECONDARY,
                         bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))

    skills = [rows[name]["skill_vs_seasonal_climatology"] * 100 for name in order]
    axes[1].barh(positions, skills, height=0.62, color=colors)
    axes[1].set_yticks(positions, [""] * len(order))
    axes[1].set_ylim(-0.6, len(order) - 1 + 1.3)
    axes[1].grid(axis="y", visible=False)
    axes[1].axvline(0.0, color=INK_SECONDARY, linewidth=1.0)
    axes[1].set_title("Skill score vs seasonal climatology (%)", color=INK, loc="left")
    axes[1].set_xlabel("positive = better than a month-of-year climatology")
    span = max(abs(min(skills)), abs(max(skills))) * 1.45 or 1.0
    axes[1].set_xlim(-span, span)
    for position, value in zip(positions, skills):
        axes[1].annotate(f"{value:+.1f}%", (value, position), textcoords="offset points",
                         xytext=(6 if value >= 0 else -6, 0), va="center",
                         ha="left" if value >= 0 else "right",
                         fontsize=8, color=INK_SECONDARY)

    # Per-variable skill: the aggregate hides that t2m is the hard one.
    axis = axes[2]
    variables = list(UNITS)
    width = 0.26
    shown = [name for name in order if name not in baseline_names]
    shown.append("Seasonal climatology (month-of-year)")
    offsets = np.arange(len(variables))
    for index, name in enumerate(shown):
        colour = INK_MUTED if name in baseline_names else SERIES[index]
        relative = [
            per_variable[name][variable]
            / per_variable["Seasonal climatology (month-of-year)"][variable]
            for variable in variables
        ]
        axis.bar(offsets + (index - 1) * width, relative, width=width * 0.92,
                 color=colour, label=name)
        for position, value in zip(offsets + (index - 1) * width, relative):
            axis.annotate(f"{value:.2f}", (position, value), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=7, color=INK_SECONDARY)
    axis.axhline(1.0, color=INK_SECONDARY, linewidth=1.0, zorder=0)
    axis.set_xticks(offsets, [f"{v}\n({UNITS[v]})" for v in variables], fontsize=8)
    axis.set_ylim(0, 1.45)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("RMSE / seasonal-climatology RMSE")
    axis.set_title("Per variable (below 1.0 = beats the seasonal cycle)",
                   color=INK, loc="left")
    axis.legend(fontsize=7, labelcolor=INK_SECONDARY, loc="lower center",
                bbox_to_anchor=(0.5, 1.06), ncol=1)

    figure.suptitle(
        "Monthly next-month forecast: the seasonal cycle is the baseline that matters",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(OUTPUT_ROOT / "baseline_skill.png", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "archive": str(ARCHIVE),
        "ensemble_size": args.ensemble_size,
        "test_targets": [str(times[index]) for index in target_indices],
        "normalized_rmse": {name: rows[name]["normalized_rmse"] for name in order},
        "skill_vs_seasonal_climatology": {
            name: rows[name]["skill_vs_seasonal_climatology"] for name in order
        },
        "raw_units_rmse": {name: per_variable[name] for name in order},
        "units": UNITS,
    }
    path = OUTPUT_ROOT / "baseline-skill.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the recency-weighted / expanded-autoencoder flow runs and plot them.

Reads the checkpoints written by the sweep in ``download/flow-matching/expanded-ae``,
scores each one on the held-out test split, and renders the figure set under
``outputs/``. Evaluation JSON is cached, so re-running only redraws the figures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from climate_diffusion.data import load_monthly_archive
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.inference import LatentFlowForecaster

ARCHIVE = Path("data/era5_monthly_states_full.npz")
RUN_ROOT = Path("download/flow-matching")
OUTPUT_ROOT = Path("outputs")
EVAL_ROOT = OUTPUT_ROOT / "expanded-ae"

# Categorical slots 1-5 of the reference palette, assigned in fixed order.
SERIES = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#52514e",
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"
# Sequential blue ramp (steps 100 -> 700) and the blue <-> red diverging pair.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "brand_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "brand_blue_red",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0938f", "#cf3736", "#7d1c1c"],
)

UNITS = {"msl": "hPa", "t2m": "K", "u10": "m/s", "v10": "m/s"}
UNIT_SCALE = {"msl": 0.01, "t2m": 1.0, "u10": 1.0, "v10": 1.0}
LONG_NAME = {
    "msl": "Mean sea level pressure",
    "t2m": "2 m temperature",
    "u10": "10 m zonal wind",
    "v10": "10 m meridional wind",
}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    color: str


VARIANT_SETS = {
    # The autoencoder-capacity sweep: recency weighting plus a widened encoder.
    "expanded-ae": [
        Variant("expanded-ae/legacy-ae", "Baseline \u00b7 uniform \u00b7 AE 1.2M", SERIES[0]),
        Variant("expanded-ae/recency-legacy-ae", "Recency \u00b7 AE 1.2M", SERIES[1]),
        Variant("expanded-ae/recency-ae-medium", "Recency \u00b7 AE 512\u00d73 \u00b7 8.5M", SERIES[2]),
        Variant("expanded-ae/recency-ae-large", "Recency \u00b7 AE 768\u00d74 \u00b7 22M", SERIES[3]),
        Variant("expanded-ae/recency-ae-xlarge", "Recency \u00b7 AE 1024\u00d76 \u00b7 55M", SERIES[4]),
    ],
    # L = unit-scale latent, S = select on forecast RMSE, E = ensemble CRPS term.
    "latent-fix": [
        Variant("expanded-ae/recency-legacy-ae", "Previous best \u00b7 AE 1.2M", SERIES[0]),
        Variant("expanded-ae/recency-ae-medium", "AE 8.5M, no fix", SERIES[1]),
        Variant("latent-fix/abl-ae-medium-sel", "AE 8.5M \u00b7 S", SERIES[2]),
        Variant("latent-fix/abl-ae-medium-norm", "AE 8.5M \u00b7 L", SERIES[3]),
        Variant("latent-fix/fix-ae-medium", "AE 8.5M \u00b7 L+S", SERIES[4]),
        Variant("latent-fix/fix-ae-medium-ens", "AE 8.5M \u00b7 L+S+E", SERIES[5]),
        Variant("latent-fix/fix-ae-large-ens", "AE 22M \u00b7 L+S+E", SERIES[6]),
        Variant("latent-fix/fix-legacy-ae", "AE 1.2M \u00b7 L+S", SERIES[7]),
        Variant("latent-fix/fix-legacy-ae-ens", "AE 1.2M \u00b7 L+S+E", SERIES[8]),
    ],
}
VARIANTS: list[Variant] = []


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "grid.color": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
        }
    )


def _checkpoint(variant: Variant) -> Path:
    name = variant.key.rsplit("/", 1)[-1]
    return RUN_ROOT / variant.key / f"{name}.pt"


def load_runs(ensemble_size: int, integration_steps: int) -> dict[str, dict]:
    """Score every variant on its held-out test split, caching the JSON."""
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    for variant in VARIANTS:
        checkpoint = _checkpoint(variant)
        evaluation_path = EVAL_ROOT / f"{variant.key.replace('/', '__')}-evaluation.json"
        if not evaluation_path.is_file():
            evaluate_flow_checkpoint(
                checkpoint,
                ARCHIVE,
                evaluation_path,
                ensemble_size=ensemble_size,
                integration_steps=integration_steps,
                seed=0,
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = json.loads(
            checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        runs[variant.key] = {
            "evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
            "metrics": json.loads(
                checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8")
            ),
            "training": payload["training"],
            "model_config": payload["model_config"],
            "loss_config": payload["loss_config"],
            "metadata": metadata,
        }
    return runs


def autoencoder_diagnostics(variant: Variant, states: np.ndarray,
                            test_targets: list[int]) -> dict[str, float]:
    """Reconstruction error and latent scale on the held-out target states.

    ``latent_std`` matters because the flow transports N(0, I) onto these codes:
    the further it sits below 1, the worse the prior/target scale mismatch.
    """
    forecaster = LatentFlowForecaster(_checkpoint(variant), device="cpu")
    mean = forecaster.state_mean.numpy()
    scale = forecaster.state_scale.numpy()
    normalized = torch.as_tensor(
        (states[test_targets] - mean[None, :]) / scale[None, :], dtype=torch.float32
    )
    with torch.inference_mode():
        latent = forecaster.model.encode_latent(normalized)
        reconstruction = forecaster.model.decode_latent(latent).numpy()
    return {
        "reconstruction_rmse": float(
            np.sqrt(np.square(reconstruction - normalized.numpy()).mean())
        ),
        "latent_std": float(latent.std()),
    }


def _best_epoch(metrics: list[dict]) -> tuple[int, float]:
    losses = [entry["validation"]["loss"] for entry in metrics]
    index = int(np.argmin(losses))
    return index + 1, losses[index]


def _direct_label(axis, x, y, text) -> None:
    axis.annotate(
        text, (x, y), textcoords="offset points", xytext=(6, 0),
        color=INK_SECONDARY, fontsize=8, va="center",
    )


def figure_training(runs: dict[str, dict], times: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    panels = [
        ("loss", "Total validation loss"),
        ("reconstruction_mse", "Validation reconstruction MSE"),
        ("flow_matching_mse", "Validation flow-matching MSE"),
    ]
    for axis, (component, title) in zip(axes.flat, panels):
        for variant in VARIANTS:
            history = runs[variant.key]["metrics"]
            epochs = [entry["epoch"] for entry in history]
            values = [entry["validation"][component] for entry in history]
            axis.plot(epochs, values, color=variant.color, label=variant.label, alpha=0.9)
            if component == "loss":
                best_epoch, best_value = _best_epoch(history)
                axis.plot(
                    best_epoch, best_value, "o", color=variant.color,
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.4,
                )
        axis.set_yscale("log")
        axis.set_xlabel("epoch")
        axis.set_title(title, color=INK, loc="left")
    axes[0, 0].legend(loc="upper right", fontsize=7, labelcolor=INK_SECONDARY,
                      ncol=1 if len(VARIANTS) <= 5 else 2)
    axes[0, 0].annotate(
        "dots mark the checkpointed epoch",
        (0.98, 0.04), xycoords="axes fraction", fontsize=7.5, color=INK_MUTED,
        ha="right",
    )

    # Recency weighting: what the sampler actually does to the training window.
    axis = axes[1, 1]
    training = next(iter(runs.values()))["training"]
    split = training["split"]
    halflife = training["recency_halflife"] or 200.0
    history_months = next(iter(runs.values()))["model_config"]["history_months"]
    train_indices = np.asarray(split["train"])
    age = train_indices[-1] - train_indices
    weights = 0.5 ** (age / halflife)
    years = (
        times[train_indices + history_months].astype("datetime64[Y]").astype(int) + 1970
    )
    axis.fill_between(years, weights, color=SERIES[0], alpha=0.18, linewidth=0)
    axis.plot(years, weights, color=SERIES[0])
    for target in (0.5, 0.25):
        year = float(np.interp(target, weights, years))
        axis.plot([year], [target], "o", color=SERIES[0], markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.4)
        _direct_label(axis, year, target, f"{target:g}x at {year:.0f}")
    axis.set_ylim(0, 1.15)
    axis.set_xlabel("target month of the training window")
    axis.set_ylabel("relative sampling weight")
    axis.set_title(
        f"Recency sampling weight (half-life {halflife:g} windows)", color=INK, loc="left"
    )

    figure.suptitle(
        "Monthly latent flow matching on ERA5 1959-2021: training behaviour",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def figure_skill(runs: dict[str, dict], reconstruction: dict[str, float],
                 output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 2.2 + 0.45 * len(VARIANTS)))
    positions = np.arange(len(VARIANTS))[::-1]

    def bars(axis, values, title, xlabel, formatter="{:.3f}"):
        axis.barh(positions, values, height=0.62, color=[v.color for v in VARIANTS])
        axis.set_yticks(positions, [v.label for v in VARIANTS], fontsize=8)
        axis.set_xlim(0, max(values) * 1.28)
        # Headroom above the top bar so the reference legend has empty space.
        axis.set_ylim(-0.6, len(VARIANTS) - 1 + 1.3)
        axis.set_title(title, color=INK, loc="left")
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", visible=False)
        for position, value in zip(positions, values):
            axis.annotate(
                formatter.format(value), (value, position),
                textcoords="offset points", xytext=(5, 0),
                va="center", fontsize=8, color=INK_SECONDARY,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2),
            )

    overall = [runs[v.key]["evaluation"]["normalized_overall"] for v in VARIANTS]
    bars(axes[0], [row["rmse"] for row in overall],
         "Held-out RMSE (normalized units)", "lower is better")
    reference = overall[0]
    handles = []
    for value, name, style in (
        (reference["persistence_rmse"], "persistence", (0, (4, 3))),
        (reference["climatology_rmse"], "climatology", (0, (1, 2))),
    ):
        line = axes[0].axvline(
            value, color=INK_MUTED, linewidth=1.2, linestyle=style, zorder=0,
            label=f"{name} {value:.3f}",
        )
        handles.append(line)
    axes[0].set_xlim(0, reference["climatology_rmse"] * 1.35)
    axes[0].legend(
        handles=handles, fontsize=7.5, labelcolor=INK_SECONDARY, loc="upper right",
    )

    bars(axes[1], [row["crps"] for row in overall],
         "Held-out ensemble CRPS", "lower is better")
    bars(axes[2], [reconstruction[v.key] for v in VARIANTS],
         "Autoencoder reconstruction RMSE", "held-out states, normalized units",
         formatter="{:.4f}")

    figure.suptitle(
        "75 held-out months (2015-11 to 2022-01): forecast skill and autoencoder fidelity",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def figure_variables(runs: dict[str, dict], output: Path) -> None:
    names = list(UNITS)
    figure, axes = plt.subplots(1, len(names), figsize=(13.5, 4.2))
    positions = np.arange(len(VARIANTS))
    for axis, name in zip(axes, names):
        values = [
            runs[v.key]["evaluation"]["by_variable_raw_units"][name]["rmse"]
            * UNIT_SCALE[name]
            for v in VARIANTS
        ]
        axis.bar(positions, values, width=0.66, color=[v.color for v in VARIANTS])
        axis.set_xticks(positions, [""] * len(VARIANTS))
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_title(f"{LONG_NAME[name]}", color=INK, loc="left", fontsize=9.5)
        axis.set_ylabel(f"RMSE ({UNITS[name]})")
        axis.grid(axis="x", visible=False)
        for position, value in zip(positions, values):
            axis.annotate(
                f"{value:.2f}", (position, value), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7.5, color=INK_SECONDARY,
            )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=v.color, label=v.label) for v in VARIANTS
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=min(len(VARIANTS), 5),
        fontsize=8, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.16),
    )
    figure.suptitle(
        "Held-out RMSE per ERA5 variable, in physical units",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _forecast_test_months(variant: Variant, states: np.ndarray, ensemble_size: int,
                          integration_steps: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return [case, ensemble, state] samples plus the matching target indices."""
    forecaster = LatentFlowForecaster(_checkpoint(variant))
    training = forecaster.training_metadata
    history_months = forecaster.config.history_months
    lead_months = int(training.get("lead_months", 1))
    test_starts = [int(value) for value in training["split"]["test"]]
    samples, targets = [], []
    for case, start in enumerate(test_starts):
        samples.append(
            forecaster.forecast(
                states[start : start + history_months],
                months=1,
                ensemble_size=ensemble_size,
                integration_steps=integration_steps,
                seed=case * ensemble_size,
            )[:, 0, :]
        )
        targets.append(start + history_months + lead_months - 1)
    return np.stack(samples), np.stack([states[i] for i in targets]), targets


def figure_maps(variant: Variant, schema: dict, samples: np.ndarray,
                truth: np.ndarray, times: np.ndarray, targets: list[int],
                output: Path) -> None:
    case = len(targets) // 2
    prediction = samples[case].mean(axis=0)
    valid_time = np.datetime64(times[targets[case]], "M")
    lookup = {item["name"]: item for item in schema["variables"]}

    figure, axes = plt.subplots(len(UNITS), 3, figsize=(11, 10.5))
    for row, name in enumerate(UNITS):
        start, end = lookup[name]["slice"]
        shape = lookup[name]["shape"]
        scale = UNIT_SCALE[name]
        actual = truth[case, start:end].reshape(shape) * scale
        predicted = prediction[start:end].reshape(shape) * scale
        error = predicted - actual
        low, high = float(min(actual.min(), predicted.min())), float(
            max(actual.max(), predicted.max())
        )
        limit = float(np.abs(error).max()) or 1.0
        panels = [
            (actual, "ERA5 truth", SEQUENTIAL, dict(vmin=low, vmax=high)),
            (predicted, f"{variant.label} mean", SEQUENTIAL, dict(vmin=low, vmax=high)),
            (error, "forecast - truth", DIVERGING,
             dict(norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))),
        ]
        for column, (field, title, cmap, kwargs) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                field, origin="lower", aspect="auto", cmap=cmap,
                extent=(0, 360, -90, 90), **kwargs,
            )
            axis.grid(visible=False)
            axis.set_xticks([0, 90, 180, 270, 360])
            axis.set_yticks([-60, -30, 0, 30, 60])
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(title, color=INK, fontsize=9.5)
            if column == 0:
                axis.set_ylabel(f"{LONG_NAME[name]}\nlatitude", fontsize=8.5)
            bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
            bar.ax.tick_params(labelsize=6.5, colors=INK_SECONDARY)
            bar.outline.set_visible(False)
            if column == 2:
                bar.set_label(UNITS[name], fontsize=7, color=INK_SECONDARY)
    figure.suptitle(
        f"Held-out month {valid_time}: ERA5 versus the flow-matching ensemble mean",
        color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def figure_timeseries(runs: dict[str, dict], schema: dict,
                      forecasts: dict[str, tuple[np.ndarray, np.ndarray, list[int]]],
                      times: np.ndarray, output: Path,
                      *, best: Variant, worst: Variant) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.8), sharex=True)
    shown, seen = [], set()
    for variant in (VARIANTS[0], best, worst):
        if variant.key not in seen:
            shown.append(variant)
            seen.add(variant.key)

    axis = axes[0]
    for variant in shown:
        rows = runs[variant.key]["evaluation"]["by_case_normalized"]
        months = np.array([np.datetime64(row["target_time"], "M") for row in rows])
        axis.plot(
            months, [row["rmse"] for row in rows],
            color=variant.color, label=variant.label, alpha=0.9,
        )
    axis.set_ylabel("normalized RMSE")
    axis.set_title("Per-month held-out error", color=INK, loc="left")
    axis.legend(
        fontsize=8, labelcolor=INK_SECONDARY, ncol=3,
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
    )

    axis = axes[1]
    lookup = {item["name"]: item for item in schema["variables"]}
    start, end = lookup["t2m"]["slice"]
    shape = lookup["t2m"]["shape"]
    latitudes = np.asarray(lookup["t2m"]["coords"]["lat"], dtype=np.float64)
    area = np.cos(np.deg2rad(latitudes))
    area = area / area.sum()

    def global_mean(states: np.ndarray) -> np.ndarray:
        grid = states[..., start:end].reshape(*states.shape[:-1], *shape)
        return (grid.mean(axis=-1) * area).sum(axis=-1)

    for variant in (best, worst):
        samples, truth, targets = forecasts[variant.key]
        months = times[targets].astype("datetime64[M]")
        member_means = global_mean(samples)
        axis.fill_between(
            months, member_means.min(axis=1), member_means.max(axis=1),
            color=variant.color, alpha=0.14, linewidth=0,
        )
        axis.plot(months, member_means.mean(axis=1), color=variant.color,
                  label=variant.label)
    axis.plot(months, global_mean(truth), color=INK, linewidth=1.4,
              label="ERA5 truth", zorder=3)
    axis.set_ylabel("area-weighted global mean 2 m temperature (K)")
    axis.set_xlabel("held-out month")
    axis.set_title("Global-mean 2 m temperature", color=INK, loc="left")
    axis.legend(
        fontsize=8, labelcolor=INK_SECONDARY, ncol=3,
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
    )
    axis.annotate(
        f"shaded = ensemble range ({samples.shape[1]} members)",
        (0.01, 0.03), xycoords="axes fraction", fontsize=7.5, color=INK_MUTED,
    )

    figure.suptitle(
        "Held-out period 2015-11 to 2022-01", color=INK, x=0.008, ha="left", fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="variant_set", choices=sorted(VARIANT_SETS),
                        default="expanded-ae")
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--integration-steps", type=int, default=32)
    args = parser.parse_args(argv)

    global VARIANTS
    VARIANTS = VARIANT_SETS[args.variant_set]
    prefix = args.variant_set.replace("-", "_")

    _style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    states, times, schema = load_monthly_archive(ARCHIVE)
    runs = load_runs(args.ensemble_size, args.integration_steps)

    forecasts = {
        variant.key: _forecast_test_months(
            variant, states, args.ensemble_size, args.integration_steps
        )
        for variant in VARIANTS
    }
    diagnostics = {
        variant.key: autoencoder_diagnostics(variant, states, forecasts[variant.key][2])
        for variant in VARIANTS
    }
    reconstruction = {
        key: row["reconstruction_rmse"] for key, row in diagnostics.items()
    }

    def skill(variant: Variant) -> float:
        return runs[variant.key]["evaluation"]["normalized_overall"]["rmse"]

    best = min(VARIANTS, key=skill)
    worst = max(VARIANTS, key=skill)
    figure_training(runs, times, OUTPUT_ROOT / f"{prefix}_training.png")
    figure_skill(runs, reconstruction, OUTPUT_ROOT / f"{prefix}_skill.png")
    figure_variables(runs, OUTPUT_ROOT / f"{prefix}_variables.png")
    samples, truth, targets = forecasts[best.key]
    figure_maps(best, schema, samples, truth, times, targets,
                OUTPUT_ROOT / f"{prefix}_maps.png")
    figure_timeseries(runs, schema, forecasts, times,
                      OUTPUT_ROOT / f"{prefix}_timeseries.png",
                      best=best, worst=worst)

    summary = {
        "archive": str(ARCHIVE),
        "ensemble_size": args.ensemble_size,
        "integration_steps": args.integration_steps,
        "variant_set": args.variant_set,
        "best_variant": best.key,
        "variants": {
            variant.key: {
                "label": variant.label,
                "model_config": runs[variant.key]["model_config"],
                "recency_halflife": runs[variant.key]["training"]["recency_halflife"],
                "latent_normalization": runs[variant.key]["training"].get(
                    "latent_normalization", False
                ),
                "select_by": runs[variant.key]["training"].get("select_by", "loss"),
                "loss_config": runs[variant.key]["loss_config"],
                "parameter_count": runs[variant.key]["metadata"]["parameter_count"],
                "autoencoder_parameter_count":
                    runs[variant.key]["metadata"]["autoencoder_parameter_count"],
                "best_epoch": _best_epoch(runs[variant.key]["metrics"])[0],
                "best_validation_loss": _best_epoch(runs[variant.key]["metrics"])[1],
                "test": runs[variant.key]["evaluation"]["normalized_overall"],
                "autoencoder_reconstruction_rmse": reconstruction[variant.key],
                "latent_std": diagnostics[variant.key]["latent_std"],
                "by_variable_raw_units": {
                    name: row["rmse"]
                    for name, row in
                    runs[variant.key]["evaluation"]["by_variable_raw_units"].items()
                },
            }
            for variant in VARIANTS
        },
    }
    summary_path = OUTPUT_ROOT / f"{args.variant_set}-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

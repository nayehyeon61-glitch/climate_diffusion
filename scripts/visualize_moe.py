"""Plot ACTUAL saved MoE metrics; no illustrative/invented learning curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="docs/results/moe-smoke/training-metrics.json")
    parser.add_argument("--summary", default="docs/results/moe-smoke/summary.json")
    parser.add_argument("--output-dir", default="docs/figures/moe-smoke")
    args = parser.parse_args(argv)
    rows = json.loads(Path(args.metrics).read_text())
    summary = json.loads(Path(args.summary).read_text()) if args.summary else None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "flow-matching-moe"})
    for phase in ("experts", "meta"):
        data = [row for row in rows if row["stage"] == phase]
        if not data:
            continue
        epochs = [row["epoch"] for row in data]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle(f"Full-state Flow Matching MoE | {phase} stage\nRecorded metrics; each stage has its own validation split")
        loss_keys = ("loss", "fm", "reconstruction", "energy", "crps")
        for key in loss_keys:
            if key in data[0]["train"]:
                axes[0, 0].plot(epochs, [r["train"][key] for r in data], label=key)
        axes[0, 0].set(title="Training objectives (raw components)", ylabel="Normalized loss")
        for key in ("balance", "diversity", "residual_l2"):
            if key in data[0]["train"]:
                axes[0, 1].plot(epochs, [r["train"][key] for r in data], label=key)
        axes[0, 1].set(title="Regularizers (raw; not loss-weighted)", ylabel="Penalty")
        prefix = "router_" if phase == "experts" else "alpha_"
        gates = sorted(k for k in data[0]["train"] if k.startswith(prefix) and k[len(prefix):].isdigit())
        for key in gates:
            axes[1, 0].plot(epochs, [r["validation"][key] for r in data], label=key)
        axes[1, 0].set(title="Validation mean gates at FM interpolation states", ylabel="Probability", ylim=(0, 1))
        for key in ("energy", "crps", "forecast_rmse", "ensemble_spread"):
            axes[1, 1].plot(epochs, [r["validation"][key] for r in data], label=key)
        axes[1, 1].set(title="Actual generated validation ensembles", ylabel="Normalized units")
        selected = min(data, key=lambda r: r["selection_score"])["epoch"]
        axes[1, 1].axvline(selected, color="gray", linestyle=":", label=f"Selected epoch {selected}")
        for ax in axes.flat:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for suffix in ("svg", "png"):
            fig.savefig(out / f"{phase}-training.{suffix}", dpi=150,
                        metadata={"Date": None} if suffix == "svg" else None)
        plt.close(fig)
    if summary:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        fig.suptitle("Synthetic smoke | identical held-out cases and seeds | NOT ERA5 forecast skill")
        modes = ("experts", "uniform", "meta")
        for index, key in enumerate(("rmse", "energy", "crps", "ensemble_spread")):
            axes[0].bar(np.arange(3) + (index - 1.5) * 0.19,
                        [summary["test"][mode][key] for mode in modes], width=0.19, label=key)
        axes[0].set(xticks=np.arange(3), xticklabels=modes, title="Test ensembles: selected checkpoints",
                    ylabel="Normalized units")
        axes[0].legend(fontsize=8)
        routing = summary["routing"]
        count = len(routing["router_mean"])
        for i, key in enumerate(("router", "alpha")):
            means, sd = np.asarray(routing[f"{key}_mean"]), np.asarray(routing[f"{key}_std"])
            # A probability plot cannot display mean±SD outside [0,1]. Label the
            # truncation explicitly; full sample SD is retained in summary.json.
            error = np.stack((np.minimum(sd, means), np.minimum(sd, 1 - means)))
            axes[1].bar(np.arange(count) + (i - 0.5) * 0.35, means, width=0.35,
                        yerr=error, capsize=3, label=f"{key}: mean ± SD (clipped to [0,1])")
        axes[1].set(xticks=np.arange(count), xticklabels=[f"E{k+1}" for k in range(count)], ylim=(0, 1),
                    title="Gates on generated final-lead ODE paths", ylabel="Probability")
        axes[1].legend(fontsize=8)
        counts = np.asarray(routing["toy_regime_by_hard_router_counts"])
        frequencies = counts / counts.sum(1, keepdims=True).clip(1)
        im = axes[2].imshow(frequencies, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (row, col), value in np.ndenumerate(frequencies):
            axes[2].text(col, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if value > 0.6 else "black")
        axes[2].set(xticks=np.arange(count), xticklabels=[f"E{k+1}" for k in range(count)],
                    yticks=[0, 1], yticklabels=["Toy mode 0", "Toy mode 1"],
                    title="Hard router frequency by toy mode\nNot proof of expert specialization")
        fig.colorbar(im, ax=axes[2], shrink=0.7)
        for suffix in ("svg", "png"):
            fig.savefig(out / f"test-diagnostics.{suffix}", dpi=150,
                        metadata={"Date": None} if suffix == "svg" else None)
        plt.close(fig)
    # Matplotlib emits trailing blanks inside SVG path strings; normalize these
    # without altering geometry so generated figures pass git diff --check.
    for path in out.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

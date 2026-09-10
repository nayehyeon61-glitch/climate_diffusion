"""Plot training curves, held-out evaluation, and routing diagnostics for one
real-ERA5 A/B/C Manifold MoE run (schema matches evaluation.py / manifold_diagnostics.py
as of commit 0e641b2). Reads only saved JSON/metrics; no checkpoint/archive access."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training(metrics_path, output):
    rows = json.loads(Path(metrics_path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    stages = ("manifold", "specialize", "joint")
    colors = {"manifold": "tab:blue", "specialize": "tab:orange", "joint": "tab:green"}
    offset = 0
    for stage in stages:
        stage_rows = [r for r in rows if r["stage"] == stage]
        if not stage_rows:
            continue
        x = offset + np.arange(1, len(stage_rows) + 1)
        train = [r["train"]["loss"] for r in stage_rows]
        val = [r["selection_score"] for r in stage_rows]
        axes[0].plot(x, train, color=colors[stage], label=f"{stage} train")
        axes[1].plot(x, val, color=colors[stage], label=f"{stage} val (selection score)")
        best = min(range(len(val)), key=lambda i: val[i])
        axes[1].scatter([x[best]], [val[best]], color=colors[stage], marker="*", s=120, zorder=5)
        offset += len(stage_rows)
        for ax in axes:
            ax.axvline(offset + 0.5, color="grey", linestyle=":", linewidth=1)
    axes[0].set(title="Train loss (A -> B -> C)", xlabel="epoch (concatenated)")
    axes[1].set(title="Validation selection score\n(* = best/saved epoch)", xlabel="epoch (concatenated)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_evaluation(validation_path, test_path, output):
    v = json.loads(Path(validation_path).read_text())["normalized_overall"]
    t = json.loads(Path(test_path).read_text())["normalized_overall"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = ["model", "persistence", "climatology"]
    for ax, split, data in ((axes[0], "validation", v), (axes[1], "test", t)):
        values = [data["rmse"], data["persistence_rmse"], data["climatology_rmse"]]
        bars = ax.bar(labels, values, color=["tab:green", "tab:grey", "tab:grey"])
        bars[0].set_color("tab:green" if values[0] < min(values[1:]) else "tab:red")
        ax.set(title=f"{split}: normalized RMSE vs baselines")
        ax.grid(alpha=0.2, axis="y")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"CRPS: val {v['crps']:.3f} / test {t['crps']:.3f}   "
                 f"coverage_80: val {v['coverage_80']:.0%} / test {t['coverage_80']:.0%} (target 80%)   "
                 f"spread/skill: val {v['spread_skill_ratio']:.2f} / test {t['spread_skill_ratio']:.2f} (target ~1)",
                 fontsize=9)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_routing(routing_path, output):
    d = json.loads(Path(routing_path).read_text())
    audit = d["generated_audit"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    gate = audit["gate_mean"]
    axes[0].bar(np.arange(len(gate)), gate, color="tab:purple")
    axes[0].set(title=f"Generated-path gate usage\n(gate entropy {audit['gate_entropy_nats']:.2f} nats, "
                       f"max ln{len(gate)}={np.log(len(gate)):.2f})",
                xlabel="expert", ylim=(0, 1))
    axes[0].grid(alpha=0.2, axis="y")
    tf = d["teacher_forced_audit"]
    metrics = {"candidate cosine\n(1.0 = collapsed)": audit["candidate_cosine"],
               "local-expert hits\nbest expert": tf["local_expert_best_fraction"],
               "gate hits\nbest expert": tf["gate_best_expert_fraction"],
               "anchor RMSE": d["anchor_rmse_evaluation"]}
    axes[1].bar(range(len(metrics)), list(metrics.values()), color="tab:orange")
    axes[1].set(xticks=range(len(metrics)), xticklabels=list(metrics.keys()), title="Specialization diagnostics")
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].grid(alpha=0.2, axis="y")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--routing", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_training(args.metrics, out / "training-abc.png")
    plot_evaluation(args.validation, args.test, out / "evaluation.png")
    plot_routing(args.routing, out / "routing.png")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot saved real loss logs; optionally audit a checkpoint on validation only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def visualize(report, checkpoint=None, archive=None):
    import matplotlib.pyplot as plt
    from smoke_temporal_moe import plot_logs
    report=Path(report)
    logs={k:json.loads((report/f"training-{k}.json").read_text()) for k in ("baseline","new")}
    comparisons={k:json.loads((report/f"validation-{k}.json").read_text()) for k in logs}
    plot_logs(logs,comparisons,report/"training-comparison.png")
    rows=[r for r in logs["new"] if r["stage"]!="manifold"]
    x=np.arange(len(rows))+1
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for key in ("fm","expert_fm","gate","balance","diversity","projection"):
        axes[0,0].plot(x,[r["train"].get(key,np.nan) for r in rows],label=key)
    axes[0,0].set(title="Preserved objectives: B then C",xlabel="epoch")
    usage=[k for k in rows[0]["train"] if k.startswith("usage_")]
    for key in usage:
        axes[0,1].plot(x,[r["train"][key] for r in rows],label=key)
    axes[0,1].set(title="Teacher-forced expert usage",ylim=(0,1))
    for label in comparisons:
        tm=comparisons[label]["temporal_overall"]
        axes[0,2].plot([0,1,2,3],[tm["tendency_mse_"+n] for n in ("msl","t2m","u10","v10")],"o-",label=label)
    axes[0,2].set(xticks=[0,1,2,3],xticklabels=["msl","t2m","u10","v10"],title="Validation scaled tendency MSE")
    for key in ("increment_spread","temporal_ramp"):
        axes[1,0].plot(x,[r["train"].get(key,np.nan) for r in rows],label=key)
    axes[1,0].set(title="Increment spread and auxiliary ramp")
    for phase in ("manifold","specialize","joint"):
        rr=[r for r in logs["new"] if r["stage"]==phase]
        axes[1,1].plot([r["epoch"] for r in rr],[r["selection_score"] for r in rr],label=phase)
    axes[1,1].set(title="Actual best-checkpoint selection scores",xlabel="phase epoch")
    if checkpoint and archive:
        from climate_diffusion.manifold_diagnostics import diagnose_manifold
        path=diagnose_manifold(checkpoint,archive,report/"routing-validation.json",split_name="validation",
                               max_cases=8,members=3,integration_steps=4)
        audit=json.loads(path.read_text())
        g=audit["generated_audit"]
        axes[1,2].bar(np.arange(len(g["gate_mean"])),g["gate_mean"])
        axes[1,2].set(title=f"Generated-path fusion weights\nCandidate cosine {g['candidate_cosine']:.3f}",xlabel="expert",ylim=(0,1))
    else:
        axes[1,2].text(.5,.5,"Use --checkpoint and --archive\nfor validation-only generated-path audit",ha="center")
        axes[1,2].set_axis_off()
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle("Synthetic diagnostics — usage is not proof of physical regime specialization")
    fig.savefig(report/"routing-and-dynamics.png",dpi=120); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",default="docs/results/temporal-120h-smoke")
    parser.add_argument("--checkpoint")
    parser.add_argument("--archive")
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    visualize(args.report_dir,args.checkpoint,args.archive)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

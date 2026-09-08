"""Render recorded manifold geometry, specialization and A/B/C learning logs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig,out,name):
    for suffix in ("svg","png"):
        path=out/f"{name}.{suffix}"
        fig.savefig(path,dpi=150,metadata={"Date":None} if suffix=="svg" else None)
        if suffix=="svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics",default="docs/results/manifold-smoke/diagnostics-joint.json")
    parser.add_argument("--metrics",default="docs/results/manifold-smoke/training-metrics.json")
    parser.add_argument("--evaluation",default="docs/results/manifold-smoke/evaluation-local.json")
    parser.add_argument("--output-dir",default="docs/figures/manifold-smoke")
    args=parser.parse_args(argv)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.fonttype":"none","svg.hashsalt":"manifold-moe-v1"})
    data=json.loads(Path(args.diagnostics).read_text())
    train,test,centers=[np.asarray(data[k]) for k in ("train_pca","test_pca","centers_pca")]
    pi_train,pi_test=np.asarray(data["train_gate"]),np.asarray(data["test_gate"])
    experts=pi_test.shape[1]
    colors=plt.get_cmap("tab10",experts)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    explained=np.asarray(data["pca_explained_variance_ratio"])*100
    fig.suptitle(f"Physics-informed embedding | stage={data['stage']} | PCA fitted on train origins\n"
                 "Routing regions in latent state space; not geographic territories or proof of specialization")
    axes[0].scatter(train[:,0],train[:,1],c=pi_train.argmax(1),cmap=colors,vmin=-0.5,vmax=experts-0.5,
                    s=16,alpha=0.45,label="Train origins")
    axes[0].scatter(test[:,0],test[:,1],c=pi_test.argmax(1),cmap=colors,vmin=-0.5,vmax=experts-0.5,
                    s=42,edgecolors="black",linewidths=0.5,label="Test origins")
    axes[0].scatter(centers[:,0],centers[:,1],marker="X",s=120,c="black",label="Fixed chart centers")
    for k,p in enumerate(centers):
        axes[0].annotate(f" E{k+1}",p,fontsize=10,fontweight="bold")
    axes[0].set_title("Observed-state chart assignments")
    axes[0].legend(fontsize=8)
    signal=np.asarray(data["test_physical_signals"])
    im=axes[1].scatter(test[:,0],test[:,1],c=signal[:,3],cmap="viridis",s=45)
    fig.colorbar(im,ax=axes[1],shrink=0.7,label="Vorticity RMS [1/s]")
    axes[1].set_title("Physical signal on the same test embedding")
    if "generated_audit" in data:
        paths=np.asarray(data["generated_audit"]["intrinsic_path_pca"])
        for k in range(min(paths.shape[1],8)):
            axes[2].plot(paths[:,k,0],paths[:,k,1],".-",alpha=0.7,linewidth=1)
            axes[2].scatter(*paths[0,k],marker="x",c="gray",s=30)
        axes[2].scatter(test[:,0],test[:,1],s=15,c="black",alpha=0.25)
        axes[2].set_title("Generated FLOW-time paths (tau 0→1)\nNot physical-time weather tracks")
    else:
        axes[2].bar(np.arange(experts),pi_test.mean(0))
        axes[2].set(title="Stage A geometric gate prior only",xlabel="Expert index",ylabel="Mean probability")
    for ax in axes[:2] if "generated_audit" not in data else axes:
        ax.set(xlabel=f"PC1 ({explained[0]:.1f}%)",ylabel=f"PC2 ({explained[1]:.1f}%)")
        ax.grid(alpha=0.15)
    save(fig,out,"manifold-geometry")
    if "teacher_forced_audit" in data:
        audit=data["teacher_forced_audit"]
        generated=data["generated_audit"]
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        fig.suptitle("Expert specialization audit | actual held-out diagnostics | partition ≠ predictive skill")
        error=np.array(audit["expert_fm_by_region"],dtype=float)
        im=axes[0].imshow(np.ma.masked_invalid(error),cmap="YlOrRd",aspect="auto")
        for (i,j),value in np.ndenumerate(error):
            axes[0].text(j,i,f"{value:.2f}" if np.isfinite(value) else "N/A",ha="center",va="center",fontsize=9)
        axes[0].set(xticks=range(experts),xticklabels=[f"E{k+1}" for k in range(experts)],
                    yticks=range(experts),yticklabels=[f"Chart {k+1} (n={n})" for k,n in enumerate(audit["region_counts"])],
                    title="FM error by chart and candidate\nLower diagonal would support local skill")
        fig.colorbar(im,ax=axes[0],shrink=0.7)
        axes[1].bar(range(experts),generated["gate_mean"],color=[colors(k) for k in range(experts)])
        axes[1].set(xticks=range(experts),xticklabels=[f"E{k+1}" for k in range(experts)],ylim=(0,1),
                    title="Gate usage on generated ODE paths",ylabel="Mean probability")
        text=(f"Gate entropy: {generated['gate_entropy_nats']:.3f} nats\n"
              f"Candidate cosine: {generated['candidate_cosine']:.3f}\n"
              f"Candidate pair MSE: {generated['candidate_pair_mse']:.3f}\n\n"
              f"Local expert is best: {audit['local_expert_best_fraction']:.1%}\n"
              f"Gate picks best expert: {audit['gate_best_expert_fraction']:.1%}\n\n"
              f"Damped metric condition: {generated['damped_metric_condition_mean']:.1f}\n"
              f"Chart distance/radius: {generated['chart_distance_over_radius_mean']:.2f}\n\n"
              "Cosine and usage alone do not establish\nmeteorological regime specialization.")
        axes[2].text(0,0.95,text,va="top",fontsize=10,linespacing=1.7)
        axes[2].axis("off")
        save(fig,out,"expert-specialization")
    if args.metrics:
        rows=json.loads(Path(args.metrics).read_text())
        phases=[p for p in ("manifold","specialize","joint") if any(r["stage"]==p for r in rows)]
        fig,axes=plt.subplots(len(phases),2,figsize=(13,3.5*len(phases)),squeeze=False,constrained_layout=True)
        fig.suptitle("Recorded A/B/C learning curves | stages have different objectives / validation populations")
        for row,phase in enumerate(phases):
            records=[r for r in rows if r["stage"]==phase]
            epochs=[r["epoch"] for r in records]
            keys=("reconstruction","physics","invariant","metric","latent_dynamics") if phase=="manifold" else (
                  "fm","expert_fm","gate","balance","projection","diversity")
            for key in keys:
                axes[row,0].plot(epochs,[r["train"][key] for r in records],label=key)
            axes[row,0].set(title=f"{phase}: raw training loss components",ylabel="Loss (symlog)")
            axes[row,0].set_yscale("symlog",linthresh=0.01)
            axes[row,0].set_ylim(bottom=0)
            keys=("loss","reconstruction","metric") if phase=="manifold" else (
                  "energy","crps","forecast_rmse","ensemble_spread","gate_entropy")
            for key in keys:
                axes[row,1].plot(epochs,[r["validation"][key] for r in records],label=key)
            best=min(records,key=lambda r:r["selection_score"])["epoch"]
            axes[row,1].axvline(best,ls=":",color="gray",label=f"Selected epoch {best}")
            axes[row,1].set(title=f"{phase}: held-out validation",ylabel="Recorded value")
            for ax in axes[row]:
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.15)
                ax.legend(fontsize=8,ncol=2)
        save(fig,out,"training-abc")
        routed = [p for p in ("specialize","joint") if p in phases]
        if routed:
            fig,axes=plt.subplots(len(routed),2,figsize=(13,3.5*len(routed)),squeeze=False,constrained_layout=True)
            fig.suptitle("Routing / fusion weights and ensemble objectives from actual training logs")
            for row,phase in enumerate(routed):
                records=[r for r in rows if r["stage"]==phase]
                epochs=[r["epoch"] for r in records]
                for k in range(experts):
                    axes[row,0].plot(epochs,[r["train"][f"usage_{k}"] for r in records],label=f"E{k+1}")
                axes[row,0].set(title=f"{phase}: mean training fusion weights",ylim=(0,1),ylabel="Probability")
                keys=("gate_entropy","responsibility_agreement","candidate_cosine") if phase=="specialize" else (
                      "energy","crps","ensemble_spread","spread_guard","anchor","pi_loss")
                for key in keys:
                    axes[row,1].plot(epochs,[r["train"][key] for r in records],label=key)
                axes[row,1].set(title=f"{phase}: routing / ensemble training diagnostics",ylabel="Recorded value")
                for ax in axes[row]:
                    ax.set_xlabel("Epoch")
                    ax.grid(alpha=0.15)
                    ax.legend(fontsize=8,ncol=2)
            save(fig,out,"routing-learning")
    if args.evaluation:
        evaluation=json.loads(Path(args.evaluation).read_text())
        if "rank_histogram_counts" in evaluation:
            fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
            ranks=np.asarray(evaluation["rank_histogram_counts"])
            ax.bar(np.arange(len(ranks)),ranks/ranks.sum())
            ax.axhline(1/len(ranks),ls="--",c="black",label="Uniform reference")
            ax.set(xlabel="Observation rank among ensemble members",ylabel="Fraction",
                   title="Held-out rank histogram\nPooled correlated cells/leads; no independent-sample confidence claim")
            ax.legend()
            save(fig,out,"rank-histogram")
    print(out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

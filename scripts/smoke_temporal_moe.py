"""Reproduce CPU 120-hour A/B/C training, loss ablation and all-member movies.

Small synthetic integration check, NOT physical ERA5 forecast skill. Refuses to
overwrite work/reports. Baseline includes the corrected noise contract, so this
isolates added losses rather than claiming an old-commit reproduction.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from smoke_moe import synthetic_archive
from climate_diffusion.train_manifold_moe import train_manifold_moe
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.time_alignment import forecast_from_checkpoint
from climate_diffusion.trajectory_output import export_trajectories
from climate_diffusion.train import _sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def plot_logs(rows, comparisons, output):
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for name, records in rows.items():
        for phase, ax in zip(("manifold","specialize","joint"),axes[0]):
            selected=[r for r in records if r["stage"]==phase]
            if selected:
                ax.plot([r["epoch"] for r in selected],[r["train"]["loss"] for r in selected],label=name+" train")
                ax.plot([r["epoch"] for r in selected],[r["validation"]["loss"] for r in selected],"--",label=name+" val")
            ax.set(title=phase,xlabel="epoch",ylabel="total objective (not same as selection score)")
    selected=[r for r in rows["new"] if r["stage"]!="manifold"]
    for key in ("loss_delta","loss_trajectory","energy","crps","loss_wind_speed","loss_wind_direction"):
        axes[1,0].plot(np.arange(len(selected))+1,[r["train"].get(key,np.nan) for r in selected],label=key)
    axes[1,0].set(title="New: unweighted terms, B then C",xlabel="epoch across stages")
    for name in ("msl","t2m","u10","v10"):
        axes[1,1].plot(np.arange(len(selected))+1,[r["train"].get("temporal_output_grad_rms_"+name,np.nan) for r in selected],label=name)
    axes[1,1].set(title="New-loss endpoint gradient RMS",yscale="log",xlabel="B then C")
    labels=list(comparisons); x=np.arange(len(labels))
    for offset,key in enumerate(("rmse","crps","ensemble_spread")):
        axes[1,2].bar(x+(offset-1)*.22,[comparisons[n]["normalized_overall"][key] for n in labels],width=.22,label=key)
    axes[1,2].set(xticks=x,xticklabels=labels,title="Same held-out validation windows")
    for ax in axes.ravel(): ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.suptitle("Synthetic 120h smoke — tiny budget, one seed; no ERA5 improvement claim")
    fig.savefig(output,dpi=120); plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir",default="outputs/temporal-120h-smoke")
    p.add_argument("--report-dir",default="docs/results/temporal-120h-smoke")
    p.add_argument("--manifold-epochs",type=int,default=10)
    p.add_argument("--expert-epochs",type=int,default=6)
    p.add_argument("--joint-epochs",type=int,default=4)
    args=p.parse_args(argv)
    torch.set_num_threads(1)
    started=time.perf_counter()
    work,report=Path(args.work_dir),Path(args.report_dir)
    work.mkdir(parents=True,exist_ok=False); report.mkdir(parents=True,exist_ok=False)
    archive,_=synthetic_archive(work,count=480)
    common=dict(batch_size=8,window_stride=8,ensemble_size=3,integration_steps=2,
                sampled_leads=2,max_validation_windows=4,learning_rate=1e-3,
                weight_decay=1e-4,seed=7,device="cpu")
    a=train_manifold_moe(archive,work/"a.pt",stage="manifold",manifold_epochs=args.manifold_epochs,
        model_options={"history_steps":6,"history_stride":1,"horizon_steps":20,
                       "num_experts":3,"manifold_dim":6,"hidden_dim":48,"context_dim":24},**common)
    payload_a=torch.load(a,map_location="cpu",weights_only=False)
    checkpoints={"a":a}; comparisons={}; logs={}
    for label in ("baseline","new"):
        extra={} if label=="baseline" else dict(delta_weight=.02,trajectory_weight=.1,
            wind_speed_weight=.01,wind_direction_weight=.005,trajectory_edges=2,
            validation_trajectory_edges=0,temporal_warmup_epochs=3,log_gradient_norms=True)
        b=train_manifold_moe(archive,work/f"{label}-b.pt",stage="specialize",init_checkpoint=a,
                            expert_epochs=args.expert_epochs,**common,**extra)
        payload_b=torch.load(b,map_location="cpu",weights_only=False)
        keys=[k for k in payload_a["model"] if k.startswith(("manifold.","physics.","reference_encoder.",
                                                             "latent_","gate.centers","gate.radius"))]
        assert all(torch.equal(payload_a["model"][k],payload_b["model"][k]) for k in keys)
        c=train_manifold_moe(archive,work/f"{label}-c.pt",stage="joint",init_checkpoint=b,
                            joint_epochs=args.joint_epochs,**common,**extra)
        checkpoints[label+"-b"]=b; checkpoints[label+"-c"]=c
        records=[]
        for path in (a,b,c): records+=json.loads(path.with_suffix(".metrics.json").read_text())
        logs[label]=records; write_json(report/f"training-{label}.json",records)
        evaluated=evaluate_flow_checkpoint(c,archive,report/f"validation-{label}.json",
            split_name="validation",ensemble_size=4,integration_steps=4,max_cases=4,seed=83,device="cpu")
        comparisons[label]=json.loads(evaluated.read_text())
    # Exercise FULL twenty-edge gradients in actual Stage B training, not just export.
    full=train_manifold_moe(archive,work/"full-window-b.pt",stage="specialize",init_checkpoint=a,
        expert_epochs=1,**{**common,"batch_size":2,"window_stride":32,"max_validation_windows":2},
        delta_weight=.02,trajectory_weight=.1,trajectory_edges=0,validation_trajectory_edges=0,
        temporal_warmup_epochs=1,log_gradient_norms=True)
    full_rows=json.loads(full.with_suffix(".metrics.json").read_text())
    assert full_rows[0]["train"]["trajectory_edges"]==20
    assert full_rows[0]["train"]["loss_delta"]>0
    write_json(report/"training-full-window.json",full_rows)
    checkpoints["full-window-b"]=full
    _,times,_=load_moe_archive(archive)
    start=comparisons["new"]["evaluation_windows"][0]
    origin=times[start+payload_a["model_config"]["history_steps"]-1]
    forecast=forecast_from_checkpoint(checkpoints["new-c"],archive,str(origin),work/"forecast.npz",
        ensemble_size=3,integration_steps=4,forecast_steps=20,seed=83,device="cpu")
    # Both exports use the exact same forecast and stable member indices.
    for interval,extension in ((6,"mp4"),(12,"gif")):
        export_trajectories(forecast,archive,report/f"members-{interval}h",interval_hours=interval,
            extension=extension,reference_label="Synthetic truth (not ERA5)")
    shutil.copy2(forecast,report/"forecast-native-6h.npz")
    plot_logs(logs,comparisons,report/"training-comparison.png")
    summary={"experiment":"From-scratch synthetic A/B/C; no ERA5 training or GPU execution",
        "shape":[480,4,4,8],"horizon_hours":120,"native_step_hours":6,"future_leads":20,
        "baseline":"same A and RNG, corrected noise, new loss weights zero; NOT original commit baseline",
        "new_training":"2-edge sub-blocks; full 20-edge validation; separate full-window B backward smoke",
        "stage_b_frozen_manifold_exact":True,
        "checkpoints":{k:{"sha256":_sha256(v),"best_epoch":torch.load(v,map_location="cpu",weights_only=False)["training"]["best_epoch"]} for k,v in checkpoints.items()},
        "archive_sha256":payload_a["training"]["archive_sha256"],
        "temporal_statistics":payload_a["training"]["temporal_statistics"],
        "validation":{k:{"state":v["normalized_overall"],"temporal":v["temporal_overall"]} for k,v in comparisons.items()},
        "test_used":False,"rendered_members":[0,1,2],"intervals_hours":[6,12],
        "elapsed_seconds":time.perf_counter()-started,
        "peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","threads":1},
        "limits":["One toy seed; not meteorological data, calibration or specialization evidence",
                  "Tiny smoke budgets do not establish superiority or fix overfitting",
                  "No actual ERA5 archive, RunPod connection or GPU benchmark available",
                  "Full-horizon objective executed for one B epoch only, not converged 120h joint law"]}
    write_json(report/"summary.json",summary)
    print(json.dumps(summary,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

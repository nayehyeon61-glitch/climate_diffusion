"""Train final A/B/C manifold model on a reproducible toy archive; no ERA5 claim."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch
import numpy as np

from smoke_moe import synthetic_archive
from climate_diffusion.train_manifold_moe import train_manifold_moe
from climate_diffusion.evaluation import evaluate_flow_checkpoint
from climate_diffusion.manifold_diagnostics import diagnose_manifold


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",default="outputs/manifold-smoke")
    parser.add_argument("--report-dir",default="docs/results/manifold-smoke")
    parser.add_argument("--manifold-epochs",type=int,default=50)
    parser.add_argument("--expert-epochs",type=int,default=40)
    parser.add_argument("--joint-epochs",type=int,default=10)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    work,report=Path(args.work_dir),Path(args.report_dir)
    work.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    archive,_=synthetic_archive(work,count=480)
    final=train_manifold_moe(archive,work/"model.pt",manifold_epochs=args.manifold_epochs,
                             expert_epochs=args.expert_epochs,joint_epochs=args.joint_epochs,
                             model_options={"history_steps":6,"history_stride":1,"horizon_steps":8,
                                            "num_experts":3,"manifold_dim":6,"hidden_dim":64,
                                            "context_dim":32,"gate_hidden_dim":64},
                             batch_size=16,window_stride=2,ensemble_size=4,integration_steps=3,
                             sampled_leads=1,max_validation_windows=16,learning_rate=1e-3,
                             seed=7,device="cpu")
    checkpoints={"manifold":work/"model.manifold.pt","specialize":work/"model.specialize.pt","joint":final}
    payloads={k:torch.load(p,map_location="cpu",weights_only=False) for k,p in checkpoints.items()}
    a,b=payloads["manifold"],payloads["specialize"]
    frozen_equal=all(torch.equal(value,b["model"][key]) for key,value in a["model"].items()
                     if key.startswith(("manifold.","physics.","reference_encoder.","latent_","gate.centers","gate.radius")))
    if not frozen_equal:
        raise AssertionError("Stage B changed the frozen manifold/geometry")
    evaluations={}
    for label,checkpoint,mode in (("stage_b",checkpoints["specialize"],"local"),
                                  ("uniform",final,"uniform"),("local",final,"local")):
        path=evaluate_flow_checkpoint(checkpoint,archive,report/f"evaluation-{label}.json",ensemble_size=8,
                                      integration_steps=8,max_cases=8,seed=83,device="cpu",moe_mode=mode)
        evaluations[label]=json.loads(path.read_text())
    for stage,checkpoint in checkpoints.items():
        diagnose_manifold(checkpoint,archive,report/f"diagnostics-{stage}.json",max_cases=32,
                           members=4,integration_steps=4,seed=83,device="cpu")
    metrics=json.loads(final.with_suffix(".metrics.json").read_text())
    (report/"training-metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    diagnostics=json.loads((report/"diagnostics-joint.json").read_text())
    summary={"experiment":"synthetic manifold A/B/C smoke; NOT ERA5 or RunPod",
             "state_shape":[480,4,4,8],"horizon_hours":48,
             "model_config":payloads["joint"]["model_config"],
             "best_epochs":{k:v["training"]["best_epoch"] for k,v in payloads.items()},
             "stage_b_frozen_manifold_exactly_equal":frozen_equal,
             "test":{k:v["normalized_overall"] for k,v in evaluations.items()},
             "teacher_forced_audit":diagnostics["teacher_forced_audit"],
             "generated_audit":{k:v for k,v in diagnostics["generated_audit"].items() if k!="intrinsic_path_pca"},
             "test_manifold_reconstruction_rmse":diagnostics["manifold_reconstruction_rmse_test"],
             "archive_sha256":payloads["joint"]["training"]["archive_sha256"],
             "checkpoint_sha256":evaluations["local"]["checkpoint_sha256"],
             "parameter_count":payloads["joint"]["training"]["parameter_count"],
             "elapsed_seconds":time.perf_counter()-started,
             "peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             "environment":{"python":platform.python_version(),"torch":torch.__version__,
                            "numpy":np.__version__,"device":"cpu","threads":1},
             "limits":["One seed, short toy horizon; partition does not prove meteorological specialization",
                       "Surface diagnostic matching, not primitive PDE or exact conservation",
                       "Decoder image may have singularities; damped lift is approximate",
                       "Leadwise conditional marginals, not a trained joint-time trajectory distribution"]}
    (report/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

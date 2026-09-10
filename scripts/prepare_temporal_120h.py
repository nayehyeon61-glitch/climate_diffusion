"""Read-only archive audit / train-only 120h statistics manifest; no model training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from climate_diffusion.manifold_moe import ManifoldMoEConfig
from climate_diffusion.moe_data import load_moe_archive, field_grid, build_moe_split
from climate_diffusion.temporal_supervision import fit_temporal_statistics, TemporalWindowDataset
from climate_diffusion.train import _sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--history-steps",type=int,default=6)
    p.add_argument("--history-stride",type=int,default=4)
    p.add_argument("--purge-windows",type=int,default=0)
    p.add_argument("--variable-weights",default='{"msl":1,"t2m":1,"u10":1,"v10":1}')
    p.add_argument("--checkpoint",help="Optional: assert a new Stage A exactly matches the preflight contract")
    args=p.parse_args(argv)
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    states,times,schema=load_moe_archive(args.archive)
    if int(schema["forecast_step_hours"])!=6:
        raise ValueError("This from-scratch profile requires a 6-hour archive")
    config=ManifoldMoEConfig(state_dim=states.shape[1],grid=field_grid(schema),
        history_steps=args.history_steps,history_stride=args.history_stride,horizon_steps=20)
    count=len(states)-config.history_span_steps-config.horizon_steps+1
    split=build_moe_split(count,20,purge_windows=args.purge_windows)
    end=split["train"][-1]+config.history_span_steps+20
    mean,scale=states[:end].mean(0),states[:end].std(0)
    scale=np.where(scale>1e-6,scale,1).astype(np.float32)
    statistics=fit_temporal_statistics(states,times,schema,scale,end,json.loads(args.variable_weights))
    sample=TemporalWindowDataset(states,times,config,[0],mean,scale,schema)[0]
    result={"profile":"temporal_120h.v1","archive_sha256":_sha256(Path(args.archive)),
        "history_steps":config.history_steps,"history_stride":config.history_stride,
        "history_span_steps":config.history_span_steps,"horizon_steps":20,"step_hours":6,
        "split":split,"normalization_span":[0,end],"temporal_statistics":statistics,
        "state_mean_sha256":hashlib.sha256(mean.tobytes()).hexdigest(),
        "state_scale_sha256":hashlib.sha256(scale.tobytes()).hexdigest(),
        "sample_shapes":{k:list(v.shape) for k,v in sample.items()},"schema":schema,
        "missingness":"archive pooled cells fully observed or fail; no target imputation",
        "checkpoint_verified":False}
    if args.checkpoint:
        import torch
        ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        assert ck["training"]["archive_sha256"]==result["archive_sha256"]
        assert ck["training"]["split"]==split
        assert ck["training"]["temporal_statistics"]==statistics
        assert np.array_equal(ck["state_mean"].numpy(),mean)
        assert np.array_equal(ck["state_scale"].numpy(),scale)
        for key in ("history_steps","history_stride","horizon_steps","step_hours"):
            assert ck["model_config"][key]==result[key]
        result["checkpoint_verified"]=True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Validated 120h/20-transition profile; unique train pairs={end-1}; manifest={output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())

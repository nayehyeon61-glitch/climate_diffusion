"""Resumable RunPod trainer for exact fixed-step vector Flow Matching archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import FlowLossConfig, FlowModelConfig
from .model import MonthlyLatentFlow
from .train import _epoch, build_purged_temporal_split


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd); temporary = Path(name)
    try:
        torch.save(payload, temporary); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixed_step_archive(path: str | Path):
    archive_path = Path(path).expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        states = archive["states"].astype(np.float32)
        times = archive["times"].astype("datetime64[ns]")
        mask = archive["observed_mask"].astype(bool) if "observed_mask" in archive.files else np.isfinite(states)
    schema = json.loads(archive_path.with_suffix(".schema.json").read_text(encoding="utf-8"))
    if schema.get("format") != "climate_diffusion.fixed_step_state.v1":
        raise ValueError("Fixed-step trainer requires climate_diffusion.fixed_step_state.v1")
    step = int(schema.get("forecast_step_hours", 0))
    if step <= 0:
        raise ValueError("Fixed-step archive requires positive forecast_step_hours")
    if states.ndim != 2 or states.shape[1] != int(schema["state_dim"]):
        raise ValueError("Fixed-step states do not match schema state_dim")
    if mask.shape != states.shape or len(times) != len(states):
        raise ValueError("Fixed-step state/mask/time shapes do not match")
    if len(times) > 1:
        actual = np.diff(times).astype("timedelta64[h]").astype(np.int64)
        if not np.all(actual == step):
            raise ValueError(f"Archive timestamps violate {step}h forecast-step contract")
    if np.isinf(states).any():
        raise ValueError("Fixed-step archive contains Inf")
    return archive_path, states, times, mask, schema, step


class FixedStepWindowDataset(Dataset):
    def __init__(self, states, mask, history_steps, lead_steps, indices, mean, scale):
        self.states=states; self.mask=mask; self.history_steps=history_steps; self.lead_steps=lead_steps
        self.indices=list(indices); self.mean=mean; self.scale=scale
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        start=self.indices[index]; target_index=start+self.history_steps+self.lead_steps-1
        history=np.asarray(self.states[start:start+self.history_steps],dtype=np.float32)
        history_mask=np.asarray(self.mask[start:start+self.history_steps],dtype=bool)
        target=np.asarray(self.states[target_index],dtype=np.float32); target_mask=np.asarray(self.mask[target_index],dtype=bool)
        history=np.where(history_mask,history,self.mean); target=np.where(target_mask,target,self.mean)
        history=(history-self.mean)/self.scale; target=(target-self.mean)/self.scale
        return {"history":torch.from_numpy(history.copy()),"target":torch.from_numpy(target.copy()),"target_mask":torch.from_numpy(target_mask.copy())}


def train_runpod_fixed_step_flow(
    archive_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    history_steps: int = 2,
    lead_steps: int = 1,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    purge_windows: int = 1,
    seed: int = 7,
    resume: bool = True,
) -> Path:
    if min(history_steps, lead_steps, epochs, batch_size) < 1:
        raise ValueError("history/lead/epochs/batch must be positive")
    archive, states, times, mask, schema, step = _load_fixed_step_archive(archive_path)
    sample_count=len(states)-history_steps-lead_steps+1
    split=build_purged_temporal_split(sample_count,validation_fraction=validation_fraction,test_fraction=test_fraction,purge_windows=purge_windows)
    last_train_target=split.train[-1]+history_steps+lead_steps-1
    train_raw=np.asarray(states[:last_train_target+1],dtype=np.float64)
    train_mask=np.asarray(mask[:last_train_target+1],dtype=bool)
    count=train_mask.sum(axis=0)
    if np.any(count==0): raise ValueError("Fixed-step train split has all-missing features")
    safe=np.where(train_mask,train_raw,0.0)
    mean=(safe.sum(axis=0)/count).astype(np.float32)
    variance=np.maximum((np.square(safe).sum(axis=0)/count)-np.square(mean.astype(np.float64)),0.0)
    scale=np.sqrt(variance).astype(np.float32); scale=np.where(scale>1e-6,scale,1.0).astype(np.float32)
    train_ds=FixedStepWindowDataset(states,mask,history_steps,lead_steps,split.train,mean,scale)
    val_ds=FixedStepWindowDataset(states,mask,history_steps,lead_steps,split.validation,mean,scale)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,generator=generator)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    config=FlowModelConfig(state_dim=states.shape[1],history_months=history_steps,latent_dim=latent_dim,hidden_dim=hidden_dim,backend="vector_mlp")
    loss_config=FlowLossConfig(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MonthlyLatentFlow(config).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    root=Path(checkpoint_dir).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    latest=root/"latest.pt"; best=root/"best.pt"; metrics=root/"metrics.jsonl"
    start_epoch=1; best_loss=float("inf"); best_epoch=0
    if resume and latest.is_file():
        payload=torch.load(latest,map_location=device,weights_only=False)
        saved=payload.get("training",{})
        if int(saved.get("forecast_step_hours",-1))!=step: raise ValueError("Resume checkpoint forecast-step mismatch")
        if payload.get("model_config")!=asdict(config): raise ValueError("Resume model configuration mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        start_epoch=int(saved["epoch"])+1; best_loss=float(saved["best_validation_loss"]); best_epoch=int(saved["best_epoch"])
    for epoch in range(start_epoch,epochs+1):
        train_metrics=_epoch(model,train_loader,loss_config,device,optimizer)
        val_metrics=_epoch(model,val_loader,loss_config,device,optimizer=None,eval_seed=seed)
        improved=val_metrics["loss"]<best_loss
        if improved: best_loss=float(val_metrics["loss"]); best_epoch=epoch
        payload={
            "format":"climate_diffusion.monthly_latent_flow.v3",
            "model":model.state_dict(),"model_config":asdict(config),"loss_config":asdict(loss_config),
            "state_mean":torch.from_numpy(mean),"state_scale":torch.from_numpy(scale),
            "auxiliary_mean":torch.empty(0),"auxiliary_scale":torch.empty(0),"schema":schema,
            "optimizer":optimizer.state_dict(),
            "training":{"epoch":epoch,"best_epoch":best_epoch,"best_validation_loss":best_loss,"history_steps":history_steps,"lead_steps":lead_steps,"forecast_step_hours":step,"archive":str(archive),"split":asdict(split)},
        }
        _atomic_save(payload,latest)
        if improved:
            frozen=dict(payload); frozen.pop("optimizer",None); frozen["training"]=dict(frozen["training"],checkpoint_role="best_frozen_candidate")
            _atomic_save(frozen,best)
            digest=_sha256(best)
            (root/"best.manifest.json").write_text(json.dumps({"format":"climate_diffusion.artifact.v2","checkpoint":"best.pt","checkpoint_sha256":digest,"forecast_step_hours":step,"schema_format":schema.get("format"),"archive":str(archive)},indent=2)+"\n",encoding="utf-8")
        with metrics.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch":epoch,"train":train_metrics,"validation":val_metrics,"best_validation_loss":best_loss,"best_epoch":best_epoch},allow_nan=False)+"\n")
        print({"epoch":epoch,"train":train_metrics,"validation":val_metrics,"best_epoch":best_epoch})
    return best


def main(argv=None):
    p=argparse.ArgumentParser(description="Train/resume exact fixed-step Flow Matching on RunPod")
    p.add_argument("--archive",required=True); p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--history-steps",type=int,default=2); p.add_argument("--lead-steps",type=int,default=1)
    p.add_argument("--latent-dim",type=int,default=64); p.add_argument("--hidden-dim",type=int,default=256)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--validation-fraction",type=float,default=.15); p.add_argument("--test-fraction",type=float,default=.15); p.add_argument("--purge-windows",type=int,default=1); p.add_argument("--seed",type=int,default=7)
    p.add_argument("--no-resume",action="store_true")
    a=p.parse_args(argv)
    result=train_runpod_fixed_step_flow(a.archive,a.checkpoint_dir,history_steps=a.history_steps,lead_steps=a.lead_steps,latent_dim=a.latent_dim,hidden_dim=a.hidden_dim,epochs=a.epochs,batch_size=a.batch_size,learning_rate=a.learning_rate,validation_fraction=a.validation_fraction,test_fraction=a.test_fraction,purge_windows=a.purge_windows,seed=a.seed,resume=not a.no_resume)
    print(result); return 0
if __name__=="__main__": raise SystemExit(main())

"""Load and sample trained latent flow checkpoints."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .config import FlowModelConfig
from .data import load_auxiliary_states, load_monthly_archive, positional_grid
from .model import MonthlyLatentFlow
from .validation import require_finite_numpy, require_finite_tensor, require_no_inf_numpy
LEGACY_MONTHLY_STEP_HOURS=30*24
class LatentFlowForecaster:
    inference_only=True
    def __init__(self,checkpoint:str|Path,*,device:str|None=None):
        self.checkpoint_path=Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file(): raise FileNotFoundError(self.checkpoint_path)
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); payload=torch.load(self.checkpoint_path,map_location=self.device,weights_only=False)
        if payload.get("format") not in {"climate_diffusion.monthly_latent_flow.v1","climate_diffusion.monthly_latent_flow.v2","climate_diffusion.monthly_latent_flow.v3","climate_diffusion.latent_flow.v3"}: raise ValueError("Unsupported climate flow checkpoint format")
        self.config=FlowModelConfig(**payload["model_config"]); self.model=MonthlyLatentFlow(self.config).to(self.device); self.model.load_state_dict(payload["model"])
        for name,p in self.model.named_parameters(): require_finite_tensor(p,f"checkpoint parameter {name}")
        self.model.eval().requires_grad_(False)
        if any(p.requires_grad for p in self.model.parameters()): raise RuntimeError("Flow checkpoint could not be frozen")
        self.state_mean=payload["state_mean"].to(self.device); self.state_scale=payload["state_scale"].to(self.device); self.auxiliary_mean=payload.get("auxiliary_mean",torch.empty(0)).to(self.device); self.auxiliary_scale=payload.get("auxiliary_scale",torch.empty(0)).to(self.device)
        for value,name in ((self.state_mean,"checkpoint state_mean"),(self.state_scale,"checkpoint state_scale"),(self.auxiliary_mean,"checkpoint auxiliary_mean"),(self.auxiliary_scale,"checkpoint auxiliary_scale")): require_finite_tensor(value,name)
        if bool((self.state_scale<=0).any()) or bool((self.auxiliary_scale<=0).any()): raise ValueError("Checkpoint normalization scales must be positive")
        self.schema=payload["schema"]
        self.coordinates=None
        if self.config.positional_channels:
            planes=positional_grid(self.schema)
            if planes is None: raise ValueError("Checkpoint encodes positional channels but its schema has no grid")
            self.coordinates=torch.as_tensor(planes,dtype=torch.float32,device=self.device).unsqueeze(0)
        self.training_metadata=payload.get("training",{}); self.checkpoint_format=str(payload["format"])
        self.forecast_step_hours=int(self.training_metadata.get("forecast_step_hours",self.schema.get("forecast_step_hours",LEGACY_MONTHLY_STEP_HOURS)))
        if self.forecast_step_hours<=0: raise ValueError("Flow checkpoint forecast_step_hours must be positive")
        self.checkpoint_sha256=self._verify_manifest()
    def _verify_manifest(self):
        h=hashlib.sha256()
        with self.checkpoint_path.open("rb") as s:
            for b in iter(lambda:s.read(1024*1024),b""): h.update(b)
        digest=h.hexdigest(); path=self.checkpoint_path.with_suffix(".manifest.json")
        if path.is_file():
            manifest=json.loads(path.read_text(encoding="utf-8")); expected=manifest.get("checkpoint_sha256")
            if expected and expected!=digest: raise ValueError(f"Checkpoint checksum mismatch for {self.checkpoint_path}")
            manifest_step=manifest.get("forecast_step_hours")
            if manifest_step is not None and int(manifest_step)!=self.forecast_step_hours: raise ValueError("Flow manifest/checkpoint forecast-step mismatch")
        return digest
    def _normalise(self,values):
        require_no_inf_numpy(values,"forecast history"); t=torch.as_tensor(np.array(values,dtype=np.float32,copy=True),device=self.device); t=torch.where(torch.isfinite(t),t,self.state_mean); n=(t-self.state_mean)/self.state_scale; require_finite_tensor(n,"normalized forecast history"); return n
    def _denormalise(self,values):
        r=values*self.state_scale+self.state_mean; require_finite_tensor(r,"denormalized flow prediction"); out=r.detach().cpu().numpy().astype(np.float32); require_finite_numpy(out,"flow forecast output"); return out
    @torch.inference_mode()
    def forecast(self,history_states:np.ndarray,*,months:int=1,ensemble_size:int=1,integration_steps:int=32,seed:int=0,history_auxiliary:np.ndarray|None=None,tile_size:tuple[int,int]|None=None,tile_overlap:int|None=None):
        history=np.asarray(history_states,dtype=np.float32)
        expected=(self.config.history_months,self.config.state_dim) if self.config.backend=="vector_mlp" else (self.config.history_months,self.config.spatial_channels,self.config.grid_height,self.config.grid_width)
        if history.shape!=expected: raise ValueError(f"Expected history shape {expected}, received {history.shape}")
        if min(months,ensemble_size,integration_steps)<1: raise ValueError("months, ensemble_size and integration_steps must be positive")
        normalized=self._normalise(history); normalized_auxiliary=None
        if self.config.auxiliary_dim:
            if history_auxiliary is None: history_auxiliary=np.broadcast_to(self.auxiliary_mean.detach().cpu().numpy(),(self.config.history_months,self.config.auxiliary_dim))
            auxiliary=torch.as_tensor(history_auxiliary,dtype=torch.float32,device=self.device); expected_aux=(self.config.history_months,self.config.auxiliary_dim)
            if auxiliary.shape!=expected_aux: raise ValueError(f"Expected auxiliary history shape {expected_aux}, received {tuple(auxiliary.shape)}")
            if bool(torch.isinf(auxiliary).any()): raise ValueError("Forecast auxiliary history contains Inf")
            auxiliary=torch.where(torch.isfinite(auxiliary),auxiliary,self.auxiliary_mean); normalized_auxiliary=(auxiliary-self.auxiliary_mean)/self.auxiliary_scale; require_finite_tensor(normalized_auxiliary,"normalized forecast auxiliary")
        if self.config.backend!="vector_mlp" and tile_size is None:
            stored=self.training_metadata.get("patch_size"); tile_size=None if stored is None else tuple(int(v) for v in stored)
        if tile_overlap is None: tile_overlap=int(self.training_metadata.get("tile_overlap",0))
        outputs=[]
        for member in range(ensemble_size):
            member_history=normalized.clone(); member_aux=None if normalized_auxiliary is None else normalized_auxiliary.clone(); gen=torch.Generator(device=self.device).manual_seed(seed+member); member_outputs=[]
            for _ in range(months):
                mh=member_history.unsqueeze(0); ma=None if member_aux is None else member_aux.unsqueeze(0)
                if self.config.backend!="vector_mlp" and tile_size is not None and (tile_size[0]<self.config.grid_height or tile_size[1]<self.config.grid_width): prediction=self.model.sample_tiled(mh,tile_size=tile_size,overlap=tile_overlap,integration_steps=integration_steps,generator=gen,history_auxiliary=ma,coordinates=self.coordinates)[0]
                else: prediction=self.model.sample(mh,integration_steps=integration_steps,generator=gen,history_auxiliary=ma,coordinates=self.coordinates)[0]
                member_outputs.append(prediction); member_history=torch.cat((member_history[1:],prediction.unsqueeze(0)),dim=0)
                if member_aux is not None: member_aux=torch.cat((member_aux[1:],member_aux[-1:]),dim=0)
            outputs.append(torch.stack(member_outputs))
        return self._denormalise(torch.stack(outputs))
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--archive",required=True); p.add_argument("--months",type=int,default=1); p.add_argument("--ensemble-size",type=int,default=1); p.add_argument("--integration-steps",type=int,default=32); p.add_argument("--seed",type=int,default=0); p.add_argument("--output",default="outputs/climate-flow-forecast.npz"); a=p.parse_args(argv)
    states,times,schema=load_monthly_archive(a.archive); f=LatentFlowForecaster(a.checkpoint); history=states[-f.config.history_months:]; auxiliary=load_auxiliary_states(a.archive,schema); ah=None if auxiliary is None else np.asarray(auxiliary[-f.config.history_months:]); predictions=f.forecast(history,months=a.months,ensemble_size=a.ensemble_size,integration_steps=a.integration_steps,seed=a.seed,history_auxiliary=ah); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,predictions=predictions,last_history_time=times[-1],checkpoint=str(f.checkpoint_path),forecast_step_hours=np.asarray(f.forecast_step_hours,dtype=np.int64)); print(out); return 0
if __name__=="__main__": raise SystemExit(main())

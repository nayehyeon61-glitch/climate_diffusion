import hashlib, json
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from climate_diffusion.config import FlowModelConfig
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner

def _checkpoint(tmp_path, step=360, manifest_step=None):
    config=FlowModelConfig(state_dim=2,history_months=2,latent_dim=2,hidden_dim=8)
    model=MonthlyLatentFlow(config); path=tmp_path/f"flow-{step}.pt"
    schema={"format":"climate_diffusion.fixed_step_state.v1" if step==360 else "climate_diffusion.monthly_state.v1","state_dim":2,"forecast_step_hours":step,"variables":[{"name":"msl","dims":["lat","lon"],"shape":[1,2],"slice":[0,2],"coords":{"lat":[30.0],"lon":[120.0,140.0]},"attrs":{}}]}
    if step!=360:
        schema["field_dim"]=2; schema["integrated_feature_names"]=[]
    torch.save({"format":"climate_diffusion.monthly_latent_flow.v3","model":model.state_dict(),"model_config":config.__dict__,"state_mean":torch.zeros(2),"state_scale":torch.ones(2),"schema":schema,"training":{"forecast_step_hours":step}},path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"checkpoint_sha256":digest,"forecast_step_hours":step if manifest_step is None else manifest_step}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return path

def _history():
    times=pd.to_datetime(["2026-07-02T00:00:00","2026-07-17T00:00:00"])
    return xr.Dataset({"msl":(("time","lat","lon"),np.ones((2,1,2),dtype=np.float32))},coords={"time":times,"lat":[30.0],"lon":[120.0,140.0]})

def test_360h_checkpoint_is_accepted(tmp_path):
    runner=FlowMatchingWeatherRunner(_checkpoint(tmp_path,360),integration_steps=1,device="cpu")
    result=runner.rollout(_history(),360)
    assert result.attrs["forecast_step_hours"]==360
    assert result.attrs["forecast_horizon_hours"]==360
    assert result.sizes["time"]==1

def test_legacy_720h_checkpoint_rejected_for_p15(tmp_path):
    runner=FlowMatchingWeatherRunner(_checkpoint(tmp_path,720),integration_steps=1,device="cpu")
    with pytest.raises(ValueError,match="720-hour"):
        runner.rollout(_history(),360)

def test_manifest_forecast_step_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="manifest/checkpoint forecast-step mismatch"):
        LatentFlowForecaster(_checkpoint(tmp_path,360,manifest_step=720),device="cpu")

def test_loaded_parameters_are_frozen(tmp_path):
    forecaster=LatentFlowForecaster(_checkpoint(tmp_path,360),device="cpu")
    assert forecaster.model.training is False
    assert all(not p.requires_grad for p in forecaster.model.parameters())
    assert all(p.grad is None for p in forecaster.model.parameters())

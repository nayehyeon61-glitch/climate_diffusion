"""WeatherNext-compatible rollout adapter backed by frozen latent Flow Matching."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from .data import aggregate_monthly_fields, auxiliary_from_dataset, reconstruct_dataset, reconstruct_spatial_dataset, spatialize_dataset, vectorize_dataset
from .fixed_step_data import sample_fixed_step_history
from .inference import LatentFlowForecaster


def _vectorize_fixed_step(dataset: xr.Dataset, schema: dict[str, object]) -> np.ndarray:
    """Vectorize legacy/minimal fixed-step snapshots using their checkpoint schema."""
    vectors=[]
    for time_index in range(dataset.sizes["time"]):
        state=np.full(int(schema["state_dim"]),np.nan,dtype=np.float32)
        for variable in schema["variables"]:
            name=variable["name"]
            if name not in dataset:
                raise ValueError(f"Initial state is missing trained variable {name!r}")
            array=dataset[name].isel(time=time_index)
            interpolation={}
            for dim,values in variable.get("coords",{}).items():
                if dim not in array.coords:
                    continue
                requested=np.asarray(values); current=np.asarray(array.coords[dim].values)
                if np.array_equal(current,requested):
                    continue
                if current.size<2:
                    raise ValueError(f"Cannot interpolate singleton coordinate {dim!r} to the training grid")
                interpolation[dim]=requested
            if interpolation:
                array=array.interp(interpolation)
            values=np.asarray(array.transpose(*variable["dims"]).values,dtype=np.float32).reshape(-1)
            start,end=variable["slice"]
            if values.size!=end-start:
                raise ValueError(f"Variable {name!r} shape does not match training schema")
            if np.isinf(values).any():
                raise ValueError(f"Inference variable {name!r} contains Inf")
            state[start:end]=values
        vectors.append(state)
    return np.stack(vectors)


class FlowMatchingWeatherRunner:
    """Inference-only replacement implementing rollout(initial_state, horizon_hours)."""
    inference_only=True
    def __init__(self,checkpoint_path:str|Path,*,integration_steps:int=32,seed:int=0,device:str|None=None):
        self.forecaster=LatentFlowForecaster(checkpoint_path,device=device)
        self.checkpoint_path=self.forecaster.checkpoint_path
        self.integration_steps=integration_steps
        self.seed=seed
        self.forecast_step_hours=self.forecaster.forecast_step_hours
    def provenance(self):
        return {"forecast_backend":"flow_matching","forecast_checkpoint":str(self.checkpoint_path),"forecast_checkpoint_kind":"flow_matching","forecast_checkpoint_sha256":self.forecaster.checkpoint_sha256,"forecast_checkpoint_format":self.forecaster.checkpoint_format,"forecast_release":"climate-diffusion","forecast_step_hours":self.forecast_step_hours,"forecast_schema_format":self.forecaster.schema.get("format",""),"weather_next_replacement":True,"inference_only":True,"parameters_frozen":True}
    def _prepare_history(self,initial_state):
        if str(self.forecaster.schema.get("format",""))=="climate_diffusion.fixed_step_state.v1":
            return sample_fixed_step_history(initial_state,self.forecast_step_hours)
        monthly,_=aggregate_monthly_fields(initial_state,complete_only=True)
        return monthly
    def rollout(self,initial_state:xr.Dataset,horizon_hours:int)->xr.Dataset:
        step=self.forecast_step_hours
        if horizon_hours<=0 or horizon_hours%step:
            raise ValueError(f"FlowMatchingWeatherRunner requires positive {step}-hour multiples")
        steps=horizon_hours//step
        if "time" not in initial_state.coords:
            raise ValueError("Flow initial state requires a time coordinate")
        state=self._prepare_history(initial_state)
        spatial=self.forecaster.schema.get("layout")=="spatial"
        fixed=str(self.forecaster.schema.get("format",""))=="climate_diffusion.fixed_step_state.v1"
        if spatial:
            vectors=spatialize_dataset(state,self.forecaster.schema)
            auxiliary=auxiliary_from_dataset(state,self.forecaster.schema,self.forecaster.auxiliary_mean.detach().cpu().numpy())
        elif fixed and "field_dim" not in self.forecaster.schema:
            vectors=_vectorize_fixed_step(state,self.forecaster.schema)
            auxiliary=None
        else:
            vectors=vectorize_dataset(state,self.forecaster.schema,integrated_defaults=np.asarray(self.forecaster.state_mean.detach().cpu(),dtype=np.float32))
            auxiliary=None
        required=self.forecaster.config.history_months
        if vectors.shape[0]<required:
            raise ValueError(f"Flow checkpoint requires {required} history states; received {vectors.shape[0]}")
        prediction=self.forecaster.forecast(vectors[-required:],months=steps,ensemble_size=1,integration_steps=self.integration_steps,seed=self.seed,history_auxiliary=None if auxiliary is None else auxiliary[-required:])[0]
        last_time=pd.Timestamp(state.time.values[-1])
        reconstruct=reconstruct_spatial_dataset if spatial else reconstruct_dataset
        outputs=[reconstruct(prediction[i],self.forecaster.schema,last_time+pd.Timedelta(hours=step*(i+1))) for i in range(steps)]
        result=xr.concat(outputs,dim="time").assign_attrs(self.provenance())
        result.attrs["forecast_horizon_hours"]=int(horizon_hours)
        return result

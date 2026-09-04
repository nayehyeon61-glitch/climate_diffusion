"""Prepare causal fixed-step atmospheric state archives for Flow Matching."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
from .data import _coarsen_global_fields, _json_values, _normalise_coordinates, _open_dataset

def sample_fixed_step_history(dataset: xr.Dataset, step_hours: int) -> xr.Dataset:
    if step_hours <= 0: raise ValueError("step_hours must be positive")
    dataset = _normalise_coordinates(dataset).sortby("time")
    if "time" not in dataset.coords: raise ValueError("Fixed-step Flow input requires a time coordinate")
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if len(times) < 2: return dataset
    available = {int(value.value) for value in times}; step = pd.Timedelta(hours=step_hours)
    selected=[]; current=times[-1]; first=times[0]
    while current >= first:
        if int(current.value) not in available: raise ValueError(f"Input history does not contain exact {step_hours}h snapshot {current}")
        selected.append(current); current -= step
    selected.reverse(); return dataset.sel(time=np.asarray(selected,dtype="datetime64[ns]"))

def prepare_fixed_step_archive(fields: str|Path, output: str|Path, *, step_hours:int=360, variables:tuple[str,...]|None=None, target_lat_points:int=18, target_lon_points:int=36)->tuple[Path,Path]:
    if step_hours <= 0: raise ValueError("step_hours must be positive")
    with _open_dataset(fields) as source:
        dataset=_normalise_coordinates(source)
        if "time" not in dataset.coords: raise ValueError("Gridded climate fields require a time coordinate")
        names=tuple(variables or dataset.data_vars); missing=sorted(set(names).difference(dataset.data_vars))
        if missing: raise ValueError(f"Missing requested variables: {missing}")
        snapshots=sample_fixed_step_history(dataset[list(names)],step_hours)
        snapshots=_coarsen_global_fields(snapshots,target_lat_points,target_lon_points).load()
    times=pd.DatetimeIndex(pd.to_datetime(snapshots.time.values))
    if len(times)<2: raise ValueError("At least two fixed-step field states are required")
    actual=np.diff(times.values).astype("timedelta64[h]").astype(np.int64)
    if not np.all(actual==step_hours): raise ValueError(f"Fixed-step archive is not uniformly {step_hours}h")
    blocks=[]; variable_schema=[]; offset=0
    for name in names:
        array=snapshots[name]; dims=tuple(d for d in array.dims if d!="time"); array=array.transpose("time",*dims)
        values=np.asarray(array.values,dtype=np.float32).reshape(len(times),-1); blocks.append(values); size=values.shape[1]
        variable_schema.append({"name":name,"dims":list(dims),"shape":[int(array.sizes[d]) for d in dims],"slice":[offset,offset+size],"coords":{d:_json_values(np.asarray(array.coords[d].values)) for d in dims if d in array.coords and array.coords[d].dims==(d,)},"attrs":{k:str(v) for k,v in array.attrs.items()}}); offset+=size
    states=np.concatenate(blocks,axis=1).astype(np.float32); mask=np.isfinite(states); states=np.where(mask,states,0.0).astype(np.float32)
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,states=states,observed_mask=mask.astype(np.float32),times=times.values.astype("datetime64[ns]"))
    schema={"format":"climate_diffusion.fixed_step_state.v1","state_dim":int(states.shape[1]),"variables":variable_schema,"forecast_step_hours":int(step_hours),"state_time_semantics":"snapshot_valid_time","aggregation":"exact_fixed_step_snapshot","missing_value_policy":"zero_with_observed_mask_no_future_statistics","target_lat_points":target_lat_points,"target_lon_points":target_lon_points}
    schema_path=out.with_suffix(".schema.json"); schema_path.write_text(json.dumps(schema,indent=2)+"\n")
    return out,schema_path

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--fields",required=True); p.add_argument("--variables",nargs="*"); p.add_argument("--step-hours",type=int,default=360); p.add_argument("--target-lat-points",type=int,default=18); p.add_argument("--target-lon-points",type=int,default=36); p.add_argument("--output",default="data/flow_states_fixed_step.npz"); a=p.parse_args(argv)
    archive,schema=prepare_fixed_step_archive(a.fields,a.output,step_hours=a.step_hours,variables=None if not a.variables else tuple(a.variables),target_lat_points=a.target_lat_points,target_lon_points=a.target_lon_points); print(f"archive={archive}\nschema={schema}"); return 0
if __name__=="__main__": raise SystemExit(main())

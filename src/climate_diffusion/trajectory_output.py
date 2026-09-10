"""120-hour physical trajectory selection, per-member maps and diagnostics.

The default export is ALL members, never an implicit ensemble mean. Existing
forecast NPZ files are read once. There is no reforecast or member reassignment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .temporal_supervision import area_weights
from .time_alignment import align_forecast, _variable, _lag_diagnostic, load_saved_forecast


def select_interval(aligned, horizon_hours=120, interval_hours=6):
    if horizon_hours <= 0 or interval_hours <= 0 or horizon_hours % interval_hours:
        raise ValueError("Horizon must be a positive multiple of output interval")
    wanted = np.arange(interval_hours, horizon_hours+1, interval_hours)
    if not np.all(np.isin(wanted, aligned.lead_hours)):
        raise ValueError("Forecast lacks exact requested physical leads; no interpolation/relabeling")
    indices = np.searchsorted(aligned.lead_hours, wanted)
    return replace(aligned, members=aligned.members[:, indices], truth=aligned.truth[indices],
                   lead_hours=aligned.lead_hours[indices], valid_times=aligned.valid_times[indices])


def member_diagnostics(aligned, member, statistics=None):
    if not 0 <= member < len(aligned.members):
        raise ValueError("Invalid member id")
    prediction, truth = aligned.members[member], aligned.truth
    dt = np.diff(np.r_[0, aligned.lead_hours])
    pred_v = np.diff(np.concatenate((aligned.origin_state[None], prediction)), axis=0)/dt[:,None]
    true_v = np.diff(np.concatenate((aligned.origin_state[None], truth)), axis=0)/dt[:,None]
    area = area_weights(aligned.schema).reshape(-1)
    by_variable = {}
    for variable in aligned.schema["variables"]:
        name = variable["name"]
        block, _ = _variable(aligned.schema, name)
        def rms(value):
            return np.sqrt((value**2*area).sum(-1))
        true_norm, pred_norm = rms(true_v[:,block]), rms(pred_v[:,block])
        threshold = 1e-8
        if statistics and name in statistics["names"]:
            threshold = max(threshold, 0.05*statistics["tendency_scale"][statistics["names"].index(name)])
        valid = true_norm > threshold
        ratio = [float(p/t) if ok else None for p,t,ok in zip(pred_norm,true_norm,valid)]
        lag = _lag_diagnostic(pred_norm,true_norm)
        if lag.get("correlation") is not None and not np.isfinite(lag["correlation"]):
            lag = {"best_lag_steps":None,"correlation":None}
        lag["best_lag_hours"] = None if lag["best_lag_steps"] is None else int(lag["best_lag_steps"]*dt[0])
        unit = variable.get("attrs",{}).get("units", {"t2m":"K","msl":"Pa","u10":"m/s","v10":"m/s"}.get(name,"source unit"))
        by_variable[name] = {"unit":unit,"tendency_unit":f"({unit})/hour",
            "state_rmse":rms(prediction[:,block]-truth[:,block]).tolist(),
            "tendency_rmse":rms(pred_v[:,block]-true_v[:,block]).tolist(),
            "prediction_tendency_rms":pred_norm.tolist(),"truth_tendency_rms":true_norm.tolist(),
            "amplitude_ratio":ratio,"ratio_valid_count":int(valid.sum()),
            "ratio_threshold":threshold,"lag_diagnostic":lag}
    us,_ = _variable(aligned.schema,"u10"); vs,_ = _variable(aligned.schema,"v10")
    speed = np.hypot(prediction[:,us],prediction[:,vs])
    truth_speed = np.hypot(truth[:,us],truth[:,vs])
    calm = statistics["calm_threshold_mps"] if statistics else 1e-3
    direction_valid = truth_speed > calm
    cosine = ((prediction[:,us]*truth[:,us]+prediction[:,vs]*truth[:,vs]) /
              (np.maximum(speed,1e-6)*np.maximum(truth_speed,1e-6))).clip(-1,1)
    direction_error = [float(((1-c)*a*ok).sum()/(a*ok).sum()) if ok.any() else None
                       for c,ok,a in zip(cosine,direction_valid,np.broadcast_to(area,cosine.shape))]
    return {"format":"climate_diffusion.member_trajectory.v1","member_id":member,
            "origin_time":str(aligned.origin_time),"valid_times":[str(v) for v in aligned.valid_times],
            "lead_hours":aligned.lead_hours.tolist(),"dt_hours":dt.tolist(),"by_variable":by_variable,
            "uv_vector_rmse_mps":np.sqrt((((prediction[:,us]-truth[:,us])**2+
                 (prediction[:,vs]-truth[:,vs])**2)*area).sum(-1)).tolist(),
            "wind_speed_rmse_mps":np.sqrt(((speed-truth_speed)**2*area).sum(-1)).tolist(),
            "wind_direction_one_minus_cosine":direction_error,
            "wind_direction_valid_count":direction_valid.sum(-1).tolist(),
            "calm_threshold_mps":calm,
            "alignment":"exact UTC; no reference shift", "wind_convention":"toward: east u, north v",
            "ratio_threshold_source":"train tendency scale" if statistics else "numerical floor only; no train statistics supplied"}


def _coastlines(axis):
    data=json.loads(Path(__file__).with_name("assets").joinpath("ne_110m_coastline.geojson").read_text())
    for feature in data["features"]:
        geometry=feature["geometry"]
        lines=[geometry["coordinates"]] if geometry["type"]=="LineString" else geometry["coordinates"]
        for line in lines:
            xy=np.asarray(line)
            for segment in np.split(xy, np.flatnonzero(np.abs(np.diff(xy[:,0]))>180)+1):
                if len(segment)>1: axis.plot(segment[:,0],segment[:,1],color="black",lw=0.45)


def plot_member_series(aligned, output):
    """Area-RMS physical tendencies: do not mix Pa/K/m/s or hide members in a mean."""
    import matplotlib.pyplot as plt
    reports=[member_diagnostics(aligned,m) for m in range(len(aligned.members))]
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout="constrained")
    for ax,variable in zip(axes.ravel(),aligned.schema["variables"]):
        name=variable["name"]
        for m,report in enumerate(reports):
            row=report["by_variable"][name]
            ax.plot(aligned.lead_hours,row["prediction_tendency_rms"],label=f"member {m}",alpha=.75)
        ax.plot(aligned.lead_hours,row["truth_tendency_rms"],"k--",label="truth",lw=2)
        ax.set(title=name,xlabel="physical lead (hours)",ylabel=row["tendency_unit"])
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Same saved forecast: individual-member tendency magnitude (not ensemble mean)")
    fig.savefig(output,dpi=120); plt.close(fig)


def render_member(aligned, member, output, *, fps=2.5, reference_label="Actual ERA5",
                  color_limits=None, quiver_scale=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    if not 0 <= member < len(aligned.members) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid member/fps")
    block,shape=_variable(aligned.schema,"t2m")
    us,_=_variable(aligned.schema,"u10"); vs,_=_variable(aligned.schema,"v10")
    coords=aligned.schema["variables"][0]["coords"]
    lon=(np.asarray(coords["lon"])+180)%360-180
    order=np.argsort(lon); lon=lon[order]; lat=np.asarray(coords["lat"])
    xx,yy=np.meshgrid(lon,lat)
    def field(value,sl): return value[...,sl].reshape(*value.shape[:-1],*shape)[...,order]
    prediction=aligned.members[member]
    if color_limits is None:
        color_limits=(min(aligned.members[...,block].min(),aligned.truth[:,block].min()),
                      max(aligned.members[...,block].max(),aligned.truth[:,block].max()))
    if color_limits[0] >= color_limits[1]: color_limits=(color_limits[0]-0.5,color_limits[1]+0.5)
    if quiver_scale is None:
        speeds=np.concatenate((np.hypot(aligned.members[...,us],aligned.members[...,vs]).ravel(),
                               np.hypot(aligned.truth[:,us],aligned.truth[:,vs]).ravel()))
        quiver_scale=max(float(np.percentile(speeds,95)),1.) / 8.0
    if not np.isfinite(quiver_scale) or quiver_scale<=0:
        raise ValueError("quiver_scale must be positive (m/s per display-degree)")
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    images,arrows=[],[]
    stride=max(1,int(max(shape)/24))
    cyclic=len(lon)>=3 and np.allclose(np.diff(lon),np.diff(lon)[0]) and np.isclose(np.diff(lon)[0]*len(lon),360)
    def scalar(value):
        values=field(value,block)
        return np.concatenate((values,values[...,:1]),-1) if cyclic else values
    plot_lon=np.r_[lon,lon[0]+360] if cyclic else lon
    for ax,values,label in zip(axes,(prediction,aligned.truth),(f"Generated member {member}",reference_label)):
        im=ax.pcolormesh(plot_lon,lat,scalar(values[0]),shading="auto",cmap="RdYlBu_r",
                         vmin=color_limits[0],vmax=color_limits[1])
        q=ax.quiver(xx[::stride,::stride],yy[::stride,::stride],
                    field(values[0],us)[::stride,::stride],field(values[0],vs)[::stride,::stride],
                    color="black",angles="xy",scale_units="xy",scale=quiver_scale,width=0.002)
        ax.quiverkey(q,0.8,-0.08,10,"10 m/s",labelpos="E")
        _coastlines(ax)
        ax.set(xlim=(-180,180),ylim=(-90,90),title=label)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        images.append(im); arrows.append(q)
    fig.colorbar(images[0],ax=axes,shrink=0.8,label="t2m (K)")
    def update(k):
        for image,arrow,values in zip(images,arrows,(prediction,aligned.truth)):
            image.set_array(scalar(values[k]).ravel())
            arrow.set_UVC(field(values[k],us)[::stride,::stride],field(values[k],vs)[::stride,::stride])
        fig.suptitle(f"Member {member} vs {reference_label} | +{aligned.lead_hours[k]}h | origin {aligned.origin_time}\n"
                     f"valid {aligned.valid_times[k]} | fixed-grid Eulerian wind")
        return images+arrows
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    update(0)
    fig.savefig(output.with_suffix(".png"),dpi=110)
    animation=FuncAnimation(fig,update,frames=len(aligned.lead_hours),interval=1000/fps)
    writer=PillowWriter(fps=fps) if output.suffix==".gif" else FFMpegWriter(fps=fps,codec="libx264",
        extra_args=["-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-pix_fmt","yuv420p","-movflags","+faststart"])
    animation.save(output,writer=writer,dpi=100)
    plt.close(fig)
    return output


def export_trajectories(forecast, archive, output_dir, *, horizon_hours=120, interval_hours=6,
                        members=None, extension="mp4", fps=2.5, reference_label="Actual ERA5"):
    if extension not in {"mp4","gif"}: raise ValueError("extension must be mp4 or gif")
    if horizon_hours<=0 or interval_hours<=0 or horizon_hours%interval_hours:
        raise ValueError("Horizon must be a positive multiple of output interval")
    full=align_forecast(forecast,archive,selected_leads=np.arange(interval_hours,horizon_hours+1,interval_hours))
    aligned=select_interval(full,horizon_hours,interval_hours)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    with np.load(forecast,allow_pickle=False) as source:
        stats=json.loads(str(source["temporal_statistics_json"].item())) if "temporal_statistics_json" in source else None
        model_step=int(source["forecast_step_hours"]) if "forecast_step_hours" in source else int(full.schema["forecast_step_hours"])
    ids=list(range(len(aligned.members))) if members is None else list(members)
    if not ids or len(set(ids))!=len(ids) or any(i<0 or i>=len(aligned.members) for i in ids):
        raise ValueError("Invalid/duplicate member selection")
    np.savez_compressed(output/"trajectory.npz",predictions=aligned.members,
        origin_time=aligned.origin_time,origin_state=aligned.origin_state,member_ids=np.arange(len(aligned.members)),
        lead_hours=aligned.lead_hours,valid_times=aligned.valid_times,forecast_step_hours=model_step,
        source_forecast_step_hours=model_step,output_interval_hours=interval_hours,
        schema_json=json.dumps(aligned.schema),temporal_statistics_json=json.dumps(stats))
    report={"format":"climate_diffusion.trajectory_export.v1","source_forecast":str(forecast),
            "source_model_step_hours":model_step,"output_interval_hours":interval_hours,
            "horizon_hours":horizon_hours,"member_count":len(aligned.members),"rendered_member_ids":ids,
            "ensemble_mean_is_not_default_output":True,"reference_label":reference_label,
            "by_variable_ensemble":{}}
    area=area_weights(aligned.schema).ravel()
    dt=np.diff(np.r_[0,aligned.lead_hours])
    prev=np.concatenate((np.broadcast_to(aligned.origin_state,aligned.members[:,:1].shape),aligned.members[:,:-1]),1)
    tendency=(aligned.members-prev)/dt[None,:,None]
    for v in aligned.schema["variables"]:
        sl,_=_variable(aligned.schema,v["name"])
        report["by_variable_ensemble"][v["name"]]={
            "state_spread_rms":np.sqrt((aligned.members[...,sl].var(0)*area).sum(-1)).tolist(),
            "tendency_spread_rms":np.sqrt((tendency[...,sl].var(0)*area).sum(-1)).tolist(),
            "mean_tendency_rms":np.sqrt((tendency[...,sl].mean(0)**2*area).sum(-1)).tolist()}
    for member in ids:
        diagnostics=member_diagnostics(aligned,member,stats)
        (output/f"member-{member:03d}.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+"\n")
        render_member(aligned,member,output/f"member-{member:03d}.{extension}",fps=fps,reference_label=reference_label)
    (output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    plot_member_series(aligned,output/"member-tendencies.png")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast",required=True)
    parser.add_argument("--archive",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--horizon-hours",type=int,default=120)
    parser.add_argument("--interval-hours",type=int,default=6)
    parser.add_argument("--members",type=int,nargs="+",help="Omit to render ALL members")
    parser.add_argument("--extension",choices=("mp4","gif"),default="mp4")
    parser.add_argument("--fps",type=float,default=2.5)
    parser.add_argument("--reference-label",default="Actual ERA5")
    args=vars(parser.parse_args(argv))
    print(export_trajectories(**args))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

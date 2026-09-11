"""Exact-valid-time diagnostics and animation for saved climate ensembles.

The three time-related quantities used by this project are deliberately kept
separate: ``dq/dtau`` is a generative Flow-Matching velocity, finite differences
below are physical-time tendencies, and u10/v10 are wind components in m/s.
This module never shifts the reference to maximize a lag score.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_monthly_archive
from .inference import LatentFlowForecaster
from .moe_data import field_grid, load_moe_archive


@dataclass(frozen=True)
class AlignedForecast:
    members: np.ndarray  # [M, H, D]
    truth: np.ndarray  # [H, D]
    origin_time: np.datetime64
    valid_times: np.ndarray
    lead_hours: np.ndarray
    schema: dict
    origin_state: np.ndarray
    sampling_contract: str = "unspecified; inspect source checkpoint before interpreting temporal coupling"


def _as_utc_ns(value) -> np.datetime64:
    """Parse a timestamp as UTC and return timezone-free datetime64[ns]."""
    import pandas as pd

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.tz_localize(None).to_datetime64().astype("datetime64[ns]")


def _validate_leads(lead_hours: np.ndarray, horizon: int) -> np.ndarray:
    raw = np.asarray(lead_hours)
    if not np.isfinite(raw).all() or not np.all(raw == raw.astype(np.int64)):
        raise ValueError("lead_hours must contain finite integer hours; no truncation")
    leads = np.asarray(lead_hours, dtype=np.int64)
    if leads.shape != (horizon,) or np.any(leads <= 0):
        raise ValueError("lead_hours must contain one positive value per forecast step")
    if len(np.unique(leads)) != len(leads) or np.any(np.diff(leads) <= 0):
        raise ValueError("lead_hours must be strictly increasing and unique")
    return leads


def load_saved_forecast(path: str | Path) -> tuple[np.ndarray, np.datetime64, np.ndarray]:
    """Read the explicit temporal contract from a forecast NPZ."""
    with np.load(path, allow_pickle=False) as saved:
        members = np.asarray(saved["predictions"], dtype=np.float32)
        if members.ndim != 3:
            raise ValueError("predictions must be [member, lead, state]")
        key = "origin_time" if "origin_time" in saved else "last_history_time"
        if key not in saved or "lead_hours" not in saved:
            raise ValueError("forecast requires origin_time/last_history_time and lead_hours")
        origin = _as_utc_ns(saved[key].item())
        leads = _validate_leads(saved["lead_hours"], members.shape[1])
        if "valid_times" in saved:
            declared = np.asarray(saved["valid_times"]).astype("datetime64[ns]")
            expected = origin + leads.astype("timedelta64[h]")
            if not np.array_equal(declared, expected):
                raise ValueError("forecast valid_times disagree with origin_time + lead_hours")
    if not np.isfinite(members).all():
        raise ValueError("forecast contains non-finite values")
    return members, origin, leads


def align_forecast(forecast_path: str | Path, archive_path: str | Path, *, selected_leads=None) -> AlignedForecast:
    """Join forecast and truth by exact UTC valid time; no nearest/lag matching."""
    members, origin, leads = load_saved_forecast(forecast_path)
    if selected_leads is not None:
        wanted = _validate_leads(np.asarray(selected_leads), len(selected_leads))
        if not np.all(np.isin(wanted, leads)):
            raise ValueError("Forecast lacks exact requested physical leads")
        indices = np.searchsorted(leads, wanted)
        members, leads = members[:, indices], leads[indices]
    states, times, schema = load_moe_archive(archive_path)
    with np.load(forecast_path, allow_pickle=False) as source:
        sampling_contract = str(source['sampling_contract'].item()) if 'sampling_contract' in source else 'unspecified'
        if "schema_json" in source and json.loads(str(source["schema_json"].item())) != schema:
            raise ValueError("Forecast variable/grid schema differs from reference archive")
    valid = origin + leads.astype("timedelta64[h]")
    positions = {value: index for index, value in enumerate(times.astype("datetime64[ns]"))}
    if len(positions) != len(times):
        raise ValueError("reference archive contains duplicate UTC timestamps")
    missing = [str(value) for value in valid if value not in positions]
    if missing:
        raise ValueError(f"reference archive lacks exact forecast valid times: {missing[:3]}")
    if origin not in positions:
        raise ValueError("reference archive lacks the exact forecast origin time")
    truth = states[[positions[value] for value in valid]]
    if members.shape[-1] != schema["state_dim"]:
        raise ValueError("forecast state dimension differs from reference schema")
    return AlignedForecast(members, truth, origin, valid, leads, schema, states[positions[origin]],sampling_contract)


def forecast_from_checkpoint(checkpoint: str | Path, archive_path: str | Path,
                             origin_time, output_path: str | Path, *, ensemble_size=4,
                             integration_steps=8, seed=0, device=None, moe_mode=None,
                             forecast_steps=None) -> Path:
    """Forecast at a requested archive origin while retaining future truth for audit."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Choose a new forecast path; preserve the saved member identities")
    forecaster = LatentFlowForecaster(checkpoint, device=device)
    loader = load_moe_archive if forecaster.is_moe else load_monthly_archive
    states, times, schema = loader(archive_path)
    forecaster.validate_archive(schema, times)
    origin = _as_utc_ns(origin_time)
    matches = np.flatnonzero(times.astype("datetime64[ns]") == origin)
    if len(matches) != 1:
        raise ValueError("origin_time must occur exactly once in the archive")
    position = int(matches[0])
    prefix = states[: position + 1]
    history = forecaster.select_history(prefix)
    steps = int(forecast_steps or (forecaster.config.horizon_steps if forecaster.is_dynamics else 1))
    leads = np.arange(1, steps + 1, dtype=np.int64) * forecaster.forecast_step_hours
    if position + steps >= len(states):
        raise ValueError("archive has insufficient post-origin truth for requested forecast")
    prediction = forecaster.forecast(history, months=steps, ensemble_size=ensemble_size,
                                     integration_steps=integration_steps, seed=seed,
                                     moe_mode=moe_mode)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predictions=prediction, origin_time=origin,
                        last_history_time=origin, lead_hours=leads,
                        valid_times=origin + leads.astype("timedelta64[h]"),
                        forecast_step_hours=np.asarray(forecaster.forecast_step_hours),
                        schema_json=json.dumps(schema),
                        temporal_statistics_json=json.dumps(forecaster.training_metadata.get("temporal_statistics")),
                        sampling_contract=np.asarray(forecaster.training_metadata.get(
                            "sampling_contract", "unspecified")))
    return output


def _variable(schema: dict, name: str) -> tuple[slice, tuple[int, ...]]:
    for variable in schema["variables"]:
        if variable["name"] == name:
            return slice(*variable["slice"]), tuple(variable["shape"])
    raise ValueError(f"variable {name!r} is absent from archive schema")


def _crps(samples: np.ndarray, target: np.ndarray) -> float:
    accuracy = np.abs(samples - target[None]).mean()
    pairwise = np.abs(samples[:, None] - samples[None, :]).mean()
    return float(accuracy - 0.5 * pairwise)


def _lag_diagnostic(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Report a diagnostic scalar lag without modifying either series."""
    if len(predicted) < 2 or np.std(predicted) == 0 or np.std(actual) == 0:
        return {"best_lag_steps": None, "correlation": None}
    best = (-np.inf, 0)
    limit = min(5, len(predicted) - 1)
    for lag in range(-limit, limit + 1):
        left = predicted[max(0, lag):len(predicted) + min(0, lag)]
        right = actual[max(0, -lag):len(actual) - max(0, lag)]
        if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0:
            corr = float(np.corrcoef(left, right)[0, 1])
            if corr > best[0]:
                best = (corr, lag)
    return {"best_lag_steps": int(best[1]), "correlation": float(best[0]),
            "contract": "diagnostic_only_no_reference_shift"}


def temporal_diagnostics(aligned: AlignedForecast) -> dict:
    members, truth, leads = aligned.members, aligned.truth, aligned.lead_hours
    mean = members.mean(0)
    spread = members.std(0)
    dt = np.diff(np.r_[0, leads]).astype(np.float64)
    previous_prediction = np.concatenate((aligned.origin_state[None], mean[:-1]), axis=0)
    previous_truth = np.concatenate((aligned.origin_state[None], truth[:-1]), axis=0)
    pred_tendency = (mean - previous_prediction) / dt[:, None]
    truth_tendency = (truth - previous_truth) / dt[:, None]
    pred_change = np.sqrt(np.nanmean(pred_tendency ** 2, axis=1))
    truth_change = np.sqrt(np.nanmean(truth_tendency ** 2, axis=1))
    rows = []
    wind = None
    names = {v["name"] for v in aligned.schema["variables"]}
    if {"u10", "v10"} <= names:
        us, _ = _variable(aligned.schema, "u10")
        vs, _ = _variable(aligned.schema, "v10")
        wind = (us, vs)
    for index, lead in enumerate(leads):
        error = mean[index] - truth[index]
        row = {"lead_hours": int(lead), "valid_time": str(aligned.valid_times[index]),
               "rmse": float(np.sqrt(np.mean(error ** 2))),
               "crps": _crps(members[:, index], truth[index]),
               "mean_spread": float(spread[index].mean()),
               "prediction_anomaly_rms_from_origin": float(np.sqrt(np.mean(
                   (mean[index] - aligned.origin_state) ** 2))),
               "truth_anomaly_rms_from_origin": float(np.sqrt(np.mean(
                   (truth[index] - aligned.origin_state) ** 2))),
               "state_tendency_rms_per_hour": float(pred_change[index]),
               "truth_tendency_rms_per_hour": float(truth_change[index])}
        if wind:
            us, vs = wind
            row["uv_vector_rmse_mps"] = float(np.sqrt(np.mean(
                (mean[index, us] - truth[index, us]) ** 2
                + (mean[index, vs] - truth[index, vs]) ** 2)))
        rows.append(row)
    ratio = pred_change / np.maximum(truth_change, 1e-12)
    member_previous = np.concatenate((np.broadcast_to(aligned.origin_state, members[:, :1].shape),
                                      members[:, :-1]), axis=1)
    member_tendency = (members - member_previous) / dt[None, :, None]
    return {"format": "climate_diffusion.time_alignment.v1",
            "origin_time": str(aligned.origin_time),
            "alignment": "exact_utc_valid_time_no_shift",
            "time_quantities": {"flow_velocity": "dq/dtau (not measured here)",
                                "state_tendency": "delta state / physical hour",
                                "wind": "u10/v10 components in m/s when source units are m/s"},
            "sampling_contract": aligned.sampling_contract,
            "mixed_unit_warning": "Legacy pooled raw-unit RMS mixes Pa/K/m/s; use per-variable member JSON for scientific interpretation",
            "median_tendency_amplitude_ratio_prediction_over_truth": float(np.median(ratio)),
            "member_tendency_rms_per_hour": np.sqrt(np.mean(member_tendency ** 2, axis=2)).tolist(),
            "lag_diagnostic": _lag_diagnostic(pred_change, truth_change),
            "by_lead": rows}


def render_animation(aligned: AlignedForecast, output_path: str | Path, *, variable="t2m",
                     u_name="u10", v_name="v10", fps=2.5, member=None) -> Path:
    """Render exact-time mean/member vs truth with fixed scales and tendencies."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    block, shape = _variable(aligned.schema, variable)
    if len(shape) != 2:
        raise ValueError("animation variable must be a 2-D lat/lon field")
    display = aligned.members.mean(0) if member is None else aligned.members[int(member)]
    if member is not None and not 0 <= int(member) < aligned.members.shape[0]:
        raise ValueError("member index is outside forecast ensemble")
    predicted, actual = display[:, block].reshape((-1, *shape)), aligned.truth[:, block].reshape((-1, *shape))
    spread = aligned.members[:, :, block].std(0).reshape((-1, *shape))
    common_min = float(min(predicted.min(), actual.min()))
    common_max = float(max(predicted.max(), actual.max()))
    error_max = float(np.max(np.abs(predicted - actual))) or 1.0
    spread_max = float(spread.max()) or 1.0
    dt = np.diff(np.r_[0, aligned.lead_hours])
    origin_field = aligned.origin_state[block].reshape(shape)
    pred_delta = np.diff(np.concatenate((origin_field[None], predicted)), axis=0) / dt[:, None, None]
    truth_delta = np.diff(np.concatenate((origin_field[None], actual)), axis=0) / dt[:, None, None]
    tendency_max = float(max(np.abs(pred_delta).max(), np.abs(truth_delta).max())) or 1.0

    coords = aligned.schema["variables"][0]["coords"]
    lon, lat = np.asarray(coords["lon"]), np.asarray(coords["lat"])
    xx, yy = np.meshgrid(lon, lat)
    wind = {v["name"] for v in aligned.schema["variables"]}
    have_wind = {u_name, v_name} <= wind
    if have_wind:
        us, _ = _variable(aligned.schema, u_name)
        vs, _ = _variable(aligned.schema, v_name)
        all_uv = np.concatenate((aligned.members[..., us].ravel(), aligned.members[..., vs].ravel(),
                                 aligned.truth[..., us].ravel(), aligned.truth[..., vs].ravel()))
        quiver_scale = float(np.nanpercentile(np.abs(all_uv), 95)) or 1.0
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    title_kind = "ensemble mean" if member is None else f"member {int(member)}"
    panels = ((predicted, f"Generated {title_kind}", common_min, common_max, "viridis"),
              (actual, "Actual ERA5", common_min, common_max, "viridis"),
              (predicted - actual, "Error", -error_max, error_max, "coolwarm"),
              (spread, "Ensemble spread", 0, spread_max, "magma"),
              (pred_delta, f"Generated delta/hour ({variable})", -tendency_max, tendency_max, "coolwarm"),
              (truth_delta, f"ERA5 delta/hour ({variable})", -tendency_max, tendency_max, "coolwarm"))
    images = []
    for axis, (fields, title, low, high, cmap) in zip(axes.ravel(), panels):
        image = axis.pcolormesh(lon, lat, fields[0], shading="auto", cmap=cmap, vmin=low, vmax=high)
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        fig.colorbar(image, ax=axis, shrink=0.72)
    quivers = []
    if have_wind:
        stride = max(1, int(max(shape) / 16))
        for axis, vector in ((axes[0, 0], display[0]), (axes[0, 1], aligned.truth[0])):
            u = vector[us].reshape(shape); v = vector[vs].reshape(shape)
            arrows = axis.quiver(xx[::stride, ::stride], yy[::stride, ::stride],
                                 u[::stride, ::stride], v[::stride, ::stride],
                                 angles="xy", scale_units="xy", scale=quiver_scale,
                                 width=0.003, color="white")
            axis.quiverkey(arrows, 0.88, -0.12, quiver_scale,
                           f"{quiver_scale:.1f} m/s", labelpos="E")
            quivers.append(arrows)

    def update(frame):
        frame_fields = (predicted[frame], actual[frame], predicted[frame] - actual[frame],
                        spread[frame], pred_delta[frame], truth_delta[frame])
        for artist, field in zip(images, frame_fields):
            artist.set_array(field.ravel())
        if have_wind:
            for arrows, vector in zip(quivers, (display[frame], aligned.truth[frame])):
                arrows.set_UVC(vector[us].reshape(shape)[::stride, ::stride],
                               vector[vs].reshape(shape)[::stride, ::stride])
        lead = int(aligned.lead_hours[frame])
        fig.suptitle(f"{variable} | origin {aligned.origin_time} | +{lead}h ({lead/24:g}d) | valid {aligned.valid_times[frame]}")
        return [*images, *quivers]

    animation = FuncAnimation(fig, update, frames=len(aligned.lead_hours), interval=1000 / fps)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps) if output.suffix.lower() == ".gif" else FFMpegWriter(fps=fps, codec="libx264")
    animation.save(output, writer=writer, dpi=110)
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--forecast", help="Saved [member,lead,state] NPZ")
    source.add_argument("--checkpoint", help="Generate a forecast before diagnosis")
    parser.add_argument("--archive", required=True, help="Fixed-step archive containing exact future truth")
    parser.add_argument("--origin-time", help="Required with --checkpoint")
    parser.add_argument("--forecast-output", default="outputs/time-alignment/forecast.npz")
    parser.add_argument("--forecast-only", action="store_true", help="Save native ensemble once, skip legacy mean renderer")
    parser.add_argument("--output", default="outputs/time-alignment/comparison.mp4")
    parser.add_argument("--report", default="outputs/time-alignment/diagnostics.json")
    parser.add_argument("--variable", default="t2m")
    parser.add_argument("--u-name", default="u10")
    parser.add_argument("--v-name", default="v10")
    parser.add_argument("--fps", type=float, default=2.5)
    parser.add_argument("--member", type=int)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--forecast-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--moe-mode")
    args = parser.parse_args(argv)
    if args.checkpoint:
        if not args.origin_time:
            parser.error("--origin-time is required with --checkpoint")
        forecast = forecast_from_checkpoint(args.checkpoint, args.archive, args.origin_time,
                                            args.forecast_output, ensemble_size=args.ensemble_size,
                                            integration_steps=args.integration_steps, seed=args.seed,
                                            device=args.device, moe_mode=args.moe_mode,
                                            forecast_steps=args.forecast_steps)
    else:
        forecast = Path(args.forecast)
    if args.forecast_only:
        print(f"forecast={forecast}")
        return 0
    aligned = align_forecast(forecast, args.archive)
    report = temporal_diagnostics(aligned)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_animation(aligned, args.output, variable=args.variable, u_name=args.u_name,
                     v_name=args.v_name, fps=args.fps, member=args.member)
    print(f"animation={args.output}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

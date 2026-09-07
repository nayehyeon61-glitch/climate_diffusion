"""Reproducible CPU two-stage smoke training, NOT an ERA5 skill experiment.

Run from repository root: PYTHONPATH=src python scripts/smoke_moe.py
Generated checkpoints/archive stay in outputs/; only JSON/SVG reports are tracked.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from climate_diffusion.fixed_step_data import prepare_fixed_step_archive
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.train_moe import train_moe
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.evaluation import evaluate_flow_checkpoint


def synthetic_archive(directory, seed=19, count=480):
    """Two toy propagation modes, shared across all four fields; labels for audit only."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-67.5, 67.5, 4)
    lon = np.arange(8) * 45.0
    yy, xx = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    # Alternating directions affect the FULL state; labels are never training input.
    regime = ((np.arange(count) // 40) % 2).astype(int)
    increments = np.where(regime == 0, 0.13, -0.18)
    phase = np.cumsum(increments)
    latent = []
    for i in range(count):
        wave = np.cos(yy) * np.sin(xx - phase[i])
        secondary = np.sin(2 * xx + 0.7 * phase[i]) * np.cos(2 * yy)
        north = np.sin(yy) * np.cos(xx - phase[i])
        latent.append(np.stack((wave + 0.25 * secondary, -0.6 * wave + 0.5 * north,
                                np.cos(yy) * np.cos(xx - phase[i]), north + 0.3 * secondary)))
    fields = np.asarray(latent) + rng.normal(0, 0.06, (count, 4, 4, 8))
    # Plausible-looking units are cosmetic; these are NOT simulated physical weather.
    fields = fields * np.array([900, 6, 8, 8])[None, :, None, None]
    fields += np.array([101000, 285, 0, 0])[None, :, None, None]
    dataset = xr.Dataset({name: (("time", "lat", "lon"), fields[:, j].astype(np.float32))
                          for j, name in enumerate(("msl", "t2m", "u10", "v10"))},
                         coords={"time": pd.date_range("2001-01-01", periods=count, freq="6h"),
                                 "lat": lat, "lon": lon})
    raw = directory / "synthetic-fields.nc"
    dataset.to_netcdf(raw, engine="scipy")
    archive, _ = prepare_fixed_step_archive(raw, directory / "synthetic-states.npz", step_hours=6,
                                           target_lat_points=4, target_lon_points=8)
    return archive, regime


@torch.no_grad()
def routing_diagnostics(checkpoint, archive, regimes, test_windows, seed=83, members=8, steps=8):
    """Record routing on final-lead GENERATED ODE paths, not teacher-forced endpoints."""
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    model = forecaster.model
    states, _, _ = load_moe_archive(archive)
    router, alpha, labels, similarities = [], [], [], []
    generator = torch.Generator().manual_seed(seed)
    for index in test_windows:
        end = index + forecaster.history_span_steps
        history = forecaster.select_history(states[index:end])
        context = model.encode_history(forecaster._normalise(history)[None]).repeat_interleave(members, 0)
        state = torch.randn(members, model.config.state_dim, generator=generator)
        lead = torch.ones(members)
        for step in range(steps):
            tau = torch.full((members,), step / steps)
            first = model.field(state, tau, context, lead)
            mid = state + 0.5 / steps * first["velocity"]
            second = model.field(mid, tau + 0.5 / steps, context, lead)
            state = state + second["velocity"] / steps
            router.append(second["router"].numpy())
            alpha.append(second["alpha"].numpy())
            labels.extend([int(regimes[end - 1])] * members)
            candidate = torch.nn.functional.normalize(second["candidates"], dim=-1)
            cosine = candidate @ candidate.transpose(1, 2)
            mask = ~torch.eye(model.config.num_experts, dtype=torch.bool)
            similarities.append(float(cosine[:, mask].mean()))
    router, alpha, labels = np.concatenate(router), np.concatenate(alpha), np.asarray(labels)
    counts = np.zeros((2, model.config.num_experts), dtype=int)
    np.add.at(counts, (labels, router.argmax(-1)), 1)
    joint = counts / counts.sum()
    independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    present = joint > 0
    mutual_information = (joint[present] * np.log(joint[present] / independent[present])).sum()
    return {"population": "final lead; midpoint evaluations of generated test-member ODE paths",
            "count": len(router), "router_mean": router.mean(0).tolist(),
            "router_std": router.std(0).tolist(), "alpha_mean": alpha.mean(0).tolist(),
            "alpha_std": alpha.std(0).tolist(),
            "router_entropy_nats": float(-(router * np.log(np.maximum(router, 1e-12))).sum(1).mean()),
            "maximum_entropy_nats": float(np.log(model.config.num_experts)),
            "candidate_cosine_mean": float(np.mean(similarities)),
            "toy_regime_by_hard_router_counts": counts.tolist(),
            "toy_regime_hard_router_mutual_information_nats": float(mutual_information),
            "caution": "Correlated ODE samples; two hand-made mode labels, not evidence of meteorological regimes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="outputs/moe-smoke")
    parser.add_argument("--report-dir", default="docs/results/moe-smoke")
    parser.add_argument("--expert-epochs", type=int, default=40)
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    work, report = Path(args.work_dir), Path(args.report_dir)
    work.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    archive, regimes = synthetic_archive(work)
    checkpoint = train_moe(archive, work / "moe.pt", expert_epochs=args.expert_epochs,
                            meta_epochs=args.meta_epochs,
                            model_options={"history_steps": 6, "history_stride": 1, "horizon_steps": 8,
                                           "num_experts": 3, "hidden_dim": 64, "context_dim": 32},
                            batch_size=16, learning_rate=1e-3, window_stride=2, ensemble_size=4,
                            integration_steps=4, expert_leads=4, meta_leads=2, seed=7, device="cpu")
    warm_path = checkpoint.with_name("moe.experts.pt")
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    final = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frozen_equal = all(torch.equal(value, final["model"][key]) for key, value in warm["model"].items()
                       if not key.startswith("meta."))
    if not frozen_equal:
        raise AssertionError("Stage 2 changed frozen expert/router/history parameters")
    comparisons = {}
    for mode in ("experts", "uniform", "meta"):
        path = evaluate_flow_checkpoint(checkpoint, archive, report / f"evaluation-{mode}.json",
                                         ensemble_size=8, integration_steps=8, seed=83,
                                         device="cpu", max_cases=8, moe_mode=mode)
        comparisons[mode] = json.loads(path.read_text())
    metrics = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
    (report / "training-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    diagnostics = routing_diagnostics(checkpoint, archive, regimes, comparisons["meta"]["test_windows"])
    summary = {"experiment": "synthetic smoke only; no ERA5 or RunPod run",
               "seed_data": 19, "seed_training": 7, "seed_evaluation": 83,
               "state_shape": [480, 4, 4, 8], "history_span_steps": 6, "horizon_steps": 8,
               "horizon_hours": 48, "model_config": final["model_config"],
               "parameter_count": final["training"]["parameter_count"],
               "experts_best_epoch": warm["training"]["best_epoch"],
               "meta_best_epoch": final["training"]["best_epoch"],
               "frozen_parameters_exactly_equal": frozen_equal,
               "split_window_counts": {k: len(v) for k, v in final["training"]["split"].items()
                                       if isinstance(v, list)},
               "normalization_span": final["training"]["normalization_span"],
               "checkpoint_sha256": comparisons["meta"]["checkpoint_sha256"],
               "warmup_checkpoint_sha256": final["training"]["warmup_checkpoint_sha256"],
               "archive_sha256": final["training"]["archive_sha256"],
               "test": {k: v["normalized_overall"] for k, v in comparisons.items()},
               "routing": diagnostics, "elapsed_seconds": time.perf_counter() - started,
               "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__, "device": "cpu", "threads": args.threads},
               "limits": ["48-hour toy horizon, NOT 15-30 day climate skill",
                          "One seed and eight held-out windows; no significance claim",
                          "No joint-time trajectory objective or physical conservation constraint",
                          "No meteorological regime-specialization claim", "No production GPU memory benchmark"]}
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

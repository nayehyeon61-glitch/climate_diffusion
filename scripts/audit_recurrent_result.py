#!/usr/bin/env python3
"""Audit a saved recurrent ERA5 result without pretending the raw truth archive exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    validation = json.loads((args.result/"validation.json").read_text())
    routing = json.loads((args.result/"validation-routing.json").read_text())
    with np.load(args.result/"forecast-6h.npz", allow_pickle=False) as stored:
        members = stored["predictions"].astype(np.float64)
        leads = stored["lead_hours"].tolist()
    pairwise = []
    for i in range(len(members)):
        for j in range(i):
            pairwise.append(np.sqrt(np.mean((members[i]-members[j])**2, axis=-1)))
    pairwise = np.stack(pairwise)
    increments = np.diff(members, axis=1)
    trace = routing["physical_trace"]
    projection = np.asarray([r["transport_projection_ratio"] for r in trace], float)
    residual_var = np.asarray([r["residual_q_variance_per_hour2"] for r in trace], float)
    drift = np.asarray([r["drift_q_rms_per_hour"] for r in trace], float)
    residual = np.asarray([r["residual_q_rms_per_hour"] for r in trace], float)
    final = np.asarray([r["final_q_rms_per_hour"] for r in trace], float)
    ranks = np.asarray(validation["rank_histogram_counts"], float)
    expected = ranks.sum()/len(ranks)
    out = {
        "format": "climate_diffusion.saved_result_audit.v1",
        "source_scope": "saved forecast + aggregate validation + four-origin routing trace; raw ERA5 unavailable",
        "lead_hours": leads,
        "member_pair_rmse_raw_state": pairwise.mean(0).tolist(),
        "member_increment_spread_raw_state": increments.std(0).mean(-1).tolist(),
        "rank_histogram": {
            "counts": ranks.astype(int).tolist(), "uniform_expected_per_bin": expected,
            "edge_to_center_ratio": float((ranks[0]+ranks[-1])/(2*ranks[len(ranks)//2])),
            "interpretation": "pooled correlated scalar ranks; U-shape is under-dispersion evidence, not full-support proof"},
        "aggregate_validation": validation["normalized_overall"],
        "noise_path": {
            "projection_ratio_mean": float(np.nanmean(projection)),
            "projection_ratio_first_last": [float(np.nanmean(projection[0])),float(np.nanmean(projection[-1]))],
            "residual_member_variance_q_per_hour2_first_last": [float(residual_var[0].mean()),float(residual_var[-1].mean())],
            "drift_q_rms_per_hour_first_last": [float(drift[0].mean()),float(drift[-1].mean())],
            "residual_q_rms_per_hour_first_last": [float(residual[0].mean()),float(residual[-1].mean())],
            "combined_q_rms_per_hour_first_last": [float(final[0].mean()),float(final[-1].mean())],
            "diagnosis": "late drift/residual cancellation is visible; causal attribution requires new ablations"},
        "unavailable_without_raw_archive": [
            "lead/variable/region/origin ensemble-mean error", "nominal coverage curve",
            "bias-corrected calibration", "drift-only/residual-only/full same-cohort scores",
            "A AE-oracle/drift/tangent capture on actual ERA5 pairs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+"\n")
    if args.figure:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        axes[0].bar(range(len(ranks)),ranks/expected,color="#4c78a8")
        axes[0].axhline(1,color="black",lw=1); axes[0].set_title("Pooled rank / uniform")
        axes[0].set_xlabel("rank bin"); axes[0].set_ylabel("ratio")
        axes[1].plot(leads,pairwise.mean(0),label="member pair RMSE")
        axes[1].set_title("Stored member separation"); axes[1].set_xlabel("lead (h)")
        axes[1].legend(frameon=False)
        axes[2].plot(leads,drift.mean((1,2)),label="drift")
        axes[2].plot(leads,residual.mean((1,2)),label="residual")
        axes[2].plot(leads,final.mean((1,2)),label="combined")
        axes[2].set_title("q tendency decomposition"); axes[2].set_xlabel("lead (h)")
        axes[2].legend(frameon=False)
        fig.tight_layout(); args.figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.figure,dpi=180); plt.close(fig)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

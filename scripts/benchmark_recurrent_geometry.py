#!/usr/bin/env python3
"""Benchmark differentiable one-step chart-geometry reuse without training."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from climate_diffusion.manifold_moe import ManifoldMoE, ManifoldMoEConfig
from climate_diffusion.recurrent_flow import integrate_flow_tau


def load_model(path: Path, device: torch.device) -> ManifoldMoE:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ManifoldMoE(ManifoldMoEConfig(**payload["model_config"]), payload["schema"],
                        payload["state_mean"], payload["state_scale"]).to(device)
    model.load_state_dict(payload["model"])
    model.set_stage("joint")
    return model


def run(model, tensors, steps, reuse):
    source, physical, context, hours = tensors
    model.zero_grad(set_to_none=True)
    physical = physical.detach().clone().requires_grad_()
    start = time.perf_counter()
    output, _ = integrate_flow_tau(model, source, physical, context, hours,
                                   integration_steps=steps, reuse_geometry=reuse)
    output.square().mean().backward()
    elapsed = time.perf_counter() - start
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return output.detach(), physical.grad.detach(), grads, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--integration-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1701)
    model = load_model(args.checkpoint, device)
    b, r = args.batch_size, model.config.manifold_dim
    tensors = (torch.randn(b, r, device=device),
               model.gate.centers[0].detach().clone().expand(b, -1).to(device),
               torch.randn(b, model.config.context_dim, device=device),
               torch.arange(b, device=device, dtype=torch.float32)*model.config.step_hours)
    # Warm both code paths. No optimizer/update is used.
    run(model, tensors, 1, False)
    run(model, tensors, 1, True)
    baseline, optimized = [], []
    reference = candidate = None
    for _ in range(args.repeats):
        reference = run(model, tensors, args.integration_steps, False)
        candidate = run(model, tensors, args.integration_steps, True)
        baseline.append(reference[3]); optimized.append(candidate[3])
    assert reference is not None and candidate is not None
    grad_l2 = sum((reference[2][k]-candidate[2][k]).square().sum() for k in reference[2]).sqrt()
    ref_l2 = sum(reference[2][k].square().sum() for k in reference[2]).sqrt().clamp_min(1e-12)
    result = {
        "format": "climate_diffusion.recurrent_geometry_benchmark.v1",
        "checkpoint": str(args.checkpoint), "device": str(device),
        "batch_size": b, "integration_steps": args.integration_steps, "repeats": args.repeats,
        "baseline_seconds_median": statistics.median(baseline),
        "optimized_seconds_median": statistics.median(optimized),
        "speedup": statistics.median(baseline)/statistics.median(optimized),
        "expected_jacobian_calls_baseline": 2*args.integration_steps,
        "expected_jacobian_calls_optimized": 1,
        "forward_max_abs_difference": float((reference[0]-candidate[0]).abs().max()),
        "physical_q_gradient_max_abs_difference": float((reference[1]-candidate[1]).abs().max()),
        "parameter_gradient_relative_l2_difference": float(grad_l2/ref_l2),
        "claim_scope": "measured CPU forward+backward; not a GPU speed or memory claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

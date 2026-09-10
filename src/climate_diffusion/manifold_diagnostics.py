"""Audit manifold locality separately from predictive skill; never fit on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inference import LatentFlowForecaster
from .moe_data import load_moe_archive
from .train import _sha256
from .model import sinusoidal_time_embedding


@torch.no_grad()
def diagnose_manifold(checkpoint_path, archive_path, output_path, *, max_cases=32, members=4,
                       integration_steps=4, seed=83, device="cpu", split_name="test"):
    if split_name not in {"test", "validation", "expert_validation"}:
        raise ValueError("Diagnostics must name a held-out evaluation split")
    if min(max_cases, members, integration_steps) < 1:
        raise ValueError("Diagnostic counts must be positive")
    f = LatentFlowForecaster(checkpoint_path, device=device)
    if not f.is_manifold:
        raise ValueError("Manifold diagnostics require a manifold_moe checkpoint")
    states, times, schema = load_moe_archive(archive_path)
    f.validate_archive(schema, times)
    if _sha256(Path(archive_path)) != f.training_metadata["archive_sha256"]:
        raise ValueError("Diagnostics require the identical training archive")
    m = f.model
    split = f.training_metadata["split"]
    config = m.config
    def choose(indices, limit):
        return [indices[k] for k in np.linspace(0,len(indices)-1,min(limit,len(indices)),dtype=int)]
    train_indices = choose(split["train"], 256)
    test_indices = choose(split[split_name], max_cases)
    def inputs(indices):
        history = np.stack([f.select_history(states[i:i+config.history_span_steps]) for i in indices])
        history = f._normalise(history)
        return history, history[:, -1], m.encode_history(history)
    train_history, train_state, train_context = inputs(train_indices)
    history, origin, context = inputs(test_indices)
    train_q, q = m.encode(train_state), m.encode(origin)
    center = train_q.mean(0).cpu().numpy()
    _, singular, vt = np.linalg.svd(train_q.cpu().numpy()-center, full_matrices=False)
    if len(vt) < 2:
        raise ValueError("At least two latent dimensions/train origins are needed for PCA")
    components = vt[:2]
    def pca(values):
        return ((values.detach().cpu().numpy()-center) @ components.T).tolist()
    width = config.time_embedding_dim
    def gate_at_observed(code, h):
        time = torch.ones(len(code),device=code.device,dtype=code.dtype)
        condition = torch.cat((h,sinusoidal_time_embedding(time,width),sinusoidal_time_embedding(time,width)),-1)
        gate,_ = m.gate(code,condition)
        return gate.exp()
    pi_train = gate_at_observed(train_q,train_context)
    pi = gate_at_observed(q,context)
    raw_features, invariants = m.physics.raw_features(origin)
    physical_signals = torch.cat((invariants,raw_features[:,1].square().mean((-2,-1)).sqrt()[:,None]),1)
    result = {"format":"climate_diffusion.manifold_diagnostics.v1", "stage":m.stage,
              "checkpoint_sha256":f.checkpoint_sha256,"test_windows":test_indices if split_name=="test" else [],
              "evaluation_split":split_name,"evaluation_windows":test_indices,
              "pca_fit":"expert train origins only; gate queries at tau=1 and final physical lead",
              "pca_explained_variance_ratio":(singular[:2]**2/(singular**2).sum()).tolist(),
              "train_pca":pca(train_q),"test_pca":pca(q),"centers_pca":pca(m.gate.centers),
              "train_gate":pi_train.cpu().tolist(),"test_gate":pi.cpu().tolist(),
              "physical_signal_names":[*m.physics.invariant_names,"vorticity_rms_s-1"],
              "test_physical_signals":physical_signals.cpu().tolist(),
              "anchor_rmse_test":float((q-m.reference_encode(origin)).square().mean().sqrt()),
              "manifold_reconstruction_rmse_test":float((m.decode(q)-origin).square().mean().sqrt()),
              "note":"Partitioning alone is not expert skill. PCA is only a 2D view, not proof of a manifold."}
    if m.stage != "manifold":
        generator = torch.Generator(device=f.device).manual_seed(seed)
        targets = f._normalise(np.stack([states[i+config.history_span_steps+config.horizon_steps-1]
                                         for i in test_indices]))
        target_q = m.encode(targets)
        source = torch.randn(target_q.shape,device=f.device,generator=generator)
        tau = source.new_full((len(source),),0.5)
        pair_q = (source+target_q)/2
        field = m.field(pair_q,tau,context,torch.ones_like(tau))
        error = (field["intrinsic_candidates"]-(target_q-source)[:,None]).square().mean(-1)
        # Row = geometry-defined region, not the identity of the best-error expert.
        regions = field["local_log_prior"].argmax(-1)
        regional_error, counts = [], []
        for k in range(config.num_experts):
            mask = regions == k
            counts.append(int(mask.sum()))
            regional_error.append(error[mask].mean(0).cpu().tolist() if bool(mask.any()) else [None]*config.num_experts)
        result["teacher_forced_audit"] = {
            "population":f"one independently noised latent FM pair per temporally correlated {split_name} origin; tau=0.5; final lead",
            "region_counts":counts,"expert_fm_by_region":regional_error,
            "local_expert_best_fraction":float((regions==error.argmin(-1)).float().mean()),
            "gate_best_expert_fraction":float((field["router"].argmax(-1)==error.argmin(-1)).float().mean())}
        # Audit actual generated ODE paths separately from teacher-forced pairs.
        n = min(8,len(context))
        h = context[:n].repeat_interleave(members,0)
        initial = torch.randn(len(h),config.manifold_dim,device=f.device,generator=generator)
        current = initial
        lead = torch.ones(len(h),device=f.device)
        gates,cosines,pairwise,normal_fraction,conditions,distances = [],[],[],[],[],[]
        path_pca = [pca(current)]
        offdiag = ~torch.eye(config.num_experts,dtype=torch.bool,device=f.device)
        for step in range(integration_steps):
            tau = current.new_full((len(current),),step/integration_steps)
            a = m.field(current,tau,h,lead)
            mid = current + a["velocity"]/(2*integration_steps)
            b = m.field(mid,tau+0.5/integration_steps,h,lead)
            current = current+b["velocity"]/integration_steps
            path_pca.append(pca(current))
            gates.append(b["router"])
            direction = torch.nn.functional.normalize(b["candidates"],dim=-1)
            cosines.append((direction@direction.transpose(1,2))[:,offdiag].mean())
            pairwise.append((b["candidates"][:,:,None]-b["candidates"][:,None,:]).square().mean(-1)[:,offdiag].mean())
            normal_fraction.append((b["raw_candidates"]-b["candidates"]).square().mean()
                                   /b["raw_candidates"].square().mean().clamp_min(1e-12))
            eigen = torch.linalg.eigvalsh(b["metric"])
            damping = config.projection_ridge*eigen.mean(-1).clamp_min(1e-8)
            conditions.append(((eigen[:,-1]+damping)/(eigen[:,0]+damping)).mean())
            distances.append(((mid[:,None]-m.gate.centers[None]).square().mean(-1).min(-1).values
                              /m.gate.radius_squared).sqrt().mean())
        gates = torch.cat(gates)
        result["generated_audit"] = {
            "population":"midpoint evaluations of actual generated final-lead ODE paths",
            "sample_count":len(gates),"gate_mean":gates.mean(0).cpu().tolist(),
            "gate_entropy_nats":float(-(gates*gates.clamp_min(1e-12).log()).sum(-1).mean()),
            "candidate_cosine":float(torch.stack(cosines).mean()),
            "candidate_pair_mse":float(torch.stack(pairwise).mean()),
            "projection_removed_fraction":float(torch.stack(normal_fraction).mean()),
            "damped_metric_condition_mean":float(torch.stack(conditions).mean()),
            "chart_distance_over_radius_mean":float(torch.stack(distances).mean()),
            "intrinsic_path_pca":path_pca}
    for key in ("pca", "gate", "physical_signals"):
        result["evaluation_"+key]=result["test_"+key]
        if split_name!="test": del result["test_"+key]
    for key in ("anchor_rmse", "manifold_reconstruction_rmse"):
        result[key+"_evaluation"]=result[key+"_test"]
        if split_name!="test": del result[key+"_test"]
    output = Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--archive",required=True)
    parser.add_argument("--output",default="outputs/manifold-moe/diagnostics.json")
    parser.add_argument("--max-cases",type=int,default=32)
    parser.add_argument("--members",type=int,default=4)
    parser.add_argument("--integration-steps",type=int,default=4)
    parser.add_argument("--seed",type=int,default=83)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--split",dest="split_name",choices=("test","validation","expert_validation"),default="test")
    args = vars(parser.parse_args(argv))
    print(diagnose_manifold(args.pop("checkpoint"),args.pop("archive"),args.pop("output"),**args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

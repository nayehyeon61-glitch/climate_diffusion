"""Read-only physical recurrence audit; no teacher-forced truth in generated paths."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .recurrent_flow import residual_target


def json_value(value):
    if isinstance(value,torch.Tensor): return json_value(value.detach().cpu().tolist())
    if isinstance(value,dict): return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_value(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


@torch.no_grad()
def diagnose_recurrent(f, states, times, output_path, *, max_cases, members,
                        integration_steps, seed, split_name):
    m,c=f.model,f.config
    indices=f.training_metadata['split'][split_name]
    indices=[indices[i] for i in np.linspace(0,len(indices)-1,min(max_cases,len(indices)),dtype=int)]
    history=f._normalise(np.stack([f.select_history(states[i:i+c.history_span_steps]) for i in indices]))
    origin=history[:,-1]; context=m.encode_history(history)
    result={'format':'climate_diffusion.recurrent_diagnostics.v1','stage':m.stage,
            'checkpoint_sha256':f.checkpoint_sha256,'evaluation_split':split_name,'evaluation_windows':indices,
            'origin_times':[str(times[i+c.history_span_steps-1]) for i in indices],
            'scope':'generated 6h recurrent prefixes; transport norms are NOT physical velocity norms',
            'initial_reconstruction_rmse':float((m.decode(m.encode(origin))-origin).square().mean().sqrt()),
            'origin_decode':'fixed origin offset; reported physical tendencies use decoded endpoints',
            'test_used':split_name=='test'}
    if m.stage!='manifold':
        next_state=f._normalise(np.stack([states[i+c.history_span_steps] for i in indices]))
        q,target=residual_target(m,origin,next_state,origin.new_full((len(origin),),c.step_hours))
        generator=torch.Generator(device=f.device).manual_seed(seed)
        source=torch.randn(q.shape,device=q.device,generator=generator)*c.residual_noise_std
        field=m.field((source+target)/2,q.new_full((len(q),),.5),context,
                      q.new_zeros(len(q)),physical_q=q)
        errors=(field['intrinsic_candidates']-(target-source)[:,None]).square().mean(-1)
        regions=field['local_log_prior'].argmax(-1)
        result['teacher_forced_audit']={
            'scope':'origin to +one archive step residual FM pair, tau=.5; not a rollout skill metric',
            'region_counts':[int((regions==k).sum()) for k in range(c.num_experts)],
            'expert_fm_by_region':[errors[regions==k].mean(0) if bool((regions==k).any()) else [None]*c.num_experts
                                  for k in range(c.num_experts)],
            'gate_best_expert_fraction':float((field['router'].argmax(-1)==errors.argmin(-1)).float().mean())}
        trace=[]
        leads=torch.arange(c.horizon_steps+1,device=f.device)[None].expand(len(origin),-1)
        path=m.sample_trajectory(context,leads,ensemble_size=members,integration_steps=integration_steps,
                                 generator=torch.Generator(device=f.device).manual_seed(seed),origin=origin,trace=trace)
        gates=torch.stack([r['gate'] for r in trace])
        result['generated_audit']={
            'population':'last tau midpoint of EACH actual physical transition; separate from sampled residual endpoints',
            'gate_mean':gates.mean((0,1,2)),
            'gate_entropy_nats':torch.stack([r['gate_entropy'] for r in trace]).mean(),
            'candidate_cosine':torch.stack([r['candidate_cosine'] for r in trace]).mean(),
            'physical_steps':c.horizon_steps,'member_count':members,
            'residual_q_variance_per_hour2':[r['residual_q_variance_per_hour2'] for r in trace]}
        # State/delta probability and tendency diagnostics use native endpoints.
        from .temporal_supervision import TemporalObjective
        truth=f._normalise(np.stack([states[i+c.history_span_steps-1:i+c.history_span_steps+c.horizon_steps]
                                    for i in indices]))
        objective=TemporalObjective(f.schema,f.state_mean,f.state_scale,f.training_metadata['temporal_statistics']).to(f.device)
        result['temporal_metrics']=objective(path,truth,origin.new_full((len(origin),c.horizon_steps),c.step_hours))
        result['physical_trace']=trace
    output=Path(output_path)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(json_value(result),indent=2,allow_nan=False)+'\n')
    return output

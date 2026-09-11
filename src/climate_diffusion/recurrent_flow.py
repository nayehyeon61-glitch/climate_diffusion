"""Physical recurrence with a conditional residual-flow sampler, no new networks.

z=encoder(x_norm), q=(z-mu_z)/sigma_z. Stage A drift is dz/day.
Residual FM transports r_tau in q/day space; its tau velocity is NOT dq/dhour.
Only the sampled endpoint r_1 is added to drift, then multiplied by dt_hours/24.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def drift_per_day(model, physical_q):
    z = physical_q * model.latent_scale + model.latent_mean
    return model.manifold.latent_drift(z) / model.latent_scale


def residual_target(model, current_state, next_state, dt_hours):
    """Teacher-forced one-step target only. Targets/coordinates cannot shrink C FM loss."""
    if (dt_hours.shape != (len(current_state),) or not bool(torch.isfinite(dt_hours).all())
            or bool((dt_hours <= 0).any())):
        raise ValueError("Residual target requires positive actual dt_hours per pair")
    with torch.no_grad():
        q = model.encode(current_state)
        target = (model.encode(next_state)-q) / (dt_hours[:, None]/24) - drift_per_day(model, q)
    return q, target


def integrate_flow_tau(model, source, physical_q, context, physical_hours, *, integration_steps,
                       mode="local"):
    """Transport residual-space noise to a residual q/day sample, midpoint tau ODE.

    physical_q is held fixed during this conditional generative solve. All K
    experts see the SAME (physical_q,r_tau); candidates fuse at EVERY tau step.
    The last field is returned only for transport diagnostics, not as a tendency.
    """
    if integration_steps < 1:
        raise ValueError("integration_steps must be positive")
    r = source * model.config.residual_noise_std
    lead = physical_hours / model.config.horizon_hours
    step_tau = 1 / integration_steps
    last = None
    for index in range(integration_steps):
        tau = r.new_full((len(r),), index*step_tau)
        a = model.field(r, tau, context, lead, mode=mode, physical_q=physical_q)
        middle = r + 0.5*step_tau*a["velocity"]
        last = model.field(middle, tau+0.5*step_tau, context, lead,
                           mode=mode, physical_q=physical_q)
        r = r + step_tau*last["velocity"]
    return r, last


def physical_step(model, physical_q, context, physical_hours, member_noise, *,
                  dt_hours, integration_steps, mode=None):
    """Euler physical transition, matching Stage A's finite-step drift objective.

    Persistent member_noise is re-used as residual FM SOURCE at each physical
    step. The changing physical_q/context condition the residual sampler.
    Zero neural FM transport does not imply zero residual: it leaves source
    noise. Use drift_only for the genuine zero-residual ablation.
    """
    if not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("dt_hours must be finite and positive")
    mode = mode or "local"
    drift = drift_per_day(model, physical_q)
    if mode == "drift_only":
        residual, audit = torch.zeros_like(drift), None
    else:
        residual, audit = integrate_flow_tau(model, member_noise, physical_q, context, physical_hours,
            integration_steps=integration_steps, mode="local" if mode == "residual_only" else mode)
    if mode == "residual_only":
        drift = torch.zeros_like(drift)
    next_q = physical_q + (dt_hours/24)*(drift+residual)
    if not bool(torch.isfinite(next_q).all()):
        raise FloatingPointError("Non-finite recurrent physical step")
    return next_q, {"drift_per_day": drift, "residual_per_day": residual,
                    "final_per_day": drift+residual, "transport": audit}


def step_diagnostics(model, q, decomposition, physical_step_index, batch_size, members):
    """Detached trace of actual generated path, with separate coordinate/units labels."""
    with torch.no_grad():
        j = model.jacobian(q.detach())
        row = {"physical_step": physical_step_index,
               "lead_hours": physical_step_index*model.config.step_hours,
               "q_input": q.detach().reshape(batch_size,members,-1).cpu(),
               "intrinsic_unit": "standardized q/hour",
               "per_variable": {}}
        row["residual_q_variance_per_hour2"] = (decomposition["residual_per_day"].detach()/24).reshape(
            batch_size,members,-1).var(1,unbiased=False).mean(-1).cpu()
        distance = (q.detach()[:,None]-model.gate.centers[None]).square().mean(-1).min(-1).values
        row["chart_distance_over_radius"] = (distance/model.gate.radius_squared).sqrt().reshape(batch_size,members).cpu()
        c,y,x = model.config.grid
        weights = model.physics.metric_weights.reshape(c,y,x)
        area = weights[0]/weights[0].sum()
        names = model.physics.variable_names
        for kind in ("drift", "residual", "final"):
            hourly = decomposition[kind+"_per_day"].detach()/24
            row[kind+"_q_rms_per_hour"] = hourly.square().mean(-1).sqrt().reshape(batch_size,members).cpu()
            # J maps intrinsic velocity to normalized full-state velocity;
            # multiplying scale restores Pa/hour, K/hour, (m/s)/hour separately.
            physical = torch.einsum("bdr,br->bd",j,hourly) * model.physics.scale
            rms = (physical.reshape(-1,c,y,x).square()*area).sum((-2,-1)).sqrt()
            for k,name in enumerate(names):
                row["per_variable"].setdefault(name,{})[kind+"_rms_per_hour"] = rms[:,k].reshape(batch_size,members).cpu()
        a = decomposition["transport"]
        if a is not None:
            pi = a["router"].detach()
            raw = a["raw_candidates"].detach()
            projected = a["candidates"].detach()
            norm = lambda v: (v.square()*model.physics.metric_weights).sum(-1).sqrt()
            rn,pn = norm(raw),norm(projected)
            direction = F.normalize(projected,dim=-1)
            off = ~torch.eye(model.config.num_experts,dtype=torch.bool,device=q.device)
            row.update(gate=pi.reshape(batch_size,members,-1).cpu(),
                gate_entropy=(-(pi*pi.clamp_min(1e-12).log()).sum(-1)).reshape(batch_size,members).cpu(),
                candidate_cosine=(direction@direction.transpose(1,2))[:,off].mean(-1).reshape(batch_size,members).cpu(),
                transport_raw_norm=rn.reshape(batch_size,members,-1).cpu(),
                transport_projected_norm=pn.reshape(batch_size,members,-1).cpu(),
                transport_projection_ratio=torch.where(rn>1e-8,pn/rn,torch.nan).reshape(batch_size,members,-1).cpu(),
                transport_unit="weighted normalized-state/day per unit tau; NOT physical tendency")
        return row


def sample_recurrent_trajectory(model, context, lead_steps, initial, *, origin,
                                integration_steps, mode=None, trace=None):
    b,m,r = initial.shape
    if origin is None or origin.shape != (b,model.config.state_dim):
        raise ValueError("Physical recurrence requires observed origin for every forecast")
    if not bool(torch.isfinite(initial).all()) or not bool(torch.isfinite(origin).all()):
        raise ValueError("Non-finite origin/member noise")
    q0 = model.encode(origin)
    q = q0[:,None].expand(-1,m,-1).reshape(b*m,r)
    shared_context = context.repeat_interleave(m,0)
    noise = initial.reshape(b*m,r)
    # Keep an origin-anchored chart, avoiding a one-off AE reconstruction jump.
    # The offset is fixed for the whole trajectory and does not alter decoder J.
    offset = origin-model.decode(q0)
    states = [origin[:,None].expand(-1,m,-1)]
    for step in range(1,int(lead_steps.max())+1):
        hours = q.new_full((len(q),),(step-1)*model.config.step_hours)
        previous_q = q
        q, decomposition = physical_step(model,q,shared_context,hours,noise,
            dt_hours=model.config.step_hours,integration_steps=integration_steps,mode=mode)
        states.append(model.decode(q).reshape(b,m,-1)+offset[:,None])
        if trace is not None:
            row = step_diagnostics(model,previous_q,decomposition,step,b,m)
            row["q_output"] = q.detach().reshape(b,m,r).cpu()
            trace.append(row)
    # No teacher forcing, re-encoding, detach or re-seeding between steps.
    path = torch.stack(states,2)
    return path.gather(2,lead_steps[:,None,:,None].expand(b,m,-1,model.config.state_dim))

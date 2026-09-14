"""Joint A+B probabilistic objective for physical recurrent Manifold MoE.

All scores consume one saved/generated trajectory graph [B,M,T+1,D].  The
teacher-forced residual-FM pass remains separate.  Physical increments are
converted to tendencies with the *actual* dt_hours and train-only scales.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch

JOINT_CHECKPOINT_FORMAT = "climate_diffusion.manifold_recurrent_joint_ab.v2"


@dataclass(frozen=True)
class LossProfile:
    fused_fm: float
    expert_fm: float = 1.0
    state_crps: float = 0.0
    transition_crps: float = 0.0
    trajectory_energy: float = 0.0
    mean_state: float = 0.0
    mean_tendency: float = 0.0
    calibration: float = 0.0
    ae_delta: float = 0.0
    decoded_drift: float = 0.0

    def validate(self) -> "LossProfile":
        values = asdict(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Loss profile weights must be finite and nonnegative")
        return self


LOSS_PROFILES = {
    "ab_control": LossProfile(fused_fm=1.0, state_crps=.5,
        trajectory_energy=.1, mean_tendency=.02),
    "v2_minimal": LossProfile(fused_fm=1.0, transition_crps=.25,
        trajectory_energy=.3, mean_tendency=.02, ae_delta=.05,
        decoded_drift=.05),
    "v2_full": LossProfile(fused_fm=.3, state_crps=1.0,
        transition_crps=.75, trajectory_energy=.75, mean_state=.1,
        mean_tendency=.1, ae_delta=.05, decoded_drift=.05),
}


def profile(name: str) -> LossProfile:
    try:
        return LOSS_PROFILES[name].validate()
    except KeyError as exc:
        raise ValueError(f"Unknown loss profile {name!r}; choose {tuple(LOSS_PROFILES)}") from exc


def _validate(samples: torch.Tensor, truth: torch.Tensor) -> tuple[int, int]:
    if samples.ndim < 3 or truth.shape != (samples.shape[0], *samples.shape[2:]):
        raise ValueError("samples/truth must be [B,M,...] and [B,...]")
    m = samples.shape[1]
    if m < 2:
        raise ValueError("Fair ensemble scores require M >= 2")
    if not bool(torch.isfinite(samples).all()) or not bool(torch.isfinite(truth).all()):
        raise FloatingPointError("Non-finite score input")
    return samples.shape[0], m


def fair_crps(samples: torch.Tensor, truth: torch.Tensor,
              weight: torch.Tensor | None = None) -> torch.Tensor:
    """Fair scalar CRPS without an O(M^2*grid) allocation.

    Uses mean|x-y| - sum_i (2i-M+1) sort(x)_i/[M(M-1)].  Conditional members
    are assumed iid.  This is a marginal score and cannot identify joint law.
    """
    _, m = _validate(samples, truth)
    accuracy = (samples-truth[:, None]).abs().mean(1)
    ordered = samples.sort(dim=1).values
    coefficient = torch.arange(1, m+1, device=samples.device,
                               dtype=samples.dtype).mul(2).sub(m+1)
    shape = (1, m) + (1,)*(samples.ndim-2)
    correction = (ordered*coefficient.reshape(shape)).sum(1)/(m*(m-1))
    score = accuracy-correction
    if weight is None:
        return score.mean()
    try:
        weighted = score*weight
    except RuntimeError as exc:
        raise ValueError("CRPS weight is not broadcastable to score coordinates") from exc
    return weighted.sum()/weight.expand_as(score).sum().clamp_min(1e-12)


def fair_energy(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Fair multivariate Energy score over flattened non-member coordinates."""
    b, m = _validate(samples, truth)
    x = samples.reshape(b,m,-1)
    y = truth.reshape(b,1,-1)
    accuracy = torch.linalg.vector_norm(x-y,dim=-1).mean(1)
    distance = torch.cdist(x,x)
    spread = distance.sum((1,2))/(m*(m-1))
    return (accuracy-.5*spread).mean()/math.sqrt(x.shape[-1])


def normalized_tendencies(samples: torch.Tensor, truth: torch.Tensor,
                          dt_hours: torch.Tensor,
                          tendency_scale: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """Map same-member increments to normalized physical tendencies."""
    _validate(samples,truth)
    if samples.ndim != 4 or samples.shape[2] < 2:
        raise ValueError("Trajectory scores require [B,M,T+1,D]")
    b,_,points,d = samples.shape
    if dt_hours.shape != (b,points-1) or tendency_scale.numel() != d:
        raise ValueError("dt_hours/tendency_scale shape mismatch")
    if not bool(torch.isfinite(dt_hours).all()) or bool((dt_hours <= 0).any()):
        raise ValueError("Every observed transition needs positive actual dt_hours")
    floor = torch.finfo(samples.dtype).eps
    scale = tendency_scale.to(samples).reshape(1,1,1,d).clamp_min(floor)
    predicted = samples.diff(2)/dt_hours[:,None,:,None]/scale
    observed = truth.diff(1)/dt_hours[:,:,None]/scale[:,0]
    return predicted,observed


def trajectory_scores(samples: torch.Tensor, truth: torch.Tensor,
                      dt_hours: torch.Tensor, tendency_scale: torch.Tensor,
                      metric: torch.Tensor | None = None,
                      *, calibration_epsilon: float = 1e-6) -> dict[str,torch.Tensor]:
    """State, transition and joint-law scores from exactly one rollout."""
    _validate(samples,truth)
    if truth.shape[1] < 2:
        raise ValueError("Known origin plus at least one future point are required")
    tendency, true_tendency = normalized_tendencies(
        samples,truth,dt_hours,tendency_scale)
    future = samples[:,:,1:]
    target_future = truth[:,1:]
    w = None if metric is None else metric.to(samples).reshape(1,1,-1)
    state_crps = fair_crps(future,target_future,w)
    transition_crps = fair_crps(tendency,true_tendency,w)

    mean_state = ((future.mean(1)-target_future).square()*
                  (1 if w is None else w)).mean()
    mean_tendency = ((tendency.mean(1)-true_tendency).square()*
                     (1 if w is None else w)).mean()

    feature_weight = torch.ones_like(truth)
    if metric is not None:
        feature_weight = metric.to(samples).sqrt().reshape(1,1,-1).expand_as(truth)
    state_feature = (future*feature_weight[:,None,1:,:]/math.sqrt(future.shape[2])).flatten(2)
    truth_state = (target_future*feature_weight[:,1:]/math.sqrt(future.shape[2])).flatten(1)
    tendency_feature = (tendency*feature_weight[:,None,1:,:]\n                        /math.sqrt(tendency.shape[2])).flatten(2)
    truth_tendency = (true_tendency*feature_weight[:,:-1]
                      /math.sqrt(tendency.shape[2])).flatten(1)
    trajectory_energy = fair_energy(
        torch.cat((state_feature,tendency_feature),-1),
        torch.cat((truth_state,truth_tendency),-1))

    # Heuristic only: pooled normalized finite-M spread/skill matching.
    m = samples.shape[1]
    variance = future.var(1,unbiased=True)
    squared_error = (future.mean(1)-target_future).square().detach()
    mask = squared_error > calibration_epsilon
    target_variance = (1+1/m)*squared_error
    ratio = (variance+calibration_epsilon)/(target_variance+calibration_epsilon)
    calibration = torch.where(mask,ratio.log().square(),torch.zeros_like(ratio))
    calibration = calibration.sum()/mask.sum().clamp_min(1)

    return {"state_crps":state_crps, "transition_crps":transition_crps,
            "trajectory_energy":trajectory_energy, "mean_state":mean_state,
            "mean_tendency":mean_tendency, "spread_skill_calibration":calibration,
            "increment_spread":tendency.std(1,unbiased=False).mean()}


def weighted_v2(components: Mapping[str,torch.Tensor],
                selected: LossProfile) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
    names = {"fm":"fused_fm", "expert_fm":"expert_fm",
             "state_crps":"state_crps", "transition_crps":"transition_crps",
             "trajectory_energy":"trajectory_energy", "mean_state":"mean_state",
             "mean_tendency":"mean_tendency",
             "spread_skill_calibration":"calibration",
             "loss_ae_delta":"ae_delta",
             "loss_finite_step_drift":"decoded_drift"}
    weighted = {}
    total = next(iter(components.values())).new_zeros(())
    for key, field in names.items():
        if key in components:
            value = components[key]*getattr(selected,field)
            weighted["weighted_"+key] = value
            total = total+value
    return total,weighted


def fixed_validation_score(components: Mapping[str,torch.Tensor]) -> torch.Tensor:
    """Profile-independent selection: state Energy + state/transition CRPS + .1 trajectory."""
    required=("energy","state_crps","transition_crps","trajectory_energy")
    missing=[name for name in required if name not in components]
    if missing:
        raise KeyError(f"Missing validation components: {missing}")
    return (components["energy"]+components["state_crps"]+
            components["transition_crps"]+.1*components["trajectory_energy"])


def module_gradient_diagnostics(components: Mapping[str,torch.Tensor],
                                groups: Mapping[str,Sequence[torch.nn.Parameter]]
                                ) -> dict[str,float]:
    """Raw per-loss gradient norm and cosine, intended for a configured first batch."""
    active={k:v for k,v in components.items() if v.requires_grad}
    result: dict[str,float]={}
    vectors: dict[tuple[str,str],torch.Tensor]={}
    for loss_name,loss in active.items():
        for group_name,parameters in groups.items():
            params=[p for p in parameters if p.requires_grad]
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vector=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                              for p,g in zip(params,grads)]) if params else loss.new_zeros(1)
            vectors[(loss_name,group_name)]=vector
            result[f"gradient_norm/{loss_name}/{group_name}"]=float(vector.norm().detach())
    keys=list(active)
    for group_name in groups:
        for i,left in enumerate(keys):
            for right in keys[i+1:]:
                a,b=vectors[(left,group_name)],vectors[(right,group_name)]
                denom=a.norm()*b.norm()
                value=(a@b/denom).detach() if float(denom)>0 else a.new_zeros(())
                result[f"gradient_cosine/{left}:{right}/{group_name}"]=float(value)
    return result

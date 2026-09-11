"""PI-AE embedding, local responsibilities, decoder-tangent field projection.

The ODE lives in standardized intrinsic coordinates q, NOT in full-state
Gaussian space with a rank-deficient tangent-only velocity. Decoder Jacobians
map each expert's full-field candidate to an intrinsic velocity; fields are
fused per member BEFORE integration. Projection is damped weighted least squares.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .moe import FieldDCT, OrthoDCT, mlp
from .model import sinusoidal_time_embedding
from .manifold_physics import SurfacePhysics

MANIFOLD_FORMAT = "climate_diffusion.manifold_moe.v1"
RECURRENT_FORMAT = "climate_diffusion.manifold_recurrent_fm.v1"


@dataclass(frozen=True)
class ManifoldMoEConfig:
    state_dim: int
    grid: tuple[int, int, int]
    history_steps: int = 6
    history_stride: int = 4
    horizon_steps: int = 120
    step_hours: int = 6
    num_experts: int = 4
    manifold_dim: int = 16
    expert_latent_dim: int = 64
    gate_hidden_dim: int = 160
    hidden_dim: int = 128
    context_dim: int = 64
    time_embedding_dim: int = 16
    gate_temperature: float = 0.7
    responsibility_temperature: float = 0.5
    locality_weight: float = 2.0
    gate_correction_limit: float = 0.5
    projection_ridge: float = 1e-3
    forecast_dynamics: str = "lead_conditioned"
    residual_noise_std: float = 1.0  # standardized intrinsic coordinates / day

    def __post_init__(self):
        object.__setattr__(self, "grid", tuple(self.grid))
        if len(self.grid) != 3 or min(self.grid) < 1 or math.prod(self.grid) != self.state_dim:
            raise ValueError("grid must be (variables, lat, lon) and multiply to state_dim")
        counts = (self.history_steps, self.history_stride, self.horizon_steps, self.step_hours,
                  self.manifold_dim, self.expert_latent_dim, self.hidden_dim, self.gate_hidden_dim,
                  self.context_dim, self.time_embedding_dim)
        if min(counts) < 1 or self.num_experts < 2 or self.manifold_dim >= self.state_dim:
            raise ValueError("Positive dimensions, K>=2 and manifold_dim < state_dim are required")
        for name in ("gate_temperature", "responsibility_temperature", "locality_weight",
                     "gate_correction_limit", "projection_ridge", "residual_noise_std"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.forecast_dynamics not in {"lead_conditioned", "recurrent_residual"}:
            raise ValueError("Unknown forecast_dynamics contract")

    @property
    def history_span_steps(self):
        return (self.history_steps - 1) * self.history_stride + 1

    @property
    def horizon_hours(self):
        return self.horizon_steps * self.step_hours


def tangent_lift(jacobian, raw_fields, weights, ridge):
    """J:[B,D,r], v:[B,K,D]. Returns a:[B,K,r], J a, pullback G.

    a = (J.T W J + eps I)^-1 J.T W v. No nonlinear decoder(v) shortcut.
    eps is relative to mean diagonal(G); ridge=0 is useful for full-rank tests.
    """
    jt_w = jacobian.transpose(-2, -1) * weights
    metric = jt_w @ jacobian
    size = metric.shape[-1]
    damping = ridge * metric.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
    system = metric + damping[:, None, None] * torch.eye(size, device=metric.device, dtype=metric.dtype)
    intrinsic = torch.linalg.solve(system, jt_w @ raw_fields.transpose(1, 2)).transpose(1, 2)
    projected = intrinsic @ jacobian.transpose(1, 2)
    return intrinsic, projected, metric


class PhysicsManifoldAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_dct = FieldDCT(config.grid)
        self.encoder = mlp(config.state_dim, config.hidden_dim, config.manifold_dim)
        self.decoder = nn.Sequential(nn.Linear(config.manifold_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, config.state_dim))
        self.latent_drift = mlp(config.manifold_dim, config.hidden_dim, config.manifold_dim)

    def encode(self, state):
        return self.encoder(self.spatial_dct(state))

    def decode(self, latent):
        return self.spatial_dct(self.decoder(latent), inverse=True)


class LocalExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        condition = config.context_dim + 2 * config.time_embedding_dim + config.manifold_dim
        self.encoder = mlp(config.state_dim + condition, config.hidden_dim, config.expert_latent_dim)
        self.velocity = mlp(config.expert_latent_dim + condition, config.hidden_dim, config.state_dim)

    def forward(self, state, condition):
        code = self.encoder(torch.cat((state, condition), -1))
        return self.velocity(torch.cat((code, condition), -1))


class LocalManifoldGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("centers", torch.zeros(config.num_experts, config.manifold_dim))
        self.register_buffer("radius_squared", torch.ones(()))
        condition = config.manifold_dim + config.context_dim + 2 * config.time_embedding_dim
        self.correction = mlp(condition, config.gate_hidden_dim, config.num_experts)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, q, condition):
        distance = (q[:, None] - self.centers[None]).square().mean(-1)
        logits = -self.config.locality_weight * distance / self.radius_squared
        local_log_prior = F.log_softmax(logits / self.config.gate_temperature, -1)
        correction = self.config.gate_correction_limit * torch.tanh(
            self.correction(torch.cat((q, condition), -1)))
        log_gate = F.log_softmax((logits + correction) / self.config.gate_temperature, -1)
        return log_gate, local_log_prior

    @torch.no_grad()
    def fit_centers(self, codes, iterations=30):
        """Deterministic farthest-first + Lloyd fit; train codes ONLY, no labels."""
        if len(codes) < self.config.num_experts:
            raise ValueError("Too few train manifold coordinates for K centers")
        centers = [codes[0]]
        while len(centers) < self.config.num_experts:
            distances = (codes[:, None] - torch.stack(centers)[None]).square().mean(-1).min(-1).values
            centers.append(codes[distances.argmax()])
        centers = torch.stack(centers)
        for _ in range(iterations):
            distances = (codes[:, None] - centers[None]).square().mean(-1)
            assignments = distances.argmin(-1)
            centers = torch.stack([codes[assignments == k].mean(0) if bool((assignments == k).any())
                                   else codes[distances.min(-1).values.argmax()]
                                   for k in range(self.config.num_experts)])
        distances = (codes[:, None] - centers[None]).square().mean(-1)
        radius = distances.min(-1).values.mean().clamp_min(0.05)
        if not bool(torch.isfinite(centers).all()):
            raise FloatingPointError("Non-finite manifold centers")
        if bool(torch.pdist(centers).min() < 1e-4):
            raise ValueError("Manifold centers collapsed; improve Stage A before specialist training")
        self.centers.copy_(centers)
        self.radius_squared.copy_(radius)


class ManifoldMoE(nn.Module):
    def __init__(self, config, schema, mean, scale):
        super().__init__()
        self.config = config
        self.manifold = PhysicsManifoldAE(config)
        self.physics = SurfacePhysics(schema, mean, scale)
        self.temporal_dct = OrthoDCT(config.history_steps)
        self.history_encoder = mlp(config.history_steps * config.manifold_dim, config.hidden_dim, config.context_dim)
        self.experts = nn.ModuleList(LocalExpert(config) for _ in range(config.num_experts))
        self.gate = LocalManifoldGate(config)
        self.reference_encoder = copy.deepcopy(self.manifold.encoder)
        self.register_buffer("latent_mean", torch.zeros(config.manifold_dim))
        self.register_buffer("latent_scale", torch.ones(config.manifold_dim))
        self.register_buffer("manifold_ready", torch.tensor(False))
        self.stage = "manifold"
        self.set_stage(self.stage)

    def set_stage(self, stage):
        if stage not in {"manifold", "specialize", "joint"}:
            raise ValueError("Stage must be manifold, specialize, or joint")
        self.stage = stage
        self.requires_grad_(False)
        if stage in {"manifold", "joint"}:
            self.manifold.requires_grad_(True)
        if stage in {"specialize", "joint"}:
            for module in (self.experts, self.gate, self.history_encoder):
                module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.reference_encoder.eval()
        if self.stage == "specialize":
            self.manifold.eval()
        if self.stage == "manifold":
            self.experts.eval()
            self.gate.eval()
            self.history_encoder.eval()
        return self

    def encode(self, state):
        return (self.manifold.encode(state) - self.latent_mean) / self.latent_scale

    def decode(self, q):
        return self.manifold.decode(q * self.latent_scale + self.latent_mean)

    def reference_encode(self, state):
        raw = self.reference_encoder(self.manifold.spatial_dct(state))
        return (raw - self.latent_mean) / self.latent_scale

    @torch.no_grad()
    def seal_manifold(self, train_states):
        raw = torch.cat([self.manifold.encode(chunk) for chunk in train_states.split(512)])
        self.latent_mean.copy_(raw.mean(0))
        self.latent_scale.copy_(raw.std(0, unbiased=False).clamp_min(0.05))
        self.gate.fit_centers((raw - self.latent_mean) / self.latent_scale)
        self.reference_encoder.load_state_dict(self.manifold.encoder.state_dict())
        self.manifold_ready.fill_(True)

    def encode_history(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.config.history_steps, self.config.state_dim):
            raise ValueError("History must be [batch, history_steps, full state_dim]")
        codes = self.encode(history)
        return self.history_encoder(self.temporal_dct(codes, dim=1).flatten(1))

    def jacobian(self, q):
        # Forward-mode AD computes D×r, preferable to D reverse-mode passes.
        # Enable transforms even if a caller requested inference_mode.
        with torch.inference_mode(False):
            normal_q = q.clone() if torch.is_inference(q) else q
            return torch.func.vmap(torch.func.jacfwd(self.decode))(normal_q)

    def field(self, q, tau, context, lead, *, mode=None, physical_q=None):
        # Recurrent mode: q is residual-space FM state (q/day), physical_q is
        # the current physical state. Never project at a residual/noise state.
        if self.config.forecast_dynamics == "recurrent_residual" and physical_q is None:
            raise ValueError("Residual FM field requires physical_q; tau is not physical time")
        chart_q = q if physical_q is None else physical_q
        if not bool(self.manifold_ready):
            raise ValueError("Pretrain and seal the manifold before expert flow evaluation")
        mode = mode or "local"
        if mode not in {"local", "uniform"} and not (mode.startswith("expert:") and
                                                       mode[7:].isdigit() and int(mode[7:]) < self.config.num_experts):
            raise ValueError("Manifold mode must be local, uniform, or expert:<zero-based index>")
        width = self.config.time_embedding_dim
        condition = torch.cat((context, sinusoidal_time_embedding(tau, width),
                               sinusoidal_time_embedding(lead, width)), -1)
        log_gate, local_log_prior = self.gate(chart_q, condition)
        pi = log_gate.exp()
        if mode == "uniform":
            pi = torch.full_like(pi, 1 / self.config.num_experts)
        elif mode.startswith("expert:"):
            pi = F.one_hot(torch.full((len(q),), int(mode[7:]), device=q.device), self.config.num_experts).to(q)
        shared_state = self.manifold.spatial_dct(self.decode(chart_q))
        expert_condition = torch.cat((q, condition), -1)
        spectral = torch.stack([expert(shared_state, expert_condition) for expert in self.experts], 1)
        raw = self.manifold.spatial_dct(spectral, inverse=True)
        jacobian = self.jacobian(chart_q)
        intrinsic, projected, metric = tangent_lift(jacobian, raw, self.physics.metric_weights,
                                                   self.config.projection_ridge)
        velocity = (pi[..., None] * intrinsic).sum(1)
        return {"velocity": velocity, "intrinsic_candidates": intrinsic, "candidates": projected,
                "raw_candidates": raw, "jacobian": jacobian, "metric": metric, "router": pi,
                "log_gate": log_gate, "local_log_prior": local_log_prior}

    def integrate(self, initial, context, lead, *, integration_steps, mode=None):
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        q = initial
        dt = 1 / integration_steps
        for index in range(integration_steps):
            tau = q.new_full((len(q),), index * dt)
            first = self.field(q, tau, context, lead, mode=mode)["velocity"]
            middle = q + 0.5 * dt * first
            second = self.field(middle, tau + 0.5 * dt, context, lead, mode=mode)["velocity"]
            q = q + dt * second
        return q

    def manifold_loss(self, state, next_state, *, physics_weight=0.1, invariant_weight=0.05,
                      metric_weight=0.1, dynamics_weight=0.1):
        if len(state) < 2 and metric_weight > 0:
            raise ValueError("Manifold metric requires at least two states; do not silently train with zero metric")
        z = self.manifold.encode(state)
        reconstruction = self.manifold.decode(z)
        metrics = self.physics.reconstruction_losses(reconstruction, state)
        # Adjacent observations use PHYSICAL time (days), never generative tau.
        next_z = self.manifold.encode(next_state)
        next_prediction = z + self.config.step_hours / 24 * self.manifold.latent_drift(z)
        dynamics = F.mse_loss(next_prediction, next_z)
        if len(state) > 1:
            latent_distance = (z - z.roll(1, 0)).square().mean(-1).clamp_min(1e-12).sqrt()
            physical_distance = self.physics.pair_distance(state, state.roll(1, 0)).detach()
            metric_loss = F.mse_loss(latent_distance, physical_distance)
        else:
            metric_loss = z.sum() * 0
        metrics.update(metric=metric_loss, latent_dynamics=dynamics)
        metrics["loss"] = (metrics["reconstruction"] + physics_weight * metrics["physics"]
                           + invariant_weight * metrics["invariant"] + metric_weight * metric_loss
                           + dynamics_weight * dynamics)
        return metrics

    def specialization_loss(self, q, target_velocity, tau, context, lead, *, gate_weight=0.2,
                            balance_weight=0.05, diversity_weight=0.001, projection_weight=0.05,
                            entropy_weight=0.01, physical_q=None):
        result = self.field(q, tau, context, lead, physical_q=physical_q)
        errors = (result["intrinsic_candidates"] - target_velocity[:, None]).square().mean(-1)
        # Detached E-step target: expert fit + immutable geometric region prior.
        responsibilities = F.softmax(result["local_log_prior"].detach()
                                      - errors.detach() / self.config.responsibility_temperature, -1)
        expert_fm = (responsibilities * errors).sum(-1).mean()
        fused_fm = F.mse_loss(result["velocity"], target_velocity)
        gate_loss = -(responsibilities * result["log_gate"]).sum(-1).mean()
        pi = result["router"]
        usage = pi.mean(0)
        balance = self.config.num_experts * usage.square().sum() - 1
        entropy = -(pi * result["log_gate"]).sum(-1)
        # Penalize near-uniform per-sample routing; global usage controlled separately.
        entropy_penalty = F.relu(entropy - 0.7 * math.log(self.config.num_experts)).square().mean()
        normed = F.normalize(result["candidates"], dim=-1, eps=1e-6)
        cosine = normed @ normed.transpose(1, 2)
        offdiag = ~torch.eye(self.config.num_experts, dtype=torch.bool, device=q.device)
        overlap = (pi[:, :, None] * pi[:, None, :]).detach()
        diversity = (F.relu(cosine[:, offdiag] - 0.95).square() * overlap[:, offdiag]).mean()
        normal_error = ((result["raw_candidates"] - result["candidates"]).square()
                        * self.physics.metric_weights).mean(-1)
        projection = (responsibilities * normal_error).sum(-1).mean()
        metrics = {"fm": fused_fm, "expert_fm": expert_fm, "gate": gate_loss, "balance": balance,
                   "diversity": diversity, "projection": projection, "entropy_penalty": entropy_penalty,
                   "gate_entropy": entropy.mean(), "candidate_cosine": cosine[:, offdiag].mean(),
                   "candidate_mse": (result["candidates"][:, :, None] - result["candidates"][:, None, :])
                                    .square().mean(-1)[:, offdiag].mean(),
                   "responsibility_agreement": (pi.argmax(-1) == responsibilities.argmax(-1)).to(q).mean()}
        metrics["loss"] = (fused_fm + expert_fm + gate_weight * gate_loss + balance_weight * balance
                           + diversity_weight * diversity + projection_weight * projection
                           + entropy_weight * entropy_penalty)
        for k in range(self.config.num_experts):
            metrics[f"usage_{k}"] = usage[k]
            metrics[f"expert_fm_{k}"] = errors[:, k].mean()
        return metrics

    def sample_trajectory(self, context, lead_steps, *, ensemble_size, integration_steps,
                          generator=None, initial=None, origin=None, mode=None, trace=None):
        """Differentiable common sampler for train/inference, [B,M,P,D].

        Integer physical steps include observed origin 0. The checkpoint horizon,
        not requested prefix/block length, defines lead normalization.
        """
        b = len(context)
        if (lead_steps.ndim != 2 or lead_steps.shape[0] != b or lead_steps.shape[1] < 1
                or lead_steps.dtype not in (torch.int32, torch.int64)
                or bool((lead_steps < 0).any()) or bool((lead_steps > self.config.horizon_steps).any())
                or ensemble_size < 1 or integration_steps < 1):
            raise ValueError("Invalid integer physical lead steps or ensemble/ODE size")
        if bool((lead_steps == 0).any()) and (origin is None or origin.shape != (b,self.config.state_dim)):
            raise ValueError("Observed origin required for physical step zero")
        if initial is None:
            initial = torch.randn(b, ensemble_size, self.config.manifold_dim, device=context.device,
                                  dtype=context.dtype, generator=generator)
        if initial.shape != (b, ensemble_size, self.config.manifold_dim):
            raise ValueError("initial must be [batch,member,manifold_dim]")
        if self.config.forecast_dynamics == "recurrent_residual":
            from .recurrent_flow import sample_recurrent_trajectory
            return sample_recurrent_trajectory(self, context, lead_steps, initial,
                origin=origin, integration_steps=integration_steps, mode=mode, trace=trace)
        common = context.repeat_interleave(ensemble_size, 0)
        noise = initial.reshape(b*ensemble_size, -1)
        outputs = []
        for steps in lead_steps.unbind(1):
            if bool((steps == 0).all()):
                value = origin[:, None].expand(-1, ensemble_size, -1)
            else:
                lead = steps.clamp_min(1).to(context).repeat_interleave(ensemble_size) / self.config.horizon_steps
                q = self.integrate(noise, common, lead, integration_steps=integration_steps, mode=mode)
                value = self.decode(q).reshape(b, ensemble_size, -1)
                if bool((steps == 0).any()):
                    value = torch.where((steps == 0)[:,None,None], origin[:,None], value)
            outputs.append(value)
        return torch.stack(outputs, 2)

    @torch.no_grad()
    def forecast(self, history, origin=None, *, ensemble_size=1, integration_steps=32,
                 lead_indices=None, generator=None, mode=None):
        if self.stage == "manifold":
            raise ValueError("Stage A checkpoint has no trained expert flow; complete Stage B first")
        if min(ensemble_size, integration_steps) < 1:
            raise ValueError("ensemble_size and integration_steps must be positive")
        leads = list(range(self.config.horizon_steps)) if lead_indices is None else list(lead_indices)
        if not leads or any(not isinstance(k, int) or not 0 <= k < self.config.horizon_steps for k in leads):
            raise ValueError("lead_indices must be within the trained horizon")
        steps = (torch.tensor(leads, device=history.device) + 1)[None].expand(len(history), -1)
        return self.sample_trajectory(self.encode_history(history), steps, ensemble_size=ensemble_size,
                                      integration_steps=integration_steps, generator=generator, mode=mode,
                                      origin=history[:, -1] if origin is None else origin)

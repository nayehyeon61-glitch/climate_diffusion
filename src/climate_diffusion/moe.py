"""Full-state regime experts, fused BEFORE each flow-time ODE update.

The ODE state and fused velocity live in standardized physical-field coordinates.
Each expert has a conditional AE bottleneck but a SEPARATE supervised velocity
head in spatial-DCT coordinates. IDCT is linear and precedes the meta learner.
There is no assumption that decoding a nonlinear latent velocity is valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import fair_ensemble_crps, sinusoidal_time_embedding

MOE_FORMAT = "climate_diffusion.flow_matching_moe.v1"


@dataclass(frozen=True)
class MoEConfig:
    state_dim: int
    grid: tuple[int, int, int]
    history_steps: int = 6
    history_stride: int = 120
    horizon_steps: int = 120
    step_hours: int = 6
    num_experts: int = 4
    expert_latent_dim: int = 64
    meta_latent_dim: int = 160
    hidden_dim: int = 256
    context_dim: int = 128
    time_embedding_dim: int = 32
    residual_limit: float = 2.0

    def __post_init__(self):
        grid = tuple(self.grid)
        object.__setattr__(self, "grid", grid)
        if len(grid) != 3 or min(grid) < 1 or math.prod(grid) != self.state_dim:
            raise ValueError("grid must be (variables, lat, lon) and multiply to state_dim")
        counts = (self.state_dim, self.history_steps, self.history_stride,
                  self.horizon_steps, self.step_hours, self.expert_latent_dim,
                  self.meta_latent_dim, self.hidden_dim, self.context_dim,
                  self.time_embedding_dim)
        if min(counts) < 1 or self.num_experts < 2:
            raise ValueError("Dimensions must be positive; num_experts must be at least 2")
        if not math.isfinite(self.residual_limit) or self.residual_limit <= 0:
            raise ValueError("residual_limit must be finite and positive")

    @property
    def history_span_steps(self):
        return (self.history_steps - 1) * self.history_stride + 1

    @property
    def horizon_hours(self):
        return self.horizon_steps * self.step_hours


class OrthoDCT(nn.Module):
    """Orthonormal DCT-II along one axis; inverse is exactly the transpose."""
    def __init__(self, size: int):
        super().__init__()
        n = torch.arange(size, dtype=torch.float64)
        matrix = torch.cos(math.pi / size * n[:, None] * (n[None, :] + 0.5))
        matrix *= math.sqrt(2.0 / size)
        matrix[0] /= math.sqrt(2.0)
        self.register_buffer("matrix", matrix)

    def forward(self, values, *, inverse=False, dim=-1):
        moved = values.movedim(dim, -1)
        matrix = self.matrix.to(dtype=values.dtype)
        result = moved @ (matrix if inverse else matrix.T)
        return result.movedim(-1, dim)


class FieldDCT(nn.Module):
    """Spatial DCT independently for every variable, never across variable names."""
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.latitude = OrthoDCT(grid[1])
        self.longitude = OrthoDCT(grid[2])

    def forward(self, flat, *, inverse=False):
        shaped = flat.reshape(*flat.shape[:-1], *self.grid)
        shaped = self.latitude(shaped, inverse=inverse, dim=-2)
        shaped = self.longitude(shaped, inverse=inverse, dim=-1)
        return shaped.reshape_as(flat)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class FullStateExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        condition_dim = config.context_dim + 2 * config.time_embedding_dim
        self.encoder = mlp(config.state_dim + condition_dim, config.hidden_dim,
                           config.expert_latent_dim)
        self.reconstruction = mlp(config.expert_latent_dim + condition_dim,
                                  config.hidden_dim, config.state_dim)
        self.velocity = mlp(config.expert_latent_dim + condition_dim,
                           config.hidden_dim, config.state_dim)

    def forward(self, spectral_state, condition, *, reconstruct=False):
        latent = self.encoder(torch.cat((spectral_state, condition), dim=-1))
        features = torch.cat((latent, condition), dim=-1)
        velocity = self.velocity(features)
        reconstruction = self.reconstruction(features) if reconstruct else None
        return velocity, reconstruction


class FlowMetaLearner(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        condition_dim = config.context_dim + 2 * config.time_embedding_dim
        self.encoder = mlp(config.state_dim * (1 + config.num_experts) + condition_dim,
                           config.hidden_dim, config.meta_latent_dim)
        self.reconstruction = mlp(config.meta_latent_dim, config.hidden_dim, config.state_dim)
        self.gate = nn.Linear(config.meta_latent_dim, config.num_experts)
        self.residual = nn.Linear(config.meta_latent_dim, config.state_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, state, candidates, condition, *, reconstruct=False):
        code = self.encoder(torch.cat((state, candidates.flatten(1), condition), dim=-1))
        alpha = self.gate(code).softmax(-1)
        residual = self.config.residual_limit * torch.tanh(self.residual(code))
        velocity = (alpha[..., None] * candidates).sum(1) + residual
        reconstructed = self.reconstruction(code) if reconstruct else None
        return velocity, alpha, residual, reconstructed


def fair_energy_score(samples, target):
    """[B,M,D] full-field energy score, normalized by sqrt(D), off-diagonal pairs."""
    members, dimension = samples.shape[1:]
    if members < 2:
        raise ValueError("Energy score requires at least two members")
    accuracy = torch.linalg.vector_norm(samples - target[:, None], dim=-1).mean()
    differences = samples[:, :, None] - samples[:, None, :]
    distance = torch.linalg.vector_norm(differences, dim=-1)
    dispersion = distance.sum() / (samples.shape[0] * members * (members - 1))
    return (accuracy - 0.5 * dispersion) / math.sqrt(dimension)


def ensemble_scores(samples, target):
    return {"energy": fair_energy_score(samples, target),
            "crps": fair_ensemble_crps(samples, target)}


class FlowMatchingMoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.temporal_dct = OrthoDCT(config.history_steps)
        self.field_dct = FieldDCT(config.grid)
        self.history_encoder = mlp(config.history_steps * config.state_dim,
                                   config.hidden_dim, config.context_dim)
        self.experts = nn.ModuleList(FullStateExpert(config) for _ in range(config.num_experts))
        self.router = mlp(config.state_dim + config.context_dim + 2 * config.time_embedding_dim,
                          config.hidden_dim, config.num_experts)
        self.meta = FlowMetaLearner(config)
        self.stage = "experts"
        self.set_stage("experts")

    def set_stage(self, stage):
        if stage not in {"experts", "meta"}:
            raise ValueError("stage must be experts or meta")
        self.stage = stage
        self.requires_grad_(False)
        if stage == "experts":
            for module in (self.history_encoder, self.experts, self.router):
                module.requires_grad_(True)
        else:
            self.meta.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "meta":
            self.history_encoder.eval()
            self.experts.eval()
            self.router.eval()
        else:
            self.meta.eval()
        return self

    def encode_history(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.config.history_steps, self.config.state_dim):
            raise ValueError("History must be [batch, history_steps, state_dim]")
        # DCT after the assembled causal history context, along HISTORY TIME.
        spectral_history = self.temporal_dct(history, dim=1)
        return self.history_encoder(spectral_history.flatten(1))

    def condition(self, context, tau, lead):
        width = self.config.time_embedding_dim
        return torch.cat((context, sinusoidal_time_embedding(tau, width),
                          sinusoidal_time_embedding(lead, width)), dim=-1)

    def field(self, state, tau, context, lead, *, mode=None, reconstruct=False):
        """All experts receive the SAME current state and flow time for each member."""
        mode = mode or self.stage
        if mode not in {"experts", "meta", "uniform"}:
            raise ValueError("mode must be experts, meta or uniform")
        condition = self.condition(context, tau, lead)
        spectral_state = self.field_dct(state)
        outputs = [expert(spectral_state, condition, reconstruct=reconstruct)
                   for expert in self.experts]
        spectral_velocities = torch.stack([pair[0] for pair in outputs], dim=1)
        # Velocity transform is linear; no nonlinear state decoder is used here.
        candidates = self.field_dct(spectral_velocities, inverse=True)
        router = self.router(torch.cat((spectral_state, condition), dim=-1)).softmax(-1)
        residual = torch.zeros_like(state)
        meta_reconstruction = None
        if mode == "meta":
            # Do NOT no_grad/detach frozen experts: stage-2 sampling needs dv/dx.
            velocity, alpha, residual, meta_reconstruction = self.meta(
                state, candidates, condition, reconstruct=reconstruct)
        else:
            alpha = router if mode == "experts" else torch.full_like(router, 1 / self.config.num_experts)
            velocity = (alpha[..., None] * candidates).sum(1)
        return {"velocity": velocity, "candidates": candidates, "router": router,
                "alpha": alpha, "residual": residual,
                "expert_reconstruction": (torch.stack([p[1] for p in outputs], 1)
                                          if reconstruct else None),
                "spectral_state": spectral_state, "meta_reconstruction": meta_reconstruction}

    def integrate(self, initial, context, lead, *, integration_steps, mode=None):
        """One midpoint ODE for each member, NOT K independent endpoint averages."""
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        state = initial
        step = 1.0 / integration_steps
        for index in range(integration_steps):
            tau = state.new_full((len(state),), index * step)
            first = self.field(state, tau, context, lead, mode=mode)["velocity"]
            middle = state + 0.5 * step * first
            second = self.field(middle, tau + 0.5 * step, context, lead, mode=mode)["velocity"]
            state = state + step * second
        return state

    def warmup_loss(self, state, target_velocity, tau, context, lead,
                    *, reconstruction_weight=0.1, balance_weight=0.05, diversity_weight=0.01):
        result = self.field(state, tau, context, lead, mode="experts", reconstruct=True)
        error = (result["candidates"] - target_velocity[:, None]).square().mean(-1)
        fm = (result["router"] * error).sum(-1).mean()
        usage = result["router"].mean(0)
        balance = self.config.num_experts * usage.square().sum() - 1.0
        directions = F.normalize(result["candidates"], dim=-1, eps=1e-6)
        similarity = directions @ directions.transpose(1, 2)
        offdiag = ~torch.eye(self.config.num_experts, dtype=torch.bool, device=state.device)
        # Bounded cosine penalty, not a reward for arbitrarily large velocities.
        diversity = F.relu(similarity[:, offdiag] - 0.8).square().mean()
        rec = F.mse_loss(result["expert_reconstruction"],
                         result["spectral_state"][:, None].expand_as(result["expert_reconstruction"]))
        loss = fm + reconstruction_weight * rec + balance_weight * balance + diversity_weight * diversity
        metrics = {"loss": loss, "fm": fm, "reconstruction": rec,
                   "balance": balance, "diversity": diversity,
                   "router_entropy": -(result["router"] * result["router"].clamp_min(1e-9).log()).sum(-1).mean()}
        for k in range(self.config.num_experts):
            metrics[f"router_{k}"] = usage[k]
            metrics[f"expert_fm_{k}"] = error[:, k].mean()
        return metrics

    def meta_loss(self, state, target_velocity, tau, context, lead, *, samples, target,
                  fm_weight=1.0, energy_weight=0.5, crps_weight=0.5,
                  diversity_weight=0.01, reconstruction_weight=0.05):
        result = self.field(state, tau, context, lead, mode="meta", reconstruct=True)
        fm = F.mse_loss(result["velocity"], target_velocity)
        rec = F.mse_loss(result["meta_reconstruction"], state)
        scores = ensemble_scores(samples, target)
        spread = samples.std(dim=1, unbiased=False)
        # A weak two-sided guard, NOT a guarantee of calibration.
        diversity = (F.relu(0.02 - spread).square() + F.relu(spread - 3.0).square()).mean()
        correction = result["residual"].square().mean()
        total = (fm_weight * fm + energy_weight * scores["energy"] + crps_weight * scores["crps"]
                 + diversity_weight * diversity + reconstruction_weight * rec + 1e-4 * correction)
        metrics = {"loss": total, "fm": fm, "reconstruction": rec, **scores,
                   "diversity": diversity, "residual_l2": correction,
                   "ensemble_spread": spread.mean()}
        for k in range(self.config.num_experts):
            metrics[f"alpha_{k}"] = result["alpha"][:, k].mean()
        return metrics

    @torch.no_grad()
    def forecast(self, history, origin=None, *, ensemble_size=1, integration_steps=32,
                 lead_indices=None, generator=None, mode=None):
        if min(ensemble_size, integration_steps) < 1:
            raise ValueError("ensemble_size and integration_steps must be positive")
        leads = list(range(self.config.horizon_steps)) if lead_indices is None else list(lead_indices)
        if not leads or any(not isinstance(k, int) or k < 0 or k >= self.config.horizon_steps for k in leads):
            raise ValueError("lead_indices must be inside the trained horizon")
        context = self.encode_history(history)
        batch = len(history)
        repeated = context.repeat_interleave(ensemble_size, dim=0)
        # Independent members, shared initial noise across leads AND experts.
        initial = torch.randn(batch * ensemble_size, self.config.state_dim,
                              dtype=history.dtype, device=history.device, generator=generator)
        outputs = []
        for index in leads:
            lead = history.new_full((len(initial),), (index + 1) / self.config.horizon_steps)
            prediction = self.integrate(initial, repeated, lead,
                                        integration_steps=integration_steps, mode=mode)
            outputs.append(prediction.reshape(batch, ensemble_size, -1))
        return torch.stack(outputs, dim=2)

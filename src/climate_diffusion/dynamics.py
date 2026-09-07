"""Physical-time latent dynamics with a flow-matching head at each step.

The monthly model in :mod:`climate_diffusion.model` jumps one archive interval
in a single shot, so its only ODE is over flow time ``tau`` in [0, 1]. This
module adds the second axis the forecast actually has:

    history -> condition c
    z(0)    = encode(x_t0)
    dz/ds   = g(z, s, c)          integrated over physical time on the step grid
    at each s_k, v_theta(w, tau, [c, z_k, s_k]) transports N(0, I) onto the
    latent of the true state, so an ensemble can be drawn at any lead time.

The deterministic ODE carries the trajectory; flow matching supplies the
stochastic spread around it. ``horizon_steps`` times ``step_hours`` is the
forecast range, e.g. 120 steps of 6h for a 720h horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import FlowModelConfig
from .model import (
    LATENT_SCALE_FLOOR,
    LATENT_SCALE_MOMENTUM,
    build_autoencoder,
    fair_ensemble_crps,
    sinusoidal_time_embedding,
)

ADAPTIVE_SOLVERS = {"dopri5", "dopri8", "bosh3", "adaptive_heun"}


@dataclass(frozen=True)
class DynamicsModelConfig:
    state_dim: int
    history_steps: int = 6
    history_stride: int = 120
    horizon_steps: int = 120
    step_hours: int = 6
    latent_dim: int = 128
    hidden_dim: int = 256
    time_embedding_dim: int = 32
    autoencoder_hidden_dim: int | None = None
    autoencoder_blocks: int = 3
    autoencoder_dropout: float = 0.0
    autoencoder_kind: str = "mlp"
    autoencoder_grid: tuple[int, int, int] | None = None
    latent_normalization: bool = True
    dynamics_solver: str = "rk4"
    dynamics_rtol: float = 1e-4
    dynamics_atol: float = 1e-4
    dynamics_adjoint: bool = True
    flow_solver: str = "midpoint"

    def __post_init__(self) -> None:
        if min(
            self.state_dim,
            self.history_steps,
            self.history_stride,
            self.horizon_steps,
            self.step_hours,
            self.latent_dim,
            self.hidden_dim,
            self.time_embedding_dim,
        ) < 1:
            raise ValueError("All dimensions and step counts must be positive")
        if self.autoencoder_hidden_dim is not None and self.autoencoder_hidden_dim < 1:
            raise ValueError("autoencoder_hidden_dim must be positive")
        if self.autoencoder_blocks < 0:
            raise ValueError("autoencoder_blocks cannot be negative")
        if not 0.0 <= self.autoencoder_dropout < 1.0:
            raise ValueError("autoencoder_dropout must be in [0, 1)")
        if min(self.dynamics_rtol, self.dynamics_atol) <= 0:
            raise ValueError("Solver tolerances must be positive")
        if self.autoencoder_kind not in {"mlp", "conv"}:
            raise ValueError("autoencoder_kind must be 'mlp' or 'conv'")
        if self.autoencoder_grid is not None:
            object.__setattr__(
                self, "autoencoder_grid",
                tuple(int(value) for value in self.autoencoder_grid),
            )

    @property
    def resolved_autoencoder_hidden_dim(self) -> int:
        return self.autoencoder_hidden_dim or self.hidden_dim

    @property
    def horizon_hours(self) -> int:
        return self.horizon_steps * self.step_hours

    @property
    def history_span_steps(self) -> int:
        """Archive steps spanned by the strided history window."""
        return (self.history_steps - 1) * self.history_stride + 1

    @property
    def adaptive_dynamics_solver(self) -> bool:
        return self.dynamics_solver in ADAPTIVE_SOLVERS


@dataclass(frozen=True)
class DynamicsLossConfig:
    reconstruction_weight: float = 1.0
    trajectory_weight: float = 1.0
    flow_weight: float = 1.0
    latent_regularization_weight: float = 1e-4
    ensemble_weight: float = 0.0
    ensemble_size: int = 0
    ensemble_steps: int = 8
    flow_steps_per_batch: int = 8

    def __post_init__(self) -> None:
        if min(
            self.reconstruction_weight,
            self.trajectory_weight,
            self.flow_weight,
            self.latent_regularization_weight,
            self.ensemble_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")
        if self.ensemble_size < 0 or self.ensemble_size == 1:
            raise ValueError("ensemble_size must be 0 (off) or at least 2")
        if min(self.ensemble_steps, self.flow_steps_per_batch) < 1:
            raise ValueError("ensemble_steps and flow_steps_per_batch must be positive")

    @property
    def ensemble_enabled(self) -> bool:
        return self.ensemble_weight > 0.0 and self.ensemble_size >= 2


class TrajectoryWindowDataset(Dataset):
    """History windows with the full future trajectory as the target.

    ``history_stride`` lets a long context sit on a fine archive: six states at
    a 120-step stride span 150 days of a 6-hourly record, including the
    origin, without feeding all 601 snapshots through the encoder.
    """

    def __init__(
        self,
        states: np.ndarray,
        config: DynamicsModelConfig,
        indices: list[int] | None = None,
    ):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.config = config
        span = config.history_span_steps
        last_start = len(states) - span - config.horizon_steps + 1
        available = list(range(max(0, last_start)))
        if not available:
            raise ValueError("Archive is too short for this history/horizon window")
        self.indices = available if indices is None else list(indices)
        allowed = set(available)
        if any(index not in allowed for index in self.indices):
            raise ValueError("Window index is outside the available states")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        config = self.config
        start = self.indices[index]
        history_positions = [
            start + offset * config.history_stride for offset in range(config.history_steps)
        ]
        origin = history_positions[-1]
        return {
            "history": self.states[history_positions],
            "origin": self.states[origin],
            "targets": self.states[origin + 1 : origin + 1 + config.horizon_steps],
        }


class HistoryEncoder(nn.Module):
    def __init__(self, config: DynamicsModelConfig):
        super().__init__()
        self.encoder = nn.GRU(config.latent_dim, config.hidden_dim, batch_first=True)

    def forward(self, history_latents: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(history_latents)
        return hidden[-1]


class LatentDynamicsField(nn.Module):
    """dz/ds over physical time, conditioned on the history summary."""

    def __init__(self, config: DynamicsModelConfig):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        input_dim = config.latent_dim + config.hidden_dim + config.time_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, config.latent_dim),
        )

    def forward(
        self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        features = sinusoidal_time_embedding(time, self.config.time_embedding_dim)
        return self.network(torch.cat((latent, condition, features), dim=-1))


class _DynamicsField(nn.Module):
    """``func(t, y)`` adapter; a Module so ``odeint_adjoint`` sees parameters."""

    def __init__(self, field: LatentDynamicsField, condition: torch.Tensor):
        super().__init__()
        self.field = field
        self.condition = condition

    def forward(self, time: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.field(latent, time.expand(latent.shape[0]).to(latent.dtype), self.condition)


class TrajectoryFlowField(nn.Module):
    """Flow-matching velocity at a given physical lead time.

    Conditioned on the history summary, the deterministic latent the dynamics
    ODE reached, and the lead time itself, so one head serves every step.
    """

    def __init__(self, config: DynamicsModelConfig):
        super().__init__()
        self.config = config
        input_dim = (
            config.latent_dim  # w_tau
            + config.hidden_dim  # history condition
            + config.latent_dim  # deterministic latent at this lead time
            + 2 * config.time_embedding_dim  # flow time and lead time
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        dynamics_latent: torch.Tensor,
        lead_time: torch.Tensor,
    ) -> torch.Tensor:
        dimension = self.config.time_embedding_dim
        features = torch.cat(
            (
                latent,
                condition,
                dynamics_latent,
                sinusoidal_time_embedding(flow_time, dimension),
                sinusoidal_time_embedding(lead_time, dimension),
            ),
            dim=-1,
        )
        return self.network(features)


class _FlowField(nn.Module):
    def __init__(self, field: TrajectoryFlowField, condition, dynamics_latent, lead_time):
        super().__init__()
        self.field = field
        self.condition = condition
        self.dynamics_latent = dynamics_latent
        self.lead_time = lead_time

    def forward(self, time: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        expanded = time.expand(latent.shape[0]).to(latent.dtype)
        return self.field(
            latent, expanded, self.condition, self.dynamics_latent, self.lead_time
        )


class LatentDynamicsFlow(nn.Module):
    """Latent ODE over physical time with a per-step flow-matching ensemble head."""

    def __init__(self, config: DynamicsModelConfig):
        super().__init__()
        self.config = config
        self.autoencoder = build_autoencoder(
            FlowModelConfig(
                state_dim=config.state_dim,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                autoencoder_hidden_dim=config.autoencoder_hidden_dim,
                autoencoder_blocks=config.autoencoder_blocks,
                autoencoder_dropout=config.autoencoder_dropout,
                autoencoder_kind=config.autoencoder_kind,
                autoencoder_grid=config.autoencoder_grid,
            )
        )
        self.history_encoder = HistoryEncoder(config)
        self.dynamics = LatentDynamicsField(config)
        self.vector_field = TrajectoryFlowField(config)
        if config.latent_normalization:
            self.register_buffer("latent_scale", torch.ones(()))
            self.register_buffer("latent_scale_count", torch.zeros((), dtype=torch.long))

    # ---- latent scale -----------------------------------------------------
    def encode_latent(self, state: torch.Tensor, *, update_scale: bool = False) -> torch.Tensor:
        latent = self.autoencoder.encode(state)
        if not self.config.latent_normalization:
            return latent
        if update_scale and self.training:
            observed = latent.detach().std(unbiased=False).clamp_min(LATENT_SCALE_FLOOR)
            if self.latent_scale_count == 0:
                self.latent_scale.fill_(float(observed))
            else:
                self.latent_scale.mul_(1.0 - LATENT_SCALE_MOMENTUM).add_(
                    LATENT_SCALE_MOMENTUM * observed
                )
            self.latent_scale_count += 1
        # Snapshot the buffer: the in-place EMA update above would otherwise
        # bump its autograd version and invalidate an earlier encode's graph.
        return latent / self.latent_scale.clone()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.config.latent_normalization:
            latent = latent * self.latent_scale.clone()
        return self.autoencoder.decode(latent)

    # ---- physical-time rollout -------------------------------------------
    def lead_times(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalized lead times on the step grid, 0 at the origin and 1 at the horizon."""
        return torch.linspace(
            0.0, 1.0, self.config.horizon_steps + 1, device=device, dtype=dtype
        )

    def rollout(self, history: torch.Tensor, origin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the latent forward over the whole horizon.

        Returns the ``[batch, horizon_steps + 1, latent]`` trajectory (index 0
        is the origin) and the history condition.
        """
        from torchdiffeq import odeint, odeint_adjoint

        config = self.config
        # Update once before any encoding so history, origin and targets share
        # the same latent coordinates throughout this forward pass.
        initial = self.encode_latent(origin, update_scale=True)
        history_latents = self.encode_latent(history)
        condition = self.history_encoder(history_latents)
        times = self.lead_times(initial.device, initial.dtype)
        solver = odeint_adjoint if config.dynamics_adjoint else odeint
        # The condition is a computed tensor, not a registered Parameter.
        # Explicitly include it so adjoint gradients reach the history GRU.
        adjoint_kwargs = (
            {"adjoint_params": (*self.dynamics.parameters(), condition)}
            if config.dynamics_adjoint else {}
        )
        trajectory = solver(
            _DynamicsField(self.dynamics, condition),
            initial,
            times,
            method=config.dynamics_solver,
            rtol=config.dynamics_rtol,
            atol=config.dynamics_atol,
            **adjoint_kwargs,
        )
        return trajectory.transpose(0, 1), condition

    # ---- flow matching at a lead time -------------------------------------
    def integrate_flow(
        self,
        condition: torch.Tensor,
        dynamics_latent: torch.Tensor,
        lead_time: torch.Tensor,
        *,
        integration_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        latent = torch.randn(
            condition.shape[0],
            self.config.latent_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        field = _FlowField(self.vector_field, condition, dynamics_latent, lead_time)
        if self.config.flow_solver != "midpoint":
            from torchdiffeq import odeint

            times = torch.linspace(
                0.0, 1.0, integration_steps + 1, device=latent.device, dtype=latent.dtype
            )
            return odeint(field, latent, times, method=self.config.flow_solver)[-1]
        step = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.as_tensor(index * step, device=latent.device, dtype=latent.dtype)
            velocity = field(time, latent)
            midpoint = latent + 0.5 * step * velocity
            latent = latent + step * field(time + 0.5 * step, midpoint)
        return latent

    # ---- training ---------------------------------------------------------
    def loss(
        self,
        history: torch.Tensor,
        origin: torch.Tensor,
        targets: torch.Tensor,
        config: DynamicsLossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        config = config or DynamicsLossConfig()
        batch_size = targets.shape[0]
        horizon = self.config.horizon_steps
        trajectory, condition = self.rollout(history, origin)

        target_latents = self.encode_latent(targets)
        reconstruction = self.decode_latent(target_latents)
        reconstruction_loss = F.mse_loss(reconstruction, targets)
        # The deterministic backbone: does the ODE reach the right latent?
        trajectory_loss = F.mse_loss(trajectory[:, 1:], target_latents)

        # Flow matching on a random subset of lead times keeps the step cost flat
        # in the horizon length.
        picks = torch.randint(
            0, horizon, (batch_size, min(config.flow_steps_per_batch, horizon)),
            device=targets.device,
        )
        rows = torch.arange(batch_size, device=targets.device)[:, None]
        chosen_target = target_latents[rows, picks].reshape(-1, self.config.latent_dim)
        chosen_dynamics = trajectory[:, 1:][rows, picks].reshape(-1, self.config.latent_dim)
        chosen_condition = condition[:, None, :].expand(
            -1, picks.shape[1], -1
        ).reshape(-1, self.config.hidden_dim)
        lead_grid = self.lead_times(targets.device, targets.dtype)[1:]
        chosen_lead = lead_grid[picks].reshape(-1)

        pairs = chosen_target.shape[0]
        source = torch.randn_like(chosen_target)
        flow_time = torch.rand(pairs, device=targets.device, dtype=targets.dtype)
        interpolated = (1.0 - flow_time[:, None]) * source + flow_time[:, None] * chosen_target
        predicted = self.vector_field(
            interpolated, flow_time, chosen_condition, chosen_dynamics, chosen_lead
        )
        flow_loss = F.mse_loss(predicted, chosen_target - source)
        latent_regularization = target_latents.square().mean()

        total = (
            config.reconstruction_weight * reconstruction_loss
            + config.trajectory_weight * trajectory_loss
            + config.flow_weight * flow_loss
            + config.latent_regularization_weight * latent_regularization
        )
        losses = {
            "reconstruction_mse": reconstruction_loss,
            "trajectory_mse": trajectory_loss,
            "flow_matching_mse": flow_loss,
            "latent_l2": latent_regularization,
        }
        if config.ensemble_enabled:
            members = config.ensemble_size
            index = int(torch.randint(0, picks.shape[1], (1,)).item())
            lead = chosen_lead.view(batch_size, -1)[:, index]
            samples = self.decode_latent(
                self.integrate_flow(
                    condition.repeat_interleave(members, dim=0),
                    chosen_dynamics.view(batch_size, -1, self.config.latent_dim)[
                        :, index
                    ].repeat_interleave(members, dim=0),
                    lead.repeat_interleave(members, dim=0),
                    integration_steps=config.ensemble_steps,
                )
            ).view(batch_size, members, -1)
            step_target = targets[rows, picks[:, index : index + 1]].squeeze(1)
            ensemble_crps = fair_ensemble_crps(samples, step_target)
            total = total + config.ensemble_weight * ensemble_crps
            losses["ensemble_crps"] = ensemble_crps
            losses["ensemble_spread"] = samples.std(dim=1).mean().detach()
        losses["loss"] = total
        return losses

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        history: torch.Tensor,
        origin: torch.Tensor,
        *,
        ensemble_size: int = 1,
        integration_steps: int = 32,
        lead_indices: list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, member, lead, state]`` samples on the physical grid."""
        if min(ensemble_size, integration_steps) < 1:
            raise ValueError("ensemble_size and integration_steps must be positive")
        leads = (
            list(range(self.config.horizon_steps))
            if lead_indices is None
            else list(lead_indices)
        )
        if not leads or any(lead < 0 or lead >= self.config.horizon_steps for lead in leads):
            raise ValueError("lead_indices must be nonempty and inside the trained horizon")
        trajectory, condition = self.rollout(history, origin)
        lead_grid = self.lead_times(origin.device, origin.dtype)[1:]
        members = []
        for _ in range(ensemble_size):
            steps = []
            for lead in leads:
                latent = self.integrate_flow(
                    condition,
                    trajectory[:, lead + 1],
                    lead_grid[lead].expand(condition.shape[0]),
                    integration_steps=integration_steps,
                    generator=generator,
                )
                steps.append(self.decode_latent(latent))
            members.append(torch.stack(steps, dim=1))
        return torch.stack(members, dim=1)

    @torch.no_grad()
    def deterministic_forecast(
        self, history: torch.Tensor, origin: torch.Tensor
    ) -> torch.Tensor:
        """Decode the ODE trajectory itself, without the stochastic head."""
        trajectory, _ = self.rollout(history, origin)
        return self.decode_latent(trajectory[:, 1:])

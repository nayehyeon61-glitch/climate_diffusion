"""Latent autoencoder and conditional vector field for monthly flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlowLossConfig, FlowModelConfig


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block used to deepen the state autoencoder."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim)
        self.project = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.project(F.silu(self.expand(self.norm(features))))
        return features + self.dropout(hidden)


class StateAutoencoder(nn.Module):
    """Compress a state vector to a latent code.

    ``blocks=0`` keeps the original three-layer MLP so earlier checkpoints load
    unchanged; a positive ``blocks`` stacks residual blocks at ``hidden_dim``
    width instead, which is what lets the latent grow without the encoder
    becoming the bottleneck.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        *,
        blocks: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks cannot be negative")
        if blocks == 0:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, state_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                *(ResidualBlock(hidden_dim, dropout) for _ in range(blocks)),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                *(ResidualBlock(hidden_dim, dropout) for _ in range(blocks)),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, state_dim),
            )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))


class GeoConv2d(nn.Module):
    """Conv2d that wraps in longitude and replicates at the poles.

    A global field is periodic east-west, so zero padding there invents an
    artificial seam at the date line; latitude has genuine boundaries instead.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        padded = F.pad(fields, (self.pad, self.pad, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(padded)


class ConvStateAutoencoder(nn.Module):
    """Grid-aware autoencoder over ``(channels, lat, lon)`` fields.

    The flat MLP encoder treats 2048 numbers as unrelated coordinates; this one
    keeps the map structure and shares weights across space, which is where the
    parameter budget actually buys reconstruction on gridded data.
    """

    def __init__(
        self,
        grid: tuple[int, int, int],
        latent_dim: int,
        hidden_dim: int,
        *,
        blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels, height, width = grid
        if height % 4 or width % 4:
            raise ValueError("Grid height and width must be divisible by 4")
        self.grid = grid
        self.width = hidden_dim
        reduced = (height // 4, width // 4)
        self.reduced = reduced

        self.stem = nn.Sequential(
            GeoConv2d(channels, hidden_dim // 2),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.AvgPool2d(2),
            GeoConv2d(hidden_dim // 2, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AvgPool2d(2),
        )
        self.encoder_blocks = nn.Sequential(
            *(
                nn.Sequential(
                    GeoConv2d(hidden_dim, hidden_dim),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                )
                for _ in range(blocks)
            )
        )
        self.to_latent = nn.Linear(hidden_dim * reduced[0] * reduced[1], latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim * reduced[0] * reduced[1])
        self.decoder_blocks = nn.Sequential(
            *(
                nn.Sequential(
                    GeoConv2d(hidden_dim, hidden_dim),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                )
                for _ in range(blocks)
            )
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            GeoConv2d(hidden_dim, hidden_dim // 2),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            GeoConv2d(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
            GeoConv2d(hidden_dim // 2, channels),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        leading = state.shape[:-1]
        fields = state.reshape(-1, *self.grid)
        features = self.encoder_blocks(self.stem(fields))
        latent = self.to_latent(features.flatten(1))
        return latent.reshape(*leading, -1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        leading = latent.shape[:-1]
        flat = latent.reshape(-1, latent.shape[-1])
        features = self.from_latent(flat).reshape(-1, self.width, *self.reduced)
        fields = self.head(self.decoder_blocks(features))
        return fields.reshape(*leading, -1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))


def build_autoencoder(config: FlowModelConfig) -> nn.Module:
    """Pick the autoencoder the config asks for; ``mlp`` stays the default."""
    if config.autoencoder_kind == "conv":
        return ConvStateAutoencoder(
            config.autoencoder_grid,
            config.latent_dim,
            config.resolved_autoencoder_hidden_dim,
            blocks=max(1, config.autoencoder_blocks),
            dropout=config.autoencoder_dropout,
        )
    return StateAutoencoder(
        config.state_dim,
        config.latent_dim,
        config.resolved_autoencoder_hidden_dim,
        blocks=config.autoencoder_blocks,
        dropout=config.autoencoder_dropout,
    )


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    if half == 0:
        return time[:, None]
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None, :] * 1000.0
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class ConditionalVectorField(nn.Module):
    def __init__(self, config: FlowModelConfig):
        super().__init__()
        self.config = config
        self.history_encoder = nn.GRU(
            config.latent_dim,
            config.hidden_dim,
            batch_first=True,
        )
        input_dim = config.latent_dim + config.hidden_dim + config.time_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def encode_condition(self, history_latents: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_encoder(history_latents)
        return hidden[-1]

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_features = sinusoidal_time_embedding(
            time, self.config.time_embedding_dim
        )
        return self.network(torch.cat((latent, condition, time_features), dim=-1))


LATENT_SCALE_MOMENTUM = 0.01
LATENT_SCALE_FLOOR = 1e-3


def fair_ensemble_crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-dimension CRPS of ``[batch, member, state]`` samples against a target.

    Uses the fair (unbiased) K(K-1) pairwise denominator, so a small training
    ensemble is not rewarded for being under-dispersed the way the biased K^2
    estimator would.
    """
    members = samples.shape[1]
    if members < 2:
        raise ValueError("fair CRPS needs at least two members")
    accuracy = (samples - target[:, None, :]).abs().mean()
    pairwise = (samples[:, :, None, :] - samples[:, None, :, :]).abs().sum(
        dim=(1, 2)
    ) / (members * (members - 1))
    return accuracy - 0.5 * pairwise.mean()


class _FlowTimeField(nn.Module):
    """Adapt the conditional vector field to the ``func(t, y)`` ODE signature.

    ``odeint_adjoint`` walks ``parameters()`` to build the backward solve, so
    this has to be a Module holding the field rather than a closure.
    """

    def __init__(self, vector_field: ConditionalVectorField, condition: torch.Tensor):
        super().__init__()
        self.vector_field = vector_field
        self.condition = condition

    def forward(self, time: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        expanded = time.expand(latent.shape[0]).to(latent.dtype)
        return self.vector_field(latent, expanded, self.condition)


class MonthlyLatentFlow(nn.Module):
    """Conditional flow matcher from Gaussian latent noise to next-month state."""

    def __init__(self, config: FlowModelConfig):
        super().__init__()
        self.config = config
        self.autoencoder = build_autoencoder(config)
        self.vector_field = ConditionalVectorField(config)
        if config.latent_normalization:
            # Flow matching transports N(0, I) onto the encoder's codes, so the
            # latent has to sit at unit scale. An expressive decoder is free to
            # shrink it far below the prior instead, and every velocity error is
            # then magnified relative to the target's own spread. The buffer
            # tracks the observed scale so the flow always sees unit variance.
            self.register_buffer("latent_scale", torch.ones(()))
            self.register_buffer("latent_scale_count", torch.zeros((), dtype=torch.long))

    def encode_latent(self, state: torch.Tensor, *, update_scale: bool = False) -> torch.Tensor:
        latent = self.autoencoder.encode(state)
        if not self.config.latent_normalization:
            return latent
        if update_scale and self.training:
            observed = latent.detach().std().clamp_min(LATENT_SCALE_FLOOR)
            if self.latent_scale_count == 0:
                # Seed from the first batch; an EMA started at 1.0 would take
                # thousands of steps to walk down to a latent two orders of
                # magnitude smaller.
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

    def loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        config: FlowLossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        config = config or FlowLossConfig()
        batch_size = target.shape[0]
        target_latent = self.encode_latent(target, update_scale=True)
        history_latents = self.encode_latent(history)
        reconstruction = self.decode_latent(target_latent)
        source_latent = torch.randn_like(target_latent)
        time = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        interpolated = (
            (1.0 - time[:, None]) * source_latent
            + time[:, None] * target_latent
        )
        target_velocity = target_latent - source_latent
        condition = self.vector_field.encode_condition(history_latents)
        predicted_velocity = self.vector_field(interpolated, time, condition)

        reconstruction_loss = F.mse_loss(reconstruction, target)
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        latent_regularization = target_latent.square().mean()
        total = (
            config.reconstruction_weight * reconstruction_loss
            + config.flow_weight * flow_loss
            + config.latent_regularization_weight * latent_regularization
        )
        losses = {
            "reconstruction_mse": reconstruction_loss,
            "flow_matching_mse": flow_loss,
            "latent_l2": latent_regularization,
        }
        if config.ensemble_enabled:
            # Score an actual generated ensemble, not a single noise draw, so
            # spread is trained rather than left to fall out of the flow fit.
            members = config.ensemble_size
            samples = self.decode_latent(
                self.integrate(
                    condition.repeat_interleave(members, dim=0),
                    integration_steps=config.ensemble_steps,
                )
            ).view(batch_size, members, -1)
            ensemble_crps = fair_ensemble_crps(samples, target)
            total = total + config.ensemble_weight * ensemble_crps
            losses["ensemble_crps"] = ensemble_crps
            losses["ensemble_spread"] = samples.std(dim=1).mean().detach()
        losses["loss"] = total
        return losses

    def integrate(
        self,
        condition: torch.Tensor,
        *,
        integration_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate the probability-flow ODE over flow time tau in [0, 1].

        Note that tau is the noise-to-data interpolation variable, not physical
        forecast time; the forecast step is one archive interval regardless of
        how finely this is solved.

        Kept free of ``no_grad`` so the ensemble scoring rule in :meth:`loss`
        can differentiate through the rollout; :meth:`sample` wraps it for
        inference. ``flow_solver="midpoint"`` keeps the original hand-rolled
        loop; any other value routes to torchdiffeq, where ``flow_adjoint``
        trades a second backward solve for O(1) memory in the step count.
        """
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        latent = torch.randn(
            condition.shape[0],
            self.config.latent_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        if self.config.flow_solver != "midpoint":
            return self._odeint_flow(latent, condition, integration_steps)
        step = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.full(
                (condition.shape[0],),
                index * step,
                device=condition.device,
                dtype=condition.dtype,
            )
            first_velocity = self.vector_field(latent, time, condition)
            midpoint = latent + 0.5 * step * first_velocity
            latent = latent + step * self.vector_field(
                midpoint, time + 0.5 * step, condition
            )
        return latent

    def _odeint_flow(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        integration_steps: int,
    ) -> torch.Tensor:
        from torchdiffeq import odeint, odeint_adjoint

        config = self.config
        field = _FlowTimeField(self.vector_field, condition)
        if config.adaptive_flow_solver:
            times = torch.tensor([0.0, 1.0], device=latent.device, dtype=latent.dtype)
        else:
            times = torch.linspace(
                0.0, 1.0, integration_steps + 1, device=latent.device, dtype=latent.dtype
            )
        solver = odeint_adjoint if config.flow_adjoint else odeint
        trajectory = solver(
            field,
            latent,
            times,
            method=config.flow_solver,
            rtol=config.flow_solver_rtol,
            atol=config.flow_solver_atol,
        )
        return trajectory[-1]

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        *,
        integration_steps: int = 32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.vector_field.encode_condition(self.encode_latent(history))
        latent = self.integrate(
            condition, integration_steps=integration_steps, generator=generator
        )
        return self.decode_latent(latent)

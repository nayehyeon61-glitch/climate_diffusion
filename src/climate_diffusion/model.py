"""Latent autoencoder and conditional vector field for monthly flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlowLossConfig, FlowModelConfig
from .spatial import (
    PeriodicConv2d,
    SpatialAutoencoder,
    SpatialResidualBlock,
    SpectralOperator2d,
    tiled_apply,
)


class StateAutoencoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
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

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))


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


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gates = PeriodicConv2d(channels * 2, channels * 2)
        self.candidate = PeriodicConv2d(channels * 2, channels)

    def forward(self, values: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((values, hidden), dim=1)).chunk(2, dim=1)
        reset, update = reset.sigmoid(), update.sigmoid()
        candidate = torch.tanh(self.candidate(torch.cat((values, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class SpatialConditionalVectorField(nn.Module):
    """ConvGRU-conditioned vector field over a spatial latent grid."""

    def __init__(self, config: FlowModelConfig):
        super().__init__()
        channels = config.spatial_latent_channels
        self.config = config
        self.history_cell = ConvGRUCell(channels)
        self.time_projection = nn.Linear(config.time_embedding_dim, channels)
        self.auxiliary_encoder = (
            nn.GRU(config.auxiliary_dim, config.hidden_dim, batch_first=True)
            if config.auxiliary_dim
            else None
        )
        self.auxiliary_projection = (
            nn.Linear(config.hidden_dim, channels) if config.auxiliary_dim else None
        )
        self.input_projection = PeriodicConv2d(channels * 3, channels)
        self.residual = SpatialResidualBlock(channels)
        self.operator = (
            SpectralOperator2d(
                channels, config.operator_modes_lat, config.operator_modes_lon
            )
            if config.backend == "spatial_operator"
            else nn.Identity()
        )
        self.output_projection = PeriodicConv2d(channels, channels)

    def encode_condition(
        self,
        history_latents: torch.Tensor,
        history_auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = torch.zeros_like(history_latents[:, 0])
        for index in range(history_latents.shape[1]):
            hidden = self.history_cell(history_latents[:, index], hidden)
        if self.auxiliary_encoder is not None:
            if history_auxiliary is None:
                raise ValueError("Spatial checkpoint requires auxiliary history")
            _, auxiliary_hidden = self.auxiliary_encoder(history_auxiliary)
            hidden = hidden + self.auxiliary_projection(auxiliary_hidden[-1])[:, :, None, None]
        return hidden

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_projection(
            sinusoidal_time_embedding(time, self.config.time_embedding_dim)
        )[:, :, None, None]
        time_features = time_features.expand(-1, -1, *latent.shape[-2:])
        values = self.input_projection(torch.cat((latent, condition, time_features), dim=1))
        values = self.residual(values)
        values = self.operator(values)
        return self.output_projection(F.silu(values))


class MonthlyLatentFlow(nn.Module):
    """Conditional flow matcher from Gaussian latent noise to next-month state."""

    def __init__(self, config: FlowModelConfig):
        super().__init__()
        self.config = config
        if config.backend == "vector_mlp":
            self.autoencoder = StateAutoencoder(
                config.state_dim, config.latent_dim, config.hidden_dim
            )
            self.vector_field = ConditionalVectorField(config)
        else:
            modes = (
                (config.operator_modes_lat, config.operator_modes_lon)
                if config.backend == "spatial_operator"
                else None
            )
            self.autoencoder = SpatialAutoencoder(
                config.spatial_channels,
                config.spatial_latent_channels,
                config.spatial_base_channels,
                config.spatial_downsample_levels,
                operator_modes=modes,
                gradient_checkpointing=config.gradient_checkpointing,
            )
            self.vector_field = SpatialConditionalVectorField(config)

    @property
    def is_spatial(self) -> bool:
        return self.config.backend != "vector_mlp"

    def loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        config: FlowLossConfig | None = None,
        history_auxiliary: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        config = config or FlowLossConfig()
        batch_size = target.shape[0]
        if self.is_spatial:
            if history.ndim != 5 or target.ndim != 4:
                raise ValueError("Spatial flow expects history [B,T,C,H,W] and target [B,C,H,W]")
            batch, months, channels, height, width = history.shape
            encoded_history = self.autoencoder.encode(
                history.reshape(batch * months, channels, height, width)
            )
            history_latents = encoded_history.reshape(
                batch, months, *encoded_history.shape[1:]
            )
            target_latent = self.autoencoder.encode(target)
            reconstruction = self.autoencoder.decode(target_latent, target.shape[-2:])
            time_shape = (batch_size, 1, 1, 1)
        else:
            history_latents = self.autoencoder.encode(history)
            target_latent = self.autoencoder.encode(target)
            reconstruction = self.autoencoder.decode(target_latent)
            time_shape = (batch_size, 1)
        source_latent = torch.randn_like(target_latent)
        time = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        interpolation_time = time.reshape(time_shape)
        interpolated = (1.0 - interpolation_time) * source_latent + interpolation_time * target_latent
        target_velocity = target_latent - source_latent
        if self.is_spatial:
            condition = self.vector_field.encode_condition(history_latents, history_auxiliary)
        else:
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
        return {
            "loss": total,
            "reconstruction_mse": reconstruction_loss,
            "flow_matching_mse": flow_loss,
            "latent_l2": latent_regularization,
        }

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        *,
        integration_steps: int = 32,
        generator: torch.Generator | None = None,
        history_auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if self.is_spatial:
            batch, months, channels, height, width = history.shape
            encoded_history = self.autoencoder.encode(
                history.reshape(batch * months, channels, height, width)
            )
            history_latents = encoded_history.reshape(
                batch, months, *encoded_history.shape[1:]
            )
            condition = self.vector_field.encode_condition(history_latents, history_auxiliary)
            latent_shape = (batch, *history_latents.shape[2:])
        else:
            history_latents = self.autoencoder.encode(history)
            condition = self.vector_field.encode_condition(history_latents)
            latent_shape = (history.shape[0], self.config.latent_dim)
        latent = torch.randn(
            *latent_shape,
            device=history.device,
            dtype=history.dtype,
            generator=generator,
        )
        step = 1.0 / integration_steps
        for index in range(integration_steps):
            time = torch.full(
                (history.shape[0],),
                index * step,
                device=history.device,
                dtype=history.dtype,
            )
            first_velocity = self.vector_field(latent, time, condition)
            midpoint = latent + 0.5 * step * first_velocity
            midpoint_time = time + 0.5 * step
            latent = latent + step * self.vector_field(
                midpoint, midpoint_time, condition
            )
        if self.is_spatial:
            return self.autoencoder.decode(latent, history.shape[-2:])
        return self.autoencoder.decode(latent)

    @torch.no_grad()
    def sample_tiled(
        self,
        history: torch.Tensor,
        *,
        tile_size: tuple[int, int],
        overlap: int,
        integration_steps: int = 32,
        generator: torch.Generator | None = None,
        history_auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_spatial:
            raise ValueError("Tiled sampling is only available for spatial backends")

        def predict(patch: torch.Tensor) -> torch.Tensor:
            return self.sample(
                patch,
                integration_steps=integration_steps,
                generator=generator,
                history_auxiliary=history_auxiliary,
            )

        return tiled_apply(history, predict, tile_size=tile_size, overlap=overlap)

"""Latent autoencoder and conditional vector field for monthly flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlowLossConfig, FlowModelConfig


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


class MonthlyLatentFlow(nn.Module):
    """Conditional flow matcher from Gaussian latent noise to next-month state."""

    def __init__(self, config: FlowModelConfig):
        super().__init__()
        self.config = config
        self.autoencoder = StateAutoencoder(
            config.state_dim, config.latent_dim, config.hidden_dim
        )
        self.vector_field = ConditionalVectorField(config)

    def loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        config: FlowLossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        config = config or FlowLossConfig()
        batch_size = target.shape[0]
        history_latents = self.autoencoder.encode(history)
        target_latent = self.autoencoder.encode(target)
        reconstruction = self.autoencoder.decode(target_latent)
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
    ) -> torch.Tensor:
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        history_latents = self.autoencoder.encode(history)
        condition = self.vector_field.encode_condition(history_latents)
        latent = torch.randn(
            history.shape[0],
            self.config.latent_dim,
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
        return self.autoencoder.decode(latent)

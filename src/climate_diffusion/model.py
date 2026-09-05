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
    set_longitude_wrap,
    tiled_apply,
)
from .validation import require_finite_tensor


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
                input_channels=config.encoder_input_channels,
            )
            self.vector_field = SpatialConditionalVectorField(config)

    @property
    def is_spatial(self) -> bool:
        return self.config.backend != "vector_mlp"

    def _configure_longitude_wrap(self, width: int) -> None:
        """Wrap longitude only when the input actually spans the whole grid."""
        if self.is_spatial:
            set_longitude_wrap(self, width == self.config.grid_width)

    def _with_coordinates(
        self, values: torch.Tensor, coordinates: torch.Tensor | None
    ) -> torch.Tensor:
        """Append the static coordinate planes the encoder expects."""
        if not self.config.positional_channels:
            return values
        if coordinates is None:
            raise ValueError(
                "This checkpoint encodes positional channels; coordinates are required"
            )
        if coordinates.shape[-3] != self.config.positional_channels:
            raise ValueError(
                f"Expected {self.config.positional_channels} positional channels, "
                f"received {coordinates.shape[-3]}"
            )
        if coordinates.shape[-2:] != values.shape[-2:]:
            raise ValueError(
                f"Coordinates cover {tuple(coordinates.shape[-2:])} but the field is "
                f"{tuple(values.shape[-2:])}"
            )
        expanded = coordinates.expand(values.shape[0], -1, -1, -1)
        return torch.cat((values, expanded), dim=1)

    def _latent_extent(self, size: int) -> int:
        """Latent rows/columns the encoder produces for a spatial extent.

        Each downsample level is a stride-2 padded convolution, which rounds up.
        """
        for _ in range(self.config.spatial_downsample_levels):
            size = -(-size // 2)
        return size

    def _draw_flow_pair(
        self,
        target_latent: torch.Tensor,
        batch_size: int,
        time_dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the flow-matching source latent and interpolation time.

        With a generator the draw is reproducible and the time is stratified
        over ``[0, 1)`` instead of independent per sample, so an evaluation pass
        estimates the flow loss with far less Monte-Carlo noise. Training keeps
        independent draws.
        """
        device = target_latent.device
        if generator is None:
            return torch.randn_like(target_latent), torch.rand(
                batch_size, device=device, dtype=time_dtype
            )
        source_latent = torch.randn(
            target_latent.shape,
            device=device,
            dtype=target_latent.dtype,
            generator=generator,
        )
        offset = torch.rand(1, device=device, dtype=time_dtype, generator=generator)
        strata = torch.arange(batch_size, device=device, dtype=time_dtype)
        return source_latent, (strata + offset) / batch_size

    def loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        config: FlowLossConfig | None = None,
        history_auxiliary: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        config = config or FlowLossConfig()
        require_finite_tensor(history, "flow loss history input")
        require_finite_tensor(target, "flow loss target input")
        if history_auxiliary is not None:
            require_finite_tensor(history_auxiliary, "flow loss auxiliary input")
        batch_size = target.shape[0]
        if self.is_spatial:
            if history.ndim != 5 or target.ndim != 4:
                raise ValueError("Spatial flow expects history [B,T,C,H,W] and target [B,C,H,W]")
            self._configure_longitude_wrap(target.shape[-1])
            batch, months, channels, height, width = history.shape
            encoded_history = self.autoencoder.encode(
                self._with_coordinates(
                    history.reshape(batch * months, channels, height, width), coordinates
                )
            )
            history_latents = encoded_history.reshape(
                batch, months, *encoded_history.shape[1:]
            )
            target_latent = self.autoencoder.encode(
                self._with_coordinates(target, coordinates)
            )
            reconstruction = self.autoencoder.decode(target_latent, target.shape[-2:])
            time_shape = (batch_size, 1, 1, 1)
        else:
            history_latents = self.autoencoder.encode(history)
            target_latent = self.autoencoder.encode(target)
            reconstruction = self.autoencoder.decode(target_latent)
            time_shape = (batch_size, 1)
        require_finite_tensor(history_latents, "encoded history latent")
        require_finite_tensor(target_latent, "encoded target latent")
        require_finite_tensor(reconstruction, "decoded target reconstruction")
        # The flow branch trains on a fixed target: without the detach the
        # encoder also learns to make its own latents easy to predict, which
        # together with the latent penalty pushes towards latent collapse.
        flow_target_latent = target_latent.detach()
        source_latent, time = self._draw_flow_pair(
            flow_target_latent, batch_size, target.dtype, generator
        )
        interpolation_time = time.reshape(time_shape)
        interpolated = (
            1.0 - interpolation_time
        ) * source_latent + interpolation_time * flow_target_latent
        target_velocity = flow_target_latent - source_latent
        if self.is_spatial:
            condition = self.vector_field.encode_condition(history_latents, history_auxiliary)
        else:
            condition = self.vector_field.encode_condition(history_latents)
        predicted_velocity = self.vector_field(interpolated, time, condition)
        require_finite_tensor(condition, "flow history condition")
        require_finite_tensor(predicted_velocity, "predicted flow velocity")

        if target_mask is None:
            reconstruction_loss = F.mse_loss(reconstruction, target)
        else:
            if target_mask.shape != target.shape:
                raise ValueError("target_mask must have the same shape as target")
            weights = target_mask.to(dtype=target.dtype)
            if not bool(weights.sum() > 0):
                raise ValueError("target_mask contains no observed values")
            reconstruction_loss = ((reconstruction - target).square() * weights).sum() / weights.sum()
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        latent_regularization = target_latent.square().mean()
        total = (
            config.reconstruction_weight * reconstruction_loss
            + config.flow_weight * flow_loss
            + config.latent_regularization_weight * latent_regularization
        )
        require_finite_tensor(reconstruction_loss, "reconstruction loss")
        require_finite_tensor(flow_loss, "flow matching loss")
        require_finite_tensor(latent_regularization, "latent regularization loss")
        require_finite_tensor(total, "total flow loss")
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
        noise: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        require_finite_tensor(history, "flow sampling history")
        if history_auxiliary is not None:
            require_finite_tensor(history_auxiliary, "flow sampling auxiliary history")
        if self.is_spatial:
            self._configure_longitude_wrap(history.shape[-1])
            batch, months, channels, height, width = history.shape
            encoded_history = self.autoencoder.encode(
                self._with_coordinates(
                    history.reshape(batch * months, channels, height, width), coordinates
                )
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
        if noise is None:
            latent = torch.randn(
                *latent_shape,
                device=history.device,
                dtype=history.dtype,
                generator=generator,
            )
        else:
            if tuple(noise.shape) != tuple(latent_shape):
                raise ValueError(
                    f"Expected noise shape {tuple(latent_shape)}, "
                    f"received {tuple(noise.shape)}"
                )
            latent = noise.to(device=history.device, dtype=history.dtype)
        require_finite_tensor(latent, "flow sampling initial latent")
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
            require_finite_tensor(latent, f"flow sampling latent step={index}")
        if self.is_spatial:
            output = self.autoencoder.decode(latent, history.shape[-2:])
        else:
            output = self.autoencoder.decode(latent)
        require_finite_tensor(output, "flow sampling decoded output")
        return output

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
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_spatial:
            raise ValueError("Tiled sampling is only available for spatial backends")
        height, width = history.shape[-2:]
        factor = 2**self.config.spatial_downsample_levels
        # One noise field for the whole globe, sliced per tile. Drawing per tile
        # instead would make each ensemble member a mosaic of independent
        # samples, and the overlap blend would average them and flatten the
        # spread exactly where tiles meet.
        latent_height, latent_width = self._latent_extent(height), self._latent_extent(width)
        global_noise = torch.randn(
            history.shape[0],
            self.config.spatial_latent_channels,
            latent_height,
            latent_width,
            device=history.device,
            dtype=history.dtype,
            generator=generator,
        )

        data_channels = history.shape[2]
        if coordinates is not None:
            # Ride along as extra history channels so tiled_apply's own cropping
            # slices the coordinate planes exactly like the data.
            planes = coordinates.expand(history.shape[0], -1, -1, -1)
            history = torch.cat(
                (history, planes.unsqueeze(1).expand(-1, history.shape[1], -1, -1, -1)),
                dim=2,
            )

        def predict(patch: torch.Tensor, lat_start: int, lon_start: int) -> torch.Tensor:
            tile_coordinates = None
            if coordinates is not None:
                tile_coordinates = patch[:, 0, data_channels:]
                patch = patch[:, :, :data_channels]
            rows = self._latent_extent(patch.shape[-2])
            columns = self._latent_extent(patch.shape[-1])
            # Latitude cannot wrap, so keep the slice inside the noise field.
            top = min(lat_start // factor, latent_height - rows)
            row_index = torch.arange(top, top + rows, device=history.device)
            column_index = torch.arange(
                lon_start // factor, lon_start // factor + columns, device=history.device
            ).remainder(latent_width)
            noise = global_noise.index_select(-2, row_index).index_select(-1, column_index)
            return self.sample(
                patch,
                integration_steps=integration_steps,
                history_auxiliary=history_auxiliary,
                noise=noise,
                coordinates=tile_coordinates,
            )

        return tiled_apply(history, predict, tile_size=tile_size, overlap=overlap)

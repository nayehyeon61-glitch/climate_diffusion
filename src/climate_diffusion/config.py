from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FlowBackend = Literal["vector_mlp", "spatial_conv", "spatial_operator"]


@dataclass(frozen=True)
class FlowModelConfig:
    state_dim: int
    history_months: int = 6
    latent_dim: int = 64
    hidden_dim: int = 256
    time_embedding_dim: int = 32
    backend: FlowBackend = "vector_mlp"
    spatial_channels: int = 0
    grid_height: int = 0
    grid_width: int = 0
    spatial_latent_channels: int = 16
    spatial_base_channels: int = 32
    spatial_downsample_levels: int = 3
    operator_modes_lat: int = 12
    operator_modes_lon: int = 24
    auxiliary_dim: int = 0
    gradient_checkpointing: bool = False
    # Static sin(lat)/cos(lon)/sin(lon) planes appended to the encoder input.
    # Defaults to 0 so checkpoints written before they existed still load.
    positional_channels: int = 0

    @property
    def encoder_input_channels(self) -> int:
        return self.spatial_channels + self.positional_channels

    def __post_init__(self) -> None:
        if self.backend not in {"vector_mlp", "spatial_conv", "spatial_operator"}:
            raise ValueError(f"Unsupported model backend: {self.backend}")
        if min(
            self.state_dim,
            self.history_months,
            self.latent_dim,
            self.hidden_dim,
            self.time_embedding_dim,
        ) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.auxiliary_dim < 0:
            raise ValueError("auxiliary_dim cannot be negative")
        if self.positional_channels not in {0, 3}:
            raise ValueError("positional_channels must be 0 or 3")
        if self.backend != "vector_mlp":
            if min(
                self.spatial_channels,
                self.grid_height,
                self.grid_width,
                self.spatial_latent_channels,
                self.spatial_base_channels,
                self.operator_modes_lat,
                self.operator_modes_lon,
            ) < 1:
                raise ValueError("Spatial backend dimensions must be positive")
            if self.spatial_downsample_levels < 0:
                raise ValueError("spatial_downsample_levels cannot be negative")


@dataclass(frozen=True)
class FlowLossConfig:
    reconstruction_weight: float = 1.0
    flow_weight: float = 1.0
    latent_regularization_weight: float = 1e-4

    def __post_init__(self) -> None:
        if min(
            self.reconstruction_weight,
            self.flow_weight,
            self.latent_regularization_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")

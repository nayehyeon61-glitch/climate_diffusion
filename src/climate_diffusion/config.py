from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowModelConfig:
    state_dim: int
    history_months: int = 6
    latent_dim: int = 64
    hidden_dim: int = 256
    time_embedding_dim: int = 32

    def __post_init__(self) -> None:
        if min(
            self.state_dim,
            self.history_months,
            self.latent_dim,
            self.hidden_dim,
            self.time_embedding_dim,
        ) < 1:
            raise ValueError("All model dimensions must be positive")


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

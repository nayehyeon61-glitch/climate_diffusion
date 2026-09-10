from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowModelConfig:
    state_dim: int
    history_months: int = 6
    latent_dim: int = 64
    hidden_dim: int = 256
    time_embedding_dim: int = 32
    autoencoder_hidden_dim: int | None = None
    autoencoder_blocks: int = 0
    autoencoder_dropout: float = 0.0
    autoencoder_kind: str = "mlp"
    autoencoder_grid: tuple[int, int, int] | None = None
    latent_normalization: bool = False
    flow_solver: str = "midpoint"
    flow_solver_rtol: float = 1e-5
    flow_solver_atol: float = 1e-5
    flow_adjoint: bool = False

    def __post_init__(self) -> None:
        if min(
            self.state_dim,
            self.history_months,
            self.latent_dim,
            self.hidden_dim,
            self.time_embedding_dim,
        ) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.autoencoder_hidden_dim is not None and self.autoencoder_hidden_dim < 1:
            raise ValueError("autoencoder_hidden_dim must be positive")
        if self.autoencoder_blocks < 0:
            raise ValueError("autoencoder_blocks cannot be negative")
        if not 0.0 <= self.autoencoder_dropout < 1.0:
            raise ValueError("autoencoder_dropout must be in [0, 1)")
        if min(self.flow_solver_rtol, self.flow_solver_atol) <= 0:
            raise ValueError("Solver tolerances must be positive")
        if self.autoencoder_kind not in {"mlp", "conv"}:
            raise ValueError("autoencoder_kind must be 'mlp' or 'conv'")
        if self.autoencoder_grid is not None:
            # asdict/JSON round-trips this as a list; keep it hashable.
            grid = tuple(int(value) for value in self.autoencoder_grid)
            if len(grid) != 3 or min(grid) < 1:
                raise ValueError("autoencoder_grid must be (channels, lat, lon)")
            if grid[0] * grid[1] * grid[2] != self.state_dim:
                raise ValueError("autoencoder_grid does not multiply to state_dim")
            object.__setattr__(self, "autoencoder_grid", grid)
        elif self.autoencoder_kind == "conv":
            raise ValueError("autoencoder_kind='conv' needs autoencoder_grid")

    @property
    def adaptive_flow_solver(self) -> bool:
        """Whether the flow solver picks its own steps (tolerance-driven)."""
        return self.flow_solver in {"dopri5", "dopri8", "bosh3", "adaptive_heun"}

    @property
    def resolved_autoencoder_hidden_dim(self) -> int:
        """Width of the state autoencoder, defaulting to the vector-field width."""
        return self.autoencoder_hidden_dim or self.hidden_dim


@dataclass(frozen=True)
class FlowLossConfig:
    reconstruction_weight: float = 1.0
    flow_weight: float = 1.0
    latent_regularization_weight: float = 1e-4
    ensemble_weight: float = 0.0
    ensemble_size: int = 0
    ensemble_steps: int = 4

    def __post_init__(self) -> None:
        if min(
            self.reconstruction_weight,
            self.flow_weight,
            self.latent_regularization_weight,
            self.ensemble_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")
        if self.ensemble_size < 0:
            raise ValueError("ensemble_size cannot be negative")
        if self.ensemble_size == 1:
            raise ValueError("ensemble_size must be 0 (off) or at least 2")
        if self.ensemble_steps < 1:
            raise ValueError("ensemble_steps must be positive")

    @property
    def ensemble_enabled(self) -> bool:
        return self.ensemble_weight > 0.0 and self.ensemble_size >= 2

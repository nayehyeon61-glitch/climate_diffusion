"""Select the original WeatherNext runner or the monthly flow replacement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import xarray as xr

from .weather_adapter import FlowMatchingWeatherRunner


class ForecastRunner(Protocol):
    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset: ...


class ForecastBackend(str, Enum):
    WEATHERNEXT = "weathernext"
    FLOW_MATCHING = "flow_matching"


@dataclass(frozen=True)
class ForecastSelectionConfig:
    backend: ForecastBackend | str = ForecastBackend.WEATHERNEXT
    flow_checkpoint: str | None = None
    integration_steps: int = 32
    seed: int = 0

    @property
    def backend_type(self) -> ForecastBackend:
        try:
            return ForecastBackend(self.backend)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ForecastBackend)
            raise ValueError(f"Unknown forecast backend; choose {choices}") from exc


def build_forecast_runner(
    config: ForecastSelectionConfig,
    *,
    weathernext_runner: ForecastRunner | None = None,
) -> ForecastRunner:
    """Keep the original WeatherNext object untouched and select one boundary."""
    if config.backend_type is ForecastBackend.WEATHERNEXT:
        if weathernext_runner is None:
            raise ValueError("weathernext backend requires an existing WeatherNext runner")
        return weathernext_runner
    if not config.flow_checkpoint:
        raise ValueError("flow_matching backend requires flow_checkpoint")
    checkpoint = Path(config.flow_checkpoint).expanduser()
    return FlowMatchingWeatherRunner(
        checkpoint,
        integration_steps=config.integration_steps,
        seed=config.seed,
    )

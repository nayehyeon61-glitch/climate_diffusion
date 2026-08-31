"""WeatherNext-compatible rollout adapter backed by monthly latent flow matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import reconstruct_dataset, vectorize_dataset
from .inference import LatentFlowForecaster

HOURS_PER_MODEL_MONTH = 30 * 24


class FlowMatchingWeatherRunner:
    """Inference-only replacement implementing rollout(initial_state, horizon_hours)."""

    inference_only = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        integration_steps: int = 32,
        seed: int = 0,
        device: str | None = None,
    ):
        self.forecaster = LatentFlowForecaster(checkpoint_path, device=device)
        self.checkpoint_path = self.forecaster.checkpoint_path
        self.integration_steps = integration_steps
        self.seed = seed

    def provenance(self) -> dict[str, object]:
        return {
            "forecast_backend": "flow_matching",
            "forecast_checkpoint": str(self.checkpoint_path),
            "forecast_step_hours": HOURS_PER_MODEL_MONTH,
            "weather_next_replacement": True,
            "inference_only": True,
        }

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        if horizon_hours <= 0 or horizon_hours % HOURS_PER_MODEL_MONTH:
            raise ValueError(
                "FlowMatchingWeatherRunner requires positive 720-hour (30-day) multiples"
            )
        months = horizon_hours // HOURS_PER_MODEL_MONTH
        if "time" not in initial_state.coords:
            raise ValueError("Monthly flow initial state requires a time coordinate")
        monthly_state = initial_state.sortby("time").resample(time="MS").mean(skipna=True)
        vectors = vectorize_dataset(
            monthly_state,
            self.forecaster.schema,
            integrated_defaults=np.asarray(
                self.forecaster.state_mean.detach().cpu(), dtype=np.float32
            ),
        )
        required = self.forecaster.config.history_months
        if vectors.shape[0] < required:
            raise ValueError(
                f"Monthly flow checkpoint requires {required} monthly history states; "
                f"received {vectors.shape[0]}"
            )
        prediction = self.forecaster.forecast(
            vectors[-required:],
            months=months,
            ensemble_size=1,
            integration_steps=self.integration_steps,
            seed=self.seed,
        )[0]
        last_time = pd.Timestamp(monthly_state.time.values[-1])
        outputs = [
            reconstruct_dataset(
                prediction[index],
                self.forecaster.schema,
                last_time + pd.DateOffset(months=index + 1),
            )
            for index in range(months)
        ]
        return xr.concat(outputs, dim="time").assign_attrs(self.provenance())

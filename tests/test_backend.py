import inspect

import pytest

from climate_diffusion.backend import ForecastSelectionConfig, build_forecast_runner
from climate_diffusion.weather_adapter import FlowMatchingWeatherRunner


class FakeWeatherNextRunner:
    def rollout(self, initial_state, horizon_hours):
        return initial_state


def test_original_weathernext_runner_is_preserved():
    original = FakeWeatherNextRunner()
    selected = build_forecast_runner(
        ForecastSelectionConfig(backend="weathernext"),
        weathernext_runner=original,
    )
    assert selected is original


def test_flow_backend_requires_checkpoint_and_has_no_fit():
    with pytest.raises(ValueError, match="flow_checkpoint"):
        build_forecast_runner(ForecastSelectionConfig(backend="flow_matching"))
    assert FlowMatchingWeatherRunner.inference_only is True
    assert not hasattr(FlowMatchingWeatherRunner, "fit")
    source = inspect.getsource(FlowMatchingWeatherRunner)
    assert "optimizer" not in source
    assert "backward" not in source

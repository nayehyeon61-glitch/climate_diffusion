import numpy as np
import pytest

from climate_diffusion.evaluation import (
    _anomaly_correlation,
    _ensemble_crps,
    _error_metrics,
    _skill_score,
    latitude_weights,
    seasonal_climatology,
)


def _spatial_schema(latitudes):
    return {"layout": "spatial", "coords": {"lat": list(latitudes), "lon": [0.0, 90.0]}}


def test_latitude_weights_follow_cosine_and_skip_vector_archives():
    assert latitude_weights({"layout": "vector"}) is None
    weights = latitude_weights(_spatial_schema([90.0, 60.0, 0.0, -60.0, -90.0]))
    assert weights.shape == (1, 5, 1)
    flat = weights.reshape(-1)
    np.testing.assert_allclose(flat[2], 1.0, atol=1e-12)
    np.testing.assert_allclose(flat[1], 0.5, atol=1e-12)
    np.testing.assert_allclose(flat[1], flat[3], atol=1e-12)
    # A pole row is clipped off zero so it stays representable, not dominant.
    assert 0 < flat[0] < 1e-5


def test_unweighted_metrics_match_the_plain_mean():
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros_like(prediction)
    metrics = _error_metrics(prediction, target)
    error = prediction.reshape(-1)
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(np.square(error).mean())))
    assert metrics["mae"] == pytest.approx(float(np.abs(error).mean()))
    assert metrics["bias"] == pytest.approx(float(error.mean()))


def test_latitude_weighting_discounts_a_polar_error():
    """The same error at the pole must count for far less than at the equator."""
    schema = _spatial_schema([90.0, 0.0])
    weights = latitude_weights(schema)
    target = np.zeros((1, 2, 2))

    polar = np.zeros_like(target)
    polar[:, 0, :] = 10.0
    equatorial = np.zeros_like(target)
    equatorial[:, 1, :] = 10.0

    polar_rmse = _error_metrics(polar, target, weights)["rmse"]
    equatorial_rmse = _error_metrics(equatorial, target, weights)["rmse"]
    assert polar_rmse < equatorial_rmse / 1000
    # Unweighted, the two are indistinguishable -- that was the defect.
    assert _error_metrics(polar, target)["rmse"] == pytest.approx(
        _error_metrics(equatorial, target)["rmse"]
    )


def test_weighted_crps_reduces_to_the_unweighted_estimator():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(6, 2, 4, 3))
    target = rng.normal(size=(2, 4, 3))
    ones = np.ones((1, 4, 1))
    assert _ensemble_crps(samples, target) == pytest.approx(
        _ensemble_crps(samples, target, ones)
    )


def test_weighted_crps_discounts_a_polar_ensemble_error():
    weights = latitude_weights(_spatial_schema([90.0, 0.0]))
    target = np.zeros((1, 2, 2))
    polar = np.zeros((4, 1, 2, 2))
    polar[:, :, 0, :] = 5.0
    equatorial = np.zeros((4, 1, 2, 2))
    equatorial[:, :, 1, :] = 5.0
    assert _ensemble_crps(polar, target, weights) < _ensemble_crps(
        equatorial, target, weights
    )


def test_anomaly_correlation_spans_perfect_to_inverted():
    climatology = np.zeros((2, 4))
    target = np.array([[1.0, -1.0, 2.0, -2.0], [0.5, -0.5, 1.5, -1.5]])
    assert _anomaly_correlation(target, target, climatology) == pytest.approx(1.0)
    assert _anomaly_correlation(-target, target, climatology) == pytest.approx(-1.0)
    # A prediction equal to climatology has no anomaly to correlate.
    assert _anomaly_correlation(climatology, target, climatology) == 0.0


def test_seasonal_climatology_uses_training_months_only():
    times = np.array(
        [f"{year}-{month:02d}-01" for year in (2000, 2001, 2002) for month in (1, 7)],
        dtype="datetime64[ns]",
    )
    states = np.array([1.0, 10.0, 3.0, 30.0, 999.0, 999.0], dtype=np.float32).reshape(6, 1)
    # Only the first four months are training; the last year must not leak in.
    climatology = seasonal_climatology(states, times, (0, 4))
    assert set(climatology) == {1, 7}
    np.testing.assert_allclose(climatology[1], [2.0])
    np.testing.assert_allclose(climatology[7], [20.0])


def test_seasonal_climatology_ignores_missing_values():
    times = np.array(["2000-03-01", "2001-03-01"], dtype="datetime64[ns]")
    states = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    climatology = seasonal_climatology(states, times, None)
    np.testing.assert_allclose(climatology[3], [2.0, 5.0])


def test_skill_score_is_the_fraction_of_baseline_error_removed():
    assert _skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert _skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert _skill_score(2.0, 1.0) == pytest.approx(-1.0)
    assert _skill_score(1.0, 0.0) == 0.0

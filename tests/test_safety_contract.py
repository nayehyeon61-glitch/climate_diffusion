import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import (
    MonthlyWindowDataset,
    load_auxiliary_states,
    load_monthly_archive,
    load_observation_mask,
    prepare_monthly_archive,
)
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.evaluation import _error_metrics
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.train import (
    _epoch,
    _train_only_statistics,
    build_raw_month_temporal_split,
)


def _write_fields(path, values, times, *, lat=(0.0,), lon=(0.0, 180.0)):
    dataset = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.asarray(values, dtype=np.float32))},
        coords={"time": times, "lat": np.asarray(lat), "lon": np.asarray(lon)},
    )
    dataset.to_netcdf(path, engine="scipy")
    return dataset


def _legacy_checkpoint(path, *, bad_scale=False):
    config = FlowModelConfig(
        state_dim=2, history_months=2, latent_dim=2, hidden_dim=8
    )
    model = MonthlyLatentFlow(config)
    torch.save(
        {
            "format": "climate_diffusion.monthly_latent_flow.v2",
            "model": model.state_dict(),
            "model_config": {
                "state_dim": 2,
                "history_months": 2,
                "latent_dim": 2,
                "hidden_dim": 8,
                "time_embedding_dim": 32,
            },
            "state_mean": torch.zeros(2),
            "state_scale": torch.tensor([float("inf"), 1.0])
            if bad_scale
            else torch.ones(2),
            "schema": {"format": "climate_diffusion.monthly_state.v1"},
        },
        path,
    )


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_vector_archive_rejects_infinity_with_variable_context(tmp_path, bad_value):
    times = pd.date_range("2020-01-01", periods=4, freq="MS")
    values = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    values[1, 0, 1] = bad_value
    fields = tmp_path / "bad.nc"
    _write_fields(fields, values, times)
    with pytest.raises(ValueError, match="msl.*Inf"):
        prepare_monthly_archive(fields, tmp_path / "archive.npz")


def test_nan_mask_is_preserved_and_imputed_from_train_only(tmp_path):
    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    values = np.arange(12, dtype=np.float32).reshape(6, 1, 2)
    values[1, 0, 0] = np.nan
    values[5, 0, 0] = 10_000.0
    fields = tmp_path / "nan.nc"
    _write_fields(fields, values, times)
    archive, _ = prepare_monthly_archive(fields, tmp_path / "archive.npz")
    states, archive_times, schema = load_monthly_archive(archive)
    mask = load_observation_mask(archive, states, schema)
    assert np.isnan(states[1, 0])
    assert not mask[1, 0]
    mean, scale = _train_only_statistics(
        states, [0, 1, 2], observation_mask=mask, spatial=False
    )
    assert mean[0] < 10.0
    dataset = MonthlyWindowDataset(
        states,
        2,
        indices=[0],
        mean=mean,
        scale=scale,
        observation_mask=mask,
        times=archive_times,
    )
    # Missing train value is imputed with the train mean and therefore normalizes to zero.
    assert dataset[0]["history"][1, 0].item() == pytest.approx(0.0)


def test_all_missing_train_feature_fails_explicitly():
    states = np.array([[np.nan, 1.0], [np.nan, 2.0], [9.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="all-missing train-only"):
        _train_only_statistics(states, [0, 1], spatial=False)


def test_missing_month_and_duplicate_time_are_rejected(tmp_path):
    missing = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"])
    fields = tmp_path / "missing.nc"
    _write_fields(fields, np.ones((3, 1, 2)), missing)
    with pytest.raises(ValueError, match="missing/non-consecutive"):
        prepare_monthly_archive(fields, tmp_path / "missing.npz")

    duplicate = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"])
    fields = tmp_path / "duplicate.nc"
    _write_fields(fields, np.ones((3, 1, 2)), duplicate)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        prepare_monthly_archive(fields, tmp_path / "duplicate.npz")


def test_out_of_order_source_is_sorted_but_windows_use_calendar_future(tmp_path):
    times = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01", "2020-04-01"])
    values = np.array([3, 1, 2, 4], dtype=np.float32)[:, None, None]
    fields = tmp_path / "unordered.nc"
    _write_fields(fields, values, times, lon=(0.0,))
    archive, _ = prepare_monthly_archive(fields, tmp_path / "archive.npz")
    states, archive_times, schema = load_monthly_archive(archive)
    mask = load_observation_mask(archive, states, schema)
    dataset = MonthlyWindowDataset(
        states,
        history_months=2,
        lead_months=1,
        indices=[0],
        mean=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
        observation_mask=mask,
        times=archive_times,
    )
    assert list(archive_times[:3]) == list(
        pd.date_range("2020-01-01", periods=3, freq="MS").values
    )
    assert dataset[0]["history"][:, 0].tolist() == [1.0, 2.0]
    assert dataset[0]["target"].item() == 3.0


def test_raw_month_split_has_no_month_overlap_with_small_purge():
    split = build_raw_month_temporal_split(
        30,
        history_months=6,
        lead_months=1,
        validation_fraction=0.2,
        test_fraction=0.1,
        purge_months=1,
    )
    consumed = {}
    for name, starts in {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }.items():
        consumed[name] = {month for start in starts for month in range(start, start + 7)}
    assert consumed["train"].isdisjoint(consumed["validation"])
    assert consumed["validation"].isdisjoint(consumed["test"])
    assert consumed["train"].isdisjoint(consumed["test"])


def test_loss_and_checkpoint_normalization_finite_guards(tmp_path):
    model = MonthlyLatentFlow(
        FlowModelConfig(state_dim=2, history_months=2, latent_dim=2, hidden_dim=8)
    )
    history = torch.zeros(1, 2, 2)
    history[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="flow loss history input"):
        model.loss(history, torch.zeros(1, 2))

    checkpoint = tmp_path / "bad-scale.pt"
    _legacy_checkpoint(checkpoint, bad_scale=True)
    with pytest.raises(FloatingPointError, match="checkpoint state_scale"):
        LatentFlowForecaster(checkpoint, device="cpu")


def _overflowing_epoch(*, scaler):
    """Run one training epoch whose gradients always overflow to +Inf."""
    from climate_diffusion.config import FlowLossConfig

    torch.manual_seed(0)
    model = MonthlyLatentFlow(
        FlowModelConfig(state_dim=2, history_months=2, latent_dim=2, hidden_dim=8)
    )
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    batches = [
        {
            "history": torch.zeros(1, 2, 2),
            "target": torch.zeros(1, 2),
        }
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = _epoch(
        model,
        batches,
        FlowLossConfig(),
        torch.device("cpu"),
        optimizer,
        scaler=scaler,
    )
    unchanged = all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    return metrics, unchanged


def test_amp_gradient_overflow_skips_the_step_instead_of_ending_the_run():
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    initial_scale = scaler.get_scale()

    metrics, parameters_unchanged = _overflowing_epoch(scaler=scaler)

    # GradScaler owns this event: it skips the update and backs the scale off.
    assert metrics["amp_overflow_steps"] == 1.0
    assert parameters_unchanged
    assert scaler.get_scale() < initial_scale


def test_non_finite_gradient_without_loss_scaling_still_stops_the_run():
    with pytest.raises(FloatingPointError, match="gradient"):
        _overflowing_epoch(scaler=None)


def test_forecast_rejects_inf_but_train_mean_imputes_nan(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    _legacy_checkpoint(checkpoint)
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    history = np.zeros((2, 2), dtype=np.float32)
    history[0, 0] = np.nan
    prediction = forecaster.forecast(history, integration_steps=1)
    assert np.isfinite(prediction).all()
    history[0, 0] = np.inf
    with pytest.raises(ValueError, match="forecast history.*Inf"):
        forecaster.forecast(history, integration_steps=1)


def test_evaluation_metric_refuses_nonfinite_predictions():
    with pytest.raises(ValueError, match="no jointly finite"):
        _error_metrics(np.array([np.nan]), np.array([1.0]))


def test_spatial_archive_preserves_nan_mask_without_float_extrema(tmp_path):
    times = pd.date_range("2020-01-01", periods=4, freq="MS")
    values = np.arange(32, dtype=np.float32).reshape(4, 1, 8)
    values[1, 0, 3] = np.nan
    fields = tmp_path / "spatial.nc"
    _write_fields(fields, values, times, lon=np.arange(8.0))
    archive, _ = prepare_monthly_archive(
        fields,
        tmp_path / "spatial",
        layout="spatial",
        target_lat_points=1,
        target_lon_points=8,
    )
    states, _, schema = load_monthly_archive(archive)
    mask = load_observation_mask(archive, states, schema)
    assert np.isnan(states[1, 0, 0, 3])
    assert not mask[1, 0, 0, 3]
    assert not np.isinf(states).any()
    assert (archive / "observed_mask.npy").is_file()
    assert json.loads((archive / "schema.json").read_text())["infinity_policy"] == "fail_fast"


def test_integrated_auxiliary_inf_fails_and_nan_is_preserved(tmp_path):
    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    fields = tmp_path / "fields.nc"
    _write_fields(fields, np.ones((6, 1, 2)), times)
    integrated = pd.DataFrame({"time": times, "pressure": np.arange(6, dtype=float)})
    integrated.loc[1, "pressure"] = np.inf
    table = tmp_path / "integrated.csv"
    integrated.to_csv(table, index=False)
    with pytest.raises(ValueError, match="integrated main-system data.*Inf"):
        prepare_monthly_archive(
            fields,
            tmp_path / "bad-aux",
            integrated=table,
            layout="spatial",
            target_lat_points=1,
            target_lon_points=2,
        )

    integrated.loc[1, "pressure"] = np.nan
    integrated.to_csv(table, index=False)
    archive, _ = prepare_monthly_archive(
        fields,
        tmp_path / "nan-aux",
        integrated=table,
        layout="spatial",
        target_lat_points=1,
        target_lon_points=2,
    )
    _, _, schema = load_monthly_archive(archive)
    auxiliary = load_auxiliary_states(archive, schema)
    assert np.isnan(auxiliary[1, 0])
    mean, _ = _train_only_statistics(
        auxiliary, [0, 1, 2], spatial=False, name="spatial auxiliary"
    )
    assert mean[0] == pytest.approx(1.0)

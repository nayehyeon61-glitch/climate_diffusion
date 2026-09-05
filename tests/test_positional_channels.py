import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import (
    MonthlyWindowDataset,
    load_monthly_archive,
    positional_grid,
    prepare_monthly_archive,
)
from climate_diffusion.inference import LatentFlowForecaster
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.train import train_flow_model


def _config(**overrides):
    base = dict(
        state_dim=2 * 8 * 16,
        history_months=3,
        backend="spatial_conv",
        spatial_channels=2,
        grid_height=8,
        grid_width=16,
        spatial_latent_channels=2,
        spatial_base_channels=2,
        spatial_downsample_levels=1,
        positional_channels=3,
    )
    base.update(overrides)
    return FlowModelConfig(**base)


def test_positional_grid_encodes_latitude_and_a_seamless_longitude():
    schema = {
        "layout": "spatial",
        "coords": {"lat": [90.0, 0.0, -90.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    }
    planes = positional_grid(schema)
    assert planes.shape == (3, 3, 4)
    sin_lat, cos_lon, sin_lon = planes
    np.testing.assert_allclose(sin_lat[:, 0], [1.0, 0.0, -1.0], atol=1e-6)
    # Latitude is constant along a row of longitude.
    assert len(np.unique(np.round(sin_lat[0], 6))) == 1
    np.testing.assert_allclose(cos_lon[0], [1.0, 0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(sin_lon[0], [0.0, 1.0, 0.0, -1.0], atol=1e-6)
    # 0 and 360 degrees must land on the same point -- no dateline discontinuity.
    wrapped = positional_grid(
        {"layout": "spatial", "coords": {"lat": [0.0], "lon": [0.0, 360.0]}}
    )
    np.testing.assert_allclose(wrapped[1][0, 0], wrapped[1][0, 1], atol=1e-6)
    np.testing.assert_allclose(wrapped[2][0, 0], wrapped[2][0, 1], atol=1e-6)
    assert positional_grid({"layout": "vector"}) is None


def test_config_rejects_an_unsupported_positional_width():
    with pytest.raises(ValueError, match="positional_channels"):
        _config(positional_channels=2)


def test_encoder_widens_while_the_decoder_still_reconstructs_data_channels():
    model = MonthlyLatentFlow(_config())
    assert model.autoencoder.input_projection.conv.in_channels == 5
    assert model.autoencoder.output_projection.conv.out_channels == 2

    coordinates = torch.randn(1, 3, 8, 16)
    losses = model.loss(
        torch.randn(2, 3, 2, 8, 16),
        torch.randn(2, 2, 8, 16),
        coordinates=coordinates,
    )
    assert torch.isfinite(losses["loss"])
    sample = model.sample(
        torch.randn(1, 3, 2, 8, 16), integration_steps=1, coordinates=coordinates
    )
    assert sample.shape == (1, 2, 8, 16)


def test_missing_or_mismatched_coordinates_are_rejected():
    model = MonthlyLatentFlow(_config())
    history = torch.randn(1, 3, 2, 8, 16)
    with pytest.raises(ValueError, match="coordinates are required"):
        model.sample(history, integration_steps=1)
    with pytest.raises(ValueError, match="positional channels"):
        model.sample(history, integration_steps=1, coordinates=torch.randn(1, 2, 8, 16))
    with pytest.raises(ValueError, match="Coordinates cover"):
        model.sample(history, integration_steps=1, coordinates=torch.randn(1, 3, 4, 8))


def test_zero_positional_channels_keeps_the_original_architecture():
    model = MonthlyLatentFlow(_config(positional_channels=0))
    assert model.autoencoder.input_projection.conv.in_channels == 2
    sample = model.sample(torch.randn(1, 3, 2, 8, 16), integration_steps=1)
    assert sample.shape == (1, 2, 8, 16)


def test_the_model_can_tell_two_latitudes_apart():
    """The same field at different latitudes must not encode identically."""
    torch.manual_seed(0)
    model = MonthlyLatentFlow(_config()).eval()
    field = torch.randn(1, 2, 8, 16)
    schema = {
        "layout": "spatial",
        "coords": {"lat": list(np.linspace(80.0, 88.0, 8)), "lon": list(np.arange(16) * 22.5)},
    }
    polar = torch.as_tensor(positional_grid(schema)).unsqueeze(0)
    schema["coords"]["lat"] = list(np.linspace(-4.0, 4.0, 8))
    equatorial = torch.as_tensor(positional_grid(schema)).unsqueeze(0)

    with torch.no_grad():
        near_pole = model.autoencoder.encode(model._with_coordinates(field, polar))
        near_equator = model.autoencoder.encode(
            model._with_coordinates(field, equatorial)
        )
    assert not torch.allclose(near_pole, near_equator)


def test_each_tile_receives_the_coordinates_of_its_own_location():
    """Tiled sampling must hand every tile the planes for where it actually sits."""
    model = MonthlyLatentFlow(_config()).eval()
    history = torch.randn(1, 3, 2, 8, 16)
    schema = {
        "layout": "spatial",
        "coords": {
            "lat": list(np.linspace(-80.0, 80.0, 8)),
            "lon": list(np.arange(16) * 22.5),
        },
    }
    planes = torch.as_tensor(positional_grid(schema)).unsqueeze(0)

    captured = []
    original = model.sample

    def record(patch, **kwargs):
        captured.append(kwargs["coordinates"])
        return original(patch, **kwargs)

    model.sample = record
    model.sample_tiled(
        history,
        tile_size=(8, 8),
        overlap=4,
        integration_steps=1,
        generator=torch.Generator().manual_seed(0),
        coordinates=planes,
    )
    model.sample = original

    # Tiles start at longitude 0, 4, 8, 12 and wrap at the dateline.
    assert len(captured) == 4
    for tile_index, start in enumerate((0, 4, 8, 12)):
        columns = (np.arange(start, start + 8) % 16).tolist()
        expected = planes[:, :, :, columns]
        torch.testing.assert_close(captured[tile_index], expected)


def test_dataset_crops_coordinates_with_the_data(tmp_path):
    months, height, width = 12, 8, 16
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    values = np.random.default_rng(0).normal(size=(months, height, width)).astype(np.float32)
    xr.Dataset(
        {"msl": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": np.linspace(-80.0, 80.0, height),
            "lon": np.arange(width) * 22.5,
        },
    ).to_netcdf(tmp_path / "fields.nc", engine="scipy")
    archive, _ = prepare_monthly_archive(
        tmp_path / "fields.nc",
        tmp_path / "archive",
        layout="spatial",
        target_lat_points=height,
        target_lon_points=width,
    )
    states, _, schema = load_monthly_archive(archive)
    planes = positional_grid(schema)
    dataset = MonthlyWindowDataset(
        states,
        3,
        1,
        mean=states.mean(axis=0),
        scale=np.ones_like(states[0]),
        coordinates=planes,
        patch_size=(4, 6),
        random_crop=False,
    )
    item = dataset[0]
    assert item["coordinates"].shape == (3, 4, 6)
    top = (height - 4) // 2
    left = (width - 6) // 2
    expected = planes[:, top : top + 4, left : left + 6]
    np.testing.assert_allclose(item["coordinates"].numpy(), expected, atol=1e-6)


def test_training_and_inference_agree_on_coordinates_end_to_end(tmp_path):
    months, height, width = 24, 8, 16
    times = pd.date_range("2000-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(0)
    values = (
        np.sin(2 * np.pi * np.arange(months) / 12)[:, None, None]
        + 0.2 * rng.normal(size=(months, height, width))
    ).astype(np.float32)
    xr.Dataset(
        {"msl": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": np.linspace(-80.0, 80.0, height),
            "lon": np.arange(width) * 22.5,
        },
    ).to_netcdf(tmp_path / "fields.nc", engine="scipy")
    archive, _ = prepare_monthly_archive(
        tmp_path / "fields.nc",
        tmp_path / "archive",
        layout="spatial",
        target_lat_points=height,
        target_lon_points=width,
    )
    checkpoint = train_flow_model(
        archive,
        tmp_path / "model.pt",
        history_months=3,
        epochs=1,
        batch_size=2,
        validation_fraction=0.2,
        test_fraction=0.2,
        purge_windows=1,
        model_backend="spatial_conv",
        spatial_base_channels=2,
        spatial_latent_channels=2,
        spatial_downsample_levels=1,
        patch_height=8,
        patch_width=8,
        tile_overlap=2,
        seed=7,
        skill_windows=1,
        skill_ensemble_size=1,
        skill_integration_steps=2,
    )
    forecaster = LatentFlowForecaster(checkpoint, device="cpu")
    assert forecaster.config.positional_channels == 3
    assert forecaster.coordinates.shape == (1, 3, height, width)
    states, _, _ = load_monthly_archive(archive)
    # Patch width 8 < grid width 16, so this runs through the tiled path.
    prediction = forecaster.forecast(states[:3], integration_steps=2)
    assert prediction.shape == (1, 1, 1, height, width)
    assert np.isfinite(prediction).all()

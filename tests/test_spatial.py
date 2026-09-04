import json

import numpy as np
import pandas as pd
import torch
import xarray as xr

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import (
    load_monthly_archive,
    prepare_monthly_archive,
    reconstruct_spatial_dataset,
    spatialize_dataset,
)
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.spatial import PeriodicConv2d, SpatialAutoencoder, tiled_apply


def spatial_config(backend="spatial_conv", height=16, width=24):
    return FlowModelConfig(
        state_dim=2 * height * width,
        history_months=3,
        latent_dim=4,
        hidden_dim=8,
        time_embedding_dim=8,
        backend=backend,
        spatial_channels=2,
        grid_height=height,
        grid_width=width,
        spatial_latent_channels=4,
        spatial_base_channels=4,
        spatial_downsample_levels=1,
        operator_modes_lat=2,
        operator_modes_lon=3,
    )


def test_spatial_flow_loss_backward_and_operator_sampling():
    torch.manual_seed(4)
    model = MonthlyLatentFlow(spatial_config("spatial_operator"))
    history = torch.randn(2, 3, 2, 16, 24)
    target = torch.randn(2, 2, 16, 24)
    losses = model.loss(history, target)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())
    sample = model.sample(history[:1], integration_steps=2)
    assert sample.shape == (1, 2, 16, 24)


def test_global_025_odd_latitude_shape_contract():
    model = SpatialAutoencoder(
        channels=1,
        latent_channels=2,
        base_channels=2,
        downsample_levels=2,
    ).eval()
    values = torch.zeros(1, 1, 721, 1440)
    with torch.no_grad():
        reconstructed = model(values)
    assert reconstructed.shape == values.shape


def test_periodic_longitude_convolution_is_roll_equivariant():
    torch.manual_seed(2)
    layer = PeriodicConv2d(1, 1)
    values = torch.randn(1, 1, 8, 12)
    shift = 4
    expected = torch.roll(layer(values), shifts=shift, dims=-1)
    actual = layer(torch.roll(values, shifts=shift, dims=-1))
    torch.testing.assert_close(actual, expected)


def test_tiled_overlap_stitching_preserves_identity_and_dateline():
    history = torch.randn(1, 3, 2, 9, 13)
    output = tiled_apply(
        history,
        lambda patch: patch[:, -1],
        tile_size=(6, 7),
        overlap=3,
    )
    torch.testing.assert_close(output, history[:, -1])


def test_spatial_archive_preserves_channels_coordinates_and_roundtrip(tmp_path):
    times = pd.date_range("2020-01-01", periods=5, freq="MS")
    values = np.arange(5 * 2 * 3 * 4, dtype=np.float32).reshape(5, 2, 3, 4)
    fields = xr.Dataset(
        {"temperature": (("time", "level", "lat", "lon"), values)},
        coords={
            "time": times,
            "level": [500, 850],
            "lat": [-90.0, 0.0, 90.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
    )
    source = tmp_path / "fields.nc"
    fields.to_netcdf(source, engine="scipy")
    archive, schema_path = prepare_monthly_archive(
        source,
        tmp_path / "spatial_archive",
        layout="spatial",
        target_lat_points=3,
        target_lon_points=4,
    )
    states, archive_times, schema = load_monthly_archive(archive)
    assert isinstance(states, np.memmap)
    assert states.shape == (5, 2, 3, 4)
    assert schema["grid_shape"] == [3, 4]
    assert schema["variables"][0]["channel_slice"] == [0, 2]
    assert json.loads(schema_path.read_text())["layout"] == "spatial"
    assert archive_times[0] == np.datetime64("2020-01-01")

    mapped = spatialize_dataset(fields, schema)
    np.testing.assert_allclose(mapped, states)
    rebuilt = reconstruct_spatial_dataset(states[-1], schema, pd.Timestamp("2020-06-01"))
    assert rebuilt["temperature"].shape == (1, 2, 3, 4)
    np.testing.assert_allclose(rebuilt["temperature"].values[0], values[-1])

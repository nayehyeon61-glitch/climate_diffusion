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
from climate_diffusion.config import FlowLossConfig
from climate_diffusion.model import MonthlyLatentFlow
from climate_diffusion.spatial import (
    PeriodicConv2d,
    SpatialAutoencoder,
    set_longitude_wrap,
    tiled_apply,
)


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


def test_sub_global_patch_pads_by_replication_instead_of_wrapping():
    """A patch narrower than the grid must not join its two edges."""
    torch.manual_seed(3)
    layer = PeriodicConv2d(1, 1)
    field = torch.randn(1, 1, 6, 32)
    patch = field[..., 4:12]

    set_longitude_wrap(layer, False)
    actual = layer(patch)

    padded = torch.nn.functional.pad(patch, (1, 1, 0, 0), mode="replicate")
    padded = torch.nn.functional.pad(padded, (0, 0, 1, 1), mode="replicate")
    torch.testing.assert_close(actual, layer.conv(padded))

    # The interior never depended on the padding mode, only the two edges did.
    set_longitude_wrap(layer, True)
    wrapped = layer(patch)
    torch.testing.assert_close(actual[..., 1:-1], wrapped[..., 1:-1])
    assert not torch.allclose(actual[..., 0], wrapped[..., 0])


def test_model_wraps_longitude_only_for_globe_spanning_input():
    config = spatial_config(height=8, width=16)
    model = MonthlyLatentFlow(config)

    def wrap_flags():
        return {
            layer.wrap_longitude
            for layer in model.modules()
            if isinstance(layer, PeriodicConv2d)
        }

    model.loss(torch.randn(2, 3, 2, 8, 16), torch.randn(2, 2, 8, 16))
    assert wrap_flags() == {True}

    model.loss(torch.randn(2, 3, 2, 8, 8), torch.randn(2, 2, 8, 8))
    assert wrap_flags() == {False}

    model.sample(torch.randn(1, 3, 2, 8, 16), integration_steps=1)
    assert wrap_flags() == {True}


def test_flow_term_does_not_backpropagate_through_the_target_encoder():
    """The flow branch trains against a detached latent, not a moving one."""
    torch.manual_seed(5)
    model = MonthlyLatentFlow(spatial_config(height=8, width=16))
    target = torch.randn(2, 2, 8, 16, requires_grad=True)
    losses = model.loss(
        torch.randn(2, 3, 2, 8, 16),
        target,
        FlowLossConfig(
            reconstruction_weight=0.0,
            flow_weight=1.0,
            latent_regularization_weight=0.0,
        ),
    )
    losses["loss"].backward()
    assert target.grad is None or torch.count_nonzero(target.grad) == 0


def test_seeded_loss_is_reproducible_and_unseeded_is_not():
    torch.manual_seed(6)
    model = MonthlyLatentFlow(spatial_config(height=8, width=16))
    history = torch.randn(4, 3, 2, 8, 16)
    target = torch.randn(4, 2, 8, 16)

    def flow_loss(seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            return float(model.loss(history, target, generator=generator)["flow_matching_mse"])

    assert flow_loss(seed=11) == flow_loss(seed=11)
    assert flow_loss(seed=11) != flow_loss(seed=12)
    assert flow_loss() != flow_loss()


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

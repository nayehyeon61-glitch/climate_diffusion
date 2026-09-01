"""Memory-safe spatial building blocks for global latitude/longitude fields."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PeriodicConv2d(nn.Module):
    """Convolution with circular longitude and replicated latitude padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("PeriodicConv2d requires an odd kernel size")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pad = self.padding
        if pad:
            values = F.pad(values, (pad, pad, 0, 0), mode="circular")
            # Replication is stable at the poles and does not invent a second seam.
            values = F.pad(values, (0, 0, pad, pad), mode="replicate")
        return self.conv(values)


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = PeriodicConv2d(channels, channels)
        self.conv2 = PeriodicConv2d(channels, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.conv1(F.silu(self.norm1(values)))
        values = self.conv2(F.silu(self.norm2(values)))
        return residual + values


class SpectralOperator2d(nn.Module):
    """Low-mode Fourier operator used at the autoencoder/vector-field bottleneck."""

    def __init__(self, channels: int, modes_lat: int, modes_lon: int) -> None:
        super().__init__()
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon
        scale = 1.0 / max(1, channels * channels)
        shape = (channels, channels, modes_lat, modes_lon)
        self.weight_north = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_south = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.local = PeriodicConv2d(channels, channels, kernel_size=1)

    @staticmethod
    def _multiply(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", values, weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        height, width = values.shape[-2:]
        spectrum = torch.fft.rfft2(values, norm="ortho")
        output = torch.zeros_like(spectrum)
        modes_lat = min(self.modes_lat, max(1, height // 2))
        modes_lon = min(self.modes_lon, spectrum.shape[-1])
        output[:, :, :modes_lat, :modes_lon] = self._multiply(
            spectrum[:, :, :modes_lat, :modes_lon],
            self.weight_north[:, :, :modes_lat, :modes_lon],
        )
        output[:, :, -modes_lat:, :modes_lon] = self._multiply(
            spectrum[:, :, -modes_lat:, :modes_lon],
            self.weight_south[:, :, :modes_lat, :modes_lon],
        )
        return self.local(values) + torch.fft.irfft2(
            output, s=(height, width), norm="ortho"
        )


class SpatialAutoencoder(nn.Module):
    """Fully convolutional autoencoder that never flattens a global field."""

    def __init__(
        self,
        channels: int,
        latent_channels: int,
        base_channels: int,
        downsample_levels: int,
        *,
        operator_modes: tuple[int, int] | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.downsample_levels = downsample_levels
        self.gradient_checkpointing = gradient_checkpointing
        self.input_projection = PeriodicConv2d(channels, base_channels)
        encoder = []
        current = base_channels
        for _ in range(downsample_levels):
            encoder.extend(
                [SpatialResidualBlock(current), PeriodicConv2d(current, current * 2, stride=2)]
            )
            current *= 2
        self.encoder_blocks = nn.Sequential(*encoder)
        self.bottleneck = SpatialResidualBlock(current)
        self.operator = (
            SpectralOperator2d(current, *operator_modes)
            if operator_modes is not None
            else nn.Identity()
        )
        self.to_latent = PeriodicConv2d(current, latent_channels, kernel_size=1)
        self.from_latent = PeriodicConv2d(latent_channels, current, kernel_size=1)
        decoder = []
        for _ in range(downsample_levels):
            decoder.append(SpatialResidualBlock(current))
            decoder.append(PeriodicConv2d(current, current // 2))
            current //= 2
        self.decoder_blocks = nn.ModuleList(decoder)
        self.output_projection = PeriodicConv2d(current, channels)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(values)
        if self.gradient_checkpointing and self.training and values.requires_grad:
            values = checkpoint(self.encoder_blocks, values, use_reentrant=False)
            values = checkpoint(self.bottleneck, values, use_reentrant=False)
            values = checkpoint(self.operator, values, use_reentrant=False)
        else:
            values = self.encoder_blocks(values)
            values = self.bottleneck(values)
            values = self.operator(values)
        return self.to_latent(values)

    def decode(self, latent: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        values = self.from_latent(latent)
        for index in range(0, len(self.decoder_blocks), 2):
            block = self.decoder_blocks[index]
            if self.gradient_checkpointing and self.training and values.requires_grad:
                values = checkpoint(block, values, use_reentrant=False)
            else:
                values = block(values)
            values = F.interpolate(values, scale_factor=2.0, mode="bilinear", align_corners=False)
            values = self.decoder_blocks[index + 1](values)
        if values.shape[-2:] != output_shape:
            # Required for the 721 latitude row, which is not divisible by powers of two.
            values = F.interpolate(values, size=output_shape, mode="bilinear", align_corners=False)
        return self.output_projection(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values), values.shape[-2:])


def tile_starts(length: int, tile: int, overlap: int, *, periodic: bool = False) -> list[int]:
    if not 0 < tile <= length:
        raise ValueError("tile must be positive and no larger than the dimension")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    step = tile - overlap
    if periodic:
        return list(range(0, length, step))
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    lat = torch.hann_window(height + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    lon = torch.hann_window(width + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    return (lat[:, None] * lon[None, :]).clamp_min(1e-3)


def tiled_apply(
    history: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile_size: tuple[int, int],
    overlap: int,
) -> torch.Tensor:
    """Apply a patch predictor and blend it into a seam-safe global field.

    ``history`` is ``[B, T, C, H, W]`` and ``predict`` returns ``[B, C, h, w]``.
    Longitude tiles wrap around the dateline; latitude tiles stop at the poles.
    """

    if history.ndim != 5:
        raise ValueError("Tiled spatial history must have shape [B,T,C,H,W]")
    height, width = history.shape[-2:]
    tile_height, tile_width = tile_size
    lat_starts = tile_starts(height, tile_height, overlap)
    lon_starts = tile_starts(width, tile_width, overlap, periodic=True)
    result = history.new_zeros((history.shape[0], history.shape[2], height, width))
    weights = history.new_zeros((1, 1, height, width))
    window = _blend_window(
        tile_height,
        tile_width,
        device=history.device,
        dtype=history.dtype,
    )[None, None]
    for lat_start in lat_starts:
        lat_index = torch.arange(lat_start, lat_start + tile_height, device=history.device)
        for lon_start in lon_starts:
            lon_index = torch.arange(
                lon_start, lon_start + tile_width, device=history.device
            ).remainder(width)
            patch = history.index_select(-2, lat_index).index_select(-1, lon_index)
            prediction = predict(patch)
            if prediction.shape[-2:] != (tile_height, tile_width):
                raise ValueError("Patch predictor changed the requested tile shape")
            for local_lon, global_lon in enumerate(lon_index.tolist()):
                result[:, :, lat_start : lat_start + tile_height, global_lon] += (
                    prediction[:, :, :, local_lon] * window[:, :, :, local_lon]
                )
                weights[:, :, lat_start : lat_start + tile_height, global_lon] += window[
                    :, :, :, local_lon
                ]
    if torch.any(weights == 0):
        raise RuntimeError("Tile configuration left uncovered grid cells")
    return result / weights

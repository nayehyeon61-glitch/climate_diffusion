"""Surface-field physical diagnostics and metric, not a primitive-equation solver.

All finite differences use REAL geographic spacing. Flow time never enters a
physical residual. Matching reconstructed diagnostics is not enforcing zero
divergence or exact conservation of an open atmosphere.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .moe_data import field_grid


class SurfacePhysics(nn.Module):
    diagnostic_names = ("divergence", "vorticity", "pressure_dx", "pressure_dy",
                        "temperature_dx", "temperature_dy", "specific_kinetic_energy")
    invariant_names = ("area_mean_msl_proxy", "area_mean_temperature", "area_mean_specific_ke")

    def __init__(self, schema, mean, scale):
        super().__init__()
        self.grid = field_grid(schema)
        names = [v["name"] for v in schema["variables"]]
        required = ("msl", "t2m", "u10", "v10")
        if not set(required).issubset(names):
            raise ValueError("Surface manifold physics requires msl, t2m, u10, v10 (Pa, K, m/s)")
        self.indices = tuple(names.index(name) for name in required)
        coords = schema["variables"][0]["coords"]
        lat, lon = np.deg2rad(coords["lat"]), np.deg2rad(coords["lon"])
        if (min(len(lat), len(lon)) < 2 or np.any(np.abs(lat) >= math.pi / 2 - 1e-4)
                or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0))
                or not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0))):
            raise ValueError("Physics requires monotone lat/lon, at least 2 cells, and no pole singularities")
        self.periodic_lon = bool(len(lon) >= 3 and np.allclose(np.diff(lon), np.diff(lon)[0])
                                 and np.isclose(abs(lon[1] - lon[0]) * len(lon), 2 * math.pi))
        for name, value in (("mean", mean), ("scale", scale), ("lat", lat), ("lon", lon)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        areas = np.cos(lat)[:, None] * abs(np.gradient(lat))[:, None] * abs(np.gradient(lon))[None, :]
        areas /= areas.mean()
        self.register_buffer("area", torch.tensor(areas, dtype=torch.float32))
        self.register_buffer("metric_weights", self.area.expand(self.grid[0], -1, -1).flatten().clone())
        self.register_buffer("feature_mean", torch.zeros(7))
        self.register_buffer("feature_scale", torch.ones(7))
        self.register_buffer("invariant_mean", torch.zeros(3))
        self.register_buffer("invariant_scale", torch.ones(3))

    def _dlon(self, value):
        if self.periodic_lon:
            return (value.roll(-1, -1) - value.roll(1, -1)) / (2 * (self.lon[1] - self.lon[0]))
        return torch.gradient(value, spacing=(self.lon,), dim=(-1,), edge_order=1)[0]

    def _dlat(self, value):
        return torch.gradient(value, spacing=(self.lat,), dim=(-2,), edge_order=1)[0]

    def raw_features(self, normalized):
        physical = (normalized * self.scale + self.mean).reshape(*normalized.shape[:-1], *self.grid)
        p, temp, u, v = [physical[..., index, :, :] for index in self.indices]
        cosine = self.lat.cos()[:, None]
        denominator = 6_371_000.0 * cosine
        divergence = (self._dlon(u) + self._dlat(v * cosine)) / denominator
        vorticity = (self._dlon(v) - self._dlat(u * cosine)) / denominator
        diagnostics = torch.stack((divergence, vorticity, self._dlon(p) / denominator,
                                   self._dlat(p) / 6_371_000.0, self._dlon(temp) / denominator,
                                   self._dlat(temp) / 6_371_000.0, 0.5 * (u.square() + v.square())), -3)
        invariants = torch.stack(((p * self.area).mean((-2, -1)),
                                  (temp * self.area).mean((-2, -1)),
                                  (diagnostics[..., -1, :, :] * self.area).mean((-2, -1))), -1)
        return diagnostics, invariants

    @torch.no_grad()
    def fit(self, train_states):
        """Called only on expert-train raw span, never validation/calibration/test."""
        diagnostics, invariants = self.raw_features(train_states)
        self.feature_mean.copy_(diagnostics.mean((0, 2, 3)))
        self.feature_scale.copy_(diagnostics.std((0, 2, 3), unbiased=False).clamp_min(1e-12))
        self.invariant_mean.copy_(invariants.mean(0))
        grid_scale = self.scale.reshape(self.grid)
        floors = torch.stack((grid_scale[self.indices[0]].mean() * 0.05,
                              grid_scale[self.indices[1]].mean() * 0.05,
                              invariants[:, 2].abs().mean() * 0.05)).clamp_min(1e-6)
        self.invariant_scale.copy_(torch.maximum(invariants.std(0, unbiased=False), floors))

    def forward(self, normalized):
        features, invariants = self.raw_features(normalized)
        return ((features - self.feature_mean[:, None, None]) / self.feature_scale[:, None, None],
                (invariants - self.invariant_mean) / self.invariant_scale)

    def field_mse(self, a, b):
        return ((a - b).square() * self.metric_weights).mean()

    def reconstruction_losses(self, prediction, target):
        a, ia = self(prediction)
        b, ib = self(target)
        return {"reconstruction": self.field_mse(prediction, target),
                "physics": ((a - b).square() * self.area).mean(),
                "invariant": (ia - ib).square().mean()}

    def pair_distance(self, a, b):
        fa, _ = self(a)
        fb, _ = self(b)
        squared = (((a - b).square() * self.metric_weights).mean(-1)
                   + 0.2 * ((fa - fb).square() * self.area).mean((-3, -2, -1)))
        return squared.clamp_min(1e-12).sqrt()

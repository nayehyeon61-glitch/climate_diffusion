"""A-step pressure-level Hybrid PINN in physical seconds and SI units.

The prepared information sidecar stores geopotential *height* in metres. This
module multiplies it by g exactly once. It enforces pressure-level momentum,
thermodynamics, continuity, and layer thickness; it is not a full-column GCM.
Sparse pressure layers do not justify a surface-pressure column-mass equation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass
class HybridPINNConfig:
    levels_hpa: tuple[int, ...] = (500, 850)
    weight: float = 0.1
    warmup_epochs: int = 1
    ramp_epochs: int = 3
    closure_weight: float = 0.01
    tendency_weight: float = 0.1
    continuity_weight: float = 0.1
    thickness_weight: float = 0.1

    def __post_init__(self):
        self.levels_hpa = tuple(self.levels_hpa)
        self.validate()

    def validate(self):
        if (len(self.levels_hpa) < 2 or len(set(self.levels_hpa)) != len(self.levels_hpa)
                or any(isinstance(p, bool) or not isinstance(p, (int, np.integer))
                       or not 0 < p <= 1100 for p in self.levels_hpa)):
            raise ValueError("PINN needs at least two distinct positive integer pressure levels in hPa")
        for name in ("warmup_epochs", "ramp_epochs"):
            value = getattr(self, name)
            minimum = 1 if name == "ramp_epochs" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"PINN {name} must be an integer >= {minimum}")
        for name in ("weight", "closure_weight", "tendency_weight", "continuity_weight", "thickness_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"PINN {name} must be finite and nonnegative")
        if self.weight <= 0 or self.closure_weight <= 0 or self.tendency_weight <= 0:
            raise ValueError("Enabled PINN requires positive weight, closure_weight and tendency_weight")


class HybridPINN(nn.Module):
    radius = 6_371_000.0
    gravity = 9.80665
    rotation = 7.292115e-5
    gas_constant = 287.05
    heat_capacity = 1004.0
    # Fixed SI characteristic rates; not fitted to validation or test data.
    momentum_scale = 1e-3       # m/s^2
    temperature_scale = 1e-4    # K/s
    continuity_scale = 1e-5    # 1/s
    thickness_scale = 100.0    # m

    def __init__(self, config, info_metadata, info_mean, info_scale, latent_dim, hidden_dim):
        super().__init__()
        config.validate()
        self.config = config
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (latent_dim, hidden_dim)):
            raise ValueError("PINN latent_dim and hidden_dim must be positive integers")
        self.latent_dim = latent_dim
        try:
            self.shape = tuple(info_metadata["shape"])
            variables = info_metadata["variables"]
            lat = np.asarray(info_metadata["grid"]["lat"], dtype=float)
            lon = np.asarray(info_metadata["grid"]["lon"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PINN requires complete information metadata with shape, variables and grid") from exc
        if (len(self.shape) != 3 or any(not isinstance(x, int) or x <= 0 for x in self.shape)
                or self.shape != (len(variables), len(lat), len(lon))):
            raise ValueError("PINN information shape must equal [variables, latitude, longitude]")
        names = [v["name"] for v in variables]
        if len(names) != len(set(names)):
            raise ValueError("PINN information variable names must be unique")
        self.levels = tuple(sorted(config.levels_hpa))
        self.nlevels = len(self.levels)
        required = [f"{name}{p}" for name in "uvtzw" for p in self.levels]
        required += ["sp", "terrain_height", "terrain_slope"]
        missing = sorted(set(required) - set(names))
        if missing:
            raise ValueError("PINN requires matched pressure-level u/v/t/z/w, sp and terrain; missing: " + ", ".join(missing))
        humidity = [f"q{p}" in names for p in self.levels]
        if any(humidity) and not all(humidity):
            raise ValueError("PINN humidity must be supplied at every selected pressure level or absent")
        self.has_humidity = all(humidity)
        self.indices = {name: [names.index(f"{name}{p}") for p in self.levels]
                        for name in "uvtzw" + ("q" if self.has_humidity else "")}
        self.sp_index = names.index("sp")
        expected = {"u": ("m/s",), "v": ("m/s",), "t": ("K",), "z": ("m",),
                    "w": ("Pa/s",), "q": ("kg/kg", "kg kg**-1", "kg kg-1", "1")}
        for key, indices in self.indices.items():
            for p, index in zip(self.levels, indices):
                var = variables[index]
                if var.get("unit") not in expected[key] or var.get("pressure_hpa") != p or var.get("kind") != "dynamic":
                    raise ValueError(f"PINN {var['name']} has noncanonical units, pressure level or kind")
        for name, unit, kind in (("sp", "Pa", "dynamic"), ("terrain_height", "m", "static"),
                                 ("terrain_slope", "1", "static")):
            var = variables[names.index(name)]
            if var.get("unit") != unit or var.get("kind") != kind or var.get("pressure_hpa") is not None:
                raise ValueError(f"PINN {name} requires canonical {unit} units and {kind} kind")
        if (lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 3
                or not np.isfinite(lat).all() or not np.isfinite(lon).all()
                or np.any(np.abs(lat) > 90)
                or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0))
                or not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0))
                or abs(lon[-1] - lon[0]) >= 360):
            raise ValueError("PINN grid requires finite monotone lat/lon, >=3 cells, and no duplicate cyclic endpoint")
        self.periodic_lon = bool(np.allclose(np.diff(lon), np.diff(lon)[0])
                                 and np.isclose(abs(lon[1] - lon[0]) * len(lon), 360))
        self.info_dim = math.prod(self.shape)
        for name, values in (("info_mean", info_mean), ("info_scale", info_scale)):
            value = torch.as_tensor(values, dtype=torch.float32).detach().clone()
            if value.shape != (self.info_dim,) or not torch.isfinite(value).all() or (name == "info_scale" and torch.any(value <= 0)):
                raise ValueError("PINN information statistics must be finite flattened arrays with positive scales")
            self.register_buffer(name, value)
        self.register_buffer("lat", torch.tensor(np.deg2rad(lat), dtype=torch.float32))
        self.register_buffer("lon", torch.tensor(np.deg2rad(lon), dtype=torch.float32))
        self.register_buffer("pressure", torch.tensor(self.levels, dtype=torch.float32) * 100.0)
        area = np.cos(np.deg2rad(lat))[:, None] * abs(np.gradient(np.deg2rad(lat)))[:, None]
        area = area * abs(np.gradient(np.deg2rad(lon)))[None, :]
        self.register_buffer("area", torch.tensor(area / area.mean(), dtype=torch.float32))
        self.register_buffer("polar_valid", torch.tensor(np.abs(np.cos(np.deg2rad(lat))) >= 1e-4))
        self.closure_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, 3 * self.nlevels * self.shape[1] * self.shape[2]))
        nn.init.zeros_(self.closure_head[-1].weight)
        nn.init.zeros_(self.closure_head[-1].bias)
        # u/v/T residuals are dimensionless after division by these scales.
        self.register_buffer("closure_scales", torch.tensor([self.momentum_scale, self.momentum_scale,
                                                              self.temperature_scale]).view(1, 3, 1, 1, 1))

    def _dlon(self, value):
        lon = self.lon.to(value)
        if self.periodic_lon:
            return (value.roll(-1, -1) - value.roll(1, -1)) / (2 * (lon[1] - lon[0]))
        return torch.gradient(value, spacing=(lon,), dim=(-1,), edge_order=1)[0]

    def _dlat(self, value):
        return torch.gradient(value, spacing=(self.lat.to(value),), dim=(-2,), edge_order=1)[0]

    def _dp(self, value):
        return torch.gradient(value, spacing=(self.pressure.to(value),), dim=(-3,), edge_order=1)[0]

    def _gradient(self, value):
        cosine = self.lat.to(value).cos().clamp_min(1e-4)[:, None]
        return self._dlon(value) / (self.radius * cosine), self._dlat(value) / self.radius

    def _divergence(self, u, v):
        cosine = self.lat.to(u).cos().clamp_min(1e-4)[:, None]
        return (self._dlon(u) + self._dlat(v * cosine)) / (self.radius * cosine)

    def _physical(self, normalized):
        return (normalized * self.info_scale + self.info_mean).reshape(-1, *self.shape)

    def _fields(self, physical):
        return {name: physical[:, indices] for name, indices in self.indices.items()}

    def _stencil_mask(self, observed0, observed1):
        sp = torch.minimum(observed0[:, self.sp_index], observed1[:, self.sp_index]).detach()
        if torch.any(sp <= 0):
            raise ValueError("PINN observed surface pressure must be positive Pa")
        valid = (self.pressure[None, :, None, None] < sp[:, None]) & self.polar_valid[None, None, :, None]
        # A conservative complete-column mask ensures all vertical derivative
        # stencils and adjacent-layer thickness constraints remain above ground.
        valid = valid.all(1, keepdim=True).expand(-1, self.nlevels, -1, -1)
        safe = valid.clone()
        safe[..., 1:, :] &= valid[..., :-1, :]
        safe[..., :-1, :] &= valid[..., 1:, :]
        if self.periodic_lon:
            safe &= valid.roll(1, -1) & valid.roll(-1, -1)
        else:
            safe[..., 1:] &= valid[..., :-1]
            safe[..., :-1] &= valid[..., 1:]
        if not bool(safe.any()):
            raise ValueError("PINN has no valid above-ground, nonpolar derivative stencils")
        return safe

    def _mean_square(self, value, mask):
        weights = mask.to(value.dtype) * self.area.to(value)
        # where also prevents invalid large values from entering the reduction.
        safe = torch.where(mask, value, torch.zeros_like(value))
        return (safe.square() * weights).sum() / weights.sum().clamp_min(torch.finfo(value.dtype).tiny)

    def _known_tendency(self, fields):
        u, v, temp, height, omega = (fields[key] for key in "uvtzw")
        lat = self.lat.to(u)[:, None]
        cosine = lat.cos().clamp_min(1e-4)
        coriolis = 2 * self.rotation * lat.sin()
        curvature = lat.sin() / (self.radius * cosine)
        def advection(value):
            east, north = self._gradient(value)
            return u * east + v * north + omega * self._dp(value)
        phi_east, phi_north = self._gradient(height * self.gravity)
        known_u = -advection(u) + (coriolis + u * curvature) * v - phi_east
        known_v = -advection(v) - coriolis * u - u.square() * curvature - phi_north
        known_t = (-advection(temp) + self.gas_constant / self.heat_capacity * temp * omega
                   / self.pressure.to(temp)[None, :, None, None])
        return torch.stack((known_u, known_v, known_t), dim=1)

    def forward(self, decoded_info0, decoded_info1, observed_info0, observed_info1, z0, dt_hours):
        """Return unweighted PINN metrics; the trainer applies config.weight/ramp.

        All information arguments are normalized flattened [batch, info_dim].
        Observed endpoints are supervision/masks, never closure conditioning.
        Midpoint decoded fields supply spatial physics. Temporal differences are
        physical-time secants; six-hour pairs are not instantaneous derivatives.
        """
        infos = (decoded_info0, decoded_info1, observed_info0, observed_info1)
        if any(x.ndim != 2 or x.shape != decoded_info0.shape or x.shape[-1] != self.info_dim for x in infos):
            raise ValueError("PINN needs matching [batch, information_features] endpoint tensors")
        if not decoded_info0.is_floating_point() or decoded_info0.shape[0] == 0:
            raise ValueError("PINN requires a nonempty floating-point batch")
        if any(not torch.isfinite(x).all() for x in infos) or not torch.isfinite(z0).all():
            raise ValueError("PINN endpoints and latent state must be finite")
        if z0.shape != (decoded_info0.shape[0], self.latent_dim):
            raise ValueError("PINN z0 must have [batch, latent_dim] shape")
        dt = torch.as_tensor(dt_hours, device=decoded_info0.device, dtype=decoded_info0.dtype)
        if dt.ndim == 0:
            dt = dt.expand(decoded_info0.shape[0])
        if dt.shape != (decoded_info0.shape[0],) or not torch.isfinite(dt).all() or torch.any(dt <= 0):
            raise ValueError("PINN dt_hours must be a positive finite scalar or [batch] vector")
        dt = dt[:, None, None, None] * 3600.0
        pred0, pred1 = self._physical(decoded_info0), self._physical(decoded_info1)
        obs0, obs1 = self._physical(observed_info0.detach()), self._physical(observed_info1.detach())
        mask = self._stencil_mask(obs0, obs1)
        state = self._fields((pred0 + pred1) * 0.5)
        before, after = self._fields(pred0), self._fields(pred1)
        closure_unit = self.closure_head(z0).reshape(-1, 3, self.nlevels, *self.shape[1:])
        closure = closure_unit * self.closure_scales
        predicted_rate = torch.stack([(after[key] - before[key]) / dt for key in "uvt"], dim=1)
        residual = (predicted_rate - self._known_tendency(state) - closure) / self.closure_scales
        momentum = 0.5 * (self._mean_square(residual[:, 0], mask) + self._mean_square(residual[:, 1], mask))
        thermal = self._mean_square(residual[:, 2], mask)
        continuity = self._divergence(state["u"], state["v"]) + self._dp(state["w"])
        continuity = self._mean_square(continuity / self.continuity_scale, mask)
        virtual_t = state["t"] * (1.0 + 0.61 * state["q"]) if self.has_humidity else state["t"]
        log_ratio = torch.log(self.pressure[1:] / self.pressure[:-1]).to(pred0)[None, :, None, None]
        expected = self.gas_constant / self.gravity * 0.5 * (virtual_t[:, :-1] + virtual_t[:, 1:]) * log_ratio
        thickness_error = state["z"][:, :-1] - state["z"][:, 1:] - expected
        thickness = self._mean_square(thickness_error / self.thickness_scale, mask[:, :-1] & mask[:, 1:])
        closure_loss = sum(self._mean_square(closure_unit[:, index], mask) for index in range(3)) / 3
        # Supervise every selected dynamic endpoint, including height/omega/sp
        # even though there is no independent prognostic PDE for these here.
        rate_scales = {"u": self.momentum_scale, "v": self.momentum_scale,
                       "t": self.temperature_scale, "z": 1e-2, "w": 1e-5, "q": 1e-7}
        tendency_terms = []
        for key, indices in self.indices.items():
            error = ((pred1[:, indices] - pred0[:, indices]) - (obs1[:, indices] - obs0[:, indices])) / dt
            tendency_terms.append(self._mean_square(error / rate_scales[key], mask))
        sp_error = ((pred1[:, self.sp_index] - pred0[:, self.sp_index])
                    - (obs1[:, self.sp_index] - obs0[:, self.sp_index])) / dt[:, 0]
        tendency_terms.append(self._mean_square(sp_error / 0.05, mask[:, 0]))
        tendency = sum(tendency_terms) / len(tendency_terms)
        total = (momentum + thermal + self.config.continuity_weight * continuity
                 + self.config.thickness_weight * thickness + self.config.closure_weight * closure_loss
                 + self.config.tendency_weight * tendency)
        metrics = {"pinn_momentum": momentum, "pinn_thermodynamic": thermal,
                   "pinn_continuity": continuity, "pinn_thickness": thickness,
                   "pinn_closure": closure_loss, "pinn_tendency": tendency, "pinn_total": total,
                   "pinn_valid_fraction": mask.to(pred0.dtype).mean().detach(),
                   "pinn_closure_normalized_rms": closure_loss.detach().sqrt()}
        # These detached physical RMS diagnostics distinguish improvements in
        # raw tendency from changes in normalized loss or a growing closure.
        for index, (key, unit) in enumerate((("u", "ms2"), ("v", "ms2"), ("t", "Ks"))):
            scale = self.closure_scales[0, index, 0, 0, 0]
            observed_rate = (obs1[:, self.indices[key]] - obs0[:, self.indices[key]]) / dt
            metrics[f"pinn_residual_{key}_rms_{unit}"] = self._mean_square(residual[:, index].detach() * scale, mask).sqrt()
            metrics[f"pinn_closure_{key}_rms_{unit}"] = self._mean_square(closure[:, index].detach(), mask).sqrt()
            metrics[f"pinn_predicted_{key}_tendency_rms_{unit}"] = self._mean_square(predicted_rate[:, index].detach(), mask).sqrt()
            metrics[f"pinn_observed_{key}_tendency_rms_{unit}"] = self._mean_square(observed_rate, mask).sqrt()
        if any(not torch.isfinite(value).all() for value in metrics.values()):
            raise ValueError("PINN produced nonfinite residuals; check decoded SI values and training stability")
        return metrics

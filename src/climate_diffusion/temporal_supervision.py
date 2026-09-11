"""Physical-time supervision; no new forecasting network or state channels.

All arrays use canonical archive variable order. Future data is supervision only.
The decoder returns states; only endpoint differences define physical tendencies.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from .moe_data import field_grid


def area_weights(schema):
    coords = schema["variables"][0]["coords"]
    lat, lon = np.deg2rad(coords["lat"]), np.deg2rad(coords["lon"])
    if min(len(lat), len(lon)) < 2:
        raise ValueError("At least two latitude/longitude points required")
    area = np.cos(lat)[:, None] * abs(np.gradient(lat))[:, None] * abs(np.gradient(lon))[None, :]
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Invalid nonpositive grid areas")
    return area / area.sum()


class TemporalWindowDataset(Dataset):
    """Causal context + x0-inclusive trajectory; 120h at 6h = 21 states.

    Raw archive is retained once, not expanded into overlapping stored windows.
    String/grid metadata stays on the dataset, not in tensor batches.
    """
    def __init__(self, states, times, config, starts, mean, scale, schema, observed_mask=None):
        self.states = np.asarray(states, dtype=np.float32)
        self.times = np.asarray(times).astype("datetime64[ns]")
        self.config, self.schema = config, schema
        self.starts = list(starts)
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        if (self.states.ndim != 2 or self.states.shape[1] != config.state_dim
                or self.times.shape != (len(states),) or np.isnat(self.times).any()
                or not np.all(np.diff(self.times) == np.timedelta64(config.step_hours, "h"))):
            raise ValueError("Trajectory states/timestamps violate fixed physical dt")
        if not np.isfinite(self.states).all():
            raise ValueError("Non-finite trajectory states")
        if observed_mask is not None and (observed_mask.shape != self.states.shape or not np.all(observed_mask == 1)):
            raise ValueError("Missing endpoint pairs: fully observed data required")
        if (self.mean.shape != (config.state_dim,) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)):
            raise ValueError("Invalid state normalization")
        count = len(states) - config.history_span_steps - config.horizon_steps + 1
        if not self.starts or any(i < 0 or i >= count for i in self.starts):
            raise ValueError("Invalid trajectory window indices")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        c, i = self.config, self.starts[index]
        o = i + c.history_span_steps - 1
        history = self.states[i:o+1:c.history_stride]
        raw = self.states[o:o+c.horizon_steps+1]
        trajectory = (raw - self.mean) / self.scale
        times = self.times[o:o+c.horizon_steps+1]
        dt = (np.diff(times) / np.timedelta64(1, "h")).astype(np.float32)
        delta = np.diff(raw, axis=0)
        return {"history": torch.as_tensor((history - self.mean) / self.scale, dtype=torch.float32),
                "origin": torch.as_tensor(trajectory[0], dtype=torch.float32),
                "targets": torch.as_tensor(trajectory[1:], dtype=torch.float32),
                "trajectory_raw": torch.from_numpy(raw.copy()),
                "delta_raw": torch.from_numpy(delta),
                "tendency_raw": torch.from_numpy(delta / dt[:, None]),
                "dt_hours": torch.from_numpy(dt),
                "pair_observed_mask": torch.ones(delta.shape, dtype=torch.bool),
                "origin_time_ns": torch.tensor(times[0].astype(np.int64)),
                "valid_time_ns": torch.from_numpy(times.astype(np.int64))}


class NonSingletonBatchSampler(Sampler):
    """Merge a final singleton into its preceding batch, including validation.

    Peak batch may be requested size + 1. Does not silently disable metric loss
    or discard observations. At least two samples are required.
    """
    def __init__(self, size, batch_size, *, shuffle=False, generator=None):
        if min(size, batch_size) < 2:
            raise ValueError("Manifold metric requires at least two samples and batch_size >= 2")
        self.size, self.batch_size, self.shuffle, self.generator = size, batch_size, shuffle, generator

    def __iter__(self):
        indices = torch.randperm(self.size, generator=self.generator).tolist() if self.shuffle else list(range(self.size))
        batches = [indices[i:i+self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size - int(self.size % self.batch_size == 1)


def fit_temporal_statistics(states, times, schema, state_scale, train_end, variable_weights=None):
    """Streaming float64 moments on unique adjacent TRAIN pairs only."""
    grid = field_grid(schema)
    names = [v["name"] for v in schema["variables"]]
    units = {"msl": {"Pa", "pa"}, "t2m": {"K", "kelvin"},
             "u10": {"m/s", "m s**-1", "m s-1"}, "v10": {"m/s", "m s**-1", "m s-1"}}
    for v in schema["variables"]:
        unit = v.get("attrs", {}).get("units")
        if v["name"] in units and unit is not None and unit not in units[v["name"]]:
            raise ValueError(f"Noncanonical units for {v['name']}: {unit}")
    if not 2 <= train_end <= len(states):
        raise ValueError("Invalid train statistics span")
    dt = np.diff(np.asarray(times)[:train_end]) / np.timedelta64(1, "h")
    if not np.isfinite(dt).all() or not np.all(dt == schema["forecast_step_hours"]):
        raise ValueError("Train statistics require fixed positive physical dt")
    area = area_weights(schema)
    total, second, count = np.zeros(grid[0]), np.zeros(grid[0]), 0
    speed2 = 0.
    ui, vi = names.index("u10"), names.index("v10")
    for start in range(0, train_end-1, 256):
        stop = min(start+256, train_end-1)
        raw = np.asarray(states[start:stop+1], dtype=np.float64).reshape(-1, *grid)
        v = np.diff(raw, axis=0) / dt[start:stop, None, None, None]
        total += (v * area).sum((0, 2, 3))
        second += (v ** 2 * area).sum((0, 2, 3))
        count += len(v)
    for start in range(0, train_end, 256):
        raw = np.asarray(states[start:min(start+256, train_end)], dtype=np.float64).reshape(-1, *grid)
        speed2 += ((raw[:, ui] ** 2 + raw[:, vi] ** 2) * area).sum()
    state_rms = np.sqrt((np.asarray(state_scale).reshape(grid) ** 2 * area).sum((1, 2)))
    floor = np.maximum(1e-3 * state_rms / float(dt[0]), 1e-8)
    b = np.maximum(np.sqrt(np.maximum(second/count - (total/count)**2, 0)), floor)
    weights = {name: 1. for name in names}
    if variable_weights:
        if set(variable_weights) - set(names):
            raise ValueError("Unknown variable weight")
        weights.update({name: float(value) for name,value in variable_weights.items()})
    if any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Variable weights must be finite and positive")
    result = {"format": "climate_diffusion.temporal_statistics.v1", "train_span": [0, train_end],
              "unique_pair_count": count, "step_hours": float(dt[0]), "names": names,
              "tendency_mean": (total/count).tolist(), "tendency_scale": b.tolist(),
              "tendency_floor": floor.tolist(), "variable_weights": weights,
              "wind_speed_scale": max(float(np.sqrt(speed2/train_end)), 1e-3),
              "unit_policy": "validate declared units; missing attrs assume Pa/K/m/s by canonical variable name"}
    result["calm_threshold_mps"] = max(0.05 * result["wind_speed_scale"], 1e-3)
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def fair_energy(features, target):
    """[B,M,F] → scalar. Features already include all metric/size weights."""
    if features.ndim != 3 or features.shape[1] < 2 or target.shape != features.shape[:1] + features.shape[2:]:
        raise ValueError("Energy requires [B,M>=2,F], target [B,F]")
    m = features.shape[1]
    accuracy = torch.linalg.vector_norm(features - target[:, None], dim=-1).mean()
    pairs = torch.linalg.vector_norm(features[:, :, None] - features[:, None, :], dim=-1)
    return accuracy - pairs.sum((1, 2)).mean() / (2*m*(m-1))


def wind_features(u, v, epsilon):
    """Toward direction: cos=u/speed east, sin=v/speed north; smooth at calm."""
    speed = torch.linalg.vector_norm(torch.stack((u, v), -1), dim=-1)
    denominator = torch.sqrt(u.square() + v.square() + epsilon**2)
    return speed, torch.stack((u/denominator, v/denominator), -1)


class TemporalObjective(nn.Module):
    def __init__(self, schema, mean, scale, statistics):
        super().__init__()
        self.grid, self.stats = field_grid(schema), statistics
        self.names = [v["name"] for v in schema["variables"]]
        check = dict(statistics)
        digest = check.pop("sha256")
        if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != digest:
            raise ValueError("Temporal statistics checksum mismatch")
        if statistics["names"] != self.names:
            raise ValueError("Temporal statistics variable order mismatch")
        area = area_weights(schema)
        weights = np.array([statistics["variable_weights"][n] for n in self.names])
        metric = (weights[:, None, None] / weights.sum() * area).reshape(-1)
        tendency = np.broadcast_to(np.array(statistics["tendency_scale"])[:, None, None], self.grid).copy()
        for name, value in (("mean", mean), ("scale", scale), ("metric", metric),
                            ("area", area), ("tendency_scale", tendency.reshape(-1))):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())

    def state_metrics(self, predicted, target, prefix="state"):
        errors = ((predicted - target).square()).reshape(-1, *self.grid)
        per = (errors * self.area).sum((-2, -1)).mean(0)
        return {f"{prefix}_mse_{n}": per[k] for k, n in enumerate(self.names)}

    def forward(self, samples, target, dt_hours, pair_mask=None):
        """Normalized states [B,M,P,D], [B,P,D]; dt [B,P-1]."""
        b, m, p, d = samples.shape
        if (m < 2 or p < 2 or target.shape != (b,p,d) or dt_hours.shape != (b,p-1)
                or not bool(torch.isfinite(dt_hours).all()) or bool((dt_hours <= 0).any())):
            raise ValueError("Invalid trajectory shapes/member count/physical dt")
        if pair_mask is not None and (pair_mask.shape != (b,p-1,d) or not bool(pair_mask.all())):
            raise ValueError("Missing endpoint pairs are not supported")
        ds = samples.diff(dim=2) * self.scale / dt_hours[:, None, :, None] / self.tendency_scale
        dt = target.diff(dim=1) * self.scale / dt_hours[:, :, None] / self.tendency_scale
        metric = self.metric.sqrt()
        features = torch.cat(((samples * metric / np.sqrt(p)).flatten(2),
                              (ds * metric / np.sqrt(p-1)).flatten(2)), -1)
        truth = torch.cat(((target * metric / np.sqrt(p)).flatten(1),
                           (dt * metric / np.sqrt(p-1)).flatten(1)), -1)
        metrics = {"loss_trajectory": fair_energy(features, truth),
                   "loss_delta": ((ds.mean(1)-dt).square()*self.metric).sum(-1).mean(),
                   "loss_delta_member": ((ds-dt[:,None]).square()*self.metric).sum(-1).mean(),
                   "delta_member_variance_penalty": (ds.var(1,unbiased=False)*self.metric).sum(-1).mean(),
                   "increment_spread": (ds.std(1, unbiased=False)*self.metric).sum(-1).mean(),
                   "trajectory_edges": samples.new_tensor(p-1)}
        metrics.update(self.state_metrics(samples.mean(1), target))
        metrics.update(self.state_metrics(ds.mean(1), dt, "tendency"))
        physical = (samples*self.scale+self.mean).reshape(b,m,p,*self.grid)
        observed = (target*self.scale+self.mean).reshape(b,p,*self.grid)
        ui, vi = self.names.index("u10"), self.names.index("v10")
        eps = self.stats["calm_threshold_mps"] * 0.01
        speed, direction = wind_features(physical[:,:,:,ui], physical[:,:,:,vi], eps)
        true_speed, true_direction = wind_features(observed[:,:,ui], observed[:,:,vi], eps)
        speed = speed / self.stats["wind_speed_scale"]
        true_speed = true_speed / self.stats["wind_speed_scale"]
        # Fair scalar CRPS, averaged over leads/cells with a common area measure.
        accuracy = (speed-true_speed[:,None]).abs().mean(1)
        spread = (speed[:,:,None]-speed[:,None,:]).abs().sum((1,2)) / (2*m*(m-1))
        metrics["loss_wind_speed"] = ((accuracy-spread)*self.area).sum((-2,-1)).mean()
        valid = (true_speed*self.stats["wind_speed_scale"] > self.stats["calm_threshold_mps"]).to(samples)
        # Truth-only calm weights: predicted calm cannot erase its direction penalty.
        direction_error = (direction.mean(1)-true_direction).square().sum(-1)
        denom = (valid*self.area).sum().clamp_min(1e-12)
        metrics["loss_wind_direction"] = (direction_error*valid*self.area).sum()/denom
        metrics["wind_direction_valid_fraction"] = valid.mean()
        return metrics


def select_block(batch, edges, generator):
    """Return normalized truth block and integer physical lead steps, including 0."""
    targets = torch.cat((batch["origin"][:, None], batch["targets"]), 1)
    b, points, _ = targets.shape
    edges = points-1 if edges == 0 else min(edges, points-1)
    if edges < 1:
        raise ValueError("trajectory_edges must be zero (full window) or positive")
    start = torch.randint(points-edges, (b,1), device=targets.device, generator=generator)
    steps = start + torch.arange(edges+1, device=targets.device)[None]
    rows = torch.arange(b, device=targets.device)[:, None]
    dt = batch["dt_hours"][rows, steps[:, :-1]]
    mask = batch["pair_observed_mask"][rows, steps[:, :-1]]
    return targets[rows, steps], steps, dt, mask

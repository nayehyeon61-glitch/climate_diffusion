"""Manufactured physical fields test SI signs, geometry, masks and gradients."""
import copy
import math

import numpy as np
import pytest
import torch

from climate_diffusion.hybrid_pinn import HybridPINN, HybridPINNConfig


def setup_pinn(levels=(500, 850), lat=None, lon=None, humidity=False):
    lat = np.linspace(-60, 60, 7) if lat is None else np.asarray(lat)
    lon = np.arange(12) * 30.0 if lon is None else np.asarray(lon)
    units = {"u": "m/s", "v": "m/s", "t": "K", "z": "m", "w": "Pa/s", "q": "kg/kg"}
    variables = [dict(name=f"{name}{p}", unit=units[name], pressure_hpa=p, kind="dynamic")
                 for name in "uvtzw" + ("q" if humidity else "") for p in levels]
    variables += [dict(name=name, unit=unit, pressure_hpa=None, kind=kind) for name, unit, kind in
                  (("sp", "Pa", "dynamic"), ("terrain_height", "m", "static"), ("terrain_slope", "1", "static"))]
    metadata = dict(variables=variables, grid=dict(lat=lat.tolist(), lon=lon.tolist()),
                    shape=[len(variables), len(lat), len(lon)])
    size = math.prod(metadata["shape"])
    model = HybridPINN(HybridPINNConfig(levels_hpa=levels), metadata, np.zeros(size), np.ones(size), 4, 8).double()
    state = torch.zeros(2, *metadata["shape"], dtype=torch.float64)
    names = [v["name"] for v in variables]
    for p in levels:
        state[:, names.index(f"t{p}")] = 270.0
        state[:, names.index(f"z{p}")] = model.gas_constant * 270 / model.gravity * math.log(1000 / p)
    state[:, names.index("sp")] = 101_000.0
    return model, state, metadata


def evaluate(model, before, after=None, observed0=None, observed1=None, dt=6.0):
    after = before if after is None else after
    observed0 = before if observed0 is None else observed0
    observed1 = after if observed1 is None else observed1
    return model(before.flatten(1), after.flatten(1), observed0.flatten(1), observed1.flatten(1),
                 torch.zeros(before.shape[0], model.latent_dim, dtype=before.dtype), dt)


def test_resting_isothermal_hydrostatic_column_and_thickness_units():
    model, state, _ = setup_pinn()
    values = evaluate(model, state)
    assert values["pinn_total"] < 1e-10
    assert values["pinn_tendency"] == 0
    assert values["pinn_valid_fraction"] == 1
    changed = state.clone()
    changed[:, model.indices["z"][0]] += 100.0  # canonical height is metres, not geopotential
    result = evaluate(model, changed)
    assert result["pinn_thickness"] == pytest.approx(1.0, rel=1e-5)
    assert result["pinn_momentum"] < 1e-12  # constant height offset has no horizontal force


def test_pressure_si_and_nonuniform_levels_support_descending_input():
    model, state, _ = setup_pinn(levels=(900, 300, 650))
    assert model.levels == (300, 650, 900)
    assert model.pressure.tolist() == [30000.0, 65000.0, 90000.0]
    value = model.pressure[None, :, None, None].expand(2, -1, *state.shape[-2:]) * 2e-6 + 0.1
    assert torch.allclose(model._dp(value), torch.full_like(value, 2e-6), atol=1e-15)
    assert evaluate(model, state)["pinn_thickness"] < 1e-10


def test_spherical_derivatives_periodic_seam_and_reversed_latitude():
    model, state, _ = setup_pinn(lat=[-70., -45., -10., 20., 60.], lon=np.arange(64) * 360 / 64)
    lon = model.lon[None, :].expand(*state.shape[-2:])
    lat = model.lat[:, None].expand(*state.shape[-2:])
    east, north = model._gradient(lon.sin() + 2 * lat)
    expected_east = lon.cos() * math.sin(2 * math.pi / 64) / (2 * math.pi / 64) / (model.radius * lat.cos())
    assert torch.allclose(east, expected_east, atol=1e-11, rtol=1e-5)
    assert torch.allclose(north, torch.full_like(north, 2 / model.radius), atol=1e-14)
    reverse, _, _ = setup_pinn(lat=np.rad2deg(model.lat.numpy())[::-1], lon=np.arange(64) * 360 / 64)
    reversed_east, reversed_north = reverse._gradient((lon.sin() + 2 * lat).flip(0))
    assert torch.allclose(reversed_east.flip(0), east, rtol=1e-5, atol=1e-12)
    assert torch.allclose(reversed_north.flip(0), north, rtol=1e-5, atol=1e-12)
    constant = torch.ones_like(lon)
    assert model._gradient(constant)[0].abs().max() == 0


def test_regional_longitude_does_not_wrap():
    model, state, _ = setup_pinn(lon=[10., 20., 35., 55., 80.])
    assert not model.periodic_lon
    longitude = model.lon[None, :].expand(*state.shape[-2:])
    assert torch.allclose(model._dlon(longitude), torch.ones_like(longitude), atol=1e-14)


def test_solid_body_spherical_momentum_balance():
    model, state, _ = setup_pinn(lat=np.linspace(-75, 75, 121), lon=np.arange(8) * 45)
    lat = model.lat[:, None]
    speed = 15.0
    u = speed * lat.cos()
    # dPhi/dphi balances both Coriolis and the spherical centrifugal term.
    phi = -(model.radius * model.rotation * speed + speed ** 2 / 2) * lat.sin().square()
    for ui, zi in zip(model.indices["u"], model.indices["z"]):
        state[:, ui] = u
        state[:, zi] += phi / model.gravity
    known = model._known_tendency(model._fields(state))
    assert known[:, 0].abs().max() < 1e-12
    assert known[:, 2].abs().max() < 1e-12
    # Finite differences approximate the analytic balance; avoid one-sided edges.
    assert known[:, 1, :, 1:-1].abs().max() < 1e-6


def test_continuity_uses_spherical_metric_and_pressure_derivative():
    model, state, _ = setup_pinn()
    u = torch.zeros(2, model.nlevels, *state.shape[-2:], dtype=state.dtype)
    v = torch.ones_like(u) * 5
    lat = model.lat[:, None]
    # Constant northward velocity has nonzero spherical divergence.
    divergence = model._divergence(u, v)
    assert divergence[:, :, 1:-1].abs().max() > 0
    assert torch.all(torch.sign(divergence[0, 0, 1:-1, 0]) == torch.sign(-lat[1:-1, 0]))
    omega = -divergence * (model.pressure[None, :, None, None] - model.pressure[0])
    assert (divergence + model._dp(omega)).abs().max() < 1e-15


def test_seconds_conversion_observed_tendency_and_gradient_path():
    model, state, _ = setup_pinn()
    observed_next = state.clone()
    observed_next[:, model.indices["t"]] += 2.16  # 1e-4 K/s over six hours
    stationary = evaluate(model, state, observed0=state, observed1=observed_next)
    assert stationary["pinn_tendency"] > 0
    prediction = observed_next.clone().requires_grad_()
    values = evaluate(model, state, prediction, observed0=state, observed1=observed_next)
    assert values["pinn_tendency"] == 0
    assert values["pinn_thermodynamic"].item() == pytest.approx(1.0, rel=1e-6)
    doubled = evaluate(model, state, prediction, dt=12.0)
    assert doubled["pinn_thermodynamic"].item() == pytest.approx(0.25, rel=1e-6)
    assert values["pinn_residual_t_rms_Ks"].item() == pytest.approx(1e-4, rel=1e-6)
    assert values["pinn_predicted_t_tendency_rms_Ks"].item() == pytest.approx(1e-4, rel=1e-6)
    assert values["pinn_observed_t_tendency_rms_Ks"].item() == pytest.approx(1e-4, rel=1e-6)
    assert values["pinn_closure_t_rms_Ks"].item() == 0
    values["pinn_total"].backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert model.closure_head[-1].bias.grad.abs().sum() > 0


def test_poles_and_below_ground_stencil_masks_have_finite_gradients():
    model, state, _ = setup_pinn(lat=np.linspace(-90, 90, 9))
    observed = state.clone()
    observed[:, model.sp_index, 4, 5] = 80_000  # 850 hPa is below terrain here
    before = observed.clone()
    after = observed.clone().requires_grad_()
    after_data = after.detach().clone()
    after_data[:, model.indices["u"][0], 4, 5] = 1e6
    after_data.requires_grad_()
    clean = evaluate(model, before, after, observed0=observed, observed1=observed)
    contaminated = evaluate(model, before, after_data, observed0=observed, observed1=observed)
    assert torch.allclose(clean["pinn_total"], contaminated["pinn_total"], atol=1e-12)
    assert 0 < contaminated["pinn_valid_fraction"] < 1
    contaminated["pinn_total"].backward()
    assert torch.isfinite(after_data.grad).all()
    assert after_data.grad[:, model.indices["u"][0], 4, 5].abs().sum() == 0
    observed[:, model.sp_index] = 40_000
    with pytest.raises(ValueError, match="no valid"):
        evaluate(model, observed)


def test_partial_humidity_and_wrong_canonical_units_fail():
    model, state, metadata = setup_pinn(humidity=True)
    assert model.has_humidity
    for field, bad_unit in (("z500", "m2 s-2"), ("w850", "m/s"), ("sp", "hPa")):
        bad = copy.deepcopy(metadata)
        next(v for v in bad["variables"] if v["name"] == field)["unit"] = bad_unit
        with pytest.raises(ValueError, match="canonical"):
            HybridPINN(model.config, bad, model.info_mean, model.info_scale, 4, 8)
    partial = copy.deepcopy(metadata)
    i = next(i for i, v in enumerate(partial["variables"]) if v["name"] == "q850")
    partial["variables"].pop(i)
    partial["shape"][0] -= 1
    with pytest.raises(ValueError, match="humidity"):
        HybridPINN(model.config, partial, torch.zeros(math.prod(partial["shape"])),
                   torch.ones(math.prod(partial["shape"])), 4, 8)


@pytest.mark.parametrize("kwargs", [dict(levels_hpa=(500, 500)), dict(levels_hpa=(500,)),
                                      dict(levels_hpa=(500.5, 850)), dict(weight=float("nan")),
                                      dict(closure_weight=-1), dict(tendency_weight=0),
                                      dict(closure_weight=0), dict(weight=0),
                                      dict(warmup_epochs=1.5), dict(ramp_epochs=True), dict(ramp_epochs=0)])
def test_config_rejects_invalid_physics_choices(kwargs):
    with pytest.raises(ValueError):
        HybridPINNConfig(**kwargs)


def test_missing_full_upper_air_fields_and_invalid_time_fail():
    model, state, metadata = setup_pinn()
    bad = copy.deepcopy(metadata)
    bad["variables"][0]["name"] = "u10"
    with pytest.raises(ValueError, match="missing"):
        HybridPINN(model.config, bad, model.info_mean, model.info_scale, 4, 8)
    for dt in (0, -6, float("nan"), [6, 6, 6]):
        with pytest.raises(ValueError, match="dt_hours"):
            evaluate(model, state, dt=dt)


def test_affine_information_normalization_preserves_physical_residuals():
    reference, state, metadata = setup_pinn()
    next_state = state.clone()
    next_state[:, reference.indices["u"]] += 3.0
    expected = evaluate(reference, state, next_state)
    mean = torch.linspace(-1, 1, reference.info_dim)
    scale = torch.linspace(0.5, 2, reference.info_dim)
    normalized = HybridPINN(reference.config, metadata, mean, scale, 4, 8).double()
    z = torch.zeros(2, 4, dtype=torch.float64)
    before = (state.flatten(1) - normalized.info_mean) / normalized.info_scale
    after = (next_state.flatten(1) - normalized.info_mean) / normalized.info_scale
    actual = normalized(before, after, before, after, z, 6)
    for key in ("pinn_momentum", "pinn_thermodynamic", "pinn_continuity", "pinn_thickness", "pinn_tendency"):
        assert torch.allclose(actual[key], expected[key], atol=1e-9, rtol=1e-6)

import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import importlib.util

spec = importlib.util.spec_from_file_location('prepare_era5_extra', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_era5_extra.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def archive(tmp_path, times=None):
    times = np.arange(np.datetime64('2001-01-31T18'), np.datetime64('2001-02-01T19'), np.timedelta64(6, 'h')) if times is None else times
    path = tmp_path / 'surface.npz'
    np.savez(path, times=times)
    coords = {'lat': [-60., 0., 60.], 'lon': [0., 90., 180., 270.]}
    schema = {'forecast_step_hours': 6, 'variables': [
        dict(name=n, dims=['lat', 'lon'], coords=coords, shape=[3, 4])
        for n in ('msl', 't2m', 'u10', 'v10')]}
    path.with_suffix('.schema.json').write_text(json.dumps(schema))
    return path


def source(request, terrain=False):
    times = pd.to_datetime([f'{request["year"][0]}-{request["month"][0]}-{d}T{h}'
                            for d in request['day'] for h in request['time']])
    lat, lon = np.array([90., 45., 0., -45., -90.]), np.arange(0., 360., 45.)
    base = np.broadcast_to(lat[:, None] / 10 + np.cos(lon[None, :] * np.pi / 180), (5, 8))
    fields = {}
    for variable in request['variable']:
        short = {'geopotential': 'z', 'u_component_of_wind': 'u', 'v_component_of_wind': 'v',
                 'temperature':'t', 'vertical_velocity':'w', 'surface_pressure':'sp'}[variable]
        if 'pressure_level' not in request:
            values = (100 + base) * p.G if short=='z' else 100000 + base
            da = xr.DataArray(np.broadcast_to(values, (len(times), 5, 8)),
                              dims=['valid_time', 'latitude', 'longitude'],
                              coords=dict(valid_time=times, latitude=lat, longitude=lon))
        else:
            lev = np.array(request['pressure_level'], dtype=int)
            values = np.broadcast_to(base, (len(times), len(lev), 5, 8)).copy()
            values += np.arange(len(times))[:, None, None, None]
            values += lev[None, :, None, None] / 10
            da = xr.DataArray(values * (p.G if short == 'z' else 1),
                              dims=['valid_time', 'pressure_level', 'latitude', 'longitude'],
                              coords=dict(valid_time=times, pressure_level=lev, latitude=lat, longitude=lon))
            da.pressure_level.attrs['units'] = 'hPa'
        da.attrs['units'] = {'z':'m**2 s**-2','u':'m s**-1','v':'m s**-1',
                             't':'K','w':'Pa s**-1','sp':'Pa'}[short]
        fields[short] = da
    return xr.Dataset(fields)


def test_plan_does_not_download(tmp_path, monkeypatch):
    path = archive(tmp_path)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=lambda: pytest.fail('network on plan')))
    plan = p.build(path, tmp_path/'extra.nc', tmp_path/'cache', 'linear', 3, False)
    assert plan['snapshots'] == 5
    assert not (tmp_path/'cache').exists()


def test_month_boundary_exact_times(tmp_path):
    times, _, _ = p.read_target(archive(tmp_path))
    jobs, _ = p.requests_for(times, 3)
    assert len(jobs) == 2
    np.testing.assert_array_equal(np.concatenate([j[0] for j in jobs]), times)
    assert jobs[0][1]['z']['pressure_level'] == ['250', '500', '850']
    assert jobs[0][1]['uv']['pressure_level'] == ['850']


def test_gap_rejected(tmp_path):
    with pytest.raises(ValueError, match='gap-free'):
        p.read_target(archive(tmp_path, np.array(['2001-01-01T00', '2001-01-01T12'], dtype='datetime64[h]')))


@pytest.mark.parametrize('which', ['pole', 'longitude'])
def test_invalid_terrain_grid(tmp_path, which):
    path = archive(tmp_path); schema = json.loads(path.with_suffix('.schema.json').read_text())
    for v in schema['variables']:
        v['coords']['lat' if which == 'pole' else 'lon'] = [-90., 0., 90.] if which == 'pole' else [0., 10., 180., 270.]
    path.with_suffix('.schema.json').write_text(json.dumps(schema))
    with pytest.raises(ValueError): p.read_target(path)


def test_convert_units_and_time(tmp_path):
    times, lat, lon = p.read_target(archive(tmp_path)); jobs, _ = p.requests_for(times, 3)
    req = jobs[0][1]['z']; path = tmp_path/'z.nc'; source(req).to_netcdf(path)
    field = p.extract(path, 'z', 850, times[:1], lat, lon, 'linear')
    assert field.attrs['units'] == 'm'
    np.testing.assert_allclose(field[0, :, 0], lat/10 + 1 + 85, atol=1e-5)
    np.testing.assert_array_equal(field.time, times[:1])
    with pytest.raises(ValueError, match='Missing exact'):
        p.extract(path, 'z', 850, times[1:2], lat, lon, 'linear')


def test_periodic_interpolation():
    field = xr.DataArray([[1., 0., -1., 0.]] * 3,
                         dims=['lat', 'lon'], coords={'lat': [-60., 0., 60.], 'lon': [0., 90., 180., 270.]})
    got = p.regrid(field, np.array([0.]), np.array([-45., 315.]), 'linear')
    np.testing.assert_allclose(got, [[.5, .5]])


def test_missing_cells_rejected():
    f = xr.DataArray([[1., np.nan, 0.]]*3, dims=['lat','lon'], coords={'lat': [-60.,0.,60.], 'lon': [0.,120.,240.]})
    with pytest.raises(ValueError, match='Missing input'):
        p.regrid(f, np.array([0.]), np.array([60.]), 'linear')


def test_coarsen_matches_known_bin_means():
    f = xr.DataArray(np.arange(32.).reshape(4,8), dims=['lat','lon'],
                     coords={'lat':[60.,20.,-20.,-60.], 'lon':np.arange(8)*45.})
    out = p.regrid(f, np.array([-40.,40.]), np.array([22.5,112.5,202.5,292.5]), 'coarsen-match')
    np.testing.assert_allclose(out, f.coarsen(lat=2,lon=2).mean().sortby('lat'))
    with pytest.raises(ValueError, match='target latitude'):
        p.regrid(f, np.array([-30.,30.]), np.array([22.5,112.5,202.5,292.5]), 'coarsen-match')


def test_complete_fake_cds_pipeline_cache_and_no_overwrite(tmp_path, monkeypatch):
    calls = []
    class Client:
        def retrieve(self, dataset, request, output):
            calls.append(dataset)
            source(request, terrain=dataset.endswith('single-levels')).to_netcdf(output)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=Client))
    path = archive(tmp_path); output = tmp_path/'extra.nc'; cache = tmp_path/'cache'
    p.build(path, output, cache, 'linear', 3, True)
    times, lat, lon = p.read_target(path)
    with xr.open_dataset(output) as ds:
        assert set(ds.data_vars) == set(p.NAMES) | {'terrain_height'}
        assert ds.terrain_height.dims == ('lat', 'lon')
        np.testing.assert_allclose(ds.terrain_height[:, 0], 100 + lat/10 + 1)
        np.testing.assert_array_equal(ds.time, times)
        np.testing.assert_array_equal(ds.lat, lat); np.testing.assert_array_equal(ds.lon, lon)
        assert all(np.isfinite(ds[v]).all() for v in ds.data_vars)
    assert len(calls) == 5
    p.build(path, tmp_path/'extra2.nc', cache, 'linear', 3, True)
    assert len(calls) == 5  # cached raw files, no new downloads
    with pytest.raises(FileExistsError): p.build(path, output, cache, 'linear', 3, True)
    cached = next(cache.glob('*.nc')); cached.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        p.build(path, tmp_path/'extra3.nc', cache, 'linear', 3, True)


def test_63_year_plan_without_download(tmp_path, monkeypatch):
    times = np.arange(np.datetime64('1959-01-01T00'), np.datetime64('2022-01-01T00'), np.timedelta64(6, 'h'))
    path = archive(tmp_path, times)
    schema = json.loads(path.with_suffix('.schema.json').read_text())
    for v in schema['variables']:
        v['coords'] = dict(lat=np.linspace(-84.375,84.375,16).tolist(), lon=np.arange(32).tolist())
        v['coords']['lon'] = (np.arange(32)*11.25).tolist()
        v['shape'] = [16,32]
    path.with_suffix('.schema.json').write_text(json.dumps(schema))
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=lambda: pytest.fail('no network for plan')))
    plan = p.build(path, tmp_path/'none.nc', tmp_path/'cache', 'linear', 3, False)
    assert plan['snapshots'] == 92044 and plan['coverage'] == 'full_archive'
    assert plan['native_uncompressed_estimate_TB'] == pytest.approx(1.9113)
    assert plan['aligned_dynamic_float32_estimate_GB'] == pytest.approx(.942531)
    probe = p.build(path, tmp_path/'probe.nc', tmp_path/'cache', 'linear', 3, False, probe_days=2)
    assert probe['snapshots'] == 8 and probe['archive_snapshots'] == 92044
    assert probe['coverage'] == 'probe_subset_not_for_full_training'


def test_probe_preserves_archive_and_marks_partial_output(tmp_path, monkeypatch):
    class Client:
        def retrieve(self, dataset, request, output):
            source(request, terrain=dataset.endswith('single-levels')).to_netcdf(output)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=Client))
    path = archive(tmp_path); original = p.sha256(path)
    p.build(path, tmp_path/'probe.nc', tmp_path/'cache', 'linear', 3, True, probe_days=1)
    assert p.sha256(path) == original
    with xr.open_dataset(tmp_path/'probe.nc') as ds:
        assert ds.sizes['time'] == 4
        assert ds.attrs['coverage'] == 'probe_subset_not_for_full_training'
        assert ds.attrs['archive_snapshots'] == 5

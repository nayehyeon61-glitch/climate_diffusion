"""PINN data contracts: real pressure velocity, matched levels and immutable shards."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_era5_extra as prep
from stream_era5_extra import produce, plan_for
from smoke_moe import synthetic_archive
from test_prepare_era5_extra import source
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.physical_information import (
    PRIMARY, _field, prepare, load_information, pinn_optional,
)


class FakeCDS:
    def __init__(self): self.calls = []

    def retrieve(self, dataset, request, output):
        self.calls.append((dataset, request))
        source(request).to_netcdf(output)


@pytest.fixture
def archive(tmp_path):
    return synthetic_archive(tmp_path, count=12)[0]


def test_pinn_requests_are_opt_in_and_co_located(archive):
    times, _, _ = prep.read_target(archive)
    old, old_terrain = prep.requests_for(times, 3)
    new, new_terrain = prep.requests_for(times, 3, pinn=True)
    assert prep.dynamic_names() == ('z850','z500','z250','u850','v850')
    assert set(old[0][1]) == {'z','uv'}
    assert old[0][1]['uv']['pressure_level'] == ['850']
    assert old[0][1]['z'] == new[0][1]['z'] and old_terrain == new_terrain
    assert new[0][1]['uv']['pressure_level'] == ['500','850']
    assert new[0][1]['tw']['pressure_level'] == ['500','850']
    assert new[0][1]['tw']['variable'] == ['temperature','vertical_velocity']
    assert new[0][1]['sp']['variable'] == ['surface_pressure']
    assert 'pressure_level' not in new[0][1]['sp']
    assert prep.request_dataset('sp') == 'reanalysis-era5-single-levels'
    assert set(prep.dynamic_names(True)) == set(PRIMARY[:-2]) | set(pinn_optional())
    three, _ = prep.requests_for(times, 3, True, (250,500,850))
    assert three[0][1]['uv']['pressure_level'] == ['250','500','850']
    assert {'t250','u250','v250','w250'}.issubset(prep.dynamic_names(True,(250,500,850)))
    with pytest.raises(ValueError, match='require'): prep.dynamic_names(False,(250,500,850))
    with pytest.raises(ValueError, match='levels'): prep.dynamic_names(True,(500,500,850))


def test_pressure_velocity_field_selection_and_unit_rejection(archive,tmp_path):
    times, lat, lon = prep.read_target(archive)
    jobs, _ = prep.requests_for(times,3,pinn=True)
    request = jobs[0][1]['tw']
    ds = source(request)
    ds = ds.assign_coords(pressure_level=ds.pressure_level*100)
    ds.pressure_level.attrs['units']='Pa'
    path=tmp_path/'omega.nc'; ds.to_netcdf(path)
    got=prep.extract(path,'w',500,times,lat,lon,'linear')
    assert got.attrs['units']=='Pa/s'
    np.testing.assert_allclose(got[0,:,0], lat/10+1+50, atol=1e-5)
    np.testing.assert_array_equal(got.time,times)
    assert _field(ds,'w850').sizes['valid_time']==len(times)
    ds.w.attrs['units']='m/s'; bad=tmp_path/'geometric-w.nc'; ds.to_netcdf(bad)
    with pytest.raises(ValueError,match='pressure velocity'):
        prep.extract(bad,'w',500,times,lat,lon,'linear')


def test_pinn_end_to_end_units_npz_and_missing_data(archive,tmp_path,monkeypatch):
    client=FakeCDS()
    monkeypatch.setitem(sys.modules,'cdsapi',types.SimpleNamespace(Client=lambda:client))
    output=tmp_path/'pinn.nc'
    plan=prep.build(archive,output,tmp_path/'cache','linear',3,True,pinn=True)
    assert plan['pinn_levels_hpa']==[500,850]
    assert plan['maximum_requests_before_cache']==5
    assert len(client.calls)==5
    with xr.open_dataset(output) as source_ds: ds=source_ds.load()
    assert set(ds.data_vars)==set(prep.dynamic_names(True)) | {'terrain_height'}
    assert ds.w850.attrs['units']=='Pa/s' and ds.sp.attrs['units']=='Pa'
    assert ds.t500.attrs['units']=='K' and ds.z500.attrs['units']=='m'
    npz=prepare(archive,output,tmp_path/'pinn.npz',pinn=True)
    _,times,schema=load_moe_archive(archive)
    data,meta=load_information(npz,archive,times,schema)
    names=[v['name'] for v in meta['variables']]
    assert names[-2:]==['terrain_height','terrain_slope']
    assert len(names)==14
    values=data.reshape(len(times),*meta['shape'])
    np.testing.assert_array_equal(values[:,names.index('z500')],ds.z500)
    np.testing.assert_array_equal(values[:,names.index('sp')],ds.sp)
    # No substitution of sea-level pressure or geometric w; no silent fill.
    missing=tmp_path/'missing.nc';ds.drop_vars('sp').to_netcdf(missing)
    with pytest.raises(ValueError,match='sp'):prepare(archive,missing,tmp_path/'missing.npz',pinn=True)
    ds.w850.attrs['units']='m/s';bad=tmp_path/'bad.nc';ds.to_netcdf(bad)
    with pytest.raises(ValueError,match='Pa/s'):prepare(archive,bad,tmp_path/'bad.npz',pinn=True)
    assert not (tmp_path/'missing.npz').exists() and not (tmp_path/'bad.npz').exists()


def test_optional_humidity_and_250_units(archive,tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'cdsapi',types.SimpleNamespace(Client=FakeCDS))
    output=tmp_path/'pinn250.nc'
    prep.build(archive,output,tmp_path/'cache','linear',3,True,pinn=True,pinn_levels=(250,500,850))
    with xr.open_dataset(output) as source_ds: ds=source_ds.load()
    for level in (250,500):
        ds[f'q{level}']=xr.full_like(ds[f't{level}'],.002)
        ds[f'q{level}'].attrs['units']='kg/kg'
    humid=tmp_path/'humid.nc';ds.to_netcdf(humid)
    npz=prepare(archive,humid,tmp_path/'humid.npz',optional=('q250','q500'),pinn=True,pinn_levels=(250,500,850))
    _,times,schema=load_moe_archive(archive)
    _,meta=load_information(npz,archive,times,schema)
    fields={v['name']:v for v in meta['variables']}
    assert fields['w250']['pressure_hpa']==250 and fields['q500']['unit']=='kg/kg'


def test_pinn_shards_resume_and_channel_statistics(archive,tmp_path):
    client=FakeCDS(); store=tmp_path/'store'
    produce(archive,store,client=client,pinn=True,delete_raw=True)
    _,times,schema=load_moe_archive(archive)
    data,meta=load_information(store,archive,times,schema)
    names=[v['name'] for v in meta['variables']]
    assert names[-2:]==['terrain_height','terrain_slope']
    assert data[:].shape==(12,14*4*8)
    mean,scale,tendency=data.statistics(8)
    assert mean.shape==scale.shape==tendency.shape==(14*4*8,)
    assert np.isfinite(mean).all() and (scale>0).all() and (tendency>0).all()
    assert not list((store/'raw').glob('*.nc'))
    before=len(client.calls)
    produce(archive,store,client=client,pinn=True,delete_raw=True)
    assert len(client.calls)==before
    with pytest.raises(ValueError,match='plan changed'):produce(archive,store,client=client)
    plan,*_=plan_for(archive,'linear',3)
    assert plan['fields']==list(PRIMARY) and 'pinn' not in plan
    pinned=json.loads((store/'plan.json').read_text())
    assert pinned['field_units']['w850']=='Pa/s' and pinned['field_units']['sp']=='Pa'

"""Strict sidecar for extra physical conditioning; surface archive is unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from .moe_data import load_moe_archive
from .temporal_supervision import area_weights

FORMAT = 'climate_diffusion.physical_information.v1'
PRIMARY = ('z850','z500','z250','u850','v850','terrain_height','terrain_slope')
OPTIONAL = ('t850','t500','u500','v500','sst','q850',
            't250','u250','v250','w250','w500','w850','q250','q500','sp')
PINN_LEVELS = (500, 850)

def pinn_optional(levels=PINN_LEVELS):
    """Fields needed in addition to PRIMARY for co-located primitive equations."""
    levels=tuple(levels)
    if levels not in ((500,850),(250,500,850)):
        raise ValueError('PINN pressure levels must be 500 850 or 250 500 850, in ascending order')
    return tuple(name for p in levels for key in ('u','v','t','z','w')
                 if (name:=f'{key}{p}') not in PRIMARY)+('sp',)

def canonical_unit(name):
    if name=='terrain_slope': return '1'
    if name.startswith('z') or name=='terrain_height': return 'm'
    if name.startswith(('u','v')): return 'm/s'
    if name.startswith('w'): return 'Pa/s'
    if name=='sp': return 'Pa'
    if name.startswith('q'): return 'kg/kg'
    return 'K'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def terrain_slope(height, lat, lon):
    """Dimensionless |grad height|, spherical metres and periodic longitude."""
    lat,lon=np.deg2rad(lat),np.deg2rad(lon)
    if min(len(lat),len(lon))<3 or np.any(np.abs(np.cos(lat))<1e-5):
        raise ValueError('Slope requires >=3 points per axis and no exact poles')
    if not np.all(np.diff(lat)>0) or not np.allclose(np.diff(lon),2*np.pi/len(lon)):
        raise ValueError('Require ascending latitude and uniform cyclic longitude without duplicate endpoint')
    north=np.gradient(height,lat,axis=-2)/6371000.
    east=(np.roll(height,-1,-1)-np.roll(height,1,-1))/(2*(lon[1]-lon[0])*6371000*np.cos(lat)[:,None])
    return np.hypot(north,east)

def _field(ds,name):
    if name in ds: return ds[name]
    if name[0] in 'zutvqw' and name[1:].isdigit() and name[0] in ds:
        a=ds[name[0]]
        level=next((d for d in ('pressure_level','level','isobaricInhPa') if d in a.dims),None)
        if level is None: raise ValueError(f'Missing pressure dimension for {name}')
        unit=a[level].attrs.get('units')
        if unit not in ('Pa','hPa','millibars'): raise ValueError('Pressure level units must be declared Pa/hPa')
        target=int(name[1:])*(100 if unit=='Pa' else 1)
        if target not in a[level].values: raise ValueError(f'Missing level for {name}')
        return a.sel({level:target})
    raise ValueError(f'Missing required physical variable: {name}')

def prepare(archive,fields,output,optional=(),pinn=False,pinn_levels=PINN_LEVELS):
    output=Path(output)
    if output.suffix!='.npz':raise ValueError('Information output must end in .npz')
    if output.exists(): raise FileExistsError(output)
    _,times,schema=load_moe_archive(archive)
    if set(optional)-set(OPTIONAL) or len(optional)!=len(set(optional)):
        raise ValueError('Unsupported or duplicate optional variable')
    if pinn:
        optional=tuple(dict.fromkeys((*pinn_optional(pinn_levels),*optional)))
    elif tuple(pinn_levels)!=PINN_LEVELS:
        raise ValueError('Custom PINN levels require pinn=True / --pinn')
    if schema['forecast_step_hours']!=6:
        raise ValueError('Physical information requires an exact 6h surface archive')
    coords=schema['variables'][0]['coords']; names=list(PRIMARY)+list(optional)
    if pinn:
        names=[n for n in names if not n.startswith('terrain_')]+['terrain_height','terrain_slope']
    arrays=[]; metadata=[]
    with xr.open_dataset(fields) as ds:
        for axis in ('lat','lon'):
            if axis not in ds.coords or not np.array_equal(ds[axis],coords[axis]):
                raise ValueError('Information grid must exactly match prepared surface grid; no implicit interpolation')
        for name in names:
            static=name.startswith('terrain_')
            if name=='terrain_slope':
                values=terrain_slope(arrays[names.index('terrain_height')][0],coords['lat'],coords['lon'])
                unit='1'; source_unit='derived height spherical gradient'
            else:
                a=_field(ds,name)
                unit=source_unit=a.attrs.get('units')
                if static:
                    if set(a.dims)!={'lat','lon'}: raise ValueError('Static terrain must not have a time axis')
                    values=a.transpose('lat','lon').values
                else:
                    if set(a.dims)!={'time','lat','lon'} or not np.array_equal(a.time.values.astype('datetime64[ns]'),times):
                        raise ValueError('Dynamic information requires exact UTC surface timestamps')
                    values=a.transpose('time','lat','lon').values
                if name.startswith('z') or name=='terrain_height':
                    if unit in ('m**2 s**-2','m2 s-2','m^2/s^2','m**2 s**(-2)'): values=values/9.80665; unit='m'
                    if unit!='m': raise ValueError(f'{name}: declare geopotential or height units')
                elif name.startswith(('u','v')):
                    if unit not in ('m/s','m s**-1','m s-1'): raise ValueError(f'{name}: expected m/s')
                    unit='m/s'
                elif name.startswith('w'):
                    if unit not in ('Pa/s','Pa s**-1','Pa s-1'): raise ValueError(f'{name}: expected pressure velocity Pa/s, not geometric m/s')
                    unit='Pa/s'
                elif name=='sp':
                    if unit!='Pa': raise ValueError('sp: expected surface pressure in Pa, not mean sea-level pressure')
                    if np.any(values<=0): raise ValueError('sp: surface pressure must be positive')
                elif name.startswith('t') or name=='sst':
                    if unit!='K': raise ValueError(f'{name}: expected K')
                elif name.startswith('q') and unit not in ('kg/kg','kg kg**-1','1'):
                    raise ValueError('Humidity must be mass fraction')
            if static: values=np.broadcast_to(values,(len(times),*values.shape))
            if not np.isfinite(values).all(): raise ValueError('Missing information cells: no imputation allowed')
            arrays.append(values.astype(np.float32))
            metadata.append({'name':name,'unit':unit,'source_unit':source_unit,'kind':'static' if static else 'dynamic',
                             'pressure_hpa':int(name[1:]) if name[0] in 'zutvqw' and name[1:].isdigit() else None})
    data=np.stack(arrays,axis=1)
    info={'format':FORMAT,'surface_sha256':digest(archive),'source_sha256':digest(fields),
          'variables':metadata,'grid':coords,'shape':list(data.shape[1:]),
          'time_policy':'UTC exact 6h; origin information fixed throughout forecast',
          'msl':'already in surface input; not duplicated'}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,data=data.reshape(len(times),-1),times=times,
                        observed_mask=np.ones((len(times),data[0].size),dtype=np.uint8),metadata_json=json.dumps(info))
    return output

def validate_information(data, meta, actual_times, observed_mask, times, schema):
    """Validate a complete sidecar or one exact-time shard; no imputation."""
    if meta['grid']!=schema['variables'][0]['coords'] or not np.array_equal(actual_times,times):
        raise ValueError('Information grid/times mismatch')
    if data.shape!=(len(times),int(np.prod(meta['shape']))) or not np.isfinite(data).all():
        raise ValueError('Information shape/nonfinite error')
    if observed_mask.shape!=data.shape or not np.all(observed_mask==1):
        raise ValueError('Missing physical information endpoint')
    names=[v['name'] for v in meta['variables']]
    if len(names)!=len(set(names)) or not set(PRIMARY).issubset(names) or set(names)-set(PRIMARY+OPTIONAL):
        raise ValueError('Invalid physical information variable schema')
    if meta['shape']!=[len(names),*schema['variables'][0]['shape']]:raise ValueError('Information schema shape mismatch')
    cells=int(np.prod(meta['shape'][1:]))
    for i,v in enumerate(meta['variables']):
        name=v['name']
        expected=('kg/kg','kg kg**-1','1') if name.startswith('q') else (canonical_unit(name),)
        if v['unit'] not in expected:raise ValueError('Noncanonical information units')
        pressure=int(name[1:]) if name[0] in 'zutvqw' and name[1:].isdigit() else None
        if v['pressure_hpa']!=pressure:raise ValueError('Information pressure level mismatch')
        if name=='sp' and np.any(data[:,i*cells:(i+1)*cells]<=0):raise ValueError('sp: surface pressure must be positive')
        expected_kind='static' if name.startswith('terrain_') else 'dynamic'
        if v['kind']!=expected_kind:raise ValueError('Static/dynamic schema mismatch')
        if v['kind']=='static' and not np.all(data[:,i*cells:(i+1)*cells]==data[:1,i*cells:(i+1)*cells]):
            raise ValueError('Static terrain changes in time')

def information_digest(path):
    """A sidecar file hash, or immutable plan + metadata identity for shards."""
    path=Path(path)
    if path.is_dir():
        return hashlib.sha256((digest(path/'plan.json')+digest(path/'metadata.json')).encode()).hexdigest()
    return digest(path)

def load_information(path,archive,times,schema):
    if Path(path).is_dir():
        from .information_shards import InformationShards
        data=InformationShards(path,archive,times,schema)
        return data,data.meta
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata_json']));data=f['data'].astype(np.float32)
        if meta['format']!=FORMAT or meta['surface_sha256']!=digest(archive): raise ValueError('Information archive hash/format mismatch')
        validate_information(data,meta,f['times'],f['observed_mask'],times,schema)
    return data,meta

def fit_information(data,meta,train_end,schema):
    """Per-channel train-only area statistics, including static spatial variation."""
    a=data[:train_end].reshape(-1,*meta['shape']).astype(np.float64)
    w=area_weights(schema);mean=(a*w).sum((-2,-1)).mean(0)
    sd=np.sqrt(((a-mean[None,:,None,None])**2*w).sum((-2,-1)).mean(0))
    scale=np.maximum(sd,np.maximum(np.abs(mean)*1e-6,1e-6))
    expand=lambda v:np.broadcast_to(v[:,None,None],meta['shape']).copy().reshape(-1).astype(np.float32)
    return expand(mean),expand(scale)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','fields','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--optional',nargs='*',default=[],choices=OPTIONAL)
    p.add_argument('--pinn',action='store_true',help='Require co-located pressure u/v/t/z/w plus actual surface pressure sp')
    p.add_argument('--pinn-levels',nargs='+',type=int,default=list(PINN_LEVELS),choices=[250,500,850])
    a=p.parse_args(argv);print(prepare(a.archive,a.fields,a.output,a.optional,a.pinn,a.pinn_levels))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prepare required ERA5 extra inputs for the separated A information model.

Default: print an archive-derived request plan, without contacting CDS.
--download: retrieve required fields, regrid in space, preserve exact UTC times.
No model training, surface-archive edits, temporal interpolation or credentials
in this script. Required fields only; optional SST/humidity etc are excluded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

G = 9.80665
NAMES = ('z850', 'z500', 'z250', 'u850', 'v850')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_target(archive):
    archive = Path(archive)
    schema = json.loads(archive.with_suffix('.schema.json').read_text())
    if schema['forecast_step_hours'] != 6:
        raise ValueError('Expected a 6-hour surface archive')
    if [v['name'] for v in schema['variables']] != ['msl', 't2m', 'u10', 'v10']:
        raise ValueError('Expected canonical surface variables msl/t2m/u10/v10')
    with np.load(archive, allow_pickle=False) as f:
        times = f['times'].astype('datetime64[ns]')
    if len(times) < 2 or np.isnat(times).any() or not np.all(np.diff(times) == np.timedelta64(6, 'h')):
        raise ValueError('Archive timestamps must be unique, increasing, gap-free UTC 6h')
    if not np.array_equal(times, times.astype('datetime64[h]').astype('datetime64[ns]')):
        raise ValueError('CDS retrieval needs exact integer-hour timestamps')
    coords = schema['variables'][0]['coords']
    for var in schema['variables']:
        if var['dims'] != ['lat', 'lon'] or var['coords'] != coords:
            raise ValueError('All surface fields must share [lat,lon] coordinates')
    lat, lon = np.asarray(coords['lat'], dtype=float), np.asarray(coords['lon'], dtype=float)
    if min(len(lat), len(lon)) < 3 or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Need finite global coordinates with at least 3 points per axis')
    if not np.all(np.diff(lat) > 0) or np.any(np.abs(lat) >= 90):
        raise ValueError('Current terrain-slope contract needs ascending latitude without exact poles; do not relabel the surface archive')
    if not np.allclose(np.diff(lon), 360 / len(lon)):
        raise ValueError('Terrain-slope contract needs uniform periodic longitude, without duplicate endpoint')
    if any(var['shape'] != [len(lat), len(lon)] for var in schema['variables']):
        raise ValueError('Surface schema coordinate/shape mismatch')
    return times, lat, lon


def requests_for(times, days_per_request):
    dates = pd.DatetimeIndex(times)
    jobs = []
    for month in dates.to_period('M').unique():
        month_times = dates[dates.to_period('M') == month]
        days = np.unique(month_times.day)
        for start in range(0, len(days), days_per_request):
            days_chunk = days[start:start + days_per_request]
            selected = month_times[np.isin(month_times.day, days_chunk)]
            common = dict(product_type=['reanalysis'], year=[f'{month.year:04}'],
                          month=[f'{month.month:02}'], day=[f'{d:02}' for d in days_chunk],
                          time=sorted(set(selected.strftime('%H:%M'))),
                          data_format='netcdf', download_format='unarchived')
            # Separate requests avoid unnecessary u/v at 250 and 500 hPa.
            jobs.append((selected.values, {
                'z': dict(common, variable=['geopotential'], pressure_level=['250', '500', '850']),
                'uv': dict(common, variable=['u_component_of_wind', 'v_component_of_wind'], pressure_level=['850']),
            }))
    first = dates[0]
    terrain = dict(product_type=['reanalysis'], variable=['geopotential'],
                   year=[f'{first.year:04}'], month=[f'{first.month:02}'],
                   day=[f'{first.day:02}'], time=[first.strftime('%H:%M')],
                   data_format='netcdf', download_format='unarchived')
    return jobs, terrain


def retrieve(client, cache, dataset, request):
    identity = json.dumps({'dataset': dataset, 'request': request}, sort_keys=True)
    path = cache / (hashlib.sha256(identity.encode()).hexdigest()[:24] + '.nc')
    record = path.with_suffix('.request.json')
    if path.exists():
        if not record.exists() or json.loads(record.read_text())['sha256'] != sha256(path):
            raise ValueError(f'Cache checksum/receipt mismatch: {path}')
        return path
    partial = path.with_suffix('.download')
    # Only a known incomplete download belonging to this exact request is removed.
    if partial.exists():
        partial.unlink()
    client.retrieve(dataset, request, str(partial))
    with xr.open_dataset(partial) as ds:
        if not ds.data_vars:
            raise ValueError('CDS returned no data variables')
    receipt = {'dataset': dataset, 'request': request, 'sha256': sha256(partial)}
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    partial.rename(path)
    return path


def canonical(ds):
    aliases = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon',
               'level': 'pressure_level', 'isobaricInhPa': 'pressure_level'}
    ds = ds.rename({k: v for k, v in aliases.items() if k in ds.dims and v not in ds.dims})
    for dim in ('number', 'expver'):
        if dim in ds.dims:
            if ds.sizes[dim] != 1:
                raise ValueError(f'Multiple {dim} values: resolve ERA5/ERA5T or ensemble products explicitly')
            ds = ds.isel({dim: 0}, drop=True)
    for dim in ('lat', 'lon'):
        if dim not in ds.coords or ds[dim].dims != (dim,):
            raise ValueError('Expected a regular ERA5 latitude/longitude grid')
    return ds


def regrid(field, lat, lon, method):
    """Spatial operation only; never fills missing inputs or interpolates time."""
    field = field.load()
    if not np.isfinite(field.values).all():
        raise ValueError('Missing input cells: no filling/interpolation across missing data')
    if method == 'coarsen-match':
        # Same block mean as climate_diffusion.data._coarsen_global_fields.
        factors = {k: max(1, int(np.ceil(field.sizes[k] / len(v)))) for k, v in (('lat', lat), ('lon', lon))}
        result = field.coarsen(factors, boundary='pad').mean(skipna=True).sortby('lat')
        result = result.assign_coords(lon=result.lon % 360).sortby('lon')
        target_lon = lon % 360
        if len(np.unique(target_lon)) != len(lon):
            raise ValueError('Duplicate target longitude')
        try:
            result = result.sel(lon=xr.DataArray(target_lon, dims='lon'))
        except KeyError as exc:
            raise ValueError('Block means do not match target longitude; inspect original surface preprocessing or explicitly select --regrid linear') from exc
        if result.sizes['lat'] != len(lat) or not np.allclose(result.lat, lat, atol=1e-8, rtol=0):
            raise ValueError('Block means do not match target latitude; inspect original surface preprocessing or explicitly select --regrid linear')
    else:
        field = field.sortby('lat').assign_coords(lon=field.lon % 360).sortby('lon')
        xlon = field.lon.values
        if len(np.unique(xlon)) != len(xlon) or not np.allclose(np.diff(xlon), 360 / len(xlon)):
            raise ValueError('Linear regridding requires a complete regular periodic source longitude grid')
        if lat.min() < float(field.lat.min()) or lat.max() > float(field.lat.max()):
            raise ValueError('No latitude extrapolation allowed')
        left = field.isel(lon=[-1]).assign_coords(lon=[xlon[-1] - 360])
        right = field.isel(lon=[0]).assign_coords(lon=[xlon[0] + 360])
        periodic = xr.concat([left, field, right], dim='lon')
        result = periodic.interp(lat=xr.DataArray(lat, dims='lat'),
                                 lon=xr.DataArray(lon % 360, dims='lon'), method='linear')
    if not np.isfinite(result.values).all():
        raise ValueError('Nonfinite spatial interpolation result')
    result = result.assign_coords(lat=lat, lon=lon)
    return result.astype('float32')


def extract(path, short, level, selected_times, lat, lon, method, terrain=False):
    with xr.open_dataset(path) as source:
        ds = canonical(source)
        if short not in ds:
            raise ValueError(f'Expected ERA5 variable {short!r}; received {list(ds.data_vars)}')
        field = ds[short]
        if level is not None:
            if 'pressure_level' not in field.dims:
                raise ValueError('Missing pressure_level dimension')
            unit = ds.pressure_level.attrs.get('units')
            if unit not in ('hPa', 'millibars', 'Pa'):
                raise ValueError('Pressure coordinate units must be hPa/Pa')
            field = field.sel(pressure_level=level * (100 if unit == 'Pa' else 1), drop=True)
        if 'time' not in field.dims or not pd.Index(field.time.values).is_unique:
            raise ValueError('Missing or duplicate source valid times')
        try:
            field = field.sel(time=selected_times)
        except KeyError as exc:
            raise ValueError('Missing exact source times; temporal interpolation is forbidden') from exc
        if set(field.dims) != {'time', 'lat', 'lon'}:
            raise ValueError(f'Unexpected source dimensions: {field.dims}')
        if terrain:
            if len(selected_times) != 1:
                raise ValueError('Static terrain retrieval must contain one timestamp')
            field = field.isel(time=0, drop=True)
        units = field.attrs.get('units')
        if short == 'z':
            if units in ('m**2 s**-2', 'm2 s-2', 'm^2/s^2', 'm**2 s**(-2)'):
                field = field / G
            elif units != 'm':
                raise ValueError(f'Unsupported geopotential/height units {units!r}')
            output_unit = 'm'
        else:
            if units not in ('m/s', 'm s**-1', 'm s-1'):
                raise ValueError(f'Unsupported wind units {units!r}')
            output_unit = 'm/s'
        result = regrid(field, lat, lon, method)
        result = result.transpose(*(('lat', 'lon') if terrain else ('time', 'lat', 'lon')))
        result.attrs = {'units': output_unit, 'source_units': units,
                        'spatial_alignment': method}
        return result


def build(archive, output, cache, method, days, download, probe_days=0):
    times, lat, lon = read_target(archive)
    archive_count = len(times)
    if probe_days < 0:
        raise ValueError('probe_days must be nonnegative')
    if probe_days:
        times = times[times < times[0] + np.timedelta64(probe_days, 'D')]
    jobs, terrain_req = requests_for(times, days)
    plan = dict(start=str(times[0]), end=str(times[-1]), snapshots=len(times),
                archive_snapshots=archive_count,
                coverage='full_archive' if len(times) == archive_count else 'probe_subset_not_for_full_training',
                grid=[len(lat), len(lon)], required_fields=list(NAMES) + ['terrain_height'],
                maximum_requests_before_cache=2 * len(jobs) + 1, spatial_alignment=method,
                native_uncompressed_estimate_TB=round(len(times) * 5 * 721 * 1440 * 4 / 1e12, 4),
                native_uncompressed_estimate_GiB=round(len(times) * 5 * 721 * 1440 * 4 / 2**30, 2),
                aligned_dynamic_float32_estimate_GB=round(len(times) * 5 * len(lat) * len(lon) * 4 / 1e9, 6),
                note='Native 0.25-degree downloads can be large. Estimate excludes request supersets and overhead; no optional fields. Exact archive times retained.')
    print(json.dumps(plan, indent=2), flush=True)
    if not download:
        print('Plan only. Add --download after configuring CDS credentials and accepting dataset terms.')
        return plan
    output, cache = Path(output), Path(cache)
    if output.exists() or output.with_suffix('.provenance.json').exists():
        raise FileExistsError('Choose new output/provenance filenames')
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    import cdsapi
    from netCDF4 import Dataset, date2num
    client = cdsapi.Client()
    sources = []
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    os.close(fd)
    try:
        with Dataset(temporary, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', len(times)); nc.createDimension('lat', len(lat)); nc.createDimension('lon', len(lon))
            lc = nc.createVariable('lat', 'f8', ('lat',)); lc[:] = lat; lc.units = 'degrees_north'
            oc = nc.createVariable('lon', 'f8', ('lon',)); oc[:] = lon; oc.units = 'degrees_east'
            tc = nc.createVariable('time', 'i8', ('time',))
            tc.units = 'hours since 1970-01-01 00:00:00'; tc.calendar = 'standard'
            tc[:] = date2num(pd.DatetimeIndex(times).to_pydatetime(), tc.units, tc.calendar)
            nc.source_surface_sha256 = sha256(archive)
            nc.coverage = plan['coverage']
            nc.archive_snapshots = archive_count
            nc.spatial_alignment = method
            nc.time_alignment = 'exact UTC source selection; no temporal interpolation'
            for name in NAMES:
                v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'), zlib=True)
                v.units = 'm' if name.startswith('z') else 'm/s'
            terrain_path = retrieve(client, cache, 'reanalysis-era5-single-levels', terrain_req)
            terrain = extract(terrain_path, 'z', None, times[:1], lat, lon, method, terrain=True)
            tv = nc.createVariable('terrain_height', 'f4', ('lat', 'lon'), zlib=True)
            tv.units = 'm'; tv[:] = terrain.values
            sources.append({'path': str(terrain_path), 'sha256': sha256(terrain_path)})
            for i, (selected, requests) in enumerate(jobs):
                files = {k: retrieve(client, cache, 'reanalysis-era5-pressure-levels', req) for k, req in requests.items()}
                indices = np.searchsorted(times, selected)
                if not np.array_equal(times[indices], selected):
                    raise ValueError('Internal archive index mismatch')
                for name in NAMES:
                    field = extract(files['z' if name.startswith('z') else 'uv'], name[0], int(name[1:]), selected, lat, lon, method)
                    nc.variables[name][indices, :, :] = field.values
                sources.extend({'path': str(p), 'sha256': sha256(p)} for p in files.values())
                nc.sync(); print(f'Aligned block {i + 1}/{len(jobs)}', flush=True)
        # Atomic no-overwrite publication on the same filesystem.
        os.link(temporary, output)
        provenance = {**plan, 'archive': str(archive), 'archive_sha256': sha256(archive),
                      'output_sha256': sha256(output), 'source_files': sources,
                      'scope': 'Required extra fields only. Not a trained model or ERA5 forecast.'}
        with open(output.with_suffix('.provenance.json'), 'x') as f:
            json.dump(provenance, f, indent=2)
        print(f'Prepared: {output}', flush=True)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', help='Default: era5-extra-aligned.nc; a probe defaults to era5-extra-probe.nc, in /workspace/data')
    parser.add_argument('--cache', default='/workspace/data/era5-extra-cds-cache')
    parser.add_argument('--regrid', choices=['linear', 'coarsen-match'], required=True,
                        help='Explicit spatial choice. linear is bilinear point interpolation, not conservative cell averaging.')
    parser.add_argument('--days-per-request', type=int, default=3)
    parser.add_argument('--probe-days', type=int, default=0,
                        help='First N days only, for authentication/format checks; NOT a full-archive training input')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.days_per_request <= 31:
        parser.error('--days-per-request must be 1..31')
    if args.probe_days < 0:
        parser.error('--probe-days must be nonnegative')
    output = args.output or ('/workspace/data/era5-extra-probe.nc' if args.probe_days else '/workspace/data/era5-extra-aligned.nc')
    build(args.archive, output, args.cache, args.regrid, args.days_per_request, args.download, args.probe_days)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Resumable CDS producer: download -> validate compact shard -> remove owned raw.

Default is a plan only. --download contacts CDS; --delete-raw then removes
verified originals. --prune-verified-raw is a separate, network-free cleanup.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from prepare_era5_extra import (
    read_target, requests_for, retrieve, extract, dynamic_names, field_spec,
    request_dataset, PINN_LEVELS,
)
from climate_diffusion.physical_information import FORMAT, digest, terrain_slope, canonical_unit
from climate_diffusion.information_shards import (
    STORE_FORMAT, InformationShards, publish, publish_json, read_json, sync_directory,
)


def plan_for(archive, method, days, pinn=False, pinn_levels=PINN_LEVELS):
    if not 1 <= days <= 31:
        raise ValueError('days-per-request must be 1..31')
    times, lat, lon = read_target(archive)
    names = dynamic_names(pinn, pinn_levels)
    jobs, terrain_request = requests_for(times, days, pinn, pinn_levels)
    ranges = []
    for selected, _ in jobs:
        start = int(np.searchsorted(times, selected[0]))
        ranges.append([start, start+len(selected)])
    plan = dict(format=STORE_FORMAT, surface_sha256=digest(archive),
                schema_sha256=digest(Path(archive).with_suffix('.schema.json')),
                times_sha256=hashlib.sha256(times.tobytes()).hexdigest(),
                start=str(times[0]), stop=str(times[-1]), snapshots=len(times),
                grid=dict(lat=lat.tolist(), lon=lon.tolist()), ranges=ranges,
                spatial_alignment=method, days_per_request=days,
                fields=list(names)+['terrain_height','terrain_slope'], time_policy='exact UTC 6h, no temporal interpolation')
    if pinn:
        plan.update(pinn=True, pinn_levels_hpa=list(pinn_levels),
                    field_units={name:canonical_unit(name) for name in plan['fields']})
    return plan, times, lat, lon, jobs, terrain_request


def raw_sources(paths):
    return [dict(name=p.name, sha256=digest(p), request=read_json(p.with_suffix('.request.json')))
            for p in paths]


def clean_raw(raw, sources):
    """Never delete an external path, symlink, or source changed since conversion."""
    for source in sources:
        name = source['name']
        if Path(name).name != name or not name.endswith('.nc'):
            raise ValueError('Unsafe source name')
        path = raw/name
        if path.is_symlink():
            raise ValueError('Refusing to delete symlink raw file')
        if not path.exists():
            continue  # already removed after a previous successful publication
        receipt = read_json(path.with_suffix('.request.json'))
        identity = json.dumps({k: receipt[k] for k in ('dataset', 'request')}, sort_keys=True)
        if name != hashlib.sha256(identity.encode()).hexdigest()[:24]+'.nc':
            raise ValueError('Raw filename does not belong to recorded request')
        if receipt != source['request'] or digest(path) != source['sha256'] or receipt['sha256'] != source['sha256']:
            raise ValueError('Raw receipt/checksum changed; nothing deleted')
        path.unlink()
        sync_directory(raw)
        # Small request receipt is retained as provenance, never used as data.


def produce(archive, store, method='linear', days=3, delete_raw=False, client=None, on_commit=None,
            pinn=False, pinn_levels=PINN_LEVELS):
    plan, times, lat, lon, jobs, terrain_request = plan_for(archive, method, days, pinn, pinn_levels)
    names = dynamic_names(pinn, pinn_levels)
    store = Path(store)
    if store.is_symlink():
        raise ValueError('Store must not be a symlink')
    store.mkdir(parents=True, exist_ok=True)
    if (store/'.producer.lock').is_symlink():
        raise ValueError('Symlink lock forbidden')
    with open(store/'.producer.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another producer owns this store') from exc
        plan_path = store/'plan.json'
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError('Archive/regrid/chunk plan changed; choose a new store')
        else:
            if any(p.name != '.producer.lock' for p in store.iterdir()):
                raise ValueError('Refusing to adopt nonempty directory without a plan')
            publish_json(plan_path, plan)
        plan_sha = digest(plan_path)
        raw = store/'raw'
        if raw.is_symlink():
            raise ValueError('Raw directory must not be a symlink')
        raw.mkdir(exist_ok=True)
        owner = raw/'owner.json'
        if not owner.exists():
            if any(raw.iterdir()):
                raise ValueError('Refusing to adopt unowned raw files')
            publish_json(owner, {'plan_sha256': plan_sha})
        if read_json(owner) != {'plan_sha256': plan_sha} or any(p.is_symlink() for p in raw.iterdir()):
            raise ValueError('Raw ownership/symlink validation failed')
        # Lazy client: a completed store resumes without credentials or network.
        def get(dataset, request):
            nonlocal client
            if client is None:
                import cdsapi
                client = cdsapi.Client()
            return retrieve(client, raw, dataset, request)

        terrain_path = store/'terrain.npz'
        if not terrain_path.exists():
            source = get('reanalysis-era5-single-levels', terrain_request)
            height = extract(source, 'z', None, times[:1], lat, lon, method, terrain=True).values
            data = np.stack((height, terrain_slope(height, lat, lon))).astype(np.float32)
            sources = raw_sources([source])
            publish(terrain_path, lambda stream: np.savez_compressed(
                stream, data=data, sources_json=json.dumps(sources), plan_sha256=plan_sha))
        if terrain_path.is_symlink():
            raise ValueError('Symlink terrain forbidden')
        with np.load(terrain_path, allow_pickle=False) as f:
            terrain = f['data'].copy()
            terrain_sources = json.loads(str(f['sources_json']))
            if str(f['plan_sha256']) != plan_sha or terrain.shape != (2, len(lat), len(lon)) or not np.isfinite(terrain).all():
                raise ValueError('Invalid committed terrain')
            if not np.allclose(terrain[1], terrain_slope(terrain[0], lat, lon), rtol=1e-6, atol=1e-9):
                raise ValueError('Terrain slope validation failed')
        variables = [dict(name=name, unit=canonical_unit(name),
                          source_unit=('validated CDS SI units; z divided by 9.80665 once; w is Pa/s; sp is Pa; slope derived'
                                       if pinn else 'validated CDS units; z divided by 9.80665; slope derived'),
                          kind='static' if name.startswith('terrain_') else 'dynamic',
                          pressure_hpa=int(name[1:]) if name[0] in 'zutvqw' and name[1:].isdigit() else None)
                     for name in plan['fields']]
        meta = dict(format=FORMAT, surface_sha256=plan['surface_sha256'], source_sha256=plan_sha,
                    variables=variables, grid=plan['grid'], shape=[len(plan['fields']),len(lat),len(lon)],
                    time_policy='UTC exact 6h; origin information fixed throughout forecast',
                    msl='already in surface input; not duplicated', terrain_sha256=digest(terrain_path))
        metadata_path = store/'metadata.json'
        if metadata_path.exists():
            if read_json(metadata_path) != meta:
                raise ValueError('Immutable metadata/terrain changed')
        else:
            publish_json(metadata_path, meta)
        # Constructor verifies terrain and immutable identity before raw removal.
        schema = read_json(Path(archive).with_suffix('.schema.json'))
        reader = InformationShards(store, archive, times, schema)
        if delete_raw:
            clean_raw(raw, terrain_sources)
        for i, (selected, requests) in enumerate(jobs):
            path, receipt = reader.chunk_path(i), reader.receipt_path(i)
            if not receipt.exists():
                if not path.exists():
                    files = {k: get(request_dataset(k), req) for k, req in requests.items()}
                    arrays = []
                    for name in names:
                        group, short, level, _ = field_spec(name)
                        arrays.append(extract(files[group], short, level, selected, lat, lon, method).values)
                    arrays.extend(np.broadcast_to(t, (len(selected), *t.shape)) for t in terrain)
                    data = np.stack(arrays, axis=1).reshape(len(selected), -1).astype(np.float32)
                    sources = raw_sources(files.values())
                    publish(path, lambda stream: np.savez_compressed(stream, data=data, times=selected,
                            observed_mask=np.ones(data.shape, np.uint8), plan_sha256=plan_sha,
                            sources_json=json.dumps(sources)))
                # Recover an orphan file after interruption between publish and receipt.
                reader.validate_chunk(i)
                publish_json(receipt, dict(plan_sha256=plan_sha, range=reader.ranges[i], sha256=digest(path)))
            reader.chunk(i)  # checks receipt, times, grid, units, all masks, static identity
            _, sources = reader.validate_chunk(i)
            if delete_raw:
                clean_raw(raw, sources)
            print(f'Committed {i+1}/{len(jobs)}: {selected[0]} .. {selected[-1]}', flush=True)
            if on_commit is not None:
                on_commit(i, reader)
        complete = store/'complete.json'
        content = dict(plan_sha256=plan_sha, shards=reader.provenance())
        if complete.exists():
            if read_json(complete) != content:
                raise ValueError('Completed store provenance changed')
        else:
            publish_json(complete, content)
    return store


def ready(store, archive, stage, history_steps=6, history_stride=4):
    """Validate all chunks needed by A/B or all stages, including future targets."""
    from climate_diffusion.moe_data import build_moe_split
    if history_steps < 1 or history_stride < 1:
        raise ValueError('History steps/stride must be positive')
    times, _, _ = read_target(archive)
    span = (history_steps-1)*history_stride+1
    split = build_moe_split(len(times)-span-20+1, 20)
    end = len(times) if stage=='all' else split['expert_validation'][-1]+span+20
    schema = read_json(Path(archive).with_suffix('.schema.json'))
    store = Path(store)
    plan = read_json(store/'plan.json')
    required = [i for i, (start, _) in enumerate(plan['ranges']) if start < end]
    for i in required:
        if not (store/'chunks'/f'{i:06d}.json').exists():
            raise FileNotFoundError(f'Waiting for committed chunk {i}')
    reader = InformationShards(store, archive, times, schema)
    for i in required:
        reader.chunk(i)
    return dict(ready_for=stage, snapshots=end, total_snapshots=len(times),
                last_required_time=str(times[end-1]), verified_chunks=len(reader.used))


def prune_verified_raw(store, archive):
    """Idempotent final reconciliation; no network, and no unverified deletion."""
    store = Path(store)
    if store.is_symlink() or (store/'.producer.lock').is_symlink():
        raise ValueError('Symlink store/lock forbidden')
    with open(store/'.producer.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Producer still running; wait before final cleanup') from exc
        times, _, _ = read_target(archive)
        schema = read_json(Path(archive).with_suffix('.schema.json'))
        reader = InformationShards(store, archive, times, schema)
        raw = store/'raw'
        if raw.is_symlink() or read_json(raw/'owner.json') != {'plan_sha256':reader.plan_sha}:
            raise ValueError('Invalid raw owner')
        sources = []
        for i in range(len(reader.ranges)):
            reader.chunk(i)
            _, original = reader.validate_chunk(i)
            sources.extend(original)
        with np.load(store/'terrain.npz',allow_pickle=False) as f:
            sources.extend(json.loads(str(f['sources_json'])))
        clean_raw(raw,sources)
    return dict(verified_sources=len(sources),remaining_nc=len(list(raw.glob('*.nc'))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--store', required=True)
    p.add_argument('--regrid', choices=['linear','coarsen-match'], default='linear')
    p.add_argument('--days-per-request', type=int, default=3)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--download', action='store_true')
    actions.add_argument('--check-ready', choices=['AB','all'])
    actions.add_argument('--prune-verified-raw', action='store_true')
    p.add_argument('--delete-raw', action='store_true')
    p.add_argument('--history-steps', type=int, default=6)
    p.add_argument('--history-stride', type=int, default=4)
    p.add_argument('--pinn', action='store_true', help='Add pressure-level u/v/t/z/w and actual surface pressure sp')
    p.add_argument('--pinn-levels', nargs='+', type=int, choices=[250,500,850], default=list(PINN_LEVELS))
    args = p.parse_args()
    if args.check_ready:
        try:
            print(json.dumps(ready(args.store,args.archive,args.check_ready,args.history_steps,args.history_stride)))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 75  # transient: not yet published, distinct from invalid data
    elif args.prune_verified_raw:
        print(json.dumps(prune_verified_raw(args.store,args.archive)))
    elif args.download:
        produce(args.archive,args.store,args.regrid,args.days_per_request,args.delete_raw,
                pinn=args.pinn,pinn_levels=args.pinn_levels)
    else:
        plan, *_ = plan_for(args.archive,args.regrid,args.days_per_request,args.pinn,args.pinn_levels)
        print(json.dumps({**plan, 'note':'PLAN ONLY; --download starts CDS requests; --delete-raw removes verified owned originals'}, indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

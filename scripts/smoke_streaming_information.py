"""Offline integration: interrupted mocked CDS + lazy shards + real A/B/C training.

No CDS credentials or external downloads. Mock source fields are temporal toy
arrays, not ERA5. The producer runs in a separate CPU process during A/B training.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from smoke_moe import synthetic_archive
from smoke_information_process import synthetic_information
from stream_era5_extra import produce, ready, prune_verified_raw
from climate_diffusion.train_information_process import main as train, load_checkpoint
from climate_diffusion.information_forecast import evaluate
from climate_diffusion.trajectory_output import export_trajectories


class ToyCDS:
    def __init__(self, fields, delay=0):
        self.fields, self.delay = str(fields), delay

    def retrieve(self, dataset, request, output):
        time.sleep(self.delay)
        times = pd.to_datetime([f'{request["year"][0]}-{request["month"][0]}-{d}T{h}'
                                for d in request['day'] for h in request['time']])
        with xr.open_dataset(self.fields) as source:
            fields = {}
            for variable in request['variable']:
                name = {'geopotential':'z', 'u_component_of_wind':'u', 'v_component_of_wind':'v'}[variable]
                if dataset.endswith('single-levels'):
                    a = source.terrain_height.expand_dims(time=times).copy()
                else:
                    levels = list(map(int, request['pressure_level']))
                    a = xr.concat([source[f'{name}{level}'].sel(time=times) for level in levels],
                                  xr.IndexVariable('pressure_level', levels)).copy()
                    a.pressure_level.attrs['units'] = 'hPa'
                a = a*9.80665 if name=='z' else a
                a.attrs['units'] = 'm2 s-2' if name=='z' else 'm/s'
                fields[name] = a
            xr.Dataset(fields).to_netcdf(output, engine='scipy')


def background_producer(archive, store, fields):
    produce(archive, store, delete_raw=True, client=ToyCDS(fields, delay=.4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--skip-render', action='store_true')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    archive, _ = synthetic_archive(out, count=320)
    synthetic_information(archive, out)
    fields, store = out/'synthetic-information.nc', out/'shards'
    class PrefixReady(Exception): pass
    def stop_when_ready(i, reader):
        try: ready(store,archive,'AB',history_stride=1)
        except FileNotFoundError: return
        raise PrefixReady()
    try:
        produce(archive,store,delete_raw=True,client=ToyCDS(fields),on_commit=stop_when_ready)
    except PrefixReady:
        pass
    prefix_shards = len(list((store/'chunks').glob('*.json')))
    assert not (store/'complete.json').exists()
    worker = mp.get_context('spawn').Process(target=background_producer,args=(archive,store,fields))
    worker.start()
    events = []
    common = ['--archive',str(archive),'--information',str(store),'--mode','enriched',
              '--profile','process','--batch-size','2','--members','2','--tau-steps','1',
              '--history-steps','6','--history-stride','1','--manifold-dim','4',
              '--hidden-dim','24','--context-dim','8','--experts','2',
              '--expert-latent-dim','8','--gate-hidden-dim','12','--max-windows','2',
              '--window-stride','8','--seed','7','--curriculum-interval','1','--gradient-audit']
    parent = None
    try:
        for stage, epochs in [('A',6),('B',1),('C',1)]:
            if stage=='C':
                worker.join(120)
                if worker.exitcode != 0: raise RuntimeError('Mock producer did not finish successfully')
                ready(store,archive,'all',history_stride=1)
            events.append(dict(stage=stage, start_seconds=time.perf_counter()-started,
                               producer_alive=worker.is_alive(),
                               available_shards=len(list((store/'chunks').glob('*.json')))))
            checkpoint = out/f'{stage.lower()}.pt'
            train(common+['--stage',stage,'--epochs',str(epochs),'--output',str(checkpoint)]
                  +([] if parent is None else ['--init',str(parent)]))
            _, payload = load_checkpoint(checkpoint)
            assert payload['information_shards'] and payload['config']['horizon_steps']==20
            parent = checkpoint
    finally:
        if worker.is_alive():worker.terminate();worker.join()
    metrics = evaluate(parent, archive, out/'validation.json',information=store,
                       max_cases=2,members=2,tau_steps=1,forecast_output=out/'forecast.npz')
    if not args.skip_render:
        for interval in (6,12):
            export_trajectories(out/'forecast.npz',archive,out/f'members-{interval}h',
                                interval_hours=interval,extension='gif',reference_label='SYNTHETIC truth')
    cleanup=prune_verified_raw(store,archive)
    assert cleanup['remaining_nc']==0
    summary = dict(scope='offline mocked CDS + real synthetic CPU training; not ERA5 skill or a GPU benchmark',
                   epochs={'A':6,'B':1,'C':1},members=2,tau_steps=1,physical_steps=20,
                   training_windows=2,validation_cases=2,prefix_shards=prefix_shards,
                   total_shards=len(list((store/'chunks').glob('*.json'))),
                   raw_nc_remaining=len(list((store/'raw').glob('*.nc'))),
                   events=events,aggregate=metrics['aggregate'],seconds=time.perf_counter()-started)
    assert summary['raw_nc_remaining']==0
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

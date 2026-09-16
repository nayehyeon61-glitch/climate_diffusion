"""Small separate A/B/C experiments; toy fields are NOT ERA5 or a climate simulator."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import xarray as xr
from smoke_moe import synthetic_archive
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.physical_information import prepare
from climate_diffusion.train_information_process import main as train_main, write_json
from climate_diffusion.information_forecast import evaluate

def synthetic_information(archive,directory):
    states,times,schema=load_moe_archive(archive)
    coords=schema['variables'][0]['coords'];lat=np.array(coords['lat']);lon=np.array(coords['lon'])
    fields=states.reshape(-1,4,len(lat),len(lon));wave=(fields[:,0]-101000)/900
    data={name:(('time','lat','lon'),value.astype('float32')) for name,value in
          dict(z850=1500+80*wave,z500=5500+100*wave,z250=10500+150*wave,
               u850=1.5*fields[:,2],v850=1.4*fields[:,3]).items()}
    data['terrain_height']=(('lat','lon'),(500+400*np.cos(np.deg2rad(lat))[:,None]*np.cos(np.deg2rad(lon))[None]).astype('float32'))
    ds=xr.Dataset(data,coords={'time':times,'lat':lat,'lon':lon})
    for name in ds:ds[name].attrs['units']='m/s' if name.startswith(('u','v')) else 'm'
    raw=directory/'synthetic-information.nc';ds.to_netcdf(raw,engine='scipy')
    return prepare(archive,raw,directory/'information.npz')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    p.add_argument('--skip-render',action='store_true');p.add_argument('--only-process',action='store_true')
    a=p.parse_args(argv);out,report=Path(a.output),Path(a.report)
    out.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);started=time.perf_counter()
    archive,_=synthetic_archive(out,count=320);information=synthetic_information(archive,out)
    profiles=['process'] if a.only_process else ['baseline','dynamics','information','process']
    results={}
    for profile in profiles:
        mode='surface' if profile in ('baseline','dynamics') else 'enriched'
        common=['--archive',str(archive),'--mode',mode,'--profile',profile,'--batch-size','2',
            '--members','4','--tau-steps','4','--history-steps','6','--history-stride','1',
            '--manifold-dim','4','--hidden-dim','24','--context-dim','8','--experts','2',
            '--expert-latent-dim','8','--gate-hidden-dim','12','--max-windows','4',
            '--window-stride','8','--seed','7','--curriculum-interval','1','--gradient-audit']
        if mode=='enriched':common+=['--information',str(information)]
        parent=None
        for stage,epochs in [('A',6),('B',2),('C',2)]:
            checkpoint=out/f'{profile}-{stage}.pt'
            train_main(common+['--output',str(checkpoint),'--stage',stage,'--epochs',str(epochs)]
                       +([] if parent is None else ['--init',str(parent)]))
            for suffix in ('.metrics.json','.metadata.json','.manifest.json'):
                shutil.copy2(checkpoint.with_suffix(suffix),report/checkpoint.with_suffix(suffix).name)
            parent=checkpoint
        result=evaluate(parent,archive,report/f'validation-{profile}.json',information=information if mode=='enriched' else None,
            max_cases=2,members=4,tau_steps=4,forecast_output=out/f'forecast-{profile}.npz')
        results[profile]=result['aggregate']
    if 'process' in profiles:
        results['process_drift_only']=evaluate(out/'process-C.pt',archive,report/'validation-drift-only.json',
            information=information,max_cases=2,members=4,tau_steps=4,drift_only=True)['aggregate']
        if not a.skip_render:
            from climate_diffusion.trajectory_output import export_trajectories
            for interval,ext in [(6,'mp4'),(12,'gif')]:
                export_trajectories(out/'forecast-process.npz',archive,report/f'members-{interval}h',
                    interval_hours=interval,extension=ext,reference_label='SYNTHETIC truth')
    summary={'scope':'toy synthetic only; NOT ERA5; no meteorological skill claim',
             'profiles':results,'epochs':{'A':6,'B':2,'C':2},'members':4,'tau_steps':4,'physical_steps':20,
             'validation_cases':2,'training_windows_per_epoch':4,'seconds':time.perf_counter()-started,
             'note':'baseline is a controlled new-wrapper ablation, not a reproduced historical ERA5/legacy ABC run'}
    write_json(report/'summary.json',summary)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,['rmse','state_crps','coverage80']):
        ax.bar(range(len(results)),[r[key] for r in results.values()]);ax.set_title(key)
        ax.set_xticks(range(len(results)),results,rotation=30,ha='right')
    fig.suptitle('SYNTHETIC validation | 2 windows, M4, tau4 | no tuning on test')
    fig.savefig(report/'comparison.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,8),layout='constrained')
    for i,stage in enumerate('ABC'):
        rows=json.loads((report/f'process-{stage}.metrics.json').read_text())
        for ax,key in zip(axes[i],['loss','state_crps','transition_crps','loss_trajectory']):
            for split in ('train','validation'):ax.plot([r['epoch'] for r in rows],[r[split][key] for r in rows],label=split)
            ax.set_title(f'{stage}: {key}');ax.legend()
    fig.suptitle('Actual run logs; A curriculum changes weighted total, not selection metric')
    fig.savefig(report/'training.png',dpi=120);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

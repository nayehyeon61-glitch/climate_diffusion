"""Read-only oracle AE delta, finite-step drift and tangent audit; NOT a forecast."""
import argparse
from pathlib import Path
import numpy as np
import torch
from climate_diffusion.train_information_process import load_checkpoint,data_contract,Windows,write_json

def audit(checkpoint,archive,output,information=None,max_pairs=16):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    model,p=load_checkpoint(checkpoint);model.eval()
    d=data_contract(archive,information,p['mode'],model.config,p)
    ds=Windows(d['states'],d['times'],model.config,d['split']['validation'],d['mean'],d['scale'],d['schema'],information=d['information'])
    pairs={};values={name:[] for name in ('truth','ae','drift','tangent_oracle')}
    for sample in ds:
        full=torch.cat((sample['origin'][None],sample['targets']),0)
        for j in range(20):
            key=int(sample['valid_time_ns'][j])
            if key in pairs:continue
            pairs[key]=True
            x,y=full[j:j+1],full[j+1:j+2];dt=sample['dt_hours'][j]
            info=sample.get('information');info=None if info is None else info[None]
            with torch.no_grad():
                z=model.raw_encode(x,info);zy=model.raw_encode(y,info)
                rx=model.core.manifold.decode(z);ry=model.core.manifold.decode(zy)
                drift=model.core.manifold.decode(z+dt/24*model.core.manifold.latent_drift(z))
                q=(z-model.core.latent_mean)/model.core.latent_scale
                truth=(y-x)*model.temporal.scale/dt
                ae=(ry-rx)*model.temporal.scale/dt;dr=(drift-rx)*model.temporal.scale/dt
            jac=model.core.jacobian(q).detach()[0]
            w=model.temporal.metric.sqrt()/model.temporal.tendency_scale
            physical_jac=jac*model.temporal.scale[:,None]
            matrix=(physical_jac*w[:,None]).double()
            target=(truth[0]*w).double()
            solution=torch.linalg.lstsq(matrix,target).solution
            tangent=(physical_jac.double()@solution).float()[None]
            for name,value in zip(values,(truth,ae,dr,tangent)):values[name].append(value[0].numpy())
            if len(pairs)>=max_pairs:break
        if len(pairs)>=max_pairs:break
    fields={k:np.stack(v).reshape(len(pairs),*model.config.grid) for k,v in values.items()}
    area=model.temporal.area.numpy();result={}
    for i,name in enumerate(model.temporal.names):
        true=fields['truth'][:,i];rms=lambda v:float(np.sqrt((v*v*area).sum((-2,-1)).mean()))
        scale=rms(true);row={'truth_rms_per_hour':scale,'zero_tendency_rmse':scale}
        for kind in ('ae','drift','tangent_oracle'):
            pred=fields[kind][:,i];row[kind+'_rmse_per_hour']=rms(pred-true)
            row[kind+'_amplitude_ratio']=rms(pred)/scale if scale>1e-8 else None
        result[name]=row
    report={'checkpoint':str(checkpoint),'split':'validation','unique_pairs':len(pairs),
            'per_variable':result,'interpretation':'AE uses both observed endpoints; oracle geometry test, NOT forecast. '
            'Tangent is instantaneous least-squares direction oracle in per-variable/area metric; not a learned 6h drift. '
            'Origin information remains fixed within each window, including future surface encoder labels.'}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','archive','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--information');p.add_argument('--max-pairs',type=int,default=16)
    print(audit(**vars(p.parse_args())))

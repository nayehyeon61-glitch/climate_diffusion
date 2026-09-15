"""Read-only preflight/checkpoint audit and exact validation origin selection."""
import argparse
import hashlib
import json
from pathlib import Path
from climate_diffusion.moe_data import load_moe_archive, validate_moe_split, SPLIT_ORDER
from climate_diffusion.train import _sha256


def inspect(archive, preflight, checkpoint=None):
    plan=json.loads(Path(preflight).read_text())
    states,times,schema=load_moe_archive(archive)
    if _sha256(Path(archive)) != plan['archive_sha256'] or schema != plan['schema']:
        raise ValueError('Archive/schema changed after preflight')
    if plan['step_hours']!=6 or plan['horizon_steps']!=20:
        raise ValueError('This manual requires 6h x 20 = 120h')
    span=plan['history_span_steps'];split=plan['split']
    validate_moe_split(split,20,len(states)-span-20+1)
    def stamp(index):return str(times[index].astype('datetime64[s]'))+'Z'
    result={'archive_sha256':plan['archive_sha256'],'state_dim':schema['state_dim'],
            'archive_utc':[stamp(0),stamp(len(times)-1)],'forecast_hours':120,
            'history_hours':(span-1)*6,'sample_shapes':plan['sample_shapes'],
            'normalization_observations_utc':[stamp(0),stamp(plan['normalization_span'][1]-1)],
            'train_unique_pairs':plan['temporal_statistics']['unique_pair_count'],
            'split_contract':'disjoint future targets; causal histories may overlap earlier splits',
            'splits':{}}
    for name in SPLIT_ORDER:
        first,last=split[name][0],split[name][-1]
        result['splits'][name]={'window_count':len(split[name]),
            'origins_utc':[stamp(first+span-1),stamp(last+span-1)],
            'future_targets_utc':[stamp(first+span),stamp(last+span+19)]}
    if checkpoint:
        import torch
        from climate_diffusion.inference import LatentFlowForecaster
        f=LatentFlowForecaster(checkpoint,device='cpu')
        p=torch.load(checkpoint,map_location='cpu',weights_only=False);t=p['training']
        if t['archive_sha256']!=plan['archive_sha256'] or t['split']!=split:
            raise ValueError('Checkpoint archive/split mismatch')
        if t['temporal_statistics']!=plan['temporal_statistics'] or t['normalization_span']!=plan['normalization_span']:
            raise ValueError('Checkpoint statistics mismatch')
        for key in ('state_mean', 'state_scale'):
            if hashlib.sha256(p[key].numpy().tobytes()).hexdigest() != plan[key + '_sha256']:
                raise ValueError('Checkpoint state normalization mismatch')
        if f.config.horizon_steps!=20 or f.config.step_hours!=6 or f.config.history_span_steps!=span:
            raise ValueError('Checkpoint time contract mismatch')
        result['checkpoint']={'sha256':_sha256(Path(checkpoint)), 'format':p['format'],
            'stage':t['stage'],'best_epoch':t['best_epoch'],'best_selection_score':t['best_selection_score'],
            'loss_options':t['loss_options'],'parent_sha256':t.get('previous_checkpoint_sha256'),
            'has_optimizer_state':any('optim' in k for k in p),
            'note':'Best epoch may differ from last metrics row; no exact optimizer resume.'}
    result['first_validation_origin_utc']=stamp(split['validation'][0]+span-1)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True);p.add_argument('--preflight',required=True)
    p.add_argument('--checkpoint');p.add_argument('--origin-only',action='store_true')
    a=p.parse_args(argv);r=inspect(a.archive,a.preflight,a.checkpoint)
    print(r['first_validation_origin_utc'] if a.origin_only else json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()

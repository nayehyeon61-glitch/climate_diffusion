"""Plot saved actual logs; can rerun without training or resampling members."""
import argparse
import json
from pathlib import Path
import numpy as np


def visualize(report):
    import matplotlib.pyplot as plt
    report=Path(report)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for label in ('baseline','recurrent'):
        rows=json.loads((report/f'training-{label}.json').read_text())
        for stage,ax in zip(('manifold','specialize','joint'),axes[0]):
            rr=[r for r in rows if r['stage']==stage]
            ax.plot([r['epoch'] for r in rr],[r['train']['loss'] for r in rr],label=label+' train')
            ax.plot([r['epoch'] for r in rr],[r['selection_score'] for r in rr],'--',label=label+' val selection')
            ax.set(title=stage,xlabel='epoch',ylabel='train objective / selection (different scales)')
    rr=[r for r in rows if r['stage']!='manifold']
    for key in ('fm','energy','crps','loss_trajectory','loss_delta','loss_delta_member','balance','diversity'):
        axes[1,0].plot(range(len(rr)),[r['train'].get(key,np.nan) for r in rr],label=key)
    axes[1,0].set(title='Recurrent raw losses (B then C)',yscale='symlog')
    for v in ('msl','t2m','u10','v10'):
        axes[1,1].plot(range(len(rr)),[r['train'].get('temporal_output_grad_rms_'+v,np.nan) for r in rr],label=v)
    axes[1,1].set(title='New-loss endpoint gradients',yscale='log')
    summary=json.loads((report/'summary.json').read_text())
    names=list(summary['validation']); x=np.arange(len(names))
    for j,key in enumerate(('rmse','crps','rms_spread','coverage_80')):
        axes[1,2].bar(x+(j-1.5)*.18,[summary['validation'][n]['state'][key] for n in names],width=.18,label=key)
    axes[1,2].set(xticks=x,xticklabels=names,title='Identical validation cases; synthetic only')
    for ax in axes.ravel(): ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(report/'training-comparison.png',dpi=120); plt.close(fig)
    diag=json.loads((report/'recurrent-diagnostics.json').read_text()); trace=diag['physical_trace']
    t=[r['lead_hours'] for r in trace]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for kind in ('drift','residual','final'):
        axes[0,0].plot(t,[np.mean(r[kind+'_q_rms_per_hour']) for r in trace],label=kind)
    axes[0,0].set(title='Physical decomposition (q/hour)',xlabel='lead hours')
    pi=np.array([r['gate'] for r in trace]).mean((1,2))
    for k in range(pi.shape[-1]): axes[0,1].plot(t,pi[:,k],label=f'expert {k}')
    axes[0,1].set(title='Generated-path gate',ylim=(0,1))
    axes[0,2].plot(t,[np.mean(r['candidate_cosine']) for r in trace],label='candidate cosine')
    axes[0,2].plot(t,[np.mean(r['gate_entropy']) for r in trace],label='gate entropy')
    axes[0,2].set(title='Similarity / entropy (not specialization proof)')
    for kind in ('raw','projected'):
        axes[1,0].plot(t,[np.mean(r['transport_'+kind+'_norm']) for r in trace],label=kind)
    axes[1,0].set(title='FM transport norms (NOT physical speed)')
    for name in ('msl','t2m','u10','v10'):
        rr=json.loads((report/'members-6h/member-000.json').read_text())['by_variable'][name]
        axes[1,1].plot(t,rr['amplitude_ratio'],label=name)
    axes[1,1].axhline(1,color='k',ls='--'); axes[1,1].set(title='Member 0: tendency amplitude / truth')
    aggregate=json.loads((report/'members-6h/summary.json').read_text())
    for name,v in aggregate['by_variable_ensemble'].items():
        # Per-variable panel values are scaled by their own mean: no Pa/K norm mixture.
        a=np.array(v['tendency_spread_rms']); axes[1,2].plot(t,a/(a.mean()+1e-12),label=name)
    axes[1,2].set(title='Temporal spread / own-variable mean')
    for ax in axes.ravel(): ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Synthetic recurrent dynamics audit — fixed member identity, no ERA5 claim')
    fig.savefig(report/'recurrent-diagnostics.png',dpi=120); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--report-dir',required=True)
    visualize(p.parse_args().report_dir)

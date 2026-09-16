"""Additional plots from saved actual logs; never reruns forecast/training."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    a=p.parse_args(argv);root=Path(a.report)
    rows=json.loads((root/'process-A.metrics.json').read_text())
    keys=['reconstruction','static_l2','information_geometry','fm','state_crps','transition_crps','loss_trajectory']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    data=rows[-1]['gradient_first_batch']
    for i,group in enumerate(['encoder','a_sampler']):
        values=[data.get(f'gradient/{k}/{group}',0) for k in keys]
        weighted=[data.get(f'gradient/weighted_{k}/{group}',0) for k in keys]
        axes[i].bar(np.arange(len(keys))-.2,values,.4,label='raw norm')
        axes[i].bar(np.arange(len(keys))+.2,weighted,.4,label='weighted norm')
        axes[i].set_xticks(np.arange(len(keys)),keys,rotation=50,ha='right');axes[i].set_title(group);axes[i].legend()
    fig.suptitle('A final epoch, first training batch | actual gradient norms')
    fig.savefig(root/'gradient.png',dpi=130);plt.close(fig)
    report=json.loads((root/'generated-routing.json').read_text());path=report['cases'][0]
    routing=path['generated_routing_by_lead'];leads=path['physical_lead_hours']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i in range(len(routing[0]['gate_mean'])):axes[0].plot(leads,[r['gate_mean'][i] for r in routing],label=f'expert {i}')
    axes[0].set_title('Generated gate mean');axes[0].legend()
    for key in ['projection_ratio','candidate_cosine']:axes[1].plot(leads,[r[key] for r in routing],label=key)
    axes[1].set_title('Transport diagnostics, NOT physical tendency');axes[1].legend()
    for k,v in path['q_per_day_rms_by_lead'].items():axes[2].plot(leads,v,label=k)
    axes[2].set_title('Physical q/day RMS');axes[2].legend()
    for ax in axes:ax.set_xlabel('Physical lead hours')
    fig.suptitle('SYNTHETIC generated C path, one origin; no regime specialization claim')
    fig.savefig(root/'routing.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()

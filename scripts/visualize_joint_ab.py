"""Plot joint A+B raw/weighted losses and fixed validation score."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--metrics",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); rows=json.loads(Path(a.metrics).read_text())
    rows=[r for r in rows if r["stage"]=="joint_ab"]
    if not rows: raise ValueError("No joint_ab rows")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r["epoch"] for r in rows],[r["selection_score"] for r in rows],marker="o")
    axes[0].set(title="Fixed validation score",xlabel="epoch")
    keys=("fm","state_crps","transition_crps","trajectory_energy","mean_state","mean_tendency")
    for k in keys:
        if k in rows[0]["train"]: axes[1].plot([r["epoch"] for r in rows],[r["train"][k] for r in rows],label=k)
    axes[1].set(title="Raw train losses",xlabel="epoch"); axes[1].legend(fontsize=7)
    fig.tight_layout(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=180)
if __name__=="__main__": main()

"""CPU smoke for fair Loss V2; the full archive smoke uses run_joint_ab_120h.sh."""
import torch
from climate_diffusion.joint_objective import trajectory_scores, weighted_v2, profile

def main():
    torch.manual_seed(83)
    samples=torch.randn(2,4,21,12,requires_grad=True)
    truth=torch.randn(2,21,12)
    scores=trajectory_scores(samples,truth,torch.full((2,20),6.),torch.ones(12),torch.ones(12))
    scores.update(fm=samples.square().mean(),expert_fm=samples.abs().mean())
    loss,weighted=weighted_v2(scores,profile("v2_full"))
    loss.backward()
    assert samples.grad is not None and torch.isfinite(samples.grad).all()
    print({"loss":float(loss),"scores":{k:float(v) for k,v in scores.items()},
           "weighted":{k:float(v) for k,v in weighted.items()}})
if __name__=="__main__":
    main()

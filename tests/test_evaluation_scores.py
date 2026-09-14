import numpy as np

from climate_diffusion.evaluation import _coverage, _energy_score, _ensemble_crps


def test_fair_and_empirical_finite_member_scores_are_explicit():
    samples=np.array([[0.],[2.]])
    target=np.array([1.])
    assert _ensemble_crps(samples,target) == .5
    assert _ensemble_crps(samples,target,fair=True) == 0.
    assert _energy_score(samples,target) == .5
    assert _energy_score(samples,target,fair=True) == 0.
    assert _coverage(samples,target,.8) == 1.

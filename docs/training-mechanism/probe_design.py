"""NumPy analytical design probes; not production training/backward tests.

Run from any cwd. Output is regenerated beside this file unless --output is given.
The Energy examples integrate exactly over finite-support probability laws;
they are not an implementation of the proposed off-diagonal MC training estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def expected_energy(support: np.ndarray, truth: np.ndarray) -> float:
    """Equal-probability distributions; include diagonals in exact expectations."""
    target_term = np.linalg.norm(support[:, None] - truth[None, :], axis=-1).mean()
    spread_term = np.linalg.norm(support[:, None] - support[None, :], axis=-1).mean()
    return float(target_term - 0.5 * spread_term)


def run() -> dict:
    checks = []
    truth = np.array([[-1.0, -1.0], [1.0, 1.0]])
    correct = truth.copy()
    wrong_coupling = np.array([[-1.0, 1.0], [1.0, -1.0]])
    marginal_good = [expected_energy(correct[:, j:j+1], truth[:, j:j+1]) for j in range(2)]
    marginal_bad = [expected_energy(wrong_coupling[:, j:j+1], truth[:, j:j+1]) for j in range(2)]
    np.testing.assert_allclose(marginal_good, marginal_bad)
    joint_good = expected_energy(correct / np.sqrt(2), truth / np.sqrt(2))
    joint_bad = expected_energy(wrong_coupling / np.sqrt(2), truth / np.sqrt(2))
    assert joint_good < joint_bad
    inc_good = expected_energy(np.diff(correct, axis=1), np.diff(truth, axis=1))
    inc_bad = expected_energy(np.diff(wrong_coupling, axis=1), np.diff(truth, axis=1))
    assert inc_good < inc_bad
    checks.append({"name": "same_marginals_different_temporal_coupling", "passed": True,
                   "marginal_scores_both": marginal_good,
                   "joint_correct_wrong": [joint_good, joint_bad],
                   "increment_correct_wrong": [inc_good, inc_bad]})

    # The actual proposed feature retains both endpoints and an increment.
    def features(x):
        return np.concatenate([x / np.sqrt(2), np.diff(x, axis=1)], axis=1)
    pair_good = expected_energy(features(correct), features(truth))
    pair_bad = expected_energy(features(wrong_coupling), features(truth))
    assert pair_good < pair_bad
    np.testing.assert_allclose(expected_energy(features(correct[::-1]), features(truth)), pair_good)
    checks.append({"name": "pair_feature_and_global_member_permutation", "passed": True,
                   "pair_correct_wrong": [pair_good, pair_bad]})

    members = np.array([[-2., 0.], [1., 3.], [4., -1.]])
    target = np.array([0.5, -0.5])
    member_mse = ((members - target) ** 2).mean()
    mean_mse = ((members.mean(0) - target) ** 2).mean()
    variance = ((members - members.mean(0)) ** 2).mean()
    np.testing.assert_allclose(member_mse, mean_mse + variance)
    checks.append({"name": "memberwise_mse_variance_penalty", "passed": True,
                   "member_mse": float(member_mse), "mean_mse": float(mean_mse),
                   "variance_penalty": float(variance)})

    amplitudes = [0., 0.5, 1., 2., 4.]
    law = np.array([[-1.], [1.]])
    scores = [expected_energy(a * law, law) for a in amplitudes]
    np.testing.assert_allclose(scores, [1., 0.75, 0.5, 1., 2.])
    checks.append({"name": "energy_does_not_reward_unbounded_spread", "passed": True,
                   "amplitudes": amplitudes, "expected_energy": scores,
                   "note": "Exact population example, not an optimization or calibration guarantee."})

    # Different channel scales and raw units; delta cancels the state mean.
    raw = np.array([[10., 280.], [22., 286.], [40., 298.]])
    mu, sigma = np.array([7., 273.]), np.array([2., 5.])
    normalized = (raw - mu) / sigma
    times = np.datetime64("2009-04-04T06:00", "h") + np.arange(3) * np.timedelta64(6, "h")
    dt = np.diff(times) / np.timedelta64(1, "h")
    delta = np.diff(raw, axis=0)
    np.testing.assert_allclose(np.diff(normalized, axis=0) * sigma, delta)
    np.testing.assert_allclose(delta / dt[:, None], [[2., 1.], [3., 2.]])
    checks.append({"name": "raw_normalized_delta_actual_dt", "passed": True,
                   "dt_hours": dt.tolist(), "tendency": (delta / dt[:, None]).tolist()})

    leads = np.arange(1, 121) * 6
    index12 = np.flatnonzero(leads % 12 == 0)
    np.testing.assert_array_equal(index12, np.arange(1, 120, 2))
    assert len(index12) == 60 and leads[index12][-1] == 720
    # M distinct trajectories, known linear physical-time tendency, origin included.
    member_slopes = np.array([1., 2., 3.])
    states = 10 + member_slopes[:, None] * leads[None, :]
    output = states[:, index12]
    anchored = np.concatenate([np.full((3, 1), 10.), output], axis=1)
    np.testing.assert_allclose(np.diff(anchored, axis=1) / 12,
                               np.broadcast_to(member_slopes[:, None], (3, 60)))
    checks.append({"name": "twelve_hour_selection_identity_and_rate", "passed": True,
                   "source_step_hours": 6, "output_interval_hours": 12,
                   "future_frames": len(index12), "first_last_lead": [12, 720],
                   "note": "Output selection only; checkpoint H and model interval stay unchanged."})

    return {"kind": "analytical_design_probes_not_model_training",
            "reviewed_commit": "a3003d1fb53f5c1fecae21ff7a8cdfcf87069096",
            "numpy_version": np.__version__, "passed": len(checks), "checks": checks,
            "not_verified": ["PyTorch gradient", "new production loss or sampler",
                             "ERA5 retraining", "RTX4090 memory", "forecast skill"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("probe-results.json"))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['passed']} analytical design probes passed; saved {args.output}")

# Time-alignment verification results

## Preserved user MP4: rendered-pixel audit only

`../time-alignment-reference/source-comparison.mp4` is 1260×482, 30 frames,
2.5 fps, 12 seconds, with labels +24h through +720h from
2018-11-16T06:00:00. Approximate fixed crops of the two rendered map interiors
give the following RGB pixel diagnostics:

| panel | mean adjacent-frame RMS | first-to-last RMS |
|---|---:|---:|
| generated ensemble mean | 1.792 | 6.651 |
| actual ERA5 | 15.269 | 21.953 |

Thus the supplied rendering visibly changes less on the generated side. These
are **pixel values**, affected by plotting, arrows and compression; they are not
t2m tendency or wind-speed measurements and cannot identify the model cause.
The original prediction/ERA5 arrays and exact plotting code are not stored in
the repository.

## Synthetic moving-field contract test

`scripts/smoke_time_alignment.py` creates a 24-hourly 30-lead field and a
deliberately slow ensemble whose mean anomaly is 0.55 of truth. The new CLI
joins every frame by exact valid time and measures a median predicted/truth
tendency-amplitude ratio of **0.549999**. The unshifted lag diagnostic returns
0 steps (correlation 0.999999). This confirms that the diagnostic detects
amplitude attenuation independently of playback fps.

- `comparison.mp4`: 1540×770, 30 frames, 2.5 fps, 12 seconds.
- `diagnostics.json`: per-lead RMSE/CRPS, spread, anomaly, physical-time
  tendency and u/v vector error plus member tendencies.
- `first.png`, `last.png`: render QA frames.

This is a deterministic synthetic verification, not an ERA5 training run and
not evidence that the existing MoE checkpoint has been recalibrated.

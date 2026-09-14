# Joint A+B Training Manual (120 h)

This branch preserves full-state PI-Manifold MoE, persistent independent
ensemble noise and 6 h x 20 physical recurrence. It does not bypass A, add a
large network, or alter WeatherNext.

## Contract

Warmup trains reconstruction, PI/invariant/metric, latent drift, decoded AE
delta, and decoded finite-step 6 h drift. The best warmup is sealed exactly
once from train-only coordinates: affine q mean/scale plus chart centers/radius.
Joint A+B then updates encoder, decoder, drift, experts, gate and history
encoder together; the default manifold LR is 0.3 of base. Coordinates and
charts remain fixed. C uses calibration data at 0.1 base LR with manifold at a
further 0.1 factor and reference anchoring. The five-way
train/expert_validation/calibration/validation/test split and embargoes remain
unchanged.

The observed residual target is

```
stopgrad((encode(next)-encode(prev))/(dt_hours/24)-drift(encode(prev)))
```

in the same sealed affine q coordinates used by the model. Only that label
branch is detached. Generated recurrence, decoder Jacobian and tangent
projection remain differentiable. The target is recomputed at every update;
there is no EMA target.

One `sample_trajectory` call creates `[B,M,T+1,D]` for all V2 scores.
`trajectory_edges=0` means full 20-step BPTT and full-window endpoint+
increment Energy. A positive edge count uses a contiguous score block but the
recurrent sampler still builds its prefix; it is not a full-window law score.

Fair state CRPS excludes the known origin. Fair transition CRPS uses
same-member endpoint differences divided by actual `dt_hours` and the
train-only tendency scale. Delta CRPS is therefore transition CRPS before the
time/scale change of units. Fair CRPS uses denominator `M(M-1)` and a sorted
member formula, and fails for M<2. It assumes conditionally iid members.
Marginal CRPS cannot identify dependence, so joint endpoint+increment Energy
is also used; Energy itself loses sensitivity in high dimension.

New AB/C runs force `delta_member_weight=0`. Member MSE decomposes into
mean-MSE plus ensemble variance and would contract spread; finite-member mean
MSE carries the same smaller variance pressure. The optional pooled
spread-skill term is a default-zero heuristic, not a proper score: it uses
unbiased ensemble variance, finite-M factor `1+1/M`, detached mean error,
near-zero masking and squared log ratio. There is no unbounded entropy/spread
reward.

Profiles:

- `ab_control`: state Energy .5, state CRPS .5, joint trajectory Energy .1, mean tendency .02.
- `v2_minimal`: FM 1, transition CRPS .25, trajectory Energy .3, mean
  tendency .02.
- `v2_full`: FM .3, state CRPS 1, transition CRPS .75, trajectory Energy
  .75, mean state .1, mean tendency .1.

Existing `energy_weight/crps_weight` still control the legacy C route; profile
weights control only `joint_ab`. Keep M=4 and tau steps=4 for the first
ablation.

## Install and immutable archive preflight

```bash
git clone --branch feature/joint-ab-loss-v2 --single-branch \
  https://github.com/nayehyeon61-glitch/climate_diffusion.git climate_diffusion_joint
cd climate_diffusion_joint
python -m pip install -e '.[test]'
pytest -q
python scripts/smoke_joint_ab.py
python scripts/prepare_temporal_120h.py --help
```

Use a new output directory. The trainer rejects an existing checkpoint or
sidecar, verifies exact UTC fixed-step times, fully observed pairs, archive
hash, schema, normalization and split continuity. Batch size must be at least
2.

## Full warmup -> AB -> C

```bash
bash scripts/run_joint_ab_120h.sh /data/era5_6h_120h.npz \
  outputs/joint-ab-v2-run-001
```

Equivalent separate stages:

```bash
python -m climate_diffusion.train_manifold_moe --archive /data/era5.npz \
 --output outputs/run/a.pt --stage manifold --forecast-dynamics recurrent_residual \
 --manifold-epochs 8 --batch-size 4 --ae-delta-weight .05 \
 --finite-step-drift-weight .05

python -m climate_diffusion.train_manifold_moe --archive /data/era5.npz \
 --init-checkpoint outputs/run/a.pt --output outputs/run/ab.pt --stage joint_ab \
 --joint-ab-epochs 30 --loss-profile v2_full --ensemble-size 4 \
 --integration-steps 4 --trajectory-edges 0 --delta-member-weight 0 \
 --manifold-lr-factor .3 --log-gradient-norms

python -m climate_diffusion.train_manifold_moe --archive /data/era5.npz \
 --init-checkpoint outputs/run/ab.pt --output outputs/run/c.pt --stage joint \
 --joint-epochs 10 --joint-lr-factor .1 --encoder-lr-factor .1
```

The best checkpoint is reloaded between stages. Optimizer state is not stored;
a later invocation is a fresh-optimizer fine-tune, not bitwise resume. Never
reuse the previous output filename.

## Validation, rendering and frozen test

```bash
python -m climate_diffusion.evaluation --checkpoint outputs/run/c.pt \
 --archive /data/era5.npz --split validation --ensemble-size 4 \
 --integration-steps 4 --output outputs/run/validation.json

python -m climate_diffusion.trajectory_output --checkpoint outputs/run/c.pt \
 --archive /data/era5.npz --ensemble-size 4 --integration-steps 4 \
 --output outputs/run/members

python scripts/visualize_joint_ab.py --metrics outputs/run/ab.metrics.json \
 --output outputs/run/joint-ab-losses.png

python -m climate_diffusion.evaluation --checkpoint outputs/run/c.pt \
 --archive /data/era5.npz --split test --ensemble-size 4 \
 --integration-steps 4 --output outputs/run/test-frozen.json
```

Choose weights and epochs using expert_validation/validation only, then freeze
configuration before the single test read. Evaluation reports state CRPS,
transition CRPS and trajectory Energy from the same saved trajectory.

## Pilot limits

For a 4090 pilot use epoch 1-2, batch >=2, M4/tau4. Full 20-step graphs may not
fit; a positive `trajectory_edges` is a stated sub-block score, not equivalent
to a full-window law. Logs record raw and weighted losses plus first-batch
module gradient norms/cosines and normalized per-variable endpoint gradients
when `--log-gradient-norms` is set.

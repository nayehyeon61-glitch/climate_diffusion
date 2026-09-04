# 500GB RunPod: practical 0.25° monthly Flow Matching

This profile is for the `feature/spatial-025-flow-autoencoder` branch and uses the
spatial memmap archive, 256×256 patch training, AMP, gradient accumulation and
resume-safe checkpoints.

## 1. Storage layout

Use `/workspace` for persistent data and checkpoints.

```text
/workspace/
├── climate_diffusion/
├── data/
│   ├── staging/
│   │   ├── raw-current-month/
│   │   └── monthly-shards/
│   └── monthly_climate_spatial_025/
├── checkpoints/
│   └── flow_matching/
│       ├── smoke-conv/
│       └── operator-025/
└── outputs/
```

Recommended operational budget for a 500 GiB volume:

- raw ERA5 staging: keep below roughly 50–80 GiB and delete after each month
- monthly 0.25° archive + observation mask: target roughly 60–100 GiB
- checkpoints / optimizer / snapshots: reserve 50–80 GiB
- downstream token cache: reserve 50–80 GiB
- keep at least 100 GiB uncommitted while building the archive

The exact size depends on years, variables, compression of the downloaded raw
files and whether monthly shards are deleted after finalization.

## 2. Environment

```bash
cd /workspace
git clone https://github.com/nayehyeon61-glitch/climate_diffusion.git
cd climate_diffusion
git checkout feature/spatial-025-flow-autoencoder
pip install -e '.[io]'
```

Check the GPU and disk before doing any large download:

```bash
nvidia-smi
df -h /workspace
```

## 3. Recommended ERA5 channels

Start with 29 channels:

Surface:

```text
msl t2m u10 v10
```

Pressure-level variables:

```text
z t u v q
```

Pressure levels:

```text
1000 850 700 500 200 hPa
```

The streaming adapter validates that all requested pressure levels exist and
keeps surface variables without adding a pressure dimension.

## 4. Storage-bounded month-by-month preprocessing

Download only one complete 6-hourly calendar month into
`/workspace/data/staging/raw-current-month`. The downloader is intentionally
kept separate from this repository; after download, convert that month into a
small restart-safe monthly shard:

```bash
prepare-era5-streaming-flow append \
  --source /workspace/data/staging/raw-current-month/*.nc \
  --staging-dir /workspace/data/staging/monthly-shards \
  --variables msl t2m u10 v10 z t u v q \
  --pressure-levels 1000 850 700 500 200 \
  --cadence-hours 6 \
  --target-lat-points 721 \
  --target-lon-points 1440
```

The append command fails when the calendar month is incomplete, contains a time
gap, misses a pressure level, changes the spatial grid, contains Inf, or has a
fully missing channel.

After the `.npz` monthly shard is created successfully, delete only the raw
month directory:

```bash
rm -rf /workspace/data/staging/raw-current-month/*
```

Repeat for every month. Do not delete `monthly-shards` yet.

### Suggested data stages

Use these stages before committing to the full archive:

```text
Stage 1: 2020–2021  -> pipeline / VRAM smoke test
Stage 2: 2015–2022  -> speed / loss / checkpoint validation
Stage 3: 1990–2025  -> full experiment
```

## 5. Finalize the monthly memmap archive

After all requested months have been converted:

```bash
prepare-era5-streaming-flow finalize \
  --staging-dir /workspace/data/staging/monthly-shards \
  --output /workspace/data/monthly_climate_spatial_025
```

Inspect it before deleting the monthly shards:

```bash
ls -lh /workspace/data/monthly_climate_spatial_025
du -sh /workspace/data/monthly_climate_spatial_025
df -h /workspace
```

The final archive has the existing training contract:

```text
monthly_climate_spatial_025/
├── states.npy            # [month, channel, 721, 1440], float32 memmap
├── observed_mask.npy     # same shape, bool
├── observed_fraction.npy
├── times.npy
├── auxiliary.npy
└── schema.json
```

When the archive has been opened/validated successfully, monthly shards can be
removed or finalization can be rerun with `--delete-shards` on a fresh output.

## 6. First real GPU smoke run: spatial_conv

Start with the 2-year archive and 10 epochs. This verifies archive loading,
patch sampling, latent Flow loss, checkpoint saving and resume.

```bash
train-runpod-climate-flow \
  --archive /workspace/data/monthly_climate_spatial_025 \
  --checkpoint-dir /workspace/checkpoints/flow_matching/smoke-conv \
  --model-backend spatial_conv \
  --history-months 6 \
  --epochs 10 \
  --batch-size 1 \
  --spatial-base-channels 32 \
  --spatial-latent-channels 16 \
  --spatial-downsample-levels 3 \
  --patch-height 256 \
  --patch-width 256 \
  --tile-overlap 64 \
  --gradient-accumulation-steps 8 \
  --min-observed-fraction 0.95 \
  --save-every-epochs 2 \
  --keep-epoch-snapshots 3 \
  --num-workers 2
```

AMP and gradient checkpointing are enabled by default in this RunPod CLI.
Disable them only for debugging with `--no-mixed-precision` or
`--no-gradient-checkpointing`.

## 7. Resume behavior

The checkpoint directory contains:

```text
checkpoint-dir/
├── latest.pt             # resume: model + optimizer + scaler + RNG + epoch
├── best.pt               # inference/frozen candidate; no optimizer state
├── epoch-XXXX.pt         # rotating recovery snapshots
├── metrics.jsonl
└── run_summary.json
```

Running the same command again resumes automatically from `latest.pt`. The
trainer refuses to resume when the archive fingerprint or model configuration
has changed.

To deliberately start a new run, use a new checkpoint directory or pass:

```bash
--no-resume
```

Do not overwrite a scientifically important run in place.

## 8. Full spatial_operator run

After the smoke run passes, switch to the Fourier operator backend:

```bash
train-runpod-climate-flow \
  --archive /workspace/data/monthly_climate_spatial_025 \
  --checkpoint-dir /workspace/checkpoints/flow_matching/operator-025 \
  --model-backend spatial_operator \
  --history-months 6 \
  --epochs 100 \
  --batch-size 1 \
  --spatial-base-channels 32 \
  --spatial-latent-channels 16 \
  --spatial-downsample-levels 3 \
  --operator-modes-lat 12 \
  --operator-modes-lon 24 \
  --patch-height 256 \
  --patch-width 256 \
  --tile-overlap 64 \
  --gradient-accumulation-steps 8 \
  --min-observed-fraction 0.95 \
  --save-every-epochs 5 \
  --keep-epoch-snapshots 3 \
  --num-workers 2
```

The effective batch is approximately `batch_size × gradient_accumulation`, so
the default above behaves like an effective batch of 8 optimizer samples while
keeping VRAM close to batch 1.

## 9. Disk safety

`train-runpod-climate-flow` refuses to start/continue when free checkpoint-disk
space drops below 25 GiB by default. Increase the margin if the same volume also
stores large evaluation rollouts:

```bash
--min-free-disk-gb 50
```

Check disk periodically:

```bash
watch -n 30 'df -h /workspace; du -sh /workspace/data /workspace/checkpoints 2>/dev/null'
```

## 10. What to inspect before the long run

The smoke run is considered valid only when all of the following hold:

1. `states.npy` is opened as a memmap and the grid is `[721, 1440]`.
2. `schema.json` records the requested pressure levels and channel contract.
3. train/validation raw month ranges do not overlap.
4. reconstruction and flow-matching losses remain finite and begin to decrease.
5. `latest.pt` appears after every epoch and a killed/restarted process resumes.
6. `best.pt` is created and can be loaded by the frozen inference path.
7. free `/workspace` space stays comfortably above the configured threshold.

Only after this checklist should the 1990–2025 archive / 100-epoch operator run
be started.

## 11. Connection to GPT-DoubleLoss

For this experiment the monthly Flow model is a pretraining/long-range model. Do
not reuse a 720h monthly endpoint as the downstream Day-15 P15 anchor. The
GPT-DoubleLoss pipeline now requires an exact 360h endpoint for that role.

Use `best.pt` as the frozen Flow artifact for monthly/long-range experiments. A
separate fixed-step 0–360h Flow checkpoint is required for direct WeatherNext
Day-15 replacement.

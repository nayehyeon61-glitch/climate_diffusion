#!/usr/bin/env bash
# Execute ONE phase at a time. No resource creation, download, or automatic test tuning.
set -euo pipefail
phase=${1:?Usage: bash scripts/run_recurrent_120h.sh prepare|A|B|C|validation|render|test}
: "${TEMPORAL_ARCHIVE:?Set an actual fixed-step ERA5 archive path}"
: "${TEMPORAL_RUN:?Set a NEW experiment directory}"
device=${TEMPORAL_DEVICE:-cuda}
batch=${TEMPORAL_BATCH:-2}
members=${TEMPORAL_MEMBERS:-4}
steps=${TEMPORAL_TAU_STEPS:-4}
edges=${TEMPORAL_EDGES:-0}
if [[ "$phase" != prepare ]]; then
  test -f "$TEMPORAL_RUN/preflight.json"
  test "$(git rev-parse HEAD)" = "$(<"$TEMPORAL_RUN/code-commit.txt")" || {
    echo 'Code commit changed within the experiment; use a new run directory.' >&2; exit 1;
  }
fi
temporal=(--ensemble-size "$members" --sampled-leads 2 --integration-steps "$steps"
  --trajectory-edges "$edges" --validation-trajectory-edges 0
  --trajectory-weight 0.1 --delta-weight 0.02 --delta-member-weight 0.001
  --wind-speed-weight 0.01 --wind-direction-weight 0.005 --temporal-warmup-epochs 5
  --trajectory-selection-weight 0.1 --log-gradient-norms)
common=(--archive "$TEMPORAL_ARCHIVE" --batch-size "$batch" --window-stride 4
  --learning-rate 0.001 --weight-decay 0.0001 --seed 7 --device "$device")
case "$phase" in
  prepare)
    test ! -e "$TEMPORAL_RUN" || { echo 'Run directory exists; choose a NEW one.' >&2; exit 1; }
    git diff --quiet
    git diff --cached --quiet
    mkdir -p "$TEMPORAL_RUN"
    git rev-parse HEAD > "$TEMPORAL_RUN/code-commit.txt"
    python -m pip freeze > "$TEMPORAL_RUN/environment.txt"
    if [[ ! -f "$TEMPORAL_ARCHIVE" ]]; then
      : "${TEMPORAL_FIELDS:?Archive missing; provide existing ERA5 NetCDF/Zarr}"
      prepare-climate-fixed-step-data --fields "$TEMPORAL_FIELDS" --variables msl t2m u10 v10 \
        --step-hours 6 --target-lat-points 18 --target-lon-points 36 --output "$TEMPORAL_ARCHIVE"
    fi
    python scripts/prepare_temporal_120h.py --archive "$TEMPORAL_ARCHIVE" \
      --output "$TEMPORAL_RUN/preflight.json" --history-steps 6 --history-stride 4
    ;;
  A)
    train-climate-manifold-moe "${common[@]}" --output "$TEMPORAL_RUN/a.pt" --stage manifold \
      --forecast-dynamics recurrent_residual --history-steps 6 --history-stride 4 --horizon-steps 20 \
      --num-experts 4 --manifold-dim 16 --expert-latent-dim 64 --gate-hidden-dim 160 \
      --residual-noise-std 1 --manifold-epochs "${TEMPORAL_A_EPOCHS:-50}" --early-stop-patience 8 \
      2>&1 | tee "$TEMPORAL_RUN/a.console.log"
    python scripts/prepare_temporal_120h.py --archive "$TEMPORAL_ARCHIVE" \
      --output "$TEMPORAL_RUN/preflight-a-verified.json" --checkpoint "$TEMPORAL_RUN/a.pt"
    ;;
  B)
    train-climate-manifold-moe "${common[@]}" "${temporal[@]}" --output "$TEMPORAL_RUN/b.pt" \
      --stage specialize --init-checkpoint "$TEMPORAL_RUN/a.pt" \
      --expert-epochs "${TEMPORAL_B_EPOCHS:-40}" --early-stop-patience 8 \
      2>&1 | tee "$TEMPORAL_RUN/b.console.log"
    ;;
  C)
    train-climate-manifold-moe "${common[@]}" "${temporal[@]}" --output "$TEMPORAL_RUN/c.pt" \
      --stage joint --init-checkpoint "$TEMPORAL_RUN/b.pt" --joint-epochs "${TEMPORAL_C_EPOCHS:-10}" \
      --joint-lr-factor 0.1 --encoder-lr-factor 0.1 --early-stop-patience 5 \
      2>&1 | tee "$TEMPORAL_RUN/c.console.log"
    ;;
  validation|test)
    evaluate-climate-flow --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
      --split "$phase" --output "$TEMPORAL_RUN/$phase.json" --ensemble-size 8 \
      --integration-steps 16 --max-cases 32 --seed 83 --device "$device"
    if [[ "$phase" == validation ]]; then
      python -m climate_diffusion.manifold_diagnostics --checkpoint "$TEMPORAL_RUN/c.pt" \
        --archive "$TEMPORAL_ARCHIVE" --output "$TEMPORAL_RUN/validation-routing.json" \
        --split validation --max-cases 4 --members 4 --integration-steps 4 --device "$device"
    fi
    ;;
  render)
    origin=$(python - <<'PY'
import os, torch
from climate_diffusion.moe_data import load_moe_archive
p=torch.load(os.environ['TEMPORAL_RUN']+'/c.pt',map_location='cpu',weights_only=False)
_,times,_=load_moe_archive(os.environ['TEMPORAL_ARCHIVE'])
print(times[p['training']['split']['validation'][0]+p['training']['history_span_steps']-1])
PY
)
    diagnose-climate-time --checkpoint "$TEMPORAL_RUN/c.pt" --archive "$TEMPORAL_ARCHIVE" \
      --origin-time "$origin" --forecast-steps 20 --ensemble-size 8 --integration-steps 16 \
      --seed 83 --device "$device" --forecast-output "$TEMPORAL_RUN/forecast-6h.npz" --forecast-only
    render-climate-trajectories --forecast "$TEMPORAL_RUN/forecast-6h.npz" --archive "$TEMPORAL_ARCHIVE" \
      --output-dir "$TEMPORAL_RUN/members-6h" --horizon-hours 120 --interval-hours 6 --extension mp4 --diagnostic-views
    render-climate-trajectories --forecast "$TEMPORAL_RUN/forecast-6h.npz" --archive "$TEMPORAL_ARCHIVE" \
      --output-dir "$TEMPORAL_RUN/members-12h" --horizon-hours 120 --interval-hours 12 --extension gif
    ;;
  *) echo 'Unknown phase; use prepare A B C validation render test' >&2; exit 1 ;;
esac

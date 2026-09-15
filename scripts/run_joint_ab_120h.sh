#!/usr/bin/env bash
# One explicit phase; defaults are a one-epoch PILOT, not a production recipe.
set -euo pipefail
phase=${1:-help}
if [[ "$phase" == help || "$phase" == --help ]]; then
  echo 'Usage: bash scripts/run_joint_ab_120h.sh prepare|warmup|AB|C|validation|forecast|render|test ARCHIVE NEW_RUN'
  echo 'DRY_RUN=1 prints commands without data access, writes or training.'
  exit 0
fi
case "$phase" in prepare|warmup|AB|C|validation|forecast|render|test) ;; *) echo 'Unknown phase; old two-argument launcher retired. Use --help.' >&2; exit 2;; esac
archive=${2:?Existing 6h archive required}; out=${3:?NEW run directory required}
python=${PYTHON:-python}; device=${JOINT_DEVICE:-cuda}; dry=${DRY_RUN:-0}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
# Preserve legacy non-pointer binary files despite their current LFS attributes.
git_read=(git -c filter.lfs.clean=cat -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false)
run() { printf '%q ' "$@"; printf '\n'; if [[ "$dry" != 1 ]]; then "$@"; fi; }
new_file() { if [[ "$dry" != 1 && -e "$1" ]]; then echo "Refusing existing output: $1" >&2; exit 2; fi; }
logged() {
  local log=$1; shift; new_file "$log"
  if [[ "$dry" == 1 ]]; then run "$@"; else run "$@" 2>&1 | tee "$log"; fi
}
common=(--archive "$archive" --batch-size "${JOINT_BATCH:-2}" --window-stride "${JOINT_STRIDE:-4}"
  --max-validation-windows "${JOINT_VAL_WINDOWS:-4}" --device "$device" --seed "${JOINT_SEED:-7}"
  --learning-rate 0.001 --weight-decay 0.0001)
trajectory=(--ensemble-size 4 --integration-steps 4 --sampled-leads 2
  --trajectory-edges "${JOINT_EDGES:-0}" --validation-trajectory-edges 0 --delta-member-weight 0)
if [[ "$dry" != 1 ]]; then
  test -f "$archive" || { echo 'Archive missing; prepare existing ERA5 fields first.' >&2; exit 2; }
  "${git_read[@]}" diff --quiet; "${git_read[@]}" diff --cached --quiet
  if [[ "$phase" != prepare ]]; then
    test -f "$out/preflight.json"
    test "$(git rev-parse HEAD)" = "$(<"$out/code-commit.txt")" || { echo 'Code changed: use a new experiment.' >&2; exit 2; }
  fi
fi
case "$phase" in
  prepare)
    if [[ "$dry" != 1 ]]; then
      test ! -e "$out" || { echo 'Choose a NEW run directory.' >&2; exit 2; }
      mkdir -p "$out"
      git rev-parse HEAD > "$out/code-commit.txt"
      "$python" -m pip freeze > "$out/environment.txt"
    fi
    run "$python" scripts/prepare_temporal_120h.py --archive "$archive" --output "$out/preflight.json" --history-steps 6 --history-stride 4
    logged "$out/calendar.log" "$python" scripts/inspect_joint_run.py --archive "$archive" --preflight "$out/preflight.json"
    ;;
  warmup)
    logged "$out/warmup.console.log" "$python" -m climate_diffusion.train_manifold_moe "${common[@]}" \
      --output "$out/warmup.pt" --stage manifold --forecast-dynamics recurrent_residual \
      --history-steps 6 --history-stride 4 --horizon-steps 20 --num-experts 4 --manifold-dim 16 \
      --expert-latent-dim 64 --gate-hidden-dim 160 --manifold-epochs "${JOINT_WARMUP_EPOCHS:-1}" \
      --ae-delta-weight 0.05 --finite-step-drift-weight 0.05 --early-stop-patience 3
    run "$python" scripts/prepare_temporal_120h.py --archive "$archive" --output "$out/preflight-warmup.json" --checkpoint "$out/warmup.pt"
    ;;
  AB)
    # --log-gradient-norms has an exhausted-generator defect: omit until repaired.
    logged "$out/ab.console.log" "$python" -m climate_diffusion.train_manifold_moe "${common[@]}" "${trajectory[@]}" \
      --output "$out/ab.pt" --init-checkpoint "$out/warmup.pt" --stage joint_ab \
      --joint-ab-epochs "${JOINT_AB_EPOCHS:-1}" --loss-profile "${JOINT_LOSS_PROFILE:-v2_minimal}" \
      --manifold-lr-factor 0.3 --anchor-weight 0.05 --early-stop-patience 8
    ;;
  C)
    # C is legacy calibration, NOT V2. Explicit options enable its temporal path.
    logged "$out/c.console.log" "$python" -m climate_diffusion.train_manifold_moe "${common[@]}" "${trajectory[@]}" \
      --output "$out/c.pt" --init-checkpoint "$out/ab.pt" --stage joint \
      --joint-epochs "${JOINT_C_EPOCHS:-1}" --joint-lr-factor 0.1 --encoder-lr-factor 0.1 --anchor-weight 1 \
      --energy-weight 0.5 --crps-weight 0.5 --delta-weight 0.02 --trajectory-weight 0.1 \
      --wind-speed-weight 0.01 --wind-direction-weight 0.005 --temporal-warmup-epochs 5 --early-stop-patience 5
    logged "$out/checkpoint-audit.log" "$python" scripts/inspect_joint_run.py --archive "$archive" --preflight "$out/preflight.json" --checkpoint "$out/c.pt"
    ;;
  validation|test)
    if [[ "$phase" == test && "${JOINT_TEST_CONFIRMED:-0}" != 1 ]]; then
      echo 'Freeze choices first, then set JOINT_TEST_CONFIRMED=1.' >&2; exit 2
    fi
    for name in ab c; do
      new_file "$out/$phase-$name.json"
      run "$python" -m climate_diffusion.evaluation --checkpoint "$out/$name.pt" --archive "$archive" \
        --split "$phase" --output "$out/$phase-$name.json" --ensemble-size "${JOINT_EVAL_MEMBERS:-4}" \
        --integration-steps "${JOINT_EVAL_TAU:-4}" --max-cases "${JOINT_EVAL_CASES:-4}" --seed 83 --device "$device"
    done
    if [[ "$phase" == validation ]]; then
      new_file "$out/validation-routing.json"
      run "$python" -m climate_diffusion.manifold_diagnostics --checkpoint "$out/c.pt" --archive "$archive" \
        --output "$out/validation-routing.json" --split validation --max-cases 2 --members 4 --integration-steps 4 --device "$device"
      new_file "$out/ab-losses.png"
      run "$python" scripts/visualize_joint_ab.py --metrics "$out/ab.metrics.json" --output "$out/ab-losses.png"
    fi
    ;;
  forecast)
    origin=${JOINT_ORIGIN:-2000-01-01T00:00:00Z}
    if [[ "$dry" != 1 && -z "${JOINT_ORIGIN:-}" ]]; then
      origin=$("$python" scripts/inspect_joint_run.py --archive "$archive" --preflight "$out/preflight.json" --checkpoint "$out/c.pt" --origin-only)
    fi
    new_file "$out/forecast-6h.npz"
    run "$python" -m climate_diffusion.time_alignment --checkpoint "$out/c.pt" --archive "$archive" \
      --origin-time "$origin" --forecast-output "$out/forecast-6h.npz" --forecast-only --forecast-steps 20 \
      --ensemble-size "${JOINT_EVAL_MEMBERS:-4}" --integration-steps "${JOINT_EVAL_TAU:-4}" --seed 83 --device "$device"
    ;;
  render)
    for interval in 6 12; do
      new_file "$out/members-${interval}h"
      run "$python" -m climate_diffusion.trajectory_output --forecast "$out/forecast-6h.npz" --archive "$archive" \
        --output-dir "$out/members-${interval}h" --horizon-hours 120 --interval-hours "$interval" \
        --extension "${JOINT_VIDEO_FORMAT:-mp4}" --reference-label "${JOINT_REFERENCE_LABEL:-Actual ERA5}"
    done
    ;;
esac

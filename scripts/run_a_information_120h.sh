#!/usr/bin/env bash
# Explicit, separately invoked stages. Never starts paid resources or overwrites weights.
set -euo pipefail
stage="${1:?Choose preflight|A|audit|B|C|validation|render|test}"
: "${ARCHIVE:?Set the existing canonical 6h surface archive path}"
: "${RUN:?Choose a NEW run directory}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-enriched}"
case "$MODE" in enriched|surface) ;; *) echo 'MODE must be enriched or surface' >&2; exit 2;; esac
PINN="${PINN:-0}"
case "$PINN" in 0|1) ;; *) echo 'PINN must be 0 or 1' >&2; exit 2;; esac
if [[ "$PINN" == 1 && "$MODE" != enriched ]]; then
  echo 'PINN requires MODE=enriched' >&2; exit 2
fi
pinn_data=()
pinn_train=()
pinn_levels=()
if [[ "$PINN" == 1 ]]; then
  read -r -a pinn_levels <<< "${PINN_LEVELS:-500 850}"
  pinn_data=(--pinn --pinn-levels "${pinn_levels[@]}")
  pinn_train=(--pinn --pinn-levels "${pinn_levels[@]}" --pinn-weight "${PINN_WEIGHT:-0.1}"
    --pinn-warmup-epochs "${PINN_WARMUP_EPOCHS:-1}" --pinn-ramp-epochs "${PINN_RAMP_EPOCHS:-3}")
fi
DEVICE="${DEVICE:-cpu}"
M="${M:-4}"; TAU="${TAU:-4}"
HISTORY_STRIDE="${HISTORY_STRIDE:-4}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
mkdir -p "$RUN"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
info=()
if [[ "$MODE" == enriched ]]; then
  : "${INFO:?Set the physical information sidecar path}"
  info=(--information "$INFO")
fi
common=(--archive "$ARCHIVE" --mode "$MODE" "${info[@]}" --device "$DEVICE"
  --history-steps 6 --history-stride "$HISTORY_STRIDE" --members "$M" --tau-steps "$TAU"
  --batch-size "${BATCH:-2}" --max-windows "$MAX_WINDOWS" --window-stride "${WINDOW_STRIDE:-4}"
  --seed "${SEED:-7}" --learning-rate "${LR:-0.001}" --gradient-audit)
case "$stage" in
  preflight)
    "$PYTHON" scripts/prepare_temporal_120h.py --archive "$ARCHIVE" --output "$RUN/preflight.json" \
      --history-steps 6 --history-stride "$HISTORY_STRIDE"
    if [[ "$MODE" == enriched && ! -e "$INFO" ]]; then
      : "${INFO_FIELDS:?Set aligned extra-field NetCDF; no download or imputation is performed}"
      "$PYTHON" -m climate_diffusion.physical_information --archive "$ARCHIVE" --fields "$INFO_FIELDS" --output "$INFO" "${pinn_data[@]}"
    fi
    if [[ "$MODE" == enriched ]]; then
      "$PYTHON" -c '
import sys
from climate_diffusion.moe_data import load_moe_archive
from climate_diffusion.physical_information import load_information, PRIMARY, pinn_optional
_,t,s=load_moe_archive(sys.argv[1])
_,m=load_information(sys.argv[2],sys.argv[1],t,s)
names=[v["name"] for v in m["variables"]]
if sys.argv[3]=="1":
    missing=set(PRIMARY+pinn_optional(map(int,sys.argv[4:])))-set(names)
    if missing: raise ValueError("PINN inputs missing: "+", ".join(sorted(missing)))
print("Validated physical inputs:",names)
' "$ARCHIVE" "$INFO" "$PINN" "${pinn_levels[@]}"
    fi
    git rev-parse HEAD > "$RUN/code-commit.txt"
    "$PYTHON" -m pip freeze > "$RUN/environment.txt"
    ;;
  A)
    extra=()
    [[ -z "${QUALITY_MAX:-}" ]] || extra+=(--a-quality-max "$QUALITY_MAX")
    [[ -z "${A_LOSS_WEIGHTS:-}" ]] || extra+=(--loss-weights "$A_LOSS_WEIGHTS")
    "$PYTHON" -m climate_diffusion.train_information_process "${common[@]}" --stage A \
      --profile "${PROFILE:-process}" --epochs "${A_EPOCHS:-60}" --curriculum-interval "${CURRICULUM_INTERVAL:-4}" \
      --output "$RUN/a.pt" "${extra[@]}" "${pinn_train[@]}"
    ;;
  audit)
    "$PYTHON" scripts/audit_information_process.py --checkpoint "$RUN/a.pt" --archive "$ARCHIVE" \
      "${info[@]}" --output "$RUN/a-audit.json" --max-pairs "${AUDIT_PAIRS:-64}" \
      --split "${AUDIT_SPLIT:-expert_validation}"
    ;;
  B|C)
    if [[ "$stage" == B ]]; then parent=a; target=b; epochs="${B_EPOCHS:-30}"; else parent=b; target=c; epochs="${C_EPOCHS:-10}"; fi
    "$PYTHON" -m climate_diffusion.train_information_process "${common[@]}" --stage "$stage" \
      --init "$RUN/$parent.pt" --output "$RUN/$target.pt" --epochs "$epochs" --b-member-weight "${B_MEMBER_WEIGHT:-0.001}"
    ;;
  validation|test)
    forecast=()
    [[ "$stage" != validation ]] || forecast=(--forecast-output "$RUN/forecast.npz")
    "$PYTHON" -m climate_diffusion.information_forecast --checkpoint "$RUN/c.pt" --archive "$ARCHIVE" \
      "${info[@]}" --output "$RUN/$stage.json" --split "$stage" --members "$M" --tau-steps "$TAU" \
      --max-cases "${EVAL_CASES:-32}" --device "$DEVICE" "${forecast[@]}"
    ;;
  render)
    for interval in 6 12; do
      "$PYTHON" -m climate_diffusion.trajectory_output --forecast "$RUN/forecast.npz" --archive "$ARCHIVE" \
        --output-dir "$RUN/members-${interval}h" --horizon-hours 120 --interval-hours "$interval" --extension mp4
    done
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2;;
esac

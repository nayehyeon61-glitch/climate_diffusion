#!/usr/bin/env bash
# One CPU producer + the unchanged, sequential A -> B -> C trainer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the existing canonical 6h surface .npz archive}"
: "${INFO:?Set a NEW or matching resumable compact-shard directory}"
: "${RUN:?Set a NEW experiment directory (never reuse interrupted A weights)}"
PYTHON="${PYTHON:-python}"
export PYTHON INFO ARCHIVE RUN
export MODE=enriched
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
START_STAGE="${START_STAGE:-A}"
THROUGH="${THROUGH:-render}"
case "$START_STAGE" in A|B|C|validation|render) ;; *) echo 'Invalid START_STAGE' >&2; exit 2;; esac
case "$THROUGH" in A|B|C|validation|render) ;; *) echo 'Invalid THROUGH' >&2; exit 2;; esac
stages=(A B C validation render)
start_index=-1; end_index=-1
for i in "${!stages[@]}"; do
  [[ "${stages[$i]}" != "$START_STAGE" ]] || start_index="$i"
  [[ "${stages[$i]}" != "$THROUGH" ]] || end_index="$i"
done
(( end_index >= start_index )) || { echo 'THROUGH must follow START_STAGE' >&2; exit 2; }
if [[ "$START_STAGE" == A ]]; then
  [[ ! -e "$RUN" ]] || { echo 'A requires a new RUN directory' >&2; exit 2; }
else
  [[ -d "$RUN" ]] || { echo 'Later stage requires an existing completed parent run' >&2; exit 2; }
fi
mkdir -p "$RUN"
if [[ "$START_STAGE" == A ]]; then
  "$PYTHON" scripts/prepare_temporal_120h.py --archive "$ARCHIVE" \
    --output "$RUN/surface-preflight.json" --history-steps 6 --history-stride "${HISTORY_STRIDE:-4}"
fi
producer=''
cleanup() {
  if [[ -n "$producer" ]] && kill -0 "$producer" 2>/dev/null; then
    kill "$producer" 2>/dev/null || true
    wait "$producer" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
  --regrid "${REGRID:-linear}" --days-per-request "${CHUNK_DAYS:-3}" \
  --download --delete-raw >> "$RUN/download.log" 2>&1 &
producer=$!

wait_ready() {
  local which="$1" began="$SECONDS" status
  while true; do
    if "$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
        --check-ready "$which" --history-stride "${HISTORY_STRIDE:-4}" \
        > "$RUN/readiness-$which.json" 2> "$RUN/readiness-wait.log"; then
      break
    else
      status=$?
    fi
    [[ "$status" == 75 ]] || { cat "$RUN/readiness-wait.log" >&2; return "$status"; }
    if ! kill -0 "$producer" 2>/dev/null; then
      wait "$producer" || true
      echo 'Producer exited before readiness; inspect download.log. Converted files remain reusable.' >&2
      return 1
    fi
    if (( SECONDS-began >= ${WAIT_TIMEOUT_SECONDS:-86400} )); then
      echo 'Readiness timeout; resume the same INFO store, choose a new RUN for unfinished A.' >&2
      return 1
    fi
    sleep 5
  done
}

wait_ready AB
if [[ "$START_STAGE" == A ]]; then
  bash scripts/run_a_information_120h.sh preflight
fi
for (( i=start_index; i<=end_index; i++ )); do
  stage="${stages[$i]}"
  if (( i>=2 )); then
    wait_ready all
    # Do not hide a producer error just because some readable files exist.
    if [[ -n "$producer" ]]; then
      wait "$producer"; producer=''
    fi
  fi
  bash scripts/run_a_information_120h.sh "$stage" 2>&1 | tee "$RUN/stream-$stage.log"
  if [[ "$stage" == A ]]; then
    bash scripts/run_a_information_120h.sh audit
  fi
done
# All requested training stages are complete. Finish publication, rather than
# silently claiming the remainder was downloaded after stopping the producer.
if [[ -n "$producer" ]]; then
  wait "$producer"
  producer=''
fi
"$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" --prune-verified-raw \
  > "$RUN/raw-cleanup.json"
echo "Completed through $THROUGH. Inspect validation before final test."
echo 'Fixed settings only: bash scripts/run_a_information_120h.sh test'

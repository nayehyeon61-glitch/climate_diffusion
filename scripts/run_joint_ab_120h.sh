#!/usr/bin/env bash
set -euo pipefail
ARCHIVE="${1:?usage: run_joint_ab_120h.sh ARCHIVE OUTPUT_DIR}"
OUT="${2:?usage: run_joint_ab_120h.sh ARCHIVE OUTPUT_DIR}"
test ! -e "$OUT" || { echo "refusing existing output directory: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
python -m climate_diffusion.train_manifold_moe \
  --archive "$ARCHIVE" --output "$OUT/model.pt" --stage ab_all \
  --forecast-dynamics recurrent_residual --manifold-epochs 8 \
  --joint-ab-epochs 30 --joint-epochs 10 --batch-size 4 \
  --ensemble-size 4 --integration-steps 4 --trajectory-edges 0 \
  --validation-trajectory-edges 0 --loss-profile v2_full \
  --ae-delta-weight .05 --finite-step-drift-weight .05 \
  --delta-member-weight 0 --manifold-lr-factor .3 \
  --joint-lr-factor .1 --encoder-lr-factor .1 --log-gradient-norms

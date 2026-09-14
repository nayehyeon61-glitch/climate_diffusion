#!/usr/bin/env bash
set -uo pipefail
cd /workspace/climate_diffusion_recurrent
source .venv/bin/activate

export TEMPORAL_ARCHIVE=/workspace/data/era5-temporal-6h.npz
export TEMPORAL_RUN=/workspace/experiments/physical-recurrent-run-001
export TEMPORAL_DEVICE=cuda
export TEMPORAL_BATCH=2
export TEMPORAL_MEMBERS=4
export TEMPORAL_TAU_STEPS=4
export TEMPORAL_EDGES=0
export TEMPORAL_A_EPOCHS=50
export TEMPORAL_A_DELTA_WEIGHT=0.05
export TEMPORAL_A_DRIFT_WEIGHT=0.05
export TEMPORAL_A_TENDENCY_MAX=1.0
export TEMPORAL_B_EPOCHS=40
export TEMPORAL_C_EPOCHS=10

echo "PIPELINE_STAGE=prepare START $(date -u +%FT%TZ)"
bash scripts/run_recurrent_120h.sh prepare
rc=$?
if [ $rc -ne 0 ]; then echo "PIPELINE_FAILED stage=prepare rc=$rc"; exit $rc; fi
echo "PIPELINE_STAGE=prepare DONE $(date -u +%FT%TZ)"

echo "PIPELINE_STAGE=A START $(date -u +%FT%TZ)"
bash scripts/run_recurrent_120h.sh A
rc=$?
if [ $rc -ne 0 ]; then echo "PIPELINE_FAILED stage=A rc=$rc"; exit $rc; fi
echo "PIPELINE_STAGE=A DONE $(date -u +%FT%TZ)"

echo "PIPELINE_STAGE=B START $(date -u +%FT%TZ)"
bash scripts/run_recurrent_120h.sh B
rc=$?
if [ $rc -ne 0 ]; then echo "PIPELINE_FAILED stage=B rc=$rc"; exit $rc; fi
echo "PIPELINE_STAGE=B DONE $(date -u +%FT%TZ)"

echo "PIPELINE_STAGE=C START $(date -u +%FT%TZ)"
bash scripts/run_recurrent_120h.sh C
rc=$?
if [ $rc -ne 0 ]; then echo "PIPELINE_FAILED stage=C rc=$rc"; exit $rc; fi
echo "PIPELINE_STAGE=C DONE $(date -u +%FT%TZ)"

echo "PIPELINE_ALL_DONE $(date -u +%FT%TZ)"

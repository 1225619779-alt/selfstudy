#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
CASE_NAME="${2:-case39}"
WORKERS="${3:-4}"
CHUNK_STEPS="${4:-8}"
START_IDX="${5:-0}"
END_IDX="${6:-16}"
OUT_ROOT="${7:-gen_data/${CASE_NAME}_smoke_parallel_v2}"

cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

env DDET_CASE_NAME="$CASE_NAME" python case39_measure_parallel_v2.py \
  --repo_root . \
  --case_name "$CASE_NAME" \
  --workers "$WORKERS" \
  --chunk_steps "$CHUNK_STEPS" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --out_root "$OUT_ROOT"

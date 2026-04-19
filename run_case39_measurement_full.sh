#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs gen_data/case39/full_chunks

LOG_PATH="logs/case39_measurement_parallel_full.log"
SUMMARY_PATH="gen_data/case39/full_parallel_summary.json"
PID_PATH="logs/case39_measurement_parallel_full.pid"

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export DDET_DATA_SEED="${DDET_DATA_SEED:-20260417}"
export DDET_MEASURE_SEED="${DDET_MEASURE_SEED:-20260417}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[runner] invoked_at=$(date -Is)"
echo "[runner] shell_pid=$$"
echo "$$" > "$PID_PATH"
echo "[runner] pwd=$(pwd)"
echo "[runner] python=$(.venv_rocm/bin/python -V 2>&1)"
echo "[runner] DDET_CASE_NAME=${DDET_CASE_NAME}"
echo "[runner] DDET_DATA_SEED=${DDET_DATA_SEED}"
echo "[runner] DDET_MEASURE_SEED=${DDET_MEASURE_SEED}"
echo "[runner] threads: OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS} VECLIB=${VECLIB_MAXIMUM_THREADS}"

.venv_rocm/bin/python -u generate_measurement_bank_parallel.py \
  --case_name case39 \
  --workers 4 \
  --chunk_size 256 \
  --seed "${DDET_MEASURE_SEED}" \
  --summary_json "${SUMMARY_PATH}" \
  --chunk_dir "gen_data/case39/full_chunks"

echo "[runner] finished_at=$(date -Is)"

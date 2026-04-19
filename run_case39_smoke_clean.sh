#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39

LOG_PATH="logs/case39_smoke_clean.log"

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

: >"$LOG_PATH"
exec >>"$LOG_PATH" 2>&1

echo "[runner] started_at=$(date -Is)"
echo "[runner] DDET_CASE_NAME=${DDET_CASE_NAME}"

if [[ ! -f saved_model/case39/checkpoint_rnn.pt ]]; then
  echo "[runner] missing saved_model/case39/checkpoint_rnn.pt"
  exit 1
fi

.venv_rocm/bin/python -u evaluation_event_trigger_clean.py \
  --tau_verify -1.0 \
  --max_total_run 128 \
  --stop_ddd_alarm_at 5

echo "[runner] finished_at=$(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs saved_model/case39

LOG_PATH="logs/case39_train_full.log"

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
echo "[runner] pwd=$(pwd)"
echo "[runner] python=$(.venv_rocm/bin/python -V 2>&1)"
echo "[runner] DDET_CASE_NAME=${DDET_CASE_NAME}"
echo "[runner] device_check=$(.venv_rocm/bin/python - <<'PY'
import torch
print({'cuda_available': torch.cuda.is_available(), 'device': ('cuda' if torch.cuda.is_available() else 'cpu')})
PY
)"

if [[ ! -f gen_data/case39/z_noise_summary.npy ]]; then
  echo "[runner] missing gen_data/case39/z_noise_summary.npy"
  exit 1
fi

.venv_rocm/bin/python -u -m models.model

echo "[runner] finished_at=$(date -Is)"

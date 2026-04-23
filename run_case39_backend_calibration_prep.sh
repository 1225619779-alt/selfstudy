#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39 reports

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export DDET_DATA_SEED="${DDET_DATA_SEED:-20260417}"
export DDET_MEASURE_SEED="${DDET_MEASURE_SEED:-20260417}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

CLEAN_VALID_CACHE="metric/case39/clean_alarm_cache_valid_mode_0_0.03_1.1.npy"
ATTACK_VALID_CACHE="metric/case39/attack_alarm_cache_valid_50_repo_compatible_mode_0_0.03_1.1.npy"

exec >>"logs/case39_backend_calibration_prep.log" 2>&1

echo "[prep] invoked_at=$(date -Is)"
echo "[prep] collecting validation clean cache"
.venv_rocm/bin/python -u collect_clean_alarm_cache.py \
  --split valid \
  --max_total_run -1 \
  --stop_ddd_alarm_at -1 \
  --next_load_mode sample_length \
  --skip_backend \
  --output "$CLEAN_VALID_CACHE" \
  > logs/case39_collect_clean_cache_valid.log 2>&1

echo "[prep] collecting validation attack cache"
.venv_rocm/bin/python -u collect_attack_alarm_cache.py \
  --split valid \
  --total_run 50 \
  --recover_input_mode repo_compatible \
  --next_load_modes sample_length offset \
  --next_load_extra 7 \
  --skip_backend \
  --output "$ATTACK_VALID_CACHE" \
  > logs/case39_collect_attack_cache_valid.log 2>&1

echo "[prep] finished_at=$(date -Is)"

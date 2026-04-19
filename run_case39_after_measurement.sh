#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs

LOG_PATH="logs/case39_after_measurement.log"
MEASURE_SUMMARY="gen_data/case39/full_parallel_summary.json"
MEASURE_Z="gen_data/case39/z_noise_summary.npy"
TRAIN_MODEL="saved_model/case39/checkpoint_rnn.pt"

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"

: >"$LOG_PATH"
exec >>"$LOG_PATH" 2>&1

echo "[watcher] started_at=$(date -Is)"
echo "[watcher] waiting for measurement summary and bank files..."

while [[ ! -f "$MEASURE_SUMMARY" || ! -f "$MEASURE_Z" ]]; do
  echo "[watcher] $(date -Is) measurement not ready yet"
  sleep 60
done

echo "[watcher] $(date -Is) measurement ready; starting training"
bash /home/pang/projects/DDET-MTD/run_case39_train_full.sh

if [[ -f "$TRAIN_MODEL" ]]; then
  echo "[watcher] $(date -Is) training finished; starting clean smoke evaluation"
  bash /home/pang/projects/DDET-MTD/run_case39_smoke_clean.sh
else
  echo "[watcher] $(date -Is) training script ended but model checkpoint is missing"
  exit 1
fi

echo "[watcher] finished_at=$(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs

LOG_PATH="logs/case39_post_train.log"
TRAIN_SERVICE="${TRAIN_SERVICE:-case39-train}"
TRAIN_MODEL="saved_model/case39/checkpoint_rnn.pt"

if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[watcher] started_at=$(date -Is)"
echo "[watcher] waiting for ${TRAIN_SERVICE} to finish..."

while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  echo "[watcher] $(date -Is) ${TRAIN_SERVICE} still active"
  sleep 60
done

echo "[watcher] $(date -Is) ${TRAIN_SERVICE} is no longer active"

if [[ -f "$TRAIN_MODEL" ]]; then
  echo "[watcher] $(date -Is) checkpoint found; starting clean smoke evaluation"
  bash /home/pang/projects/DDET-MTD/run_case39_smoke_clean.sh
else
  echo "[watcher] $(date -Is) checkpoint missing; refusing to start smoke evaluation"
  exit 1
fi

echo "[watcher] finished_at=$(date -Is)"

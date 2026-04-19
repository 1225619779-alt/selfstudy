#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs

LOG_PATH="logs/case39_post_tau.log"
TAU_SERVICE="${TAU_SERVICE:-case39-tau-select}"
SUMMARY_PATH="${SUMMARY_PATH:-metric/case39/tau_selection_joint_valid/tau_selection_summary.txt}"

if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[watcher] started_at=$(date -Is)"
echo "[watcher] waiting for ${TAU_SERVICE} to finish..."

while systemctl --user is-active --quiet "$TAU_SERVICE"; do
  echo "[watcher] $(date -Is) ${TAU_SERVICE} still active"
  sleep 60
done

echo "[watcher] $(date -Is) ${TAU_SERVICE} is no longer active"

if [[ -f "$SUMMARY_PATH" ]]; then
  echo "[watcher] $(date -Is) tau summary found; starting clean suite"
  bash /home/pang/projects/DDET-MTD/run_case39_clean_suite_from_tau.sh "$SUMMARY_PATH"
else
  echo "[watcher] $(date -Is) tau summary missing; refusing to start clean suite"
  exit 1
fi

echo "[watcher] finished_at=$(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs

LOG_PATH="logs/case39_post_clean_suite.log"
SUMMARY_PATH="${1:-metric/case39/tau_selection_joint_valid/tau_selection_summary.txt}"

if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[watcher] started_at=$(date -Is)"
echo "[watcher] waiting for case39-posttau to finish"

while systemctl --user is-active --quiet case39-posttau; do
  echo "[watcher] $(date -Is) case39-posttau still active"
  sleep 60
done

echo "[watcher] $(date -Is) case39-posttau is no longer active"

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[watcher] tau summary missing: $SUMMARY_PATH" >&2
  exit 1
fi

echo "[watcher] tau summary found; starting post-clean support chain"
/usr/bin/bash /home/pang/projects/DDET-MTD/run_case39_attack_support_from_clean.sh "$SUMMARY_PATH"
echo "[watcher] finished_at=$(date -Is)"

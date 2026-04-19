#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs

LOG_PATH="logs/case39_post_support_perf_probe.log"
if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[watcher] started_at=$(date -Is)"
echo "[watcher] waiting for case39-postclean to finish"

while systemctl --user is-active --quiet case39-postclean; do
  echo "[watcher] $(date -Is) case39-postclean still active"
  sleep 60
done

echo "[watcher] $(date -Is) case39-postclean is no longer active"
echo "[watcher] starting performance probe"
/usr/bin/bash /home/pang/projects/DDET-MTD/run_case39_perf_probe.sh
echo "[watcher] finished_at=$(date -Is)"

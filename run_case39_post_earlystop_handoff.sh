#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs
LOG_PATH="logs/case39_post_earlystop_handoff.log"
exec > >(tee "$LOG_PATH") 2>&1

echo "[post-earlystop-handoff] invoked_at=$(date -Is)"

while systemctl --user is-active --quiet case39-earlystop; do
  sleep 30
done

echo "[post-earlystop-handoff] detected_case39_earlystop_inactive_at=$(date -Is)"

.venv_rocm/bin/python analyze_case39_backend_calibration_consistency.py
.venv_rocm/bin/python render_case39_backend_calibration_early_stop_handoff.py

echo "[post-earlystop-handoff] finished_at=$(date -Is)"

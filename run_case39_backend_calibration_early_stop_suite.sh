#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs
LOG_PATH="logs/case39_backend_calibration_early_stop_suite.log"
exec > >(tee "$LOG_PATH") 2>&1

echo "[early-stop-suite] invoked_at=$(date -Is)"

bash ./run_case39_best_candidate_clean_spot_rerun.sh
bash ./run_case39_targeted_backend_check.sh
.venv_rocm/bin/python analyze_case39_backend_calibration_consistency.py
.venv_rocm/bin/python render_case39_backend_calibration_early_stop_handoff.py

echo "[early-stop-suite] finished_at=$(date -Is)"

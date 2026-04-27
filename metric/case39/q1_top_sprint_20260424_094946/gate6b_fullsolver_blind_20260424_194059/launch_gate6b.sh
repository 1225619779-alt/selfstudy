#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD-q1-case39

OUT=/home/pang/projects/DDET-MTD-q1-case39/metric/case39/q1_top_sprint_20260424_094946/gate6b_fullsolver_blind_20260424_194059
PY=/home/pang/projects/DDET-MTD/.venv_rocm/bin/python
SCRIPT="$OUT/gate6b_fullsolver_manager.py"

setsid "$PY" "$SCRIPT" > "$OUT/gate6b_manager.nohup.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$OUT/gate6b_manager.pid"
echo "$pid"

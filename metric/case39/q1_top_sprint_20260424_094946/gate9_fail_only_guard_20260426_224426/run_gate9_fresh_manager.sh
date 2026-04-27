#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD-q1-case39
export DDET_CASE_NAME=case39

/home/pang/projects/DDET-MTD/.venv_rocm/bin/python \
  metric/case39/q1_top_sprint_20260424_094946/gate9_fail_only_guard_20260426_224426/gate9_fresh_manager.py


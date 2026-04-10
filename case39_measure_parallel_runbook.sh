#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
WORKERS="${2:-8}"
CASE_NAME="${3:-case39}"

cd "$REPO_ROOT"

printf '\n[1/2] 先做 64 步 smoke，避免长跑前才发现报错\n'
env DDET_CASE_NAME="$CASE_NAME" python case39_measure_parallel.py \
  --repo_root . \
  --case_name "$CASE_NAME" \
  --workers "$WORKERS" \
  --start_idx 0 \
  --end_idx 64 \
  --out_root "gen_data/${CASE_NAME}_smoke_parallel"

printf '\n[2/2] smoke 成功后，开始全量并行 measurement 生成\n'
env DDET_CASE_NAME="$CASE_NAME" python case39_measure_parallel.py \
  --repo_root . \
  --case_name "$CASE_NAME" \
  --workers "$WORKERS"

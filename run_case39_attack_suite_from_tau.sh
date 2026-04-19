#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39

SUMMARY_PATH="${1:-metric/case39/tau_selection_joint_valid/tau_selection_summary.txt}"
USE_ROUNDED="${USE_ROUNDED:-1}"

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export DDET_DATA_SEED="${DDET_DATA_SEED:-20260417}"
export DDET_MEASURE_SEED="${DDET_MEASURE_SEED:-20260417}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[runner] missing tau summary: $SUMMARY_PATH" >&2
  exit 1
fi

TAU_INFO="$(
  SUMMARY_PATH="$SUMMARY_PATH" USE_ROUNDED="$USE_ROUNDED" .venv_rocm/bin/python - <<'PY'
import os
import re
from pathlib import Path

text = Path(os.environ["SUMMARY_PATH"]).read_text(encoding="utf-8")

def grab(name: str) -> str:
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
    if not m:
        raise SystemExit(f"missing {name} in summary")
    return m.group(1)

use_rounded = os.environ.get("USE_ROUNDED", "1") == "1"
main_key = "tau_main_rounded_3dp" if use_rounded else "tau_main"
strict_key = "tau_strict_rounded_3dp" if use_rounded else "tau_strict"
print(grab(main_key))
print(grab(strict_key))
PY
)"

TAU_MAIN="$(printf '%s\n' "$TAU_INFO" | sed -n '1p')"
TAU_STRICT="$(printf '%s\n' "$TAU_INFO" | sed -n '2p')"

SUITE_LOG="logs/case39_attack_suite_from_tau.log"
if [[ ! -f "$SUITE_LOG" ]]; then
  : >"$SUITE_LOG"
fi
exec >>"$SUITE_LOG" 2>&1

echo "[runner] invoked_at=$(date -Is)"
echo "[runner] summary_path=$SUMMARY_PATH"
echo "[runner] tau_main=$TAU_MAIN"
echo "[runner] tau_strict=$TAU_STRICT"
echo "[runner] use_rounded=$USE_ROUNDED"

run_attack() {
  local tau="$1"
  local label="$2"
  local log_path="logs/case39_attack_${label}.log"
  local metric_path="metric/case39/metric_event_trigger_tau_${tau}_mode_0_0.03_1.1.npy"
  local summary_path="logs/case39_attack_${label}_summary.log"

  : >"$log_path"
  : >"$summary_path"

  echo "[runner] starting attack ${label} tau=${tau}"
  .venv_rocm/bin/python -u evaluation_event_trigger_attack_cli.py \
    --tau_verify "$tau" \
    --total_run 50 \
    --output "$metric_path" >>"$log_path" 2>&1

  echo "[runner] summarizing attack ${label} tau=${tau}"
  .venv_rocm/bin/python -u summarize_attack_support_metric.py \
    --metric "$metric_path" >"$summary_path" 2>&1

  echo "[runner] finished attack ${label} tau=${tau}"
}

run_attack "-1.0" "baseline"
run_attack "$TAU_MAIN" "main"
run_attack "$TAU_STRICT" "strict"

echo "[runner] finished_at=$(date -Is)"

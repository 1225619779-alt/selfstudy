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

BASELINE_METRIC="metric/case39/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
MAIN_METRIC="metric/case39/metric_event_trigger_clean_tau_${TAU_MAIN}_mode_0_0.03_1.1.npy"
STRICT_METRIC="metric/case39/metric_event_trigger_clean_tau_${TAU_STRICT}_mode_0_0.03_1.1.npy"

CLEAN_SCORE_METRIC="metric/case39/metric_clean_alarm_scores_full.npy"
ATTACK_SCORE_METRIC="metric/case39/metric_attack_alarm_scores_50.npy"
ABLATION_NPY="metric/case39/gate_ablation_case39.npy"
ABLATION_CSV="metric/case39/gate_ablation_case39.csv"

LOG_PATH="logs/case39_attack_support.log"
if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

echo "[runner] invoked_at=$(date -Is)"
echo "[runner] summary_path=$SUMMARY_PATH"
echo "[runner] tau_main=$TAU_MAIN"
echo "[runner] tau_strict=$TAU_STRICT"
echo "[runner] baseline_metric=$BASELINE_METRIC"
echo "[runner] main_metric=$MAIN_METRIC"
echo "[runner] strict_metric=$STRICT_METRIC"

for p in "$BASELINE_METRIC" "$MAIN_METRIC" "$STRICT_METRIC"; do
  if [[ ! -f "$p" ]]; then
    echo "[runner] required clean metric missing: $p" >&2
    exit 1
  fi
done

echo "[runner] collecting clean alarm scores"
.venv_rocm/bin/python -u collect_clean_alarm_scores.py \
  --max_total_run -1 \
  --stop_ddd_alarm_at -1 \
  --output "$CLEAN_SCORE_METRIC"

echo "[runner] collecting attack alarm scores"
.venv_rocm/bin/python -u collect_attack_alarm_scores.py \
  --total_run 50 \
  --output "$ATTACK_SCORE_METRIC"

echo "[runner] comparing clean baseline vs main"
.venv_rocm/bin/python -u compare_gate_results_clean.py \
  --baseline "$BASELINE_METRIC" \
  --gated "$MAIN_METRIC" > logs/case39_compare_clean_main.log 2>&1

echo "[runner] analyzing matched-budget ablation"
.venv_rocm/bin/python -u analyze_gate_ablation.py \
  --clean_scores "$CLEAN_SCORE_METRIC" \
  --attack_scores "$ATTACK_SCORE_METRIC" \
  --baseline_clean_metric "$BASELINE_METRIC" \
  --main_clean_metric "$MAIN_METRIC" \
  --strict_clean_metric "$STRICT_METRIC" \
  --out_npy "$ABLATION_NPY" \
  --out_csv "$ABLATION_CSV"

echo "[runner] finished_at=$(date -Is)"

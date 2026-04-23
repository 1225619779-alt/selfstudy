#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs reports metric/case39 metric/case39/next_load_idx_old_offset7

SUMMARY_PATH="${1:-metric/case39/tau_selection_joint_valid/tau_selection_summary.txt}"

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
  SUMMARY_PATH="$SUMMARY_PATH" .venv_rocm/bin/python - <<'PY'
import os
import re
from pathlib import Path

text = Path(os.environ["SUMMARY_PATH"]).read_text(encoding="utf-8")

def grab(name: str) -> str:
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
    if not m:
        raise SystemExit(f"missing {name} in summary")
    return m.group(1)

print(grab("tau_main"))
print(grab("tau_strict"))
PY
)"

TAU_MAIN="$(printf '%s\n' "$TAU_INFO" | sed -n '1p')"
TAU_STRICT="$(printf '%s\n' "$TAU_INFO" | sed -n '2p')"

CLEAN_BASELINE="metric/case39/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
CLEAN_MAIN="metric/case39/metric_event_trigger_clean_tau_${TAU_MAIN}_mode_0_0.03_1.1.npy"
CLEAN_STRICT="metric/case39/metric_event_trigger_clean_tau_${TAU_STRICT}_mode_0_0.03_1.1.npy"

ATTACK_BASELINE="metric/case39/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy"
ATTACK_MAIN="metric/case39/metric_event_trigger_tau_${TAU_MAIN}_mode_0_0.03_1.1.npy"
ATTACK_STRICT="metric/case39/metric_event_trigger_tau_${TAU_STRICT}_mode_0_0.03_1.1.npy"

OLD_ATTACK_BASELINE="metric/case39/next_load_idx_old_offset7/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy"
OLD_ATTACK_MAIN="metric/case39/next_load_idx_old_offset7/metric_event_trigger_tau_${TAU_MAIN}_mode_0_0.03_1.1.npy"
OLD_ATTACK_STRICT="metric/case39/next_load_idx_old_offset7/metric_event_trigger_tau_${TAU_STRICT}_mode_0_0.03_1.1.npy"

CLEAN_CACHE="metric/case39/clean_alarm_cache_test_mode_0_0.03_1.1.npy"
ATTACK_CACHE="metric/case39/attack_alarm_cache_test_50_repo_compatible_mode_0_0.03_1.1.npy"

SUITE_LOG="logs/case39_hygiene_suite.log"
exec >>"$SUITE_LOG" 2>&1

echo "[runner] invoked_at=$(date -Is)"
echo "[runner] summary_path=$SUMMARY_PATH"
echo "[runner] tau_main_exact=$TAU_MAIN"
echo "[runner] tau_strict_exact=$TAU_STRICT"

materialize_clean() {
  local tau="$1"
  local label="$2"
  local output="$3"
  local log_path="logs/case39_exact_tau_clean_${label}.log"
  : >"$log_path"
  echo "[runner] materializing clean ${label} tau=${tau}"
  .venv_rocm/bin/python -u materialize_clean_metric_from_cache.py \
    --cache "$CLEAN_CACHE" \
    --tau_verify "$tau" \
    --output "$output" >>"$log_path" 2>&1
  echo "[runner] finished clean ${label} tau=${tau}"
}

materialize_attack() {
  local tau="$1"
  local label="$2"
  local output="$3"
  local next_load_mode="$4"
  local log_path="logs/case39_${label}.log"
  : >"$log_path"
  echo "[runner] materializing attack ${label} tau=${tau} next_load_mode=${next_load_mode}"
  .venv_rocm/bin/python -u materialize_attack_metric_from_cache.py \
    --cache "$ATTACK_CACHE" \
    --tau_verify "$tau" \
    --next_load_mode "$next_load_mode" \
    --output "$output" >>"$log_path" 2>&1
  echo "[runner] finished attack ${label} tau=${tau}"
}

echo "[runner] === collect clean cache once ==="
.venv_rocm/bin/python -u collect_clean_alarm_cache.py \
  --split test \
  --max_total_run -1 \
  --stop_ddd_alarm_at -1 \
  --next_load_mode sample_length \
  --output "$CLEAN_CACHE" \
  > logs/case39_collect_clean_cache.log 2>&1

echo "[runner] === exact tau clean materialization ==="
materialize_clean "-1.0" "baseline" "$CLEAN_BASELINE"
materialize_clean "$TAU_MAIN" "main" "$CLEAN_MAIN"
materialize_clean "$TAU_STRICT" "strict" "$CLEAN_STRICT"

echo "[runner] === collect attack cache once ==="
.venv_rocm/bin/python -u collect_attack_alarm_cache.py \
  --split test \
  --total_run 50 \
  --recover_input_mode repo_compatible \
  --next_load_modes sample_length offset \
  --next_load_extra 7 \
  --output "$ATTACK_CACHE" \
  > logs/case39_collect_attack_cache.log 2>&1

echo "[runner] === exact tau attack materialization (sample_length default) ==="
materialize_attack "-1.0" "exact_tau_attack_baseline" "$ATTACK_BASELINE" "sample_length"
materialize_attack "$TAU_MAIN" "exact_tau_attack_main" "$ATTACK_MAIN" "sample_length"
materialize_attack "$TAU_STRICT" "exact_tau_attack_strict" "$ATTACK_STRICT" "sample_length"

echo "[runner] === exact tau summary reports ==="
.venv_rocm/bin/python -u write_case39_exact_tau_reports.py \
  --clean-baseline "$CLEAN_BASELINE" \
  --clean-main "$CLEAN_MAIN" \
  --clean-strict "$CLEAN_STRICT" \
  --attack-baseline "$ATTACK_BASELINE" \
  --attack-main "$ATTACK_MAIN" \
  --attack-strict "$ATTACK_STRICT" \
  --output-clean-md reports/case39_exact_tau_clean_summary.md \
  --output-attack-md reports/case39_exact_tau_attack_summary.md

echo "[runner] === failure aware statistics ==="
.venv_rocm/bin/python -u generate_failure_aware_summary.py \
  --clean-baseline "$CLEAN_BASELINE" \
  --clean-main "$CLEAN_MAIN" \
  --clean-strict "$CLEAN_STRICT" \
  --attack-baseline "$ATTACK_BASELINE" \
  --attack-main "$ATTACK_MAIN" \
  --attack-strict "$ATTACK_STRICT" \
  --output-json metric/case39/failure_aware_summary.json \
  --output-csv metric/case39/failure_aware_summary.csv \
  --output-md reports/case39_failure_aware_statistics.md

echo "[runner] === recover_input_mode equivalence ==="
.venv_rocm/bin/python -u check_recover_input_mode_equivalence.py \
  --tau_main "$TAU_MAIN" \
  --tau_strict "$TAU_STRICT" \
  --output_csv metric/case39/recover_input_mode_equivalence.csv \
  --output_json metric/case39/recover_input_mode_equivalence.json \
  --output_md reports/recover_input_mode_equivalence.md \
  > logs/case39_recover_input_mode_equivalence.log 2>&1

echo "[runner] === next_load_idx old offset=7 suite ==="
materialize_attack "-1.0" "next_load_old_baseline" "$OLD_ATTACK_BASELINE" "offset"
materialize_attack "$TAU_MAIN" "next_load_old_main" "$OLD_ATTACK_MAIN" "offset"
materialize_attack "$TAU_STRICT" "next_load_old_strict" "$OLD_ATTACK_STRICT" "offset"

echo "[runner] === next_load_idx A/B summary ==="
.venv_rocm/bin/python -u summarize_next_load_idx_ab.py \
  --old-baseline "$OLD_ATTACK_BASELINE" \
  --old-main "$OLD_ATTACK_MAIN" \
  --old-strict "$OLD_ATTACK_STRICT" \
  --new-baseline "$ATTACK_BASELINE" \
  --new-main "$ATTACK_MAIN" \
  --new-strict "$ATTACK_STRICT" \
  --output-csv metric/case39/next_load_idx_ab_check.csv \
  --output-md reports/next_load_idx_ab_check.md \
  --output-json metric/case39/next_load_idx_ab_check.json

echo "[runner] finished_at=$(date -Is)"

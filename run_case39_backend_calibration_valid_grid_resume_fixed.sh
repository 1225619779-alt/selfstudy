#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs reports metric/case39/backend_calibration_valid_grid metric/case39/backend_calibration_valid_grid/candidates

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export DDET_DATA_SEED="${DDET_DATA_SEED:-20260417}"
export DDET_MEASURE_SEED="${DDET_MEASURE_SEED:-20260417}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export DDET_MTD_VERBOSE="${DDET_MTD_VERBOSE:-0}"
export DDET_MTD_IS_WORST="${DDET_MTD_IS_WORST:-0}"

SUMMARY_PATH="metric/case39/tau_selection_joint_valid/tau_selection_summary.txt"
CLEAN_VALID_CACHE="metric/case39/clean_alarm_cache_valid_mode_0_0.03_1.1.npy"
ATTACK_VALID_CACHE="metric/case39/attack_alarm_cache_valid_50_repo_compatible_mode_0_0.03_1.1.npy"
OUT_DIR="metric/case39/backend_calibration_valid_grid/candidates"
LOG_PATH="logs/case39_backend_calibration_valid_grid_resume_fixed.log"

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[grid-resume] missing tau summary: $SUMMARY_PATH" >&2
  exit 1
fi
if [[ ! -f "$CLEAN_VALID_CACHE" || ! -f "$ATTACK_VALID_CACHE" ]]; then
  echo "[grid-resume] missing validation caches" >&2
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
        raise SystemExit(f"missing {name}")
    return m.group(1)
print(grab("tau_main"))
print(grab("tau_strict"))
PY
)"
TAU_MAIN="$(printf '%s\n' "$TAU_INFO" | sed -n '1p')"
TAU_STRICT="$(printf '%s\n' "$TAU_INFO" | sed -n '2p')"

exec >>"$LOG_PATH" 2>&1

echo "[grid-resume] invoked_at=$(date -Is)"
echo "[grid-resume] tau_main=$TAU_MAIN"
echo "[grid-resume] tau_strict=$TAU_STRICT"
echo "[grid-resume] mtd_verbose=$DDET_MTD_VERBOSE"
echo "[grid-resume] mtd_is_worst=$DDET_MTD_IS_WORST"
echo "[grid-resume] policy=reuse_clean_rerun_attack_and_summary"

for X_FACTS_RATIO in 0.2 0.3 0.5; do
  for VARRHO in 0.015 0.02 0.03; do
    for UPPER_SCALE in 1.00 1.05 1.10; do
      export DDET_MTD_X_FACTS_RATIO="$X_FACTS_RATIO"
      export DDET_MTD_VARRHO="$VARRHO"
      export DDET_MTD_UPPER_SCALE="$UPPER_SCALE"
      export DDET_MTD_MULTI_RUN_NO="15"

      TAG="xf_${X_FACTS_RATIO}_var_${VARRHO}_up_${UPPER_SCALE}_mr_15"
      CLEAN_OUT="$OUT_DIR/${TAG}_clean.npy"
      ATTACK_OUT="$OUT_DIR/${TAG}_attack.npy"
      SUMMARY_OUT="$OUT_DIR/${TAG}.json"

      echo "[grid-resume] running $TAG"

      if [[ -f "$CLEAN_OUT" ]]; then
        echo "[grid-resume] reuse clean cache for $TAG"
      else
        .venv_rocm/bin/python -u recompute_clean_backend_from_cache.py \
          --input-cache "$CLEAN_VALID_CACHE" \
          --output-cache "$CLEAN_OUT" \
          --next_load_mode sample_length \
          --next_load_extra 7 \
          --log_every 100 \
          > "logs/${TAG}_clean.log" 2>&1
      fi

      .venv_rocm/bin/python -u recompute_attack_backend_from_cache.py \
        --input-cache "$ATTACK_VALID_CACHE" \
        --output-cache "$ATTACK_OUT" \
        --next_load_modes sample_length \
        --next_load_extra 7 \
        --log_every 20 \
        > "logs/${TAG}_attack.log" 2>&1

      .venv_rocm/bin/python -u summarize_backend_candidate_from_caches.py \
        --clean-cache "$CLEAN_OUT" \
        --attack-cache "$ATTACK_OUT" \
        --tau-main "$TAU_MAIN" \
        --tau-strict "$TAU_STRICT" \
        --x-facts-ratio "$X_FACTS_RATIO" \
        --varrho "$VARRHO" \
        --upper-scale "$UPPER_SCALE" \
        --multi-run-no 15 \
        --output-json "$SUMMARY_OUT"

      echo "[grid-resume] finished $TAG"
    done
  done
done

.venv_rocm/bin/python -u summarize_backend_calibration_valid_grid.py \
  --input-dir "$OUT_DIR" \
  --output-csv metric/case39/backend_calibration_valid_grid.csv \
  --output-json metric/case39/backend_calibration_valid_grid.json \
  --output-md reports/case39_backend_calibration_valid_grid.md

echo "[grid-resume] finished_at=$(date -Is)"

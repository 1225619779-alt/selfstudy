#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39/backend_calibration_targeted reports metric/case39/backend_calibration_valid_grid/candidates

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
LOG_PATH="logs/case39_targeted_backend_check.log"

TAU_INFO="$(
  SUMMARY_PATH="$SUMMARY_PATH" .venv_rocm/bin/python - <<'PY'
import os
import re
from pathlib import Path
text = Path(os.environ["SUMMARY_PATH"]).read_text(encoding="utf-8")
def grab(name: str) -> str:
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
    print(m.group(1))
grab("tau_main")
grab("tau_strict")
PY
)"
TAU_MAIN="$(printf '%s\n' "$TAU_INFO" | sed -n '1p')"
TAU_STRICT="$(printf '%s\n' "$TAU_INFO" | sed -n '2p')"

exec > >(tee "$LOG_PATH") 2>&1

echo "[targeted] invoked_at=$(date -Is)"
echo "[targeted] tau_main=$TAU_MAIN"
echo "[targeted] tau_strict=$TAU_STRICT"

for UPPER_SCALE in 1.00 1.05 1.10; do
  export DDET_MTD_X_FACTS_RATIO="0.5"
  export DDET_MTD_VARRHO="0.015"
  export DDET_MTD_UPPER_SCALE="$UPPER_SCALE"
  export DDET_MTD_MULTI_RUN_NO="15"

  TAG="xf_0.5_var_0.015_up_${UPPER_SCALE}_mr_15"
  CLEAN_OUT="$OUT_DIR/${TAG}_clean.npy"
  ATTACK_OUT="$OUT_DIR/${TAG}_attack.npy"
  SUMMARY_OUT="$OUT_DIR/${TAG}.json"

  echo "[targeted] running $TAG"

  .venv_rocm/bin/python -u recompute_clean_backend_from_cache.py \
    --input-cache "$CLEAN_VALID_CACHE" \
    --output-cache "$CLEAN_OUT" \
    --next_load_mode sample_length \
    --next_load_extra 7 \
    --log_every 100 \
    > "logs/${TAG}_clean.log" 2>&1

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
    --x-facts-ratio 0.5 \
    --varrho 0.015 \
    --upper-scale "$UPPER_SCALE" \
    --multi-run-no 15 \
    --output-json "$SUMMARY_OUT"

  echo "[targeted] finished $TAG"
done

echo "[targeted] finished_at=$(date -Is)"

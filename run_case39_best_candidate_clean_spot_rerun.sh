#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39/spot_rerun reports

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
export DDET_MTD_X_FACTS_RATIO="0.2"
export DDET_MTD_VARRHO="0.015"
export DDET_MTD_UPPER_SCALE="1.05"
export DDET_MTD_MULTI_RUN_NO="15"

OUT_CACHE="metric/case39/spot_rerun/xf_0.2_var_0.015_up_1.05_mr_15_clean_from_scratch.npy"
OUT_METRIC="metric/case39/spot_rerun/xf_0.2_var_0.015_up_1.05_mr_15_main_exact_metric.npy"
LOG_PATH="logs/case39_best_candidate_clean_spot_rerun.log"

TAU_MAIN="$(
  .venv_rocm/bin/python - <<'PY'
import re
from pathlib import Path
text = Path("metric/case39/tau_selection_joint_valid/tau_selection_summary.txt").read_text(encoding="utf-8")
m = re.search(r"tau_main\s*=\s*([0-9.]+)", text)
print(m.group(1))
PY
)"

exec > >(tee "$LOG_PATH") 2>&1

echo "[spot-rerun] invoked_at=$(date -Is)"
echo "[spot-rerun] tau_main=$TAU_MAIN"
echo "[spot-rerun] cache=$OUT_CACHE"

.venv_rocm/bin/python -u collect_clean_alarm_cache.py \
  --split valid \
  --seed_base 20260324 \
  --next_load_mode sample_length \
  --next_load_extra 7 \
  --log_every 100 \
  --output "$OUT_CACHE"

.venv_rocm/bin/python -u materialize_clean_metric_from_cache.py \
  --cache "$OUT_CACHE" \
  --tau_verify "$TAU_MAIN" \
  --output "$OUT_METRIC"

echo "[spot-rerun] finished_at=$(date -Is)"

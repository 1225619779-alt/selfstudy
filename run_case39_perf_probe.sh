#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD

mkdir -p logs metric/case39/perf_probe

LOG_PATH="logs/case39_perf_probe.log"
if [[ ! -f "$LOG_PATH" ]]; then
  : >"$LOG_PATH"
fi
exec >>"$LOG_PATH" 2>&1

export DDET_CASE_NAME="${DDET_CASE_NAME:-case39}"
export DDET_DATA_SEED="${DDET_DATA_SEED:-20260417}"
export DDET_MEASURE_SEED="${DDET_MEASURE_SEED:-20260417}"

echo "[probe] invoked_at=$(date -Is)"
echo "[probe] case_name=${DDET_CASE_NAME}"
echo "[probe] data_seed=${DDET_DATA_SEED}"
echo "[probe] measure_seed=${DDET_MEASURE_SEED}"

THREADS_LIST=(1 4 8)
MAX_TOTAL_RUN="${MAX_TOTAL_RUN:-256}"
STOP_DDD_ALARM_AT="${STOP_DDD_ALARM_AT:-20}"

SUMMARY_JSON="metric/case39/perf_probe/summary.json"
SUMMARY_TXT="metric/case39/perf_probe/summary.txt"

: >"$SUMMARY_TXT"

for threads in "${THREADS_LIST[@]}"; do
  export OMP_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  export VECLIB_MAXIMUM_THREADS="$threads"

  out_path="metric/case39/perf_probe/clean_scores_threads${threads}.npy"
  run_log="logs/case39_perf_probe_threads${threads}.log"

  echo "[probe] threads=${threads} start_at=$(date -Is)" | tee -a "$SUMMARY_TXT"
  start_ts=$(date +%s)

  .venv_rocm/bin/python -u collect_clean_alarm_scores.py \
    --max_total_run "$MAX_TOTAL_RUN" \
    --stop_ddd_alarm_at "$STOP_DDD_ALARM_AT" \
    --output "$out_path" >"$run_log" 2>&1

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  THREADS="$threads" OUT_PATH="$out_path" ELAPSED="$elapsed" .venv_rocm/bin/python - <<'PY' | tee -a "$SUMMARY_TXT"
import json
import os
import numpy as np

threads = int(os.environ["THREADS"])
path = os.environ["OUT_PATH"]
elapsed = int(os.environ["ELAPSED"])
d = np.load(path, allow_pickle=True).item()
g = d["group_key"]
row = {
    "threads": threads,
    "elapsed_sec": elapsed,
    "total_clean_sample": int(d["total_clean_sample"][g]),
    "total_DDD_alarm": int(d["total_DDD_alarm"][g]),
    "recover_fail_count": int(np.sum(np.asarray(d["recover_fail"][g], dtype=bool))),
    "mean_score_phys_l2": float(np.nanmean(np.asarray(d["score_phys_l2"][g], dtype=float))),
}
print(json.dumps(row, ensure_ascii=True))
PY
done

.venv_rocm/bin/python - <<'PY' >"$SUMMARY_JSON"
import json
from pathlib import Path
import numpy as np

base = Path("metric/case39/perf_probe")
paths = sorted(base.glob("clean_scores_threads*.npy"))
items = []
payloads = []
for path in paths:
    d = np.load(path, allow_pickle=True).item()
    g = d["group_key"]
    payloads.append((path.name, d))
    items.append({
        "path": path.name,
        "total_clean_sample": int(d["total_clean_sample"][g]),
        "total_DDD_alarm": int(d["total_DDD_alarm"][g]),
        "clean_alarm_idx": list(map(int, d["clean_alarm_idx"][g])),
        "ddd_loss_alarm": np.asarray(d["ddd_loss_alarm"][g], dtype=float).tolist(),
        "score_phys_l2": np.asarray(d["score_phys_l2"][g], dtype=float).tolist(),
    })

reference = items[0] if items else None
consistency = []
if reference is not None:
    for item in items[1:]:
      same = (
          item["total_clean_sample"] == reference["total_clean_sample"]
          and item["total_DDD_alarm"] == reference["total_DDD_alarm"]
          and item["clean_alarm_idx"] == reference["clean_alarm_idx"]
          and np.allclose(np.asarray(item["ddd_loss_alarm"]), np.asarray(reference["ddd_loss_alarm"]), equal_nan=True)
          and np.allclose(np.asarray(item["score_phys_l2"]), np.asarray(reference["score_phys_l2"]), equal_nan=True)
      )
      consistency.append({
          "reference": reference["path"],
          "candidate": item["path"],
          "consistent": bool(same),
      })

print(json.dumps({
    "files": items,
    "consistency": consistency,
}, ensure_ascii=True, indent=2))
PY

echo "[probe] summary_json=$SUMMARY_JSON"
echo "[probe] summary_txt=$SUMMARY_TXT"
echo "[probe] finished_at=$(date -Is)"

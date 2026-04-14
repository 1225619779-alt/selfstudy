#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="metric/case39/postrun_audits/${TS}"
mkdir -p "$OUT_DIR"

echo "postrun_out_dir=$OUT_DIR"

# Try to discover the latest stage stamp created by earlier native runs.
STAMP=""
for cand in /tmp/case39_resume_step3_*.stamp /tmp/case39_native_stage2_*.stamp /tmp/case39_native_stage2_*.tmpstamp; do
  if [[ -e "$cand" ]]; then
    STAMP="$cand"
  fi
done
if [[ -n "$STAMP" ]]; then
  echo "using_stamp=$STAMP"
else
  echo "using_stamp=NONE_FOUND"
fi

audit_slice() {
  local s="$1"
  local e="$2"
  local SLICE_DIR="$OUT_DIR/slice_${s}_${e}"
  mkdir -p "$SLICE_DIR"
  echo "[audit] slice ${s}:${e}"
  python - "$SLICE_DIR" "$s" "$e" <<'PY'
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
s = int(sys.argv[2]); e = int(sys.argv[3])

z = np.load('gen_data/case39/z_noise_summary.npy')
v = np.load('gen_data/case39/v_est_summary.npy')
succ = np.load('gen_data/case39/success_summary.npy')

if e > len(z):
    raise ValueError(f"slice end {e} exceeds total length {len(z)}")

np.save(out / 'z_noise_summary.npy', z[s:e], allow_pickle=False)
np.save(out / 'v_est_summary.npy', v[s:e], allow_pickle=False)
np.save(out / 'success_summary.npy', succ[s:e], allow_pickle=False)
print({'slice_saved': str(out), 'n_steps': int(e-s)})
PY

  env DDET_CASE_NAME=case39 python case39_measure_v2_audit.py \
    --repo_root . \
    --case_name case39 \
    --parallel_out_root "$SLICE_DIR" \
    --start_idx "$s" \
    --end_idx "$e" \
    --output "$OUT_DIR/case39_measure_v2_audit_${s}_${e}.json"
}

# exact-match sampled audits from the FULL run outputs
audit_slice 0 16
audit_slice 128 144
audit_slice 4096 4112
audit_slice 20000 20016
audit_slice 34000 34016

# fair runtime benchmarks: same script, only workers differ
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[bench] workers=1 over 256 steps"
env DDET_CASE_NAME=case39 python case39_measure_parallel_v2.py \
  --repo_root . \
  --case_name case39 \
  --workers 1 \
  --chunk_steps 16 \
  --start_idx 0 \
  --end_idx 256 \
  --out_root "$OUT_DIR/fair_bench_workers1_256"

echo "[bench] workers=4 over 256 steps"
env DDET_CASE_NAME=case39 python case39_measure_parallel_v2.py \
  --repo_root . \
  --case_name case39 \
  --workers 4 \
  --chunk_steps 16 \
  --start_idx 0 \
  --end_idx 256 \
  --out_root "$OUT_DIR/fair_bench_workers4_256"

# anti-write audit if we have a usable stamp
if [[ -n "$STAMP" ]]; then
  echo "[audit] anti-write checks"
  find metric/case14 -type f -newer "$STAMP" -print > "$OUT_DIR/anti_write_q1_case14.txt" || true
  find /home/pang/projects/DDET-MTD/metric/case14 -type f -newer "$STAMP" -print > "$OUT_DIR/anti_write_oldrepo_case14.txt" || true
else
  : > "$OUT_DIR/anti_write_q1_case14.txt"
  : > "$OUT_DIR/anti_write_oldrepo_case14.txt"
fi

# compact result snapshot
python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])

def loadj(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

v1 = loadj('metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json')
v2 = loadj('metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json')

def slot_stats(obj, slot):
    ps = obj['slot_budget_aggregates'][str(slot)]['policy_stats']
    res = {}
    for m in ['phase3_oracle_upgrade','phase3_proposed','topk_expected_consequence']:
        res[m] = {
            'mean_recall': ps[m]['weighted_attack_recall_no_backend_fail']['mean'],
            'mean_unnecessary': ps[m]['unnecessary_mtd_count']['mean'],
            'mean_cost': ps[m]['average_service_cost_per_step']['mean'],
            'mean_served_ratio': ps[m]['pred_expected_consequence_served_ratio']['mean'],
        }
    res['paired'] = obj['slot_budget_aggregates'][str(slot)]['paired_stats']
    return res

summary = {
    'native_case39_stage': 'native_clean_attack_test_with_frozen_case14_dev',
    'v1_schedule': v1['confirm_manifest']['schedule'],
    'v2_schedule': v2['confirm_manifest']['schedule'],
    'slot1_v1': slot_stats(v1, 1),
    'slot2_v1': slot_stats(v1, 2),
    'slot1_v2': slot_stats(v2, 1),
    'slot2_v2': slot_stats(v2, 2),
}

# merged 8-holdout averages (v1/v2 each have 4 holdouts)
merged = {}
for slot in ['1','2']:
    merged[slot] = {}
    for m in ['phase3_oracle_upgrade','phase3_proposed','topk_expected_consequence']:
        a = v1['slot_budget_aggregates'][slot]['policy_stats'][m]
        b = v2['slot_budget_aggregates'][slot]['policy_stats'][m]
        merged[slot][m] = {
            'mean_recall': (a['weighted_attack_recall_no_backend_fail']['mean'] + b['weighted_attack_recall_no_backend_fail']['mean'])/2,
            'mean_unnecessary': (a['unnecessary_mtd_count']['mean'] + b['unnecessary_mtd_count']['mean'])/2,
            'mean_cost': (a['average_service_cost_per_step']['mean'] + b['average_service_cost_per_step']['mean'])/2,
            'mean_served_ratio': (a['pred_expected_consequence_served_ratio']['mean'] + b['pred_expected_consequence_served_ratio']['mean'])/2,
        }
summary['merged_8_holdouts'] = merged

with open(out / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

lines = []
lines.append(f"postrun_out_dir={out}")
lines.append("== merged_8_holdouts ==")
for slot in ['1','2']:
    lines.append(f"-- slot_budget={slot} --")
    for m in ['phase3_oracle_upgrade','phase3_proposed','topk_expected_consequence']:
        x = summary['merged_8_holdouts'][slot][m]
        lines.append(f"{m}: recall={x['mean_recall']:.6f}, unnecessary={x['mean_unnecessary']:.3f}, cost={x['mean_cost']:.6f}, served_ratio={x['mean_served_ratio']:.6f}")
with open(out / 'summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
PY

echo "DONE. postrun outputs:"
ls -lh "$OUT_DIR"

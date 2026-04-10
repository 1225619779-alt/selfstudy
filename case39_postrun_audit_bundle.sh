#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_FOR_FILE="${WAIT_FOR_FILE:-}"
WAIT_STABLE_SEC="${WAIT_STABLE_SEC:-20}"
STAMP_PATH="${STAMP_PATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DDET_CASE_NAME=case39

if [[ -z "$STAMP_PATH" ]]; then
  STAMP_PATH="$(ls -1t /tmp/case39_*stage2*.stamp /tmp/case39_native_stage2_*.stamp /tmp/case39_bridge_*.stamp 2>/dev/null | head -n 1 || true)"
fi

wait_for_pid() {
  local pid="$1"
  echo "[wait] waiting for PID $pid to exit ..."
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
  echo "[wait] PID $pid exited."
}

wait_for_file_stable() {
  local path="$1"
  local stable_sec="$2"
  local last_mtime=""
  local stable_start=""
  echo "[wait] waiting for file to appear and stabilize: $path"
  while true; do
    if [[ -f "$path" ]]; then
      local mtime now
      mtime="$(stat -c %Y "$path" 2>/dev/null || true)"
      if [[ -n "$mtime" && "$mtime" == "$last_mtime" ]]; then
        if [[ -z "$stable_start" ]]; then stable_start="$(date +%s)"; fi
        now="$(date +%s)"
        if (( now - stable_start >= stable_sec )); then
          echo "[wait] file stable for >= ${stable_sec}s: $path"
          break
        fi
      else
        last_mtime="$mtime"
        stable_start=""
      fi
    fi
    sleep 10
  done
}

if [[ -n "$WAIT_FOR_PID" ]]; then
  wait_for_pid "$WAIT_FOR_PID"
elif [[ -n "$WAIT_FOR_FILE" ]]; then
  wait_for_file_stable "$WAIT_FOR_FILE" "$WAIT_STABLE_SEC"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="metric/case39/postrun_audits/${TS}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/postrun.log"
SUMMARY_JSON="$OUT_DIR/summary.json"
SUMMARY_TXT="$OUT_DIR/summary.txt"

echo "postrun_out_dir=$OUT_DIR" | tee "$LOG"

audit_slice() {
  local s="$1" e="$2" out_json="$OUT_DIR/case39_measure_v2_audit_${s}_${e}.json"
  echo "[audit] slice ${s}:${e}" | tee -a "$LOG"
  env DDET_CASE_NAME=case39 python case39_measure_v2_audit.py \
    --repo_root . \
    --case_name case39 \
    --parallel_out_root gen_data/case39 \
    --start_idx "$s" \
    --end_idx "$e" \
    --output "$out_json" >> "$LOG" 2>&1
}

# 1) sampled exact-match audits on the finished full output
for s in 0 128 4096 20000 34000; do
  e=$((s+16))
  audit_slice "$s" "$e"
done

# 2) fair runtime comparison under the same v2 implementation
SEQ_OUT="gen_data/case39_bench_sequential_v2_256"
PAR_OUT="gen_data/case39_bench_parallel_v2_256_postrun"
rm -rf "$SEQ_OUT" "$PAR_OUT"

echo "[bench] workers=1 chunk_steps=16 0:256" | tee -a "$LOG"
env DDET_CASE_NAME=case39 python case39_measure_parallel_v2.py \
  --repo_root . --case_name case39 --workers 1 --chunk_steps 16 \
  --start_idx 0 --end_idx 256 --out_root "$SEQ_OUT" >> "$LOG" 2>&1

echo "[bench] workers=4 chunk_steps=16 0:256" | tee -a "$LOG"
env DDET_CASE_NAME=case39 python case39_measure_parallel_v2.py \
  --repo_root . --case_name case39 --workers 4 --chunk_steps 16 \
  --start_idx 0 --end_idx 256 --out_root "$PAR_OUT" >> "$LOG" 2>&1

OUT_DIR="$OUT_DIR" SUMMARY_JSON="$SUMMARY_JSON" STAMP_PATH="$STAMP_PATH" python - <<'PY'
from __future__ import annotations
import json, os
from pathlib import Path

repo = Path('.').resolve()
out_dir = Path(os.environ['OUT_DIR'])
stamp_path = os.environ.get('STAMP_PATH','')
summary_json = Path(os.environ['SUMMARY_JSON'])

def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))

summary = {
    'repo_root': str(repo),
    'postrun_dir': str(out_dir),
    'checkpoint_case39_exists': (repo/'saved_model'/'case39'/'checkpoint_rnn.pt').exists(),
    'clean_bank_exists': (repo/'metric'/'case39'/'metric_clean_alarm_scores_full.npy').exists(),
    'attack_bank_exists': (repo/'metric'/'case39'/'metric_attack_alarm_scores_400.npy').exists(),
    'native_confirm_candidates': {},
    'audits': {},
    'benchmarks': {},
    'anti_write': {},
}
for k,p in {
    'v1_native_clean_attack_test': repo/'metric'/'case39'/'phase3_oracle_confirm_v1_native_clean_attack_test'/'aggregate_summary.json',
    'v2_native_clean_attack_test': repo/'metric'/'case39'/'phase3_oracle_confirm_v2_native_clean_attack_test'/'aggregate_summary.json',
    'v1_bridge_or_existing': repo/'metric'/'case39'/'phase3_oracle_confirm_v1'/'aggregate_summary.json',
    'v2_bridge_or_existing': repo/'metric'/'case39'/'phase3_oracle_confirm_v2'/'aggregate_summary.json',
}.items():
    summary['native_confirm_candidates'][k] = {'path': str(p), 'exists': p.exists()}

for p in sorted(out_dir.glob('case39_measure_v2_audit_*_*.json')):
    d = load_json(p)
    if d is None:
        continue
    summary['audits'][p.name] = {
        'success_exact_equal': d['agreement']['success_exact_equal'],
        'z_max_abs_diff': d['agreement']['z_max_abs_diff'],
        'v_max_abs_diff': d['agreement']['v_max_abs_diff'],
        'seq_sec_per_iter_mean': d['sequential_runtime']['sec_per_iter_mean'],
    }

for k,p in {
    'workers1': repo/'gen_data'/'case39_bench_sequential_v2_256'/'parallel_measure_report.json',
    'workers4': repo/'gen_data'/'case39_bench_parallel_v2_256_postrun'/'parallel_measure_report.json',
}.items():
    d = load_json(p)
    if d is None:
        continue
    summary['benchmarks'][k] = {
        'path': str(p),
        'sec_per_iter_effective': d['sec_per_iter_effective'],
        'elapsed_sec': d['elapsed_sec'],
        'success_rate_raw': d['success_rate_raw'],
        'n_forward_filled': d['n_forward_filled'],
    }

if stamp_path:
    sp = Path(stamp_path)
    summary['anti_write']['stamp_path'] = str(sp)
    if sp.exists():
        q1_root = repo/'metric'/'case14'
        old_root = Path('/home/pang/projects/DDET-MTD/metric/case14')
        summary['anti_write']['q1_case14_newer_than_stamp'] = [str(p) for p in q1_root.rglob('*') if p.is_file() and p.stat().st_mtime > sp.stat().st_mtime]
        summary['anti_write']['old_repo_case14_newer_than_stamp'] = [str(p) for p in old_root.rglob('*') if p.is_file() and p.stat().st_mtime > sp.stat().st_mtime] if old_root.exists() else None
    else:
        summary['anti_write']['warning'] = 'stamp path does not exist'
else:
    summary['anti_write']['warning'] = 'no stamp path provided or auto-detected'

summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
PY

SUMMARY_JSON="$SUMMARY_JSON" python - <<'PY' > "$SUMMARY_TXT"
from __future__ import annotations
import json, os
from pathlib import Path
p = Path(os.environ['SUMMARY_JSON'])
d = json.loads(p.read_text(encoding='utf-8'))
print('=== case39 postrun audit bundle ===')
print('postrun_dir:', d['postrun_dir'])
print('checkpoint_case39_exists:', d['checkpoint_case39_exists'])
print('clean_bank_exists:', d['clean_bank_exists'])
print('attack_bank_exists:', d['attack_bank_exists'])
print('\n--- native confirm candidates ---')
for k, v in d['native_confirm_candidates'].items():
    print(f"{k}: {v['exists']} -> {v['path']}")
print('\n--- sampled audits ---')
for k, v in sorted(d['audits'].items()):
    print(f"{k}: success_exact_equal={v['success_exact_equal']}, z_max_abs_diff={v['z_max_abs_diff']}, v_max_abs_diff={v['v_max_abs_diff']}, seq_sec_per_iter_mean={v['seq_sec_per_iter_mean']}")
print('\n--- fair runtime ---')
for k, v in d['benchmarks'].items():
    print(f"{k}: sec_per_iter_effective={v['sec_per_iter_effective']}, elapsed_sec={v['elapsed_sec']}, success_rate_raw={v['success_rate_raw']}, n_forward_filled={v['n_forward_filled']}")
print('\n--- anti-write ---')
for k, v in d['anti_write'].items():
    print(f"{k}: {v}")
PY

echo "=== case39 postrun audit bundle ==="
echo "summary_txt=$SUMMARY_TXT"
echo "summary_json=$SUMMARY_JSON"
echo "log=$LOG"
cat "$SUMMARY_TXT"

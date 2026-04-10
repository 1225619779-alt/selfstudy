#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
DEV_SUMMARY="${2:-$REPO_ROOT/metric/case14/phase3_oracle_family/screen_train_val_summary.json}"
WORKERS="${3:-4}"
CHUNK_STEPS="${4:-16}"
TOTAL_STEPS="${5:-35134}"

cd "$REPO_ROOT"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run() {
  echo ">>> $*"
  "$@"
}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

STAMP="/tmp/case39_native_continue_v2_$(date +%s).stamp"
touch "$STAMP"
log "STAMP=$STAMP"

log "1/8 full native case39 measurement generation with audited v2"
rm -f gen_data/case39/z_noise_summary.npy gen_data/case39/v_est_summary.npy gen_data/case39/success_summary.npy
run env DDET_CASE_NAME=case39 python case39_measure_parallel_v2.py \
  --repo_root . \
  --case_name case39 \
  --workers "$WORKERS" \
  --chunk_steps "$CHUNK_STEPS" \
  --start_idx 0 \
  --end_idx "$TOTAL_STEPS" \
  --out_root gen_data/case39

log "2/8 quick shape sanity check"
run python - <<'PY'
import numpy as np
z = np.load('gen_data/case39/z_noise_summary.npy')
v = np.load('gen_data/case39/v_est_summary.npy')
s = np.load('gen_data/case39/success_summary.npy')
print({'z_shape': z.shape, 'v_shape': v.shape, 's_shape': s.shape, 'success_rate': float(s.mean())})
PY

log "3/8 train native case39 checkpoint"
mkdir -p saved_model/case39
run env DDET_CASE_NAME=case39 python models/model.py

log "4/8 replace case39 clean/attack symlinks with native outputs"
for p in \
  metric/case39/metric_clean_alarm_scores_full.npy \
  metric/case39/metric_attack_alarm_scores_400.npy \
  metric/case39/mixed_bank_test_smoke.npy
  do
    if [[ -L "$p" ]]; then
      rm "$p"
    fi
  done
run env DDET_CASE_NAME=case39 python collect_clean_alarm_scores.py \
  --output metric/case39/metric_clean_alarm_scores_full.npy
run env DDET_CASE_NAME=case39 python collect_attack_alarm_scores.py \
  --total_run 400 \
  --output metric/case39/metric_attack_alarm_scores_400.npy

log "5/8 remove symlinked blind-confirm test banks so they can be regenerated natively"
find metric/case39/phase3_confirm_blind_v1/banks -maxdepth 1 -type l -name 'mixed_bank_test_*.npy' -delete 2>/dev/null || true
find metric/case39/phase3_confirm_blind_v2/banks -maxdepth 1 -type l -name 'mixed_bank_test_*.npy' -delete 2>/dev/null || true

log "6/8 rebuild native test banks + baseline holdout summaries from existing manifests"
run env DDET_CASE_NAME=case39 python rebuild_confirm_holdouts_from_manifest.py \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --force
run env DDET_CASE_NAME=case39 python rebuild_confirm_holdouts_from_manifest.py \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --force

log "7/8 fixed-winner oracle confirm with frozen case14 dev summary"
if [[ ! -f "$DEV_SUMMARY" ]]; then
  echo "ERROR: missing dev summary: $DEV_SUMMARY" >&2
  exit 1
fi
run python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --dev_screen_summary "$DEV_SUMMARY" \
  --output metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test
run python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --dev_screen_summary "$DEV_SUMMARY" \
  --output metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test

log "8/8 anti-write audit + key outputs"
find metric/case14 -type f -newer "$STAMP" -print || true
find /home/pang/projects/DDET-MTD/metric/case14 -type f -newer "$STAMP" -print || true
ls -l saved_model/case39/checkpoint_rnn.pt
ls -l metric/case39/metric_clean_alarm_scores_full.npy
ls -l metric/case39/metric_attack_alarm_scores_400.npy
ls -l metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json
ls -l metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json

echo "STAMP=$STAMP"
log "DONE"

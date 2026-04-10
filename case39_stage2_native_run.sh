#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
OLD_REPO_ROOT="${2:-/home/pang/projects/DDET-MTD}"
DEV_SUMMARY="${3:-$REPO_ROOT/metric/case14/phase3_oracle_family/screen_train_val_summary.json}"

cd "$REPO_ROOT"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run() {
  echo ">>> $*"
  "$@"
}

log "1/10 apply native stage-2 patch"
run python case39_stage2_native_patch.py .

log "2/10 stage raw CSVs from old repo into current repo raw_data"
mkdir -p gen_data/raw_data saved_model/case39
for f in load.csv pv.csv; do
  src="$OLD_REPO_ROOT/gen_data/raw_data/$f"
  dst="gen_data/raw_data/$f"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: missing raw asset: $src" >&2
    exit 1
  fi
  rm -f "$dst"
  ln -s "$src" "$dst"
  ls -l "$dst"
done

log "3/10 build five basic case39 npy assets"
run env DDET_CASE_NAME=case39 python generate_case_basic_npy.py

log "4/10 generate measurement arrays under gen_data/case39"
run env DDET_CASE_NAME=case39 python gen_data/gen_data.py

log "5/10 train native case39 checkpoint"
mkdir -p saved_model/case39
run env DDET_CASE_NAME=case39 python models/model.py

log "6/10 replace case39 clean/attack symlinks with native outputs"
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

log "7/10 remove symlinked blind-confirm test banks so they can be regenerated natively"
find metric/case39/phase3_confirm_blind_v1/banks -maxdepth 1 -type l -name 'mixed_bank_test_*.npy' -delete 2>/dev/null || true
find metric/case39/phase3_confirm_blind_v2/banks -maxdepth 1 -type l -name 'mixed_bank_test_*.npy' -delete 2>/dev/null || true

log "8/10 rebuild native test banks + baseline holdout summaries from existing manifests"
run env DDET_CASE_NAME=case39 python rebuild_confirm_holdouts_from_manifest.py \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --force
run env DDET_CASE_NAME=case39 python rebuild_confirm_holdouts_from_manifest.py \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --force

log "9/10 run fixed-winner oracle confirm with frozen case14 dev summary"
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

log "10/10 quick inventory"
ls -l saved_model/case39/checkpoint_rnn.pt
ls -l metric/case39/metric_clean_alarm_scores_full.npy
ls -l metric/case39/metric_attack_alarm_scores_400.npy
ls -l metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json
ls -l metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json

log "DONE"

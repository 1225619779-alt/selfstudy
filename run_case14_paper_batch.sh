#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the repo root.
# This script freezes the current paper worldline:
#   tau_main   = 0.03318
#   tau_strict = 0.03675
# and runs clean paired results, score collection, matched-budget ablation,
# attack ARR summary, and mixed-timeline summaries.

TAU_MAIN="0.03318"
TAU_STRICT="0.03675"
CASE_NAME="case14"
VAR_RHO="0.03"
UPPER_SCALE="1.1"
MIXED_SCHEDULE="clean:80;att-1-0.2:30;clean:40;att-2-0.2:30;clean:40;att-3-0.3:30;clean:80"
SEED_CLEAN="20260324"
SEED_ATTACK="20260324"
SEED_MIXED="20260331"
RUN_MIXED="1"   # set to 0 if you want to skip mixed timeline in a quick run

ROOT_DIR="$(pwd)"
LOG_DIR="logs"
TMP_DIR="$LOG_DIR/.tmp_case14_batch"
mkdir -p "$LOG_DIR" "$TMP_DIR" "metric/${CASE_NAME}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/case14_paper_batch_${STAMP}.log"
UPLOAD_CHECKLIST="$LOG_DIR/case14_paper_batch_${STAMP}_upload_checklist.txt"
ZIP_FILE="$LOG_DIR/case14_paper_batch_${STAMP}_for_upload.zip"
ARR_TXT="metric/${CASE_NAME}/attack_arr_summary_${TAU_MAIN}_${TAU_STRICT}.txt"

BASELINE_CLEAN="metric/${CASE_NAME}/metric_event_trigger_clean_tau_-1.0_mode_0_${VAR_RHO}_${UPPER_SCALE}.npy"
MAIN_CLEAN="metric/${CASE_NAME}/metric_event_trigger_clean_tau_${TAU_MAIN}_mode_0_${VAR_RHO}_${UPPER_SCALE}.npy"
STRICT_CLEAN="metric/${CASE_NAME}/metric_event_trigger_clean_tau_${TAU_STRICT}_mode_0_${VAR_RHO}_${UPPER_SCALE}.npy"

CLEAN_SCORES="metric/${CASE_NAME}/metric_clean_alarm_scores_full.npy"
ATTACK_SCORES="metric/${CASE_NAME}/metric_attack_alarm_scores_50.npy"
ABLATION_NPY="metric/${CASE_NAME}/metric_gate_ablation_summary_${TAU_MAIN}_${TAU_STRICT}.npy"
ABLATION_CSV="metric/${CASE_NAME}/metric_gate_ablation_summary_${TAU_MAIN}_${TAU_STRICT}.csv"

BASELINE_MIXED="metric/${CASE_NAME}/metric_mixed_timeline_tau_-1.0.npy"
MAIN_MIXED="metric/${CASE_NAME}/metric_mixed_timeline_tau_${TAU_MAIN}.npy"
STRICT_MIXED="metric/${CASE_NAME}/metric_mixed_timeline_tau_${TAU_STRICT}.npy"
MAIN_MIXED_DIR="metric/${CASE_NAME}/plots_mixed_timeline_compare_main"
STRICT_MIXED_DIR="metric/${CASE_NAME}/plots_mixed_timeline_compare_strict"

required_files=(
  "evaluation_event_trigger_clean.py"
  "compare_gate_results_clean.py"
  "collect_clean_alarm_scores.py"
  "collect_attack_alarm_scores.py"
  "analyze_gate_ablation.py"
  "evaluation_mixed_timeline.py"
  "compare_mixed_timeline.py"
  "saved_model/${CASE_NAME}/checkpoint_rnn.pt"
)

section() {
  printf '\n%s\n' "================================================================================" | tee -a "$LOG_FILE"
  printf '%s\n' "$1" | tee -a "$LOG_FILE"
  printf '%s\n' "================================================================================" | tee -a "$LOG_FILE"
}

run_and_capture() {
  local label="$1"
  local mode="$2"
  shift 2
  local raw="$TMP_DIR/${label}.raw.log"

  section "$label"
  printf '+ %q' "$@" | tee -a "$LOG_FILE"
  printf '\n' | tee -a "$LOG_FILE"

  if "$@" >"$raw" 2>&1; then
    case "$mode" in
      summary_from_marker)
        awk '/^==== Summary/{flag=1} flag' "$raw" | tee -a "$LOG_FILE"
        ;;
      full)
        cat "$raw" | tee -a "$LOG_FILE"
        ;;
      tail40)
        tail -n 40 "$raw" | tee -a "$LOG_FILE"
        ;;
      *)
        cat "$raw" | tee -a "$LOG_FILE"
        ;;
    esac
  else
    echo "[ERROR] Command failed." | tee -a "$LOG_FILE"
    cat "$raw" | tee -a "$LOG_FILE"
    exit 1
  fi
}

section "CASE14 paper batch started"
echo "repo_root: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "tau_main: $TAU_MAIN" | tee -a "$LOG_FILE"
echo "tau_strict: $TAU_STRICT" | tee -a "$LOG_FILE"
echo "run_mixed: $RUN_MIXED" | tee -a "$LOG_FILE"

echo "Checking required files..." | tee -a "$LOG_FILE"
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" | tee -a "$LOG_FILE"
    exit 1
  fi
done
echo "All required files found." | tee -a "$LOG_FILE"

run_and_capture "clean_baseline" summary_from_marker \
  python evaluation_event_trigger_clean.py --tau_verify -1.0 --max_total_run -1 --stop_ddd_alarm_at -1 --seed_base "$SEED_CLEAN"

run_and_capture "clean_main" summary_from_marker \
  python evaluation_event_trigger_clean.py --tau_verify "$TAU_MAIN" --max_total_run -1 --stop_ddd_alarm_at -1 --seed_base "$SEED_CLEAN"

run_and_capture "clean_strict" summary_from_marker \
  python evaluation_event_trigger_clean.py --tau_verify "$TAU_STRICT" --max_total_run -1 --stop_ddd_alarm_at -1 --seed_base "$SEED_CLEAN"

run_and_capture "compare_clean_main" full \
  python compare_gate_results_clean.py --baseline "$BASELINE_CLEAN" --gated "$MAIN_CLEAN"

run_and_capture "compare_clean_strict" full \
  python compare_gate_results_clean.py --baseline "$BASELINE_CLEAN" --gated "$STRICT_CLEAN"

run_and_capture "collect_clean_scores" summary_from_marker \
  python collect_clean_alarm_scores.py --max_total_run -1 --stop_ddd_alarm_at -1 --output "$CLEAN_SCORES"

run_and_capture "collect_attack_scores" summary_from_marker \
  python collect_attack_alarm_scores.py --total_run 50 --seed_base "$SEED_ATTACK" --output "$ATTACK_SCORES"

run_and_capture "analyze_gate_ablation" full \
  python analyze_gate_ablation.py \
    --clean_scores "$CLEAN_SCORES" \
    --attack_scores "$ATTACK_SCORES" \
    --baseline_clean_metric "$BASELINE_CLEAN" \
    --main_clean_metric "$MAIN_CLEAN" \
    --strict_clean_metric "$STRICT_CLEAN" \
    --out_npy "$ABLATION_NPY" \
    --out_csv "$ABLATION_CSV"

section "attack_arr_summary"
python - <<PY > "$ARR_TXT"
import numpy as np

path = r"$ATTACK_SCORES"
data = np.load(path, allow_pickle=True).item()

groups = sorted(data["score_phys_l2"].keys(), key=lambda s: eval(s))

for tau in [$TAU_MAIN, $TAU_STRICT]:
    print(f"=== tau = {tau} ===")
    kept_all = 0
    alarm_all = 0
    for g in groups:
        n_alarm = int(data["total_DDD_alarm"][g])
        scores = np.asarray(data["score_phys_l2"][g], dtype=float)
        kept = int(np.sum(np.isfinite(scores) & (scores >= tau)))
        arr = kept / n_alarm if n_alarm else float("nan")
        kept_all += kept
        alarm_all += n_alarm
        print(f"{g:>8}  front_end_alarms={n_alarm:3d}  kept={kept:3d}  ARR={arr:.4f}")
    overall = kept_all / alarm_all if alarm_all else float("nan")
    print(f"Overall  front_end_alarms={alarm_all:3d}  kept={kept_all:3d}  ARR={overall:.4f}")
    print()
PY
cat "$ARR_TXT" | tee -a "$LOG_FILE"

if [[ "$RUN_MIXED" == "1" ]]; then
  run_and_capture "mixed_baseline" summary_from_marker \
    python evaluation_mixed_timeline.py \
      --tau_verify -1.0 \
      --schedule "$MIXED_SCHEDULE" \
      --start_offset 0 \
      --seed_base "$SEED_MIXED" \
      --next_load_extra 0 \
      --output "$BASELINE_MIXED"

  run_and_capture "mixed_main" summary_from_marker \
    python evaluation_mixed_timeline.py \
      --tau_verify "$TAU_MAIN" \
      --schedule "$MIXED_SCHEDULE" \
      --start_offset 0 \
      --seed_base "$SEED_MIXED" \
      --next_load_extra 0 \
      --output "$MAIN_MIXED"

  run_and_capture "mixed_strict" summary_from_marker \
    python evaluation_mixed_timeline.py \
      --tau_verify "$TAU_STRICT" \
      --schedule "$MIXED_SCHEDULE" \
      --start_offset 0 \
      --seed_base "$SEED_MIXED" \
      --next_load_extra 0 \
      --output "$STRICT_MIXED"

  run_and_capture "compare_mixed_main" full \
    python compare_mixed_timeline.py --baseline "$BASELINE_MIXED" --gated "$MAIN_MIXED" --output_dir "$MAIN_MIXED_DIR"

  run_and_capture "compare_mixed_strict" full \
    python compare_mixed_timeline.py --baseline "$BASELINE_MIXED" --gated "$STRICT_MIXED" --output_dir "$STRICT_MIXED_DIR"
fi

section "writing upload checklist"
cat > "$UPLOAD_CHECKLIST" <<EOF2
# Upload this ZIP to me first. If the ZIP is too large, upload the listed files individually.
$LOG_FILE
$ARR_TXT
$BASELINE_CLEAN
$MAIN_CLEAN
$STRICT_CLEAN
$CLEAN_SCORES
$ATTACK_SCORES
$ABLATION_NPY
$ABLATION_CSV
$BASELINE_MIXED
$MAIN_MIXED
$STRICT_MIXED
$MAIN_MIXED_DIR/mixed_timeline_compare_summary.txt
$STRICT_MIXED_DIR/mixed_timeline_compare_summary.txt
EOF2

export UPLOAD_CHECKLIST ZIP_FILE
python - <<'PY'
import os
import zipfile

manifest = os.environ['UPLOAD_CHECKLIST']
zip_path = os.environ['ZIP_FILE']

with open(manifest, 'r', encoding='utf-8') as f:
    paths = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in paths:
        if os.path.exists(path):
            zf.write(path, arcname=path)

print(f"Created zip: {zip_path}")
PY

echo "zip_file: $ZIP_FILE" | tee -a "$LOG_FILE"
echo "upload_checklist: $UPLOAD_CHECKLIST" | tee -a "$LOG_FILE"
section "CASE14 paper batch finished"

echo
echo "Done. Upload this file to me first:"
echo "  $ZIP_FILE"
echo
echo "If the zip is too large, upload instead:"
echo "  $UPLOAD_CHECKLIST"

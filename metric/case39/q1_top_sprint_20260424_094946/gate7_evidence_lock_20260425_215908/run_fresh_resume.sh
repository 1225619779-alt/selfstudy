#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD-q1-case39

OUT=/home/pang/projects/DDET-MTD-q1-case39/metric/case39/q1_top_sprint_20260424_094946/gate7_evidence_lock_20260425_215908
PY=/home/pang/projects/DDET-MTD/.venv_rocm/bin/python
SCRIPT="$OUT/resume_checkpointed_evaluation_mixed_timeline.py"
SCHEDULE='att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120'
RESUME="$OUT/partials/gate6c_seed20260711_step218_input.partial.npy"

export DDET_CASE_NAME=case39
export PYTHONPATH=/home/pang/projects/DDET-MTD-q1-case39:${PYTHONPATH:-}

timeout --signal=TERM 54000s "$PY" "$SCRIPT" \
  --tau_verify -1 \
  --schedule "$SCHEDULE" \
  --seed_base 20260711 \
  --start_offset 1500 \
  --output "$OUT/fresh_banks/mixed_bank_test_fresh_fullsolver_540_seed20260711_off1500.npy" \
  --partial_output "$OUT/partials/mixed_bank_test_fresh_fullsolver_540_seed20260711_off1500.resume.partial.npy" \
  --runtime_jsonl "$OUT/logs/runtime_resume_steps.jsonl" \
  --checkpoint_every 5 \
  --max_wall_seconds 50400 \
  --resume_from_partial "$RESUME" \
  > "$OUT/logs/fresh_fullsolver_resume.log" 2>&1

rc=$?
echo "$rc" > "$OUT/fresh_fullsolver_resume_exit_code.txt"
exit "$rc"

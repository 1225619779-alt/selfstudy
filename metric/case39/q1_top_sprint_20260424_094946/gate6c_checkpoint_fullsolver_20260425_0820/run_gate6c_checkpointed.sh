#!/usr/bin/env bash
set -euo pipefail

cd /home/pang/projects/DDET-MTD-q1-case39

OUT=/home/pang/projects/DDET-MTD-q1-case39/metric/case39/q1_top_sprint_20260424_094946/gate6c_checkpoint_fullsolver_20260425_0820
PY=/home/pang/projects/DDET-MTD/.venv_rocm/bin/python
SCRIPT="$OUT/checkpointed_evaluation_mixed_timeline.py"
SCHEDULE='att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120'

export DDET_CASE_NAME=case39
export PYTHONPATH=/home/pang/projects/DDET-MTD-q1-case39:${PYTHONPATH:-}

timeout --signal=TERM 45000s "$PY" "$SCRIPT" \
  --tau_verify -1 \
  --schedule "$SCHEDULE" \
  --seed_base 20260711 \
  --start_offset 1500 \
  --output "$OUT/fresh_banks/mixed_bank_test_fresh_checkpointed_540_seed20260711_off1500.npy" \
  --partial_output "$OUT/partials/mixed_bank_test_fresh_checkpointed_540_seed20260711_off1500.partial.npy" \
  --runtime_jsonl "$OUT/logs/runtime_steps.jsonl" \
  --checkpoint_every 1 \
  --max_wall_seconds 43200 \
  > "$OUT/logs/checkpointed_fullsolver.log" 2>&1

rc=$?
echo "$rc" > "$OUT/gate6c_exit_code.txt"
exit "$rc"

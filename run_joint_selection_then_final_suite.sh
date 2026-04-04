#!/usr/bin/env bash
set -euo pipefail

# Full automatic pipeline:
# 1) validation-only joint tau selection
# 2) parse tau_main / tau_strict
# 3) run clean / attack / mixed / compare sequentially with logs
#
# Example:
#   bash run_joint_selection_then_final_suite.sh
#   bash run_joint_selection_then_final_suite.sh \
#       --total_run_attack 50 \
#       --main_overall_arr_min 0.90 --main_protected_arr_min 0.95 \
#       --strict_overall_arr_min 0.85 --strict_protected_arr_min 0.90

mkdir -p logs

python select_tau_joint_valid.py "$@" 2>&1 | tee logs/joint_tau_selection.log

SUMMARY="metric/case14/tau_selection_joint_valid/tau_selection_summary.txt"
if [[ ! -f "$SUMMARY" ]]; then
  echo "ERROR: summary file not found: $SUMMARY"
  exit 1
fi

TAU_MAIN=$(grep '^tau_main=' "$SUMMARY" | head -n 1 | cut -d'=' -f2)
TAU_STRICT=$(grep '^tau_strict=' "$SUMMARY" | head -n 1 | cut -d'=' -f2)

if [[ -z "$TAU_MAIN" || -z "$TAU_STRICT" ]]; then
  echo "ERROR: failed to parse tau_main / tau_strict from $SUMMARY"
  exit 1
fi

echo "Parsed tau_main=$TAU_MAIN"
echo "Parsed tau_strict=$TAU_STRICT"

bash run_selected_tau_suite.sh "$TAU_MAIN" "$TAU_STRICT"

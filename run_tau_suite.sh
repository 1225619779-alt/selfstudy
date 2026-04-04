#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_tau_suite.sh 0.03922229775000409 0.07000326254419229
#
# This script runs clean / attack / mixed sequentially with explicit logs
# and explicit mixed-timeline output names so files are not overwritten.

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_tau_suite.sh <TAU_MAIN> <TAU_STRICT>"
  exit 1
fi

TAU_MAIN="$1"
TAU_STRICT="$2"

mkdir -p logs metric/case14/plots_mixed_timeline_compare_tau_main metric/case14/plots_mixed_timeline_compare_tau_strict

echo "[1/9] CLEAN baseline"
python evaluation_event_trigger_clean.py --tau_verify -1.0 --max_total_run -1 --stop_ddd_alarm_at -1 2>&1 | tee logs/clean_tau_-1.0.log

echo "[2/9] CLEAN main"
python evaluation_event_trigger_clean.py --tau_verify "${TAU_MAIN}" --max_total_run -1 --stop_ddd_alarm_at -1 2>&1 | tee "logs/clean_tau_${TAU_MAIN}.log"

echo "[3/9] CLEAN strict"
python evaluation_event_trigger_clean.py --tau_verify "${TAU_STRICT}" --max_total_run -1 --stop_ddd_alarm_at -1 2>&1 | tee "logs/clean_tau_${TAU_STRICT}.log"

echo "[4/9] ATTACK main"
python evaluation_event_trigger_attack_cli.py --tau_verify "${TAU_MAIN}" --total_run 50 2>&1 | tee "logs/attack_tau_${TAU_MAIN}.log"

echo "[5/9] ATTACK strict"
python evaluation_event_trigger_attack_cli.py --tau_verify "${TAU_STRICT}" --total_run 50 2>&1 | tee "logs/attack_tau_${TAU_STRICT}.log"

echo "[6/9] MIXED baseline"
python evaluation_mixed_timeline.py --tau_verify -1.0 --output metric/case14/metric_mixed_timeline_tau_-1.0.npy 2>&1 | tee logs/mixed_tau_-1.0.log

echo "[7/9] MIXED main"
python evaluation_mixed_timeline.py --tau_verify "${TAU_MAIN}" --output metric/case14/metric_mixed_timeline_tau_main.npy 2>&1 | tee "logs/mixed_tau_${TAU_MAIN}.log"

echo "[8/9] MIXED strict"
python evaluation_mixed_timeline.py --tau_verify "${TAU_STRICT}" --output metric/case14/metric_mixed_timeline_tau_strict.npy 2>&1 | tee "logs/mixed_tau_${TAU_STRICT}.log"

echo "[9/9] MIXED compare"
python compare_mixed_timeline.py \
  --baseline metric/case14/metric_mixed_timeline_tau_-1.0.npy \
  --gated metric/case14/metric_mixed_timeline_tau_main.npy \
  --output_dir metric/case14/plots_mixed_timeline_compare_tau_main 2>&1 | tee logs/mixed_compare_main.log

python compare_mixed_timeline.py \
  --baseline metric/case14/metric_mixed_timeline_tau_-1.0.npy \
  --gated metric/case14/metric_mixed_timeline_tau_strict.npy \
  --output_dir metric/case14/plots_mixed_timeline_compare_tau_strict 2>&1 | tee logs/mixed_compare_strict.log

echo "Done. Useful files:"
echo "  logs/*.log"
echo "  metric/case14/*summary.txt"
echo "  metric/case14/plots_mixed_timeline_compare_tau_main/mixed_timeline_compare_summary.txt"
echo "  metric/case14/plots_mixed_timeline_compare_tau_strict/mixed_timeline_compare_summary.txt"

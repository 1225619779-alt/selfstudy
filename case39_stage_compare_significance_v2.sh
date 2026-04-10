#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
python case39_stage_compare_significance_v2.py \
  --transfer_summary metric/case39/postrun_audits/20260409_231456/summary.json \
  --transfer_v1 metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json \
  --transfer_v2 metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json \
  --local_protected_summary metric/case39_localretune_protectedec/postrun_bundle/summary.json \
  --local_protected_v1 metric/case39_localretune_protectedec/phase3_oracle_confirm_v1/aggregate_summary.json \
  --local_protected_v2 metric/case39_localretune_protectedec/phase3_oracle_confirm_v2/aggregate_summary.json \
  --local_unconstrained_summary metric/case39_localretune/postrun_bundle/summary.json \
  --local_unconstrained_v1 metric/case39_localretune/phase3_oracle_confirm_v1/aggregate_summary.json \
  --local_unconstrained_v2 metric/case39_localretune/phase3_oracle_confirm_v2/aggregate_summary.json \
  --output metric/case39_compare/stage_compare_significance_v2.json

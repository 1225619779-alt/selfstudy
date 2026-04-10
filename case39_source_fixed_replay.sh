#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
CASE14_SUMMARY="${2:-metric/case14/phase3_oracle_family/screen_train_val_summary.json}"
TEMPLATE_V1="${3:-metric/case39_source_anchor/phase3_oracle_confirm_v1/aggregate_summary.json}"
TEMPLATE_V2="${4:-metric/case39_source_anchor/phase3_oracle_confirm_v2/aggregate_summary.json}"
OUTPUT_ROOT="${5:-metric/case39_source_fixed_replay}"

cd "$REPO_ROOT"
python case39_source_fixed_replay.py \
  --repo_root . \
  --case14_summary "$CASE14_SUMMARY" \
  --template_v1_agg "$TEMPLATE_V1" \
  --template_v2_agg "$TEMPLATE_V2" \
  --output_root "$OUTPUT_ROOT"

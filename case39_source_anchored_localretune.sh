#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
SOURCE_SCREEN="${2:-metric/case14/phase3_oracle_family/screen_train_val_summary.json}"
LOCAL_PROTECTED_SCREEN="${3:-metric/case39_localretune_protectedec/oracle_family/screen_train_val_summary_forced_oracle_protected_ec.json}"
LOCAL_PROTECTED_V1="${4:-metric/case39_localretune_protectedec/phase3_oracle_confirm_v1/aggregate_summary.json}"
LOCAL_PROTECTED_V2="${5:-metric/case39_localretune_protectedec/phase3_oracle_confirm_v2/aggregate_summary.json}"

cd "$REPO_ROOT"

OUTROOT="metric/case39_source_anchor"
mkdir -p "$OUTROOT/oracle_family"

if [[ ! -f "$SOURCE_SCREEN" ]]; then
  echo "ERROR: source screen summary not found: $SOURCE_SCREEN" >&2
  exit 1
fi
if [[ ! -f "$LOCAL_PROTECTED_SCREEN" ]]; then
  echo "ERROR: local protected screen summary not found: $LOCAL_PROTECTED_SCREEN" >&2
  exit 1
fi
if [[ ! -f "$LOCAL_PROTECTED_V1" || ! -f "$LOCAL_PROTECTED_V2" ]]; then
  echo "ERROR: local protected aggregate summaries not found." >&2
  echo "Expected: $LOCAL_PROTECTED_V1 and $LOCAL_PROTECTED_V2" >&2
  exit 1
fi

echo "[1/5] build source-anchored local-retune screen summary"
python case39_source_anchored_retune.py \
  --source_screen "$SOURCE_SCREEN" \
  --local_screen "$LOCAL_PROTECTED_SCREEN" \
  --output "$OUTROOT/oracle_family/screen_train_val_summary_source_anchored.json"

echo "[2/5] reconstruct manifests from local-protected aggregate summaries"
python - <<'PY'
import json
from pathlib import Path
pairs = [
    ("metric/case39_localretune_protectedec/phase3_oracle_confirm_v1/aggregate_summary.json", "metric/case39_source_anchor/manifest_v1_from_local_protected.json"),
    ("metric/case39_localretune_protectedec/phase3_oracle_confirm_v2/aggregate_summary.json", "metric/case39_source_anchor/manifest_v2_from_local_protected.json"),
]
for src, dst in pairs:
    data = json.loads(Path(src).read_text(encoding='utf-8'))
    manifest = data['confirm_manifest']
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"wrote {out}")
PY

echo "[3/5] rerun oracle confirm on local-native holdouts with source-anchored protected-EC"
python run_phase3_oracle_confirm.py \
  --manifest "$OUTROOT/manifest_v1_from_local_protected.json" \
  --dev_screen_summary "$OUTROOT/oracle_family/screen_train_val_summary_source_anchored.json" \
  --output "$OUTROOT/phase3_oracle_confirm_v1"

python run_phase3_oracle_confirm.py \
  --manifest "$OUTROOT/manifest_v2_from_local_protected.json" \
  --dev_screen_summary "$OUTROOT/oracle_family/screen_train_val_summary_source_anchored.json" \
  --output "$OUTROOT/phase3_oracle_confirm_v2"

echo "[4/5] build merged summary + fallback hash audit"
python case39_protocol_bundle.py \
  --v1 "$OUTROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
  --v2 "$OUTROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
  --asset_protocol metric/case39/asset_protocol.json \
  --out_dir "$OUTROOT/postrun_bundle" \
  --stage case39_source_anchored_localretune

echo "[5/5] compact stage comparison"
python case39_stage_compare_bundle.py \
  --frozen_dev_summary metric/case39/postrun_audits/20260409_231456/summary.json \
  --localretune_help_summary metric/case39_localretune/postrun_bundle/summary.json \
  --localretune_protectedec_summary metric/case39_localretune_protectedec/postrun_bundle/summary.json \
  --output_json "$OUTROOT/postrun_bundle/stage_compare_against_existing.json" \
  --output_md "$OUTROOT/postrun_bundle/stage_compare_against_existing.md"

echo
ls -lh \
  "$OUTROOT/oracle_family/screen_train_val_summary_source_anchored.json" \
  "$OUTROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
  "$OUTROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
  "$OUTROOT/postrun_bundle/summary.json" \
  "$OUTROOT/postrun_bundle/fallback_case14_hash_audit.json"

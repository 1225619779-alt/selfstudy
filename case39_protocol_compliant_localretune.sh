#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

LOCAL_SCREEN="metric/case39_localretune/oracle_family/screen_train_val_summary.json"
LOCAL_AGG_V1="metric/case39_localretune/phase3_oracle_confirm_v1/aggregate_summary.json"
LOCAL_AGG_V2="metric/case39_localretune/phase3_oracle_confirm_v2/aggregate_summary.json"
OUT_ROOT="metric/case39_localretune_protectedec"
FORCED_VARIANT="oracle_protected_ec"

if [[ ! -f "$LOCAL_SCREEN" ]]; then
  echo "ERROR: missing $LOCAL_SCREEN" >&2
  exit 1
fi
if [[ ! -f "$LOCAL_AGG_V1" || ! -f "$LOCAL_AGG_V2" ]]; then
  echo "ERROR: missing local-retune aggregate summaries under metric/case39_localretune" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/oracle_family" "$OUT_ROOT/phase3_confirm_blind_v1" "$OUT_ROOT/phase3_confirm_blind_v2"

python case39_force_variant_from_screen.py \
  --input "$LOCAL_SCREEN" \
  --variant "$FORCED_VARIANT" \
  --output "$OUT_ROOT/oracle_family/screen_train_val_summary_forced_${FORCED_VARIANT}.json"

python - <<'PY'
import json
from pathlib import Path
pairs = [
    (Path("metric/case39_localretune/phase3_oracle_confirm_v1/aggregate_summary.json"), Path("metric/case39_localretune_protectedec/phase3_confirm_blind_v1/manifest.json")),
    (Path("metric/case39_localretune/phase3_oracle_confirm_v2/aggregate_summary.json"), Path("metric/case39_localretune_protectedec/phase3_confirm_blind_v2/manifest.json")),
]
for src, out in pairs:
    data = json.loads(src.read_text(encoding="utf-8"))
    manifest = data.get("confirm_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"confirm_manifest missing in {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
PY

python run_phase3_oracle_confirm.py \
  --manifest "$OUT_ROOT/phase3_confirm_blind_v1/manifest.json" \
  --dev_screen_summary "$OUT_ROOT/oracle_family/screen_train_val_summary_forced_${FORCED_VARIANT}.json" \
  --output "$OUT_ROOT/phase3_oracle_confirm_v1"

python run_phase3_oracle_confirm.py \
  --manifest "$OUT_ROOT/phase3_confirm_blind_v2/manifest.json" \
  --dev_screen_summary "$OUT_ROOT/oracle_family/screen_train_val_summary_forced_${FORCED_VARIANT}.json" \
  --output "$OUT_ROOT/phase3_oracle_confirm_v2"

python case39_protocol_bundle.py \
  --v1 "$OUT_ROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
  --v2 "$OUT_ROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
  --asset_protocol metric/case39/asset_protocol.json \
  --out_dir "$OUT_ROOT/postrun_bundle" \
  --stage case39_localretune_protocol_compliant_oracle_protected_ec

printf '\nDONE. Key outputs:\n'
ls -lh \
  "$OUT_ROOT/oracle_family/screen_train_val_summary_forced_${FORCED_VARIANT}.json" \
  "$OUT_ROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
  "$OUT_ROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
  "$OUT_ROOT/postrun_bundle/summary.json" \
  "$OUT_ROOT/postrun_bundle/fallback_case14_hash_audit.json"

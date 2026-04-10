#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
OUT_DIR="${2:-}"

cd "$REPO_ROOT"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="metric/case39/postrun_audits/$(date +%Y%m%d_%H%M%S)_finalize"
fi
mkdir -p "$OUT_DIR"

python - <<'PY' "$OUT_DIR"
import hashlib, json, os, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
asset_protocol = Path('metric/case39/asset_protocol.json')
summary_json = Path('metric/case39/postrun_audits')

if not asset_protocol.exists():
    raise SystemExit('Missing metric/case39/asset_protocol.json')

with open(asset_protocol, 'r', encoding='utf-8') as f:
    ap = json.load(f)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

checks = []
all_ok = True
for group_name in ['assets', 'holdout_test_banks']:
    group = ap.get(group_name, {})
    for key, rec in group.items():
        src = Path(rec['source_path'])
        expected = rec.get('sha256')
        exists = src.exists()
        current = sha256_file(src) if exists else None
        ok = exists and (expected == current)
        if not ok:
            all_ok = False
        checks.append({
            'group': group_name,
            'key': key,
            'source_path': str(src),
            'exists': exists,
            'expected_sha256': expected,
            'current_sha256': current,
            'hash_match': ok,
        })

payload = {
    'method': 'case39_fallback_case14_hash_audit',
    'asset_protocol_path': str(asset_protocol.resolve()),
    'all_hashes_match_asset_protocol': all_ok,
    'n_checks': len(checks),
    'checks': checks,
}

with open(out_dir / 'fallback_case14_hash_audit.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

with open(out_dir / 'fallback_case14_hash_audit.txt', 'w', encoding='utf-8') as f:
    f.write(f"all_hashes_match_asset_protocol={all_ok}\n")
    f.write(f"n_checks={len(checks)}\n")
    for row in checks:
        f.write(f"[{row['group']}] {row['key']} hash_match={row['hash_match']} source={row['source_path']}\n")

# find most recent postrun summary if present
postrun_root = Path('metric/case39/postrun_audits')
summary_candidates = sorted(postrun_root.glob('*/summary.json'), key=lambda p: p.stat().st_mtime, reverse=True)
brief = {
    'stage': None,
    'merged_8_holdouts': None,
    'used_summary_path': None,
}
if summary_candidates:
    p = summary_candidates[0]
    try:
        d = json.load(open(p, 'r', encoding='utf-8'))
        brief['stage'] = d.get('native_case39_stage') or d.get('stage')
        brief['merged_8_holdouts'] = d.get('merged_8_holdouts')
        brief['used_summary_path'] = str(p.resolve())
    except Exception:
        pass

with open(out_dir / 'current_stage_brief.json', 'w', encoding='utf-8') as f:
    json.dump(brief, f, ensure_ascii=False, indent=2)

with open(out_dir / 'current_stage_brief.txt', 'w', encoding='utf-8') as f:
    f.write(f"stage={brief['stage']}\n")
    f.write(f"used_summary_path={brief['used_summary_path']}\n")
    if isinstance(brief['merged_8_holdouts'], dict):
        for slot, slot_payload in brief['merged_8_holdouts'].items():
            f.write(f"[{slot}]\n")
            for method, vals in slot_payload.items():
                f.write(
                    f"  {method}: recall={vals['mean_recall']:.6f}, unnecessary={vals['mean_unnecessary']:.3f}, cost={vals['mean_cost']:.6f}, served_ratio={vals['mean_served_ratio']:.6f}\n"
                )
PY

echo "finalize_out_dir=$OUT_DIR"
ls -l "$OUT_DIR"/fallback_case14_hash_audit.* "$OUT_DIR"/current_stage_brief.*

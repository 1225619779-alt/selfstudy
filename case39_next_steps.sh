#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

mkdir -p metric/case39/preflight

python phase3_case39_capability_audit.py \
  --repo_root . \
  --output metric/case39/preflight/case39_capability_audit_rerun.json

python case39_native_readiness_audit.py \
  --repo_root . \
  --output metric/case39/preflight/case39_native_readiness.json

python - <<'PY'
import json
from pathlib import Path
paths = [
    Path('metric/case39/preflight/case39_capability_audit_rerun.json'),
    Path('metric/case39/preflight/case39_native_readiness.json'),
]
for p in paths:
    print(f"\n=== {p} ===")
    d = json.loads(p.read_text(encoding='utf-8'))
    for k in ['status', 'repo_root', 'next_step']:
        if k in d:
            print(f"{k}: {d[k]}")
    if 'checkpoint' in d:
        print('checkpoint_exists:', d['checkpoint'].get('exists'))
        print('expected_checkpoint:', d['checkpoint'].get('expected_path'))
        print('checkpoint_hits_under_repo:', d['checkpoint'].get('checkpoint_rnn_hits_under_repo'))
PY

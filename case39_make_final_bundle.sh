#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"
python case39_stage_compare_bundle.py --repo_root . --output metric/case39_compare/case39_stage_compare_bundle.json
python case39_make_final_bundle.py --repo_root . --bundle_json metric/case39_compare/case39_stage_compare_bundle.json --output_dir metric/case39_final_bundle

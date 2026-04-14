#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"
python case39_make_final_bundle_v2.py .

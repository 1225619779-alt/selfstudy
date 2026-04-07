#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
cd "$repo_root"

echo '== hardcoded case14 hits =='
git grep -nE 'metric/case14|/case14|\bcase14\b' || true

echo
echo '== likely runner/manifest/bank hotspots =='
git grep -nE 'manifest|bank|runner|case root|repo_root|metric/' -- '*.py' '*.sh' '*.json' '*.yaml' '*.yml' || true

echo
echo '== python entrypoints with argparse =='
git grep -nE 'ArgumentParser|add_argument\(' -- '*.py' || true

echo
echo '== shell entrypoints =='
find . -type f -name '*.sh' -maxdepth 5 -print | sort

#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/pang/projects}"
OUT_DIR="${2:-./metric/case39/preflight}"
mkdir -p "$OUT_DIR"
{
  echo "# root=$ROOT"
  echo "## case39-named files"
  find "$ROOT" -type f | grep -i 'case39' || true
  echo
  echo "## checkpoint_rnn.pt hits"
  find "$ROOT" -type f -name 'checkpoint_rnn.pt' || true
  echo
  echo "## likely raw data hits (.xlsx .mat .csv .npy .pt .pth .pkl) containing case39 in path"
  find "$ROOT" -type f \( -name '*.xlsx' -o -name '*.mat' -o -name '*.csv' -o -name '*.npy' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \) | grep -i 'case39' || true
} | tee "$OUT_DIR/case39_external_asset_scan.txt"

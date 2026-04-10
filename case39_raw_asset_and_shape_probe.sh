#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
SEARCH_ROOT="${2:-/home/pang}"
OUT_DIR="${3:-metric/case39/preflight/native_raw_probe}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

echo "[1/4] probing pypower case39 structural facts"
python - <<'PY' > "$OUT_DIR/case39_shape_probe.txt"
from pypower.api import case39, ext2int, bustypes
from configs.config_mea_idx import define_mea_idx_noise

case = case39()
case_int = ext2int(case)
ref_idx, pv_idx, pq_idx = bustypes(case_int['bus'], case_int['gen'])
idx, no_mea, noise_sigma = define_mea_idx_noise(case, choice='HALF_RTU')

print('n_bus =', len(case['bus']))
print('n_branch =', len(case['branch']))
print('ref_idx_0based =', ref_idx.tolist())
print('pv_bus_0based =', pv_idx.tolist())
print('pq_bus_count =', len(pq_idx))
print('feature_size_half_rtu =', int(no_mea))
print('noise_sigma_len =', int(len(noise_sigma)))
print('expected_saved_model_path = saved_model/case39/checkpoint_rnn.pt')
print('expected_gen_data_root = gen_data/case39')
PY

echo "[2/4] searching raw load/PV CSV assets"
find "$SEARCH_ROOT" -type f \( \
  -iname 'load.csv' -o \
  -iname 'pv.csv' -o \
  -iname 'load_high.csv' -o \
  -iname 'pv_high.csv' \
\) | sort > "$OUT_DIR/raw_csv_hits.txt" || true

echo "[3/4] searching notebooks/scripts that mention those raw assets"
grep -RIn "load_high.csv\|pv_high.csv\|load.csv\|pv.csv\|DateTime" \
  . /home/pang/projects/DDET-MTD /home/pang/projects/DDET-MTD-q1-case39 \
  > "$OUT_DIR/raw_asset_code_hits.txt" || true

echo "[4/4] checking current repo raw_data directory"
{
  echo '=== gen_data/raw_data tree ==='
  find gen_data/raw_data -maxdepth 3 -type f | sort || true
  echo
  echo '=== saved_model tree ==='
  find saved_model -maxdepth 3 -type f | sort || true
} > "$OUT_DIR/local_raw_tree.txt"

echo "native raw probe written to: $OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

OUT="metric/case39/preflight/native_path_probe"
mkdir -p "$OUT"

echo "[1/8] listing key trees"
{
  echo "=== saved_model tree ==="
  find saved_model -maxdepth 3 -type f | sort || true
  echo
  echo "=== gen_data tree ==="
  find gen_data -maxdepth 4 -type f | sort || true
  echo
  echo "=== metric/case39 tree ==="
  find metric/case39 -maxdepth 4 -type f | sort || true
  echo
  echo "=== old repo case39 helper hits ==="
  ls -l /home/pang/projects/DDET-MTD/bootstrap_q1_case39_workspace.sh 2>/dev/null || true
  ls -l /home/pang/projects/DDET-MTD/phase3_case39_preflight.py 2>/dev/null || true
} > "$OUT/tree_listing.txt"

echo "[2/8] CLI help probes"
python run.py -h > "$OUT/run_help.txt" 2>&1 || true
python gen_data/gen_data.py -h > "$OUT/gen_data_help.txt" 2>&1 || true
python collect_clean_alarm_scores.py -h > "$OUT/collect_clean_help.txt" 2>&1 || true
python collect_attack_alarm_scores.py -h > "$OUT/collect_attack_help.txt" 2>&1 || true
python evaluation_mixed_timeline.py -h > "$OUT/evaluation_mixed_timeline_help.txt" 2>&1 || true
python /home/pang/projects/DDET-MTD/phase3_case39_preflight.py -h > "$OUT/oldrepo_case39_preflight_help.txt" 2>&1 || true

echo "[3/8] grep key symbols"
grep -RIn \
  "case14\|case39\|checkpoint_rnn\|saved_model\|raw_data\|torch.save\|torch.load\|add_argument\|ArgumentParser\|gen_data\|mixed_bank_fit\|mixed_bank_eval" \
  configs gen_data models utils \
  run.py collect_clean_alarm_scores.py collect_attack_alarm_scores.py evaluation_mixed_timeline.py \
  /home/pang/projects/DDET-MTD/bootstrap_q1_case39_workspace.sh \
  /home/pang/projects/DDET-MTD/phase3_case39_preflight.py \
  > "$OUT/key_grep.txt" || true

echo "[4/8] dump config.py"
sed -n '1,240p' configs/config.py > "$OUT/config.py.txt"

echo "[5/8] dump nn_setting.py"
sed -n '1,240p' configs/nn_setting.py > "$OUT/nn_setting.py.txt"

echo "[6/8] dump run.py"
sed -n '1,320p' run.py > "$OUT/run.py.txt"

echo "[7/8] dump gen_data/gen_data.py"
sed -n '1,320p' gen_data/gen_data.py > "$OUT/gen_data.py.txt"

echo "[8/8] dump load_data + old-repo helpers"
sed -n '1,320p' utils/load_data.py > "$OUT/load_data.py.txt"
sed -n '1,320p' /home/pang/projects/DDET-MTD/bootstrap_q1_case39_workspace.sh > "$OUT/oldrepo_bootstrap_q1_case39_workspace.sh.txt" 2>/dev/null || true
sed -n '1,320p' /home/pang/projects/DDET-MTD/phase3_case39_preflight.py > "$OUT/oldrepo_phase3_case39_preflight.py.txt" 2>/dev/null || true

printf 'native probe written to: %s\n' "$OUT"
SH
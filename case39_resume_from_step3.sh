#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
DEV_SUMMARY="${2:?Usage: bash case39_resume_from_step3.sh <repo_root> <case14_dev_summary_json>}"
cd "$REPO_ROOT"

STAMP="/tmp/case39_resume_from_step3_$(date +%s).stamp"
touch "$STAMP"
echo "STAMP=$STAMP"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"

python - <<'PY'
from pathlib import Path
p = Path('utils/load_data.py')
text = p.read_text(encoding='utf-8')
orig = text
text = text.replace("np.load('gen_data/case14/z_noise_summary.npy')", "np.load(f'gen_data/{sys_config[\"case_name\"]}/z_noise_summary.npy')")
text = text.replace("np.load('gen_data/case14/v_est_summary.npy')", "np.load(f'gen_data/{sys_config[\"case_name\"]}/v_est_summary.npy')")
text = text.replace("np.load('gen_data/case14/success_summary.npy')", "np.load(f'gen_data/{sys_config[\"case_name\"]}/success_summary.npy')")
if text != orig:
    p.write_text(text, encoding='utf-8')
    print('patched utils/load_data.py measurement paths -> case-aware')
else:
    print('utils/load_data.py already case-aware (or patch not needed)')
PY

mkdir -p saved_model/case39

echo "[1/7] train native case39 checkpoint"
env DDET_CASE_NAME=case39 python -m models.model
ls -lh saved_model/case39/checkpoint_rnn.pt

for p in \
  metric/case39/metric_clean_alarm_scores_full.npy \
  metric/case39/metric_attack_alarm_scores_400.npy
 do
  if [ -L "$p" ] || [ -f "$p" ]; then rm -f "$p"; fi
done

echo "[2/7] collect native clean bank"
env DDET_CASE_NAME=case39 python collect_clean_alarm_scores.py \
  --output metric/case39/metric_clean_alarm_scores_full.npy
ls -lh metric/case39/metric_clean_alarm_scores_full.npy

echo "[3/7] collect native attack bank (400)"
env DDET_CASE_NAME=case39 python collect_attack_alarm_scores.py \
  --total_run 400 \
  --output metric/case39/metric_attack_alarm_scores_400.npy
ls -lh metric/case39/metric_attack_alarm_scores_400.npy

echo "[4/7] regenerate native blind confirm test banks from manifests"
python - <<'PY'
from pathlib import Path
import json, os, subprocess, sys
repo = Path('.').resolve()
env = os.environ.copy()
env['DDET_CASE_NAME'] = 'case39'
env['PYTHONPATH'] = str(repo)
for manifest_rel in [Path('metric/case39/phase3_confirm_blind_v1/manifest.json'), Path('metric/case39/phase3_confirm_blind_v2/manifest.json')]:
    manifest_path = repo / manifest_rel
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for h in manifest['holdouts']:
        out = repo / h['test_bank']
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() or out.is_symlink():
            out.unlink()
        cmd = [sys.executable, 'evaluation_mixed_timeline.py',
               '--tau_verify', '-1',
               '--schedule', h['schedule'],
               '--seed_base', str(h['seed_base']),
               '--start_offset', str(h['start_offset']),
               '--output', str(out)]
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True, env=env)
PY

echo "[5/7] rerun frozen-regime holdout summaries for v1/v2"
python - <<'PY'
from pathlib import Path
import json, os, subprocess, sys
repo = Path('.').resolve()
env = os.environ.copy()
env['DDET_CASE_NAME'] = 'case39'
env['PYTHONPATH'] = str(repo)
for manifest_rel in [Path('metric/case39/phase3_confirm_blind_v1/manifest.json'), Path('metric/case39/phase3_confirm_blind_v2/manifest.json')]:
    manifest_path = repo / manifest_rel
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fr = manifest['frozen_regime']
    for h in manifest['holdouts']:
        out = repo / h['result_npy']
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        summary = Path(str(out).replace('.npy', '.summary.json'))
        if summary.exists():
            summary.unlink()
        cmd = [sys.executable, 'evaluation_budget_scheduler_phase3_holdout.py',
               '--clean_bank', manifest['clean_bank'],
               '--attack_bank', manifest['attack_bank'],
               '--train_bank', manifest['train_bank'],
               '--val_bank', manifest['val_bank'],
               '--test_bank', h['test_bank'],
               '--output', h['result_npy'],
               '--slot_budget_list', *[str(x) for x in fr['slot_budget_list']],
               '--max_wait_steps', str(fr['max_wait_steps']),
               '--decision_step_group', str(fr['decision_step_group']),
               '--busy_time_quantile', str(fr['busy_time_quantile']),
               '--cost_budget_window_steps', str(fr['cost_budget_window_steps']),
               '--cost_budget_quantile', str(fr['cost_budget_quantile'])]
        if fr.get('use_cost_budget', False):
            cmd.append('--use_cost_budget')
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True, env=env)
PY

echo "[6/7] rerun oracle confirm with frozen case14 dev summary"
env DDET_CASE_NAME=case39 python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --dev_screen_summary "$DEV_SUMMARY" \
  --output metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test

env DDET_CASE_NAME=case39 python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --dev_screen_summary "$DEV_SUMMARY" \
  --output metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test

echo "[7/7] anti-write audit"
find metric/case14 -type f -newer "$STAMP" -print
find /home/pang/projects/DDET-MTD/metric/case14 -type f -newer "$STAMP" -print

echo "DONE. Key outputs:"
ls -lh saved_model/case39/checkpoint_rnn.pt
ls -lh metric/case39/metric_clean_alarm_scores_full.npy
ls -lh metric/case39/metric_attack_alarm_scores_400.npy
ls -lh metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json
ls -lh metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json

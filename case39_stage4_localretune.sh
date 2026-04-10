#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
REFERENCE_SUMMARY="${2:-metric/case39/postrun_audits/20260409_231456/summary.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DDET_CASE_NAME=case39

STAMP="/tmp/case39_stage4_localretune_$(date +%s).stamp"
touch "$STAMP"

echo "STAMP=$STAMP"

LOCAL_ROOT="metric/case39_localretune"
mkdir -p "$LOCAL_ROOT" "$LOCAL_ROOT/oracle_family" "$LOCAL_ROOT/phase3_confirm_blind_v1/results" "$LOCAL_ROOT/phase3_confirm_blind_v2/results"

# build fit/eval schedules from existing blind manifests to stay distribution-aligned but avoid test leakage via new seeds/offsets
readarray -t SCHEDULES < <($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
v1 = json.load(open('metric/case39/phase3_confirm_blind_v1/manifest.json','r',encoding='utf-8'))
v2 = json.load(open('metric/case39/phase3_confirm_blind_v2/manifest.json','r',encoding='utf-8'))
fit_schedule = v1['confirm_families'][0]['schedule'] + ';' + v2['confirm_families'][0]['schedule']
val_schedule = v1['confirm_families'][1]['schedule'] + ';' + v2['confirm_families'][1]['schedule']
print(fit_schedule)
print(val_schedule)
PY
)
FIT_SCHEDULE="${SCHEDULES[0]}"
VAL_SCHEDULE="${SCHEDULES[1]}"

echo "[1/7] generate native local-retune fit/eval banks"
echo "FIT_SCHEDULE=$FIT_SCHEDULE"
echo "VAL_SCHEDULE=$VAL_SCHEDULE"
$PYTHON_BIN evaluation_mixed_timeline.py \
  --tau_verify -1 \
  --schedule "$FIT_SCHEDULE" \
  --seed_base 20260711 \
  --start_offset 1500 \
  --output "$LOCAL_ROOT/mixed_bank_fit_native.npy"

$PYTHON_BIN evaluation_mixed_timeline.py \
  --tau_verify -1 \
  --schedule "$VAL_SCHEDULE" \
  --seed_base 20260721 \
  --start_offset 3200 \
  --output "$LOCAL_ROOT/mixed_bank_eval_native.npy"

echo "[2/7] build local-retune manifests"
$PYTHON_BIN - <<'PY'
import json
from pathlib import Path
local_root = Path('metric/case39_localretune')
for src_path, tag in [
    ('metric/case39/phase3_confirm_blind_v1/manifest.json', 'v1'),
    ('metric/case39/phase3_confirm_blind_v2/manifest.json', 'v2'),
]:
    m = json.load(open(src_path,'r',encoding='utf-8'))
    m['train_bank'] = 'metric/case39_localretune/mixed_bank_fit_native.npy'
    m['val_bank'] = 'metric/case39_localretune/mixed_bank_eval_native.npy'
    for h in m['holdouts']:
        stem = Path(h['result_npy']).name.replace('.npy','')
        out_dir = local_root / f'phase3_confirm_blind_{tag}' / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)
        h['result_npy'] = str((out_dir / f'{stem}_localretune.npy').as_posix())
        h['result_summary'] = str((out_dir / f'{stem}_localretune.summary.json').as_posix())
    out_path = local_root / f'phase3_confirm_blind_{tag}' / 'manifest.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(m,f,ensure_ascii=False,indent=2)
# minimal screen manifest; holdouts intentionally empty because screen_only uses only clean/attack/train/val/frozen_regime
screen_manifest = {
    'workdir': str(Path('.').resolve()),
    'case_name': 'case39',
    'clean_bank': 'metric/case39/metric_clean_alarm_scores_full.npy',
    'attack_bank': 'metric/case39/metric_attack_alarm_scores_400.npy',
    'train_bank': 'metric/case39_localretune/mixed_bank_fit_native.npy',
    'val_bank': 'metric/case39_localretune/mixed_bank_eval_native.npy',
    'schedule': 'case39_localretune_screen_only',
    'confirm_families': [],
    'frozen_regime': {
        'decision_step_group': 1,
        'busy_time_quantile': 0.65,
        'use_cost_budget': False,
        'cost_budget_window_steps': 20,
        'cost_budget_quantile': 0.6,
        'slot_budget_list': [1,2],
        'max_wait_steps': 10,
    },
    'holdouts': [],
}
with open(local_root / 'screen_manifest.json','w',encoding='utf-8') as f:
    json.dump(screen_manifest,f,ensure_ascii=False,indent=2)
print(json.dumps({'screen_manifest': str((local_root / 'screen_manifest.json').resolve())}, ensure_ascii=False, indent=2))
PY

echo "[3/7] screen oracle family locally on native case39 fit/eval"
$PYTHON_BIN run_phase3_oracle_family_multi_holdout.py \
  --workdir . \
  --manifest "$LOCAL_ROOT/screen_manifest.json" \
  --output "$LOCAL_ROOT/oracle_family" \
  --screen-only

echo "[4/7] rerun v1 holdout summaries under native local-retune fit/eval"
$PYTHON_BIN case39_rerun_holdout_manifest.py \
  --workdir . \
  --python "$PYTHON_BIN" \
  --manifest "$LOCAL_ROOT/phase3_confirm_blind_v1/manifest.json" \
  --overwrite

echo "[5/7] rerun v2 holdout summaries under native local-retune fit/eval"
$PYTHON_BIN case39_rerun_holdout_manifest.py \
  --workdir . \
  --python "$PYTHON_BIN" \
  --manifest "$LOCAL_ROOT/phase3_confirm_blind_v2/manifest.json" \
  --overwrite

echo "[6/7] oracle confirm on local-retune screen summary"
$PYTHON_BIN run_phase3_oracle_confirm.py \
  --manifest "$LOCAL_ROOT/phase3_confirm_blind_v1/manifest.json" \
  --dev_screen_summary "$LOCAL_ROOT/oracle_family/screen_train_val_summary.json" \
  --output "$LOCAL_ROOT/phase3_oracle_confirm_v1"

$PYTHON_BIN run_phase3_oracle_confirm.py \
  --manifest "$LOCAL_ROOT/phase3_confirm_blind_v2/manifest.json" \
  --dev_screen_summary "$LOCAL_ROOT/oracle_family/screen_train_val_summary.json" \
  --output "$LOCAL_ROOT/phase3_oracle_confirm_v2"

echo "[7/7] significance bundle + fallback anti-write/hash audit"
$PYTHON_BIN case39_significance_bundle.py \
  --v1 "$LOCAL_ROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
  --v2 "$LOCAL_ROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
  --output_dir "$LOCAL_ROOT/postrun_bundle" \
  --label case39_fully_native_localretune \
  --reference_summary "$REFERENCE_SUMMARY" \
  --reference_label native_clean_attack_test_with_frozen_case14_dev

$PYTHON_BIN - <<'PY'
import hashlib, json
from pathlib import Path
proto = json.load(open('metric/case39/asset_protocol.json','r',encoding='utf-8'))
checks = []
for group_name in ['assets','holdout_test_banks']:
    group = proto.get(group_name, {})
    for key, meta in group.items():
        src = Path(meta['source_path'])
        cur = None
        ok = False
        if src.exists():
            h = hashlib.sha256()
            with open(src,'rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
            cur = h.hexdigest()
            ok = (cur == meta['sha256'])
        checks.append({'group': group_name,'key': key,'source_path': str(src),'exists': src.exists(),'expected_sha256': meta['sha256'],'current_sha256': cur,'hash_match': ok})
out = {
    'method': 'case39_localretune_fallback_case14_hash_audit',
    'all_hashes_match_asset_protocol': all(c['hash_match'] for c in checks if c['exists']),
    'n_checks': len(checks),
    'checks': checks,
}
out_dir = Path('metric/case39_localretune/postrun_bundle')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir/'fallback_case14_hash_audit.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
with open(out_dir/'fallback_case14_hash_audit.txt','w',encoding='utf-8') as f:
    f.write(f"all_hashes_match_asset_protocol={out['all_hashes_match_asset_protocol']}\n")
    f.write(f"n_checks={out['n_checks']}\n")
    for c in checks:
        f.write(f"[{c['group']}] {c['key']} hash_match={c['hash_match']} source={c['source_path']}\n")
print(json.dumps({'fallback_hash_audit_json': str((out_dir/'fallback_case14_hash_audit.json').resolve())}, ensure_ascii=False, indent=2))
PY

echo "DONE. Key outputs:"
ls -lh "$LOCAL_ROOT/oracle_family/screen_train_val_summary.json" \
       "$LOCAL_ROOT/phase3_oracle_confirm_v1/aggregate_summary.json" \
       "$LOCAL_ROOT/phase3_oracle_confirm_v2/aggregate_summary.json" \
       "$LOCAL_ROOT/postrun_bundle/summary.json" \
       "$LOCAL_ROOT/postrun_bundle/fallback_case14_hash_audit.json"

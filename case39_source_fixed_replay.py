from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def build_confirm_manifest(template_agg: Dict[str, Any]) -> Dict[str, Any]:
    manifest = template_agg.get('confirm_manifest')
    if not isinstance(manifest, dict):
        raise ValueError('aggregate summary does not contain confirm_manifest')
    return manifest


def run_confirm(repo_root: Path, manifest_path: Path, dev_screen_summary: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        'run_phase3_oracle_confirm.py',
        '--manifest', str(manifest_path),
        '--dev_screen_summary', str(dev_screen_summary),
        '--output', str(output_dir),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)


def _stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'raw_deltas': []}
    return {
        'mean': mean(vals),
        'std': pstdev(vals) if len(vals) > 1 else 0.0,
        'min': min(vals),
        'max': max(vals),
        'raw_deltas': vals,
    }


def merge_aggregates(v1: Dict[str, Any], v2: Dict[str, Any], stage_name: str, used_summary_path: str) -> Dict[str, Any]:
    holdouts = []
    for agg in (v1, v2):
        holdouts.extend(agg.get('per_holdout_results', []))

    slot_map: Dict[str, Dict[str, List[Dict[str, float]]]] = {'1': {}, '2': {}}
    methods = ['phase3_oracle_upgrade', 'phase3_proposed', 'topk_expected_consequence']
    for slot in ['1', '2']:
        for m in methods:
            slot_map[slot][m] = []

    paired: Dict[str, Any] = {'1': {}, '2': {}}
    for slot in ['1', '2']:
        paired[slot] = {
            'oracle_vs_phase3': {'recall': [], 'unnecessary': [], 'cost': [], 'served_ratio': [], 'delay_p95': []},
            'oracle_vs_topk_expected': {'recall': [], 'unnecessary': [], 'cost': [], 'served_ratio': [], 'delay_p95': []},
        }

    for h in holdouts:
        slot_budget_results = h.get('slot_budget_results', {})
        for slot in ['1', '2']:
            r = slot_budget_results.get(slot)
            if not isinstance(r, dict):
                continue
            for m in methods:
                if m in r:
                    slot_map[slot][m].append(r[m])
            if 'phase3_oracle_upgrade' in r and 'phase3_proposed' in r:
                a = r['phase3_oracle_upgrade']; b = r['phase3_proposed']
                paired[slot]['oracle_vs_phase3']['recall'].append(a['weighted_attack_recall_no_backend_fail'] - b['weighted_attack_recall_no_backend_fail'])
                paired[slot]['oracle_vs_phase3']['unnecessary'].append(a['unnecessary_mtd_count'] - b['unnecessary_mtd_count'])
                paired[slot]['oracle_vs_phase3']['cost'].append(a['average_service_cost_per_step'] - b['average_service_cost_per_step'])
                paired[slot]['oracle_vs_phase3']['served_ratio'].append(a['pred_expected_consequence_served_ratio'] - b['pred_expected_consequence_served_ratio'])
                paired[slot]['oracle_vs_phase3']['delay_p95'].append(a['queue_delay_p95'] - b['queue_delay_p95'])
            if 'phase3_oracle_upgrade' in r and 'topk_expected_consequence' in r:
                a = r['phase3_oracle_upgrade']; b = r['topk_expected_consequence']
                paired[slot]['oracle_vs_topk_expected']['recall'].append(a['weighted_attack_recall_no_backend_fail'] - b['weighted_attack_recall_no_backend_fail'])
                paired[slot]['oracle_vs_topk_expected']['unnecessary'].append(a['unnecessary_mtd_count'] - b['unnecessary_mtd_count'])
                paired[slot]['oracle_vs_topk_expected']['cost'].append(a['average_service_cost_per_step'] - b['average_service_cost_per_step'])
                paired[slot]['oracle_vs_topk_expected']['served_ratio'].append(a['pred_expected_consequence_served_ratio'] - b['pred_expected_consequence_served_ratio'])
                paired[slot]['oracle_vs_topk_expected']['delay_p95'].append(a['queue_delay_p95'] - b['queue_delay_p95'])

    merged: Dict[str, Any] = {}
    for slot in ['1', '2']:
        merged[slot] = {}
        for m in methods:
            rows = slot_map[slot][m]
            if not rows:
                continue
            merged[slot][m] = {
                'mean_recall': mean([x['weighted_attack_recall_no_backend_fail'] for x in rows]),
                'mean_unnecessary': mean([x['unnecessary_mtd_count'] for x in rows]),
                'mean_cost': mean([x['average_service_cost_per_step'] for x in rows]),
                'mean_served_ratio': mean([x['pred_expected_consequence_served_ratio'] for x in rows]),
            }
        for comp in paired[slot].values():
            for k, vals in list(comp.items()):
                comp[k] = _stats(vals)

    return {
        'stage': stage_name,
        'n_holdouts': len(holdouts),
        'merged_8_holdouts': merged,
        'paired': paired,
        'used_summary_path': used_summary_path,
    }


def fallback_hash_audit(asset_protocol_path: Path) -> Dict[str, Any]:
    ap = load_json(asset_protocol_path)
    checks = []
    for group_name in ('assets', 'holdout_test_banks'):
        group = ap.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for key, meta in group.items():
            src = Path(meta['source_path'])
            expected = meta.get('sha256')
            exists = src.exists()
            current = sha256_file(src) if exists else None
            checks.append({
                'group': group_name,
                'key': key,
                'source_path': str(src),
                'exists': exists,
                'expected_sha256': expected,
                'current_sha256': current,
                'hash_match': exists and expected == current,
            })
    return {
        'method': 'case39_source_fixed_fallback_case14_hash_audit',
        'all_hashes_match_asset_protocol': all(c['hash_match'] for c in checks),
        'n_checks': len(checks),
        'checks': checks,
    }


def audit_text(audit: Dict[str, Any]) -> str:
    lines = [
        f"all_hashes_match_asset_protocol={audit['all_hashes_match_asset_protocol']}",
        f"n_checks={audit['n_checks']}",
    ]
    for c in audit['checks']:
        lines.append(f"[{c['group']}] {c['key']} hash_match={c['hash_match']} source={c['source_path']}")
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description='Replay case14 frozen winner on native case39 test banks without local search.')
    p.add_argument('--repo_root', default='.')
    p.add_argument('--case14_summary', default='metric/case14/phase3_oracle_family/screen_train_val_summary.json')
    p.add_argument('--template_v1_agg', default='metric/case39_source_anchor/phase3_oracle_confirm_v1/aggregate_summary.json')
    p.add_argument('--template_v2_agg', default='metric/case39_source_anchor/phase3_oracle_confirm_v2/aggregate_summary.json')
    p.add_argument('--output_root', default='metric/case39_source_fixed_replay')
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    case14_summary = (repo_root / args.case14_summary).resolve() if not Path(args.case14_summary).is_absolute() else Path(args.case14_summary)
    v1_agg_path = (repo_root / args.template_v1_agg).resolve() if not Path(args.template_v1_agg).is_absolute() else Path(args.template_v1_agg)
    v2_agg_path = (repo_root / args.template_v2_agg).resolve() if not Path(args.template_v2_agg).is_absolute() else Path(args.template_v2_agg)
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    v1_template = load_json(v1_agg_path)
    v2_template = load_json(v2_agg_path)

    v1_manifest = build_confirm_manifest(v1_template)
    v2_manifest = build_confirm_manifest(v2_template)
    v1_manifest_path = output_root / 'confirm_manifest_v1.json'
    v2_manifest_path = output_root / 'confirm_manifest_v2.json'
    save_json(v1_manifest_path, v1_manifest)
    save_json(v2_manifest_path, v2_manifest)

    out_v1 = output_root / 'phase3_oracle_confirm_v1'
    out_v2 = output_root / 'phase3_oracle_confirm_v2'

    run_confirm(repo_root, v1_manifest_path, case14_summary, out_v1)
    run_confirm(repo_root, v2_manifest_path, case14_summary, out_v2)

    agg_v1 = load_json(out_v1 / 'aggregate_summary.json')
    agg_v2 = load_json(out_v2 / 'aggregate_summary.json')

    stage_name = 'case39_source_fixed_case14winner_native_test'
    summary = merge_aggregates(agg_v1, agg_v2, stage_name, str(case14_summary))

    postrun = output_root / 'postrun_bundle'
    postrun.mkdir(parents=True, exist_ok=True)
    save_json(postrun / 'summary.json', summary)
    write_text(
        postrun / 'summary.txt',
        '\n'.join([
            f"stage={summary['stage']}",
            f"used_summary_path={summary['used_summary_path']}",
            *[
                f"[{slot}]\n  " + '\n  '.join(
                    f"{m}: recall={vals['mean_recall']:.6f}, unnecessary={vals['mean_unnecessary']:.3f}, cost={vals['mean_cost']:.6f}, served_ratio={vals['mean_served_ratio']:.6f}"
                    for m, vals in summary['merged_8_holdouts'][slot].items()
                )
                for slot in ['1', '2']
            ],
        ]) + '\n'
    )

    asset_protocol_path = repo_root / 'metric/case39/asset_protocol.json'
    audit = fallback_hash_audit(asset_protocol_path)
    save_json(postrun / 'fallback_case14_hash_audit.json', audit)
    write_text(postrun / 'fallback_case14_hash_audit.txt', audit_text(audit))

    print(json.dumps({
        'status': 'OK',
        'output_root': str(output_root),
        'summary_json': str(postrun / 'summary.json'),
        'audit_json': str(postrun / 'fallback_case14_hash_audit.json'),
    }, indent=2))


if __name__ == '__main__':
    main()

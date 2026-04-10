#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def safe_load_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = load_json(path)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def find_stage_summary(repo_root: Path, target_stage: str) -> Optional[Path]:
    candidates = []
    for p in repo_root.glob('metric/**/summary.json'):
        obj = safe_load_dict(p)
        if not obj:
            continue
        stage = obj.get('stage') or obj.get('label')
        if stage == target_stage:
            candidates.append(p)
    # prefer shortest path depth then newest mtime
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), -p.stat().st_mtime))
    return candidates[0]


def find_hash_audit(repo_root: Path, target_method_substr: str) -> Optional[Path]:
    candidates = []
    for p in repo_root.glob('metric/**/*.json'):
        obj = safe_load_dict(p)
        if not obj:
            continue
        method = obj.get('method', '')
        if target_method_substr in method and 'hash_audit' in method:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: -p.stat().st_mtime)
    return candidates[0]


def stage_metric(summary: Dict[str, Any], slot: str, method: str, metric: str) -> Optional[float]:
    try:
        return summary['merged_8_holdouts'][slot][method][metric]
    except Exception:
        return None


def build_claim(bundle: Dict[str, Any]) -> str:
    transfer = bundle['stages'].get('transfer_frozen_dev', {})
    source_fixed = bundle['stages'].get('source_fixed_replay', {})
    source_anchor = bundle['stages'].get('source_anchor', {})
    local_protected = bundle['stages'].get('local_protected', {})
    local_unconstrained = bundle['stages'].get('local_unconstrained', {})

    t1 = stage_metric(transfer, '1', 'phase3_oracle_upgrade', 'mean_recall')
    t2 = stage_metric(transfer, '2', 'phase3_oracle_upgrade', 'mean_recall')
    sf1 = stage_metric(source_fixed, '1', 'phase3_oracle_upgrade', 'mean_recall')
    sf2 = stage_metric(source_fixed, '2', 'phase3_oracle_upgrade', 'mean_recall')
    sa1 = stage_metric(source_anchor, '1', 'phase3_oracle_upgrade', 'mean_recall')
    sa2 = stage_metric(source_anchor, '2', 'phase3_oracle_upgrade', 'mean_recall')
    lp1 = stage_metric(local_protected, '1', 'phase3_oracle_upgrade', 'mean_recall')
    lp2 = stage_metric(local_protected, '2', 'phase3_oracle_upgrade', 'mean_recall')
    lu1 = stage_metric(local_unconstrained, '1', 'phase3_oracle_upgrade', 'mean_recall')
    lu2 = stage_metric(local_unconstrained, '2', 'phase3_oracle_upgrade', 'mean_recall')

    lines = []
    lines.append('# Claim recommendation')
    lines.append('')
    lines.append('Primary result: use **native_clean_attack_test_with_frozen_case14_dev** as the main paper result.')
    lines.append('')
    lines.append('Interpretation: native case39 local selection drifts toward over-conservative policies, while source-frozen transfer acts like regularization and yields the most useful risk–cost trade-off.')
    lines.append('')
    lines.append('Recommended stage roles:')
    lines.append('- Main result: transfer_frozen_dev')
    lines.append('- Mechanism/isolation: source_fixed_replay')
    lines.append('- Repair attempt: source_anchor')
    lines.append('- Protocol-compliant negative control: local_protected')
    lines.append('- Protocol-violating stress test: local_unconstrained')
    lines.append('')
    lines.append('Key oracle recalls (slot1 / slot2):')
    lines.append(f'- transfer_frozen_dev: {t1} / {t2}')
    lines.append(f'- source_fixed_replay: {sf1} / {sf2}')
    lines.append(f'- source_anchor: {sa1} / {sa2}')
    lines.append(f'- local_protected: {lp1} / {lp2}')
    lines.append(f'- local_unconstrained: {lu1} / {lu2}')
    lines.append('')
    lines.append('Next best step is not another heavy run. Focus on paper packaging, significance presentation, and artifact cleanup.')
    return '\n'.join(lines)


def format_num(x: Any) -> str:
    if x is None:
        return 'NA'
    if isinstance(x, float):
        return f'{x:.6f}'
    return str(x)


def stage_row(name: str, summary: Dict[str, Any]) -> str:
    s1 = summary.get('merged_8_holdouts', {}).get('1', {})
    s2 = summary.get('merged_8_holdouts', {}).get('2', {})
    o1 = s1.get('phase3_oracle_upgrade', {})
    o2 = s2.get('phase3_oracle_upgrade', {})
    return '| {} | {} | {} | {} | {} | {} | {} |'.format(
        name,
        format_num(o1.get('mean_recall')),
        format_num(o1.get('mean_unnecessary')),
        format_num(o1.get('mean_cost')),
        format_num(o2.get('mean_recall')),
        format_num(o2.get('mean_unnecessary')),
        format_num(o2.get('mean_cost')),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='Build final case39 bundle from already-computed stage summaries.')
    ap.add_argument('repo_root', nargs='?', default='.')
    ap.add_argument('--output_dir', default='metric/case39_final_bundle')
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        'transfer_frozen_dev': 'native_clean_attack_test_with_frozen_case14_dev',
        'source_fixed_replay': 'case39_source_fixed_case14winner_native_test',
        'source_anchor': 'case39_source_anchored_localretune',
        'local_protected': 'case39_localretune_protocol_compliant_oracle_protected_ec',
        'local_unconstrained': 'case39_fully_native_localretune',
    }

    stages: Dict[str, Dict[str, Any]] = {}
    stage_paths: Dict[str, str] = {}
    missing = []
    for label, stage in wanted.items():
        p = find_stage_summary(repo_root, stage)
        if p is None:
            missing.append({'label': label, 'stage': stage})
            continue
        obj = safe_load_dict(p)
        if obj is None:
            missing.append({'label': label, 'stage': stage, 'path': str(p)})
            continue
        stages[label] = obj
        stage_paths[label] = str(p)

    bundle: Dict[str, Any] = {
        'method': 'case39_make_final_bundle_v2',
        'repo_root': str(repo_root),
        'stage_paths': stage_paths,
        'stages': stages,
        'missing': missing,
        'transfer_hash_audit': None,
        'local_hash_audit': None,
        'claim_recommendation': None,
    }

    # audits (optional)
    for p in repo_root.glob('metric/**/*.json'):
        obj = safe_load_dict(p)
        if not obj:
            continue
        method = obj.get('method', '')
        if method == 'case39_fallback_case14_hash_audit':
            bundle['transfer_hash_audit'] = {'path': str(p), 'all_hashes_match_asset_protocol': obj.get('all_hashes_match_asset_protocol')}
        elif method == 'case39_localretune_fallback_case14_hash_audit':
            bundle['local_hash_audit'] = {'path': str(p), 'all_hashes_match_asset_protocol': obj.get('all_hashes_match_asset_protocol')}
        elif method == 'case39_source_fixed_fallback_case14_hash_audit':
            bundle['source_fixed_hash_audit'] = {'path': str(p), 'all_hashes_match_asset_protocol': obj.get('all_hashes_match_asset_protocol')}

    bundle['claim_recommendation'] = build_claim(bundle)

    with (out_dir / 'final_stage_bundle.json').open('w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    md = []
    md.append('# Case39 final stage table')
    md.append('')
    md.append('| Stage | slot1 recall | slot1 unnecessary | slot1 cost | slot2 recall | slot2 unnecessary | slot2 cost |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for key in ['transfer_frozen_dev', 'source_fixed_replay', 'source_anchor', 'local_protected', 'local_unconstrained']:
        if key in stages:
            md.append(stage_row(key, stages[key]))
    (out_dir / 'final_stage_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out_dir / 'claim_recommendation.md').write_text(bundle['claim_recommendation'] + '\n', encoding='utf-8')

    print(json.dumps({
        'status': 'OK',
        'output_dir': str(out_dir),
        'stage_labels_found': list(stages.keys()),
        'missing': missing,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

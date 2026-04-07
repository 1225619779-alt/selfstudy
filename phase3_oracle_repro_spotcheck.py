from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: str | Path) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _index_reference_per_holdout(reference_confirm: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    rows = reference_confirm.get('per_holdout_results', [])
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        tag = str(row['tag'])
        out[tag] = row
    return out


def _subset_manifest(manifest: Dict[str, object], holdout_tags: Iterable[str]) -> Dict[str, object]:
    tags = set(str(t) for t in holdout_tags)
    new_manifest = dict(manifest)
    new_manifest['holdouts'] = [h for h in manifest['holdouts'] if str(h['tag']) in tags]
    if not new_manifest['holdouts']:
        raise ValueError('No holdouts selected from manifest.')
    return new_manifest


def _compare_slot_payload(rerun: Dict[str, object], ref: Dict[str, object]) -> Tuple[float, List[str]]:
    metrics = [
        ('phase3_oracle_upgrade', 'weighted_attack_recall_no_backend_fail'),
        ('phase3_oracle_upgrade', 'unnecessary_mtd_count'),
        ('phase3_oracle_upgrade', 'queue_delay_p95'),
        ('phase3_oracle_upgrade', 'average_service_cost_per_step'),
        ('phase3_oracle_upgrade', 'pred_expected_consequence_served_ratio'),
        ('phase3_proposed', 'weighted_attack_recall_no_backend_fail'),
        ('phase3_proposed', 'unnecessary_mtd_count'),
        ('phase3_proposed', 'queue_delay_p95'),
        ('phase3_proposed', 'average_service_cost_per_step'),
        ('best_threshold', 'weighted_attack_recall_no_backend_fail'),
        ('best_threshold', 'unnecessary_mtd_count'),
        ('topk_expected_consequence', 'weighted_attack_recall_no_backend_fail'),
        ('topk_expected_consequence', 'unnecessary_mtd_count'),
    ]
    max_err = 0.0
    issues: List[str] = []
    for policy, metric in metrics:
        if policy not in rerun or policy not in ref:
            issues.append(f'missing_policy:{policy}')
            continue
        if metric not in rerun[policy] or metric not in ref[policy]:
            issues.append(f'missing_metric:{policy}.{metric}')
            continue
        a = float(rerun[policy][metric])
        b = float(ref[policy][metric])
        err = abs(a - b)
        max_err = max(max_err, err)
        if err > 1e-12:
            issues.append(f'{policy}.{metric}: rerun={a}, ref={b}, abs_err={err}')
    if str(rerun.get('best_threshold_name')) != str(ref.get('best_threshold_name')):
        issues.append(
            f"best_threshold_name: rerun={rerun.get('best_threshold_name')}, ref={ref.get('best_threshold_name')}"
        )
    return max_err, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Rerun a small subset of oracle-confirm holdouts and compare the rerun output '
            'against an existing reference confirm aggregate_summary.json.'
        )
    )
    parser.add_argument('--manifest', required=True, help='Original confirm manifest json (v1 or v2).')
    parser.add_argument('--reference_confirm', required=True, help='Existing confirm aggregate_summary.json for the same manifest.')
    parser.add_argument('--dev_screen_summary', required=True, help='Dev screen summary from phase3_oracle_family.')
    parser.add_argument('--holdout_tags', nargs='+', required=True, help='One or more holdout tags to rerun.')
    parser.add_argument('--output_dir', required=True, help='Directory to write the mini-manifest, rerun output, and spotcheck summary.')
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    reference_confirm = _load_json(args.reference_confirm)
    reference_by_tag = _index_reference_per_holdout(reference_confirm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mini_manifest = _subset_manifest(manifest, args.holdout_tags)
    mini_manifest_path = output_dir / 'mini_manifest.json'
    with open(mini_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(mini_manifest, f, ensure_ascii=False, indent=2)

    # Import lazily so the script can still be linted on machines without the project deps loaded.
    from phase3_oracle_confirm_core import run_phase3_oracle_confirm

    rerun_dir = output_dir / 'rerun_output'
    result = run_phase3_oracle_confirm(
        confirm_manifest_path=str(mini_manifest_path),
        dev_screen_summary_path=args.dev_screen_summary,
        output_dir=str(rerun_dir),
    )
    rerun_summary = _load_json(result['aggregate_summary_path'])

    checks: Dict[str, object] = {}
    overall_max_err = 0.0
    overall_issues: List[str] = []

    for row in rerun_summary.get('per_holdout_results', []):
        tag = str(row['tag'])
        ref_row = reference_by_tag.get(tag)
        if ref_row is None:
            checks[tag] = {'status': 'FAIL', 'issues': ['tag_missing_in_reference']}
            overall_issues.append(f'{tag}: missing in reference confirm')
            continue

        tag_max_err = 0.0
        tag_issues: List[str] = []
        for slot, rerun_slot_payload in row['slot_budget_results'].items():
            ref_slot_payload = ref_row['slot_budget_results'].get(str(slot))
            if ref_slot_payload is None:
                tag_issues.append(f'slot_{slot}: missing in reference')
                continue
            slot_max_err, slot_issues = _compare_slot_payload(rerun_slot_payload, ref_slot_payload)
            tag_max_err = max(tag_max_err, slot_max_err)
            tag_issues.extend([f'slot_{slot}: {x}' for x in slot_issues])

        checks[tag] = {
            'status': 'PASS' if not tag_issues else 'FAIL',
            'max_abs_error': tag_max_err,
            'issues': tag_issues,
        }
        overall_max_err = max(overall_max_err, tag_max_err)
        overall_issues.extend([f'{tag}: {x}' for x in tag_issues])

    summary = {
        'method': 'phase3_oracle_repro_spotcheck',
        'manifest': args.manifest,
        'reference_confirm': args.reference_confirm,
        'dev_screen_summary': args.dev_screen_summary,
        'mini_manifest_path': str(mini_manifest_path.resolve()),
        'rerun_aggregate_summary_path': str(Path(result['aggregate_summary_path']).resolve()),
        'selected_holdout_tags': [str(t) for t in args.holdout_tags],
        'n_selected_holdouts': len(args.holdout_tags),
        'checks': checks,
        'max_abs_error': overall_max_err,
        'issues': overall_issues,
        'status': 'PASS' if not overall_issues else 'FAIL',
    }

    out_path = output_dir / 'repro_spotcheck_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({'repro_spotcheck_summary_path': str(out_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

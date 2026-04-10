from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

CASE39_PATTERNS = [
    'case39',
    'case39.xlsx',
    'case39.py',
    '14bus_config.yaml',  # if GridForge present, user can adapt example configs
]


def _find_matching_files(root: Path, needle: str, max_hits: int = 30) -> List[str]:
    hits: List[str] = []
    if not root.exists():
        return hits
    needle_lower = needle.lower()
    try:
        for p in root.rglob('*'):
            if len(hits) >= max_hits:
                break
            try:
                name = p.name.lower()
            except Exception:
                continue
            if needle_lower in name:
                hits.append(str(p))
    except Exception:
        return hits
    return hits


def _repo_probe(root: Path, label: str) -> Dict[str, object]:
    exists = root.exists()
    probe = {
        'label': label,
        'path': str(root),
        'exists': exists,
        'case39_hits': [],
        'gridforge_indicators': [],
        'recommended_role': None,
    }
    if not exists:
        return probe

    case_hits: List[str] = []
    for needle in ['case39', 'case39.xlsx', 'case39.py']:
        case_hits.extend(_find_matching_files(root, needle, max_hits=10))
    # de-dup while preserving order
    seen = set()
    case_hits_dedup = []
    for x in case_hits:
        if x not in seen:
            case_hits_dedup.append(x)
            seen.add(x)
    probe['case39_hits'] = case_hits_dedup[:20]

    grid_hits = []
    for needle in ['gridforge', 'prepare.py', 'construct_grid_config', 'construct_grid_data']:
        grid_hits.extend(_find_matching_files(root, needle, max_hits=10))
    seen = set()
    grid_hits_dedup = []
    for x in grid_hits:
        if x not in seen:
            grid_hits_dedup.append(x)
            seen.add(x)
    probe['gridforge_indicators'] = grid_hits_dedup[:20]

    name = root.name.lower()
    if 'lapso' in name or 'gridforge' in name:
        probe['recommended_role'] = 'preferred_case39_asset_source'
    elif 'robust_mtd' in name:
        probe['recommended_role'] = 'later_second_backend_or_robustness_source'
    else:
        probe['recommended_role'] = 'unknown'
    return probe


def _choose_recommendation(probes: List[Dict[str, object]]) -> Dict[str, object]:
    preferred = None
    for p in probes:
        if p['exists'] and p['case39_hits'] and p['recommended_role'] == 'preferred_case39_asset_source':
            preferred = p
            break
    fallback = None
    for p in probes:
        if p['exists'] and p['case39_hits']:
            fallback = p
            break

    if preferred is not None:
        return {
            'status': 'READY_WITH_EXTERNAL_ASSETS',
            'recommended_source_label': preferred['label'],
            'recommended_source_path': preferred['path'],
            'why': 'Found case39 assets in a paper-backed external repo suitable for system expansion.',
        }
    if fallback is not None:
        return {
            'status': 'READY_WITH_LOCAL_CASE39_ASSETS',
            'recommended_source_label': fallback['label'],
            'recommended_source_path': fallback['path'],
            'why': 'Found case39-named assets locally, though not in the preferred GridForge/LAPSO path.',
        }
    return {
        'status': 'MISSING_CASE39_ASSETS',
        'recommended_source_label': None,
        'recommended_source_path': None,
        'why': 'No case39 assets detected locally. Clone xuwkk/gridforge or xuwkk/lapso_exp, then rerun preflight.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Preflight check for case39 system expansion. Detect local/external case39 assets and recommend the asset source.'
    )
    parser.add_argument('--repo_root', default='.', help='Current q1_scheduler repo root.')
    parser.add_argument('--gridforge_root', default='../gridforge', help='Optional local GridForge clone path.')
    parser.add_argument('--lapso_root', default='../lapso_exp', help='Optional local LAPSO_EXP clone path.')
    parser.add_argument('--robust_mtd_root', default='../Robust_MTD', help='Optional local Robust_MTD clone path.')
    parser.add_argument('--output', required=True, help='Output json path.')
    args = parser.parse_args()

    probes = [
        _repo_probe(Path(args.repo_root).resolve(), 'current_q1_scheduler_repo'),
        _repo_probe(Path(args.gridforge_root).resolve(), 'gridforge_repo'),
        _repo_probe(Path(args.lapso_root).resolve(), 'lapso_exp_repo'),
        _repo_probe(Path(args.robust_mtd_root).resolve(), 'robust_mtd_repo'),
    ]

    recommendation = _choose_recommendation(probes)

    summary = {
        'method': 'phase3_case39_preflight',
        'probes': probes,
        'recommendation': recommendation,
        'next_step_if_ready': (
            'Use the recommended source only as infrastructure/testbed generation, not as a competing baseline. '
            'Then build case39 banks and run the fixed oracle_protected_ec vs phase3/best-threshold/topk stack.'
        ),
        'note': (
            'This is a readiness check, not a scientific result. It tells us whether case39 expansion can start '
            'and which local repo path should be used for asset generation.'
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({'case39_preflight_summary_path': str(out_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

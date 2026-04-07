#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

SKIP_DIR_NAMES = {
    '.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
    '.venv', '.venv_q1', '.venv_rocm', '.ipynb_checkpoints'
}
TEXT_SUFFIXES = {'.py', '.json', '.md', '.txt', '.yaml', '.yml', '.sh'}
CASE14_PATTERNS = [
    'case14',
    'metric/case14',
    'case14/',
    'case14_',
]
CASE39_PATTERNS = [
    'case39',
    'metric/case39',
    'case39/',
    'case39_',
]
CORE_HINTS = (
    'evaluation', 'scheduler', 'phase3', 'holdout', 'select_', 'run_', 'metric/'
)


def should_skip_dir(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def classify_file(rel_path: str) -> str:
    if rel_path.startswith('metric/'):
        return 'artifact_or_result'
    if any(h in rel_path for h in CORE_HINTS):
        return 'likely_code_or_runner'
    return 'other_text_file'


def read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None


def find_hits(text: str, patterns: List[str]) -> List[str]:
    hits = []
    lower = text.lower()
    for p in patterns:
        if p in lower:
            hits.append(p)
    return hits


def scan_repo(repo_root: Path) -> Tuple[List[Dict], List[Dict]]:
    case14_files: List[Dict] = []
    case39_files: List[Dict] = []

    for path in repo_root.rglob('*'):
        if not path.is_file():
            continue
        if should_skip_dir(path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        text = read_text_safe(path)
        if text is None:
            continue

        rel = path.relative_to(repo_root).as_posix()
        hits14 = find_hits(text, CASE14_PATTERNS)
        hits39 = find_hits(text, CASE39_PATTERNS)

        if hits14:
            case14_files.append({
                'path': rel,
                'classification': classify_file(rel),
                'matched_patterns': hits14,
            })
        if hits39:
            case39_files.append({
                'path': rel,
                'classification': classify_file(rel),
                'matched_patterns': hits39,
            })

    case14_files.sort(key=lambda x: (x['classification'], x['path']))
    case39_files.sort(key=lambda x: (x['classification'], x['path']))
    return case14_files, case39_files


def summarize_counts(rows: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r['classification']] = out.get(r['classification'], 0) + 1
    return out


def build_recommendations(case14_rows: List[Dict], case39_rows: List[Dict]) -> List[str]:
    notes: List[str] = []
    code_case14 = [r for r in case14_rows if r['classification'] == 'likely_code_or_runner']
    metric_case14 = [r for r in case14_rows if r['path'].startswith('metric/case14/')]
    code_case39 = [r for r in case39_rows if r['classification'] == 'likely_code_or_runner']

    if code_case14:
        notes.append(
            'There are likely code/runner files with case14-hardcoded strings. '
            'Expect a small bridge patch rather than zero-code case39 support.'
        )
    if metric_case14:
        notes.append(
            'Keep all existing metric/case14 assets frozen. Create a fresh metric/case39 root instead of editing case14 outputs.'
        )
    if not code_case39:
        notes.append(
            'The repo has little or no native case39-specific runner logic. A dedicated case39 bridge layer is recommended.'
        )
    notes.append(
        'Do not mutate the current shared DDET-MTD workspace in place. Use a separate git worktree or clone for case39 expansion.'
    )
    notes.append(
        'First bridge goal: parameterize case root / bank paths / manifest generation. Do not rewrite the phase3 policy shell.'
    )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case14_rows, case39_rows = scan_repo(repo_root)
    code_case14 = [r for r in case14_rows if r['classification'] == 'likely_code_or_runner']
    status = 'NEEDS_CASE39_BRIDGE' if code_case14 else 'LIGHT_BRIDGE_POSSIBLE'

    summary = {
        'method': 'phase3_case39_capability_audit',
        'repo_root': str(repo_root),
        'status': status,
        'counts': {
            'case14_total_files': len(case14_rows),
            'case39_total_files': len(case39_rows),
            'case14_by_classification': summarize_counts(case14_rows),
            'case39_by_classification': summarize_counts(case39_rows),
        },
        'sample_case14_code_or_runner_hits': code_case14[:25],
        'sample_case39_hits': case39_rows[:25],
        'recommendations': build_recommendations(case14_rows, case39_rows),
        'next_step': (
            'Create an isolated case39 worktree, then patch only the minimal path/case-parameter plumbing needed to generate '
            'metric/case39 banks and run the fixed oracle_protected_ec vs phase3/best-threshold/topk stack.'
        ),
    }

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'status': summary['status'],
        'output': str(output_path),
        'case14_total_files': summary['counts']['case14_total_files'],
        'case39_total_files': summary['counts']['case39_total_files'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

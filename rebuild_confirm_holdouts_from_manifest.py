from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str], cwd: Path) -> None:
    print('$', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate native test banks and baseline summaries from an existing confirm manifest.')
    parser.add_argument('--manifest', required=True, help='Path to confirm manifest json.')
    parser.add_argument('--force', action='store_true', help='Regenerate even if outputs already exist.')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    workdir = manifest_path.parent.parent.parent.resolve()
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding='utf-8'))

    for hold in manifest['holdouts']:
        test_bank = workdir / hold['test_bank']
        result_npy = workdir / hold['result_npy']
        result_summary = workdir / hold['result_summary']

        if test_bank.exists() and test_bank.is_symlink():
            test_bank.unlink()
        if args.force and result_npy.exists():
            result_npy.unlink()
        if args.force and result_summary.exists():
            result_summary.unlink()

        if args.force or not test_bank.exists():
            test_bank.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable,
                'evaluation_mixed_timeline.py',
                '--tau_verify', '-1',
                '--schedule', str(hold['schedule']),
                '--seed_base', str(hold['seed_base']),
                '--start_offset', str(hold['start_offset']),
                '--output', str(test_bank),
            ], cwd=workdir)
        else:
            print(f'[skip] existing test bank: {test_bank}')

        if args.force or (not result_npy.exists()) or (not result_summary.exists()):
            result_npy.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable,
                'evaluation_budget_scheduler_phase3_holdout.py',
                '--clean_bank', str(workdir / manifest['clean_bank']),
                '--attack_bank', str(workdir / manifest['attack_bank']),
                '--train_bank', str(workdir / manifest['train_bank']),
                '--val_bank', str(workdir / manifest['val_bank']),
                '--test_bank', str(test_bank),
                '--slot_budget_list', *[str(x) for x in manifest['frozen_regime']['slot_budget_list']],
                '--decision_step_group', str(manifest['frozen_regime']['decision_step_group']),
                '--busy_time_quantile', str(manifest['frozen_regime']['busy_time_quantile']),
                '--output', str(result_npy),
            ], cwd=workdir)
        else:
            print(f'[skip] existing baseline summary: {result_summary}')

    print(json.dumps({'manifest': str(manifest_path), 'done': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

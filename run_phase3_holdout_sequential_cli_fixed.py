#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _score(compact: Dict[str, Any]) -> float:
    return (
        float(compact['weighted_attack_recall_no_backend_fail'])
        - 0.005 * float(compact['unnecessary_mtd_count'])
        - 0.01 * float(compact['queue_delay_p95'])
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], *, cwd: Path) -> None:
    print('\n' + '=' * 96)
    print('[RUN]', ' '.join(cmd))
    print('=' * 96)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _choose_regime(regimes: List[Dict[str, Any]], slot_budgets: List[int], prefer_no_cost: bool, prefer_no_cost_margin: float) -> Dict[str, Any]:
    def avg_selection_score(entry: Dict[str, Any]) -> float:
        vals = []
        for sb in slot_budgets:
            proposed = entry['slot_budgets'][str(sb)]['proposed']
            vals.append(_score(proposed))
        return sum(vals) / max(len(vals), 1)

    ranked = sorted(regimes, key=avg_selection_score, reverse=True)
    chosen = ranked[0]

    if prefer_no_cost:
        best_nocost = None
        for entry in ranked:
            if not bool(entry['use_cost_budget']):
                best_nocost = entry
                break
        if best_nocost is not None and avg_selection_score(best_nocost) >= avg_selection_score(chosen) - prefer_no_cost_margin:
            chosen = best_nocost

    chosen = dict(chosen)
    chosen['avg_selection_score'] = round(avg_selection_score(chosen), 6)
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Sequential runner for phase-3 holdout protocol (CLI-safe version).')
    p.add_argument('--workdir', type=str, default='.')
    p.add_argument('--python', type=str, default=sys.executable)
    p.add_argument('--clean_bank', type=str, default='metric/case14/metric_clean_alarm_scores_full.npy')
    p.add_argument('--attack_bank', type=str, default='metric/case14/metric_attack_alarm_scores_400.npy')
    p.add_argument('--train_bank', type=str, default='metric/case14/mixed_bank_fit.npy')
    p.add_argument('--val_bank', type=str, default='metric/case14/mixed_bank_eval.npy')
    p.add_argument('--test_bank', type=str, default='metric/case14/mixed_bank_test_holdout.npy')
    p.add_argument('--ranking_output', type=str, default='metric/case14/phase3_val_regime_ranking_holdout.json')
    p.add_argument('--holdout_output', type=str, default='metric/case14/budget_scheduler_phase3_holdout_auto.npy')
    p.add_argument('--slot_budget_list', type=int, nargs='*', default=[1, 2])
    p.add_argument('--decision_step_group_list', type=int, nargs='*', default=[1, 2])
    p.add_argument('--busy_time_quantile_list', type=float, nargs='*', default=[0.35, 0.50, 0.65])
    p.add_argument('--use_cost_budget_modes', type=str, nargs='*', default=['off', 'on'])
    p.add_argument('--cost_budget_quantile_list', type=float, nargs='*', default=[0.50, 0.60])
    p.add_argument('--cost_budget_window_steps', type=int, default=20)
    p.add_argument('--max_wait_steps', type=int, default=10)
    p.add_argument('--n_bins', type=int, default=20)
    p.add_argument('--rng_seed', type=int, default=20260402)
    p.add_argument('--prefer_no_cost', action='store_true')
    p.add_argument('--prefer_no_cost_margin', type=float, default=0.01)
    p.add_argument('--reuse_ranking', action='store_true')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    os.chdir(workdir)

    required = [
        workdir / 'phase3_holdout_core.py',
        workdir / 'select_regime_phase3_val.py',
        workdir / 'evaluation_budget_scheduler_phase3_holdout.py',
        workdir / 'scheduler' / 'policies_phase3.py',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print('[ERROR] Missing required files:')
        for m in missing:
            print('  -', m)
        return 1

    ranking_path = (workdir / args.ranking_output).resolve()
    holdout_path = (workdir / args.holdout_output).resolve()
    holdout_summary_path = Path(str(holdout_path).replace('.npy', '.summary.json'))
    _ensure_parent(ranking_path)
    _ensure_parent(holdout_path)

    print(f'[INFO] Workdir: {workdir}')
    print(f'[INFO] Python : {args.python}')
    print(f'[INFO] Ranking: {ranking_path}')
    print(f'[INFO] Holdout: {holdout_path}')

    if not (ranking_path.exists() and args.reuse_ranking and not args.force):
        cmd = [
            args.python, 'select_regime_phase3_val.py',
            '--clean_bank', args.clean_bank,
            '--attack_bank', args.attack_bank,
            '--train_bank', args.train_bank,
            '--val_bank', args.val_bank,
            '--slot_budget_list', *[str(x) for x in args.slot_budget_list],
            '--decision_step_group_list', *[str(x) for x in args.decision_step_group_list],
            '--busy_time_quantile_list', *[str(x) for x in args.busy_time_quantile_list],
            '--use_cost_budget_modes', *[str(x) for x in args.use_cost_budget_modes],
            '--cost_budget_quantile_list', *[str(x) for x in args.cost_budget_quantile_list],
            '--cost_budget_window_steps', str(args.cost_budget_window_steps),
            '--max_wait_steps', str(args.max_wait_steps),
            '--n_bins', str(args.n_bins),
            '--rng_seed', str(args.rng_seed),
            '--output', str(ranking_path.relative_to(workdir)),
        ]
        _run(cmd, cwd=workdir)
    else:
        print(f'[SKIP] Reusing ranking file: {ranking_path}')

    with ranking_path.open('r', encoding='utf-8') as f:
        regimes = json.load(f)

    chosen = _choose_regime(regimes, args.slot_budget_list, args.prefer_no_cost, args.prefer_no_cost_margin)
    print('\n' + '#' * 96)
    print('[SELECTED REGIME FOR HOLDOUT]')
    print(json.dumps(chosen, ensure_ascii=False, indent=2))
    print('#' * 96)

    if holdout_path.exists() and holdout_summary_path.exists() and not args.force:
        print(f'[SKIP] Holdout outputs already exist: {holdout_path}')
        return 0

    cmd = [
        args.python, 'evaluation_budget_scheduler_phase3_holdout.py',
        '--clean_bank', args.clean_bank,
        '--attack_bank', args.attack_bank,
        '--train_bank', args.train_bank,
        '--val_bank', args.val_bank,
        '--test_bank', args.test_bank,
        '--slot_budget_list', *[str(x) for x in args.slot_budget_list],
        '--decision_step_group', str(chosen['decision_step_group']),
        '--busy_time_quantile', str(chosen['busy_time_quantile']),
        '--max_wait_steps', str(args.max_wait_steps),
        '--n_bins', str(args.n_bins),
        '--rng_seed', str(args.rng_seed),
        '--output', str(holdout_path.relative_to(workdir)),
    ]
    if bool(chosen['use_cost_budget']):
        cmd.extend([
            '--use_cost_budget',
            '--cost_budget_window_steps', str(args.cost_budget_window_steps),
            '--cost_budget_quantile', str(chosen['cost_budget_quantile']),
        ])

    _run(cmd, cwd=workdir)
    print('\n[ALL DONE] Holdout selection + final test finished successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from phase3_holdout_core import run_train_tune_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware holdout evaluation for phase-3 consequence-aware hard-constrained scheduler."
    )
    parser.add_argument("--clean_bank", type=str, required=True)
    parser.add_argument("--attack_bank", type=str, required=True)
    parser.add_argument("--train_bank", type=str, required=True)
    parser.add_argument("--val_bank", type=str, required=True)
    parser.add_argument("--test_bank", type=str, required=True)
    parser.add_argument("--output", type=str, default=f"metric/{os.environ.get("DDET_CASE_NAME", "case14")}/budget_scheduler_phase3_holdout.npy")
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--max_wait_steps", type=int, default=10)
    parser.add_argument("--decision_step_group", type=int, default=1)
    parser.add_argument("--busy_time_quantile", type=float, default=0.50)
    parser.add_argument("--use_cost_budget", action="store_true")
    parser.add_argument("--cost_budget_window_steps", type=int, default=20)
    parser.add_argument("--cost_budget_quantile", type=float, default=0.60)
    parser.add_argument("--threshold_quantiles", type=float, nargs="*", default=[0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--adaptive_gain_scale_list", type=float, nargs="*", default=[0.0, 0.10, 0.20, 0.40])
    parser.add_argument("--consequence_blend_verify", type=float, default=0.70)
    parser.add_argument("--consequence_mode", type=str, default="conditional", choices=["conditional", "expected"])
    parser.add_argument("--objective_clean_penalty", type=float, default=0.60)
    parser.add_argument("--objective_delay_penalty", type=float, default=0.15)
    parser.add_argument("--objective_queue_penalty", type=float, default=0.10)
    parser.add_argument("--objective_cost_penalty", type=float, default=0.05)
    parser.add_argument("--vq_v_grid", type=float, nargs="*", default=[1.0, 2.0, 4.0])
    parser.add_argument("--vq_age_grid", type=float, nargs="*", default=[0.0, 0.10, 0.20])
    parser.add_argument("--vq_urgency_grid", type=float, nargs="*", default=[0.0, 0.10, 0.20])
    parser.add_argument("--vq_fail_grid", type=float, nargs="*", default=[0.0, 0.05])
    parser.add_argument("--vq_busy_grid", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--vq_cost_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    parser.add_argument("--vq_clean_grid", type=float, nargs="*", default=[0.0, 0.20, 0.50])
    parser.add_argument("--vq_admission_threshold_grid", type=float, nargs="*", default=[-0.10, 0.0, 0.10])
    parser.add_argument("--rng_seed", type=int, default=20260402)
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main() -> None:
    args = parse_args()
    ensure_parent(args.output)
    ns = SimpleNamespace(
        clean_bank=args.clean_bank,
        attack_bank=args.attack_bank,
        train_bank=args.train_bank,
        tune_bank=args.val_bank,
        eval_bank=args.test_bank,
        output=args.output,
        n_bins=args.n_bins,
        slot_budget_list=list(args.slot_budget_list),
        max_wait_steps=args.max_wait_steps,
        decision_step_group=args.decision_step_group,
        busy_time_quantile=args.busy_time_quantile,
        use_cost_budget=args.use_cost_budget,
        cost_budget_window_steps=args.cost_budget_window_steps,
        cost_budget_quantile=args.cost_budget_quantile,
        threshold_quantiles=list(args.threshold_quantiles),
        adaptive_gain_scale_list=list(args.adaptive_gain_scale_list),
        consequence_blend_verify=args.consequence_blend_verify,
        consequence_mode=args.consequence_mode,
        objective_clean_penalty=args.objective_clean_penalty,
        objective_delay_penalty=args.objective_delay_penalty,
        objective_queue_penalty=args.objective_queue_penalty,
        objective_cost_penalty=args.objective_cost_penalty,
        vq_v_grid=list(args.vq_v_grid),
        vq_age_grid=list(args.vq_age_grid),
        vq_urgency_grid=list(args.vq_urgency_grid),
        vq_fail_grid=list(args.vq_fail_grid),
        vq_busy_grid=list(args.vq_busy_grid),
        vq_cost_grid=list(args.vq_cost_grid),
        vq_clean_grid=list(args.vq_clean_grid),
        vq_admission_threshold_grid=list(args.vq_admission_threshold_grid),
        rng_seed=args.rng_seed,
    )
    results = run_train_tune_eval(ns)
    np.save(args.output, results, allow_pickle=True)
    summary_path = args.output.replace('.npy', '.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    print(f"Saved holdout result npy: {args.output}")
    print(f"Saved holdout summary json: {summary_path}")
    print("\n==== Holdout environment diagnostics ====")
    print(json.dumps(_to_jsonable(results['environment']), ensure_ascii=False, indent=2))
    print("\n==== Holdout TEST compact summaries ====")
    for slot_budget, payload in results['slot_budget_results'].items():
        print(f"\n-- slot_budget={slot_budget} --")
        print('busy_time_unit', payload['busy_time_unit'])
        print('window_cost_budget', payload['window_cost_budget'])
        for policy, compact in payload['eval_compact'].items():
            print(policy, compact)


if __name__ == '__main__':
    main()

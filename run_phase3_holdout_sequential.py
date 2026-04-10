#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")


def _score_one(compact: Dict[str, Any]) -> float:
    return (
        float(compact["weighted_attack_recall_no_backend_fail"])
        - 0.005 * float(compact["unnecessary_mtd_count"])
        - 0.01 * float(compact["queue_delay_p95"])
    )


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _build_ns(args: argparse.Namespace, *, decision_step_group: int, busy_time_quantile: float, use_cost_budget: bool, cost_budget_quantile: float, eval_bank: str) -> SimpleNamespace:
    return SimpleNamespace(
        clean_bank=args.clean_bank,
        attack_bank=args.attack_bank,
        train_bank=args.train_bank,
        tune_bank=args.val_bank,
        eval_bank=eval_bank,
        output=args.holdout_output,
        n_bins=args.n_bins,
        slot_budget_list=list(args.slot_budget_list),
        max_wait_steps=args.max_wait_steps,
        decision_step_group=int(decision_step_group),
        busy_time_quantile=float(busy_time_quantile),
        use_cost_budget=bool(use_cost_budget),
        cost_budget_window_steps=int(args.cost_budget_window_steps),
        cost_budget_quantile=float(cost_budget_quantile),
        threshold_quantiles=[0.50, 0.60, 0.70, 0.80, 0.90],
        adaptive_gain_scale_list=[0.0, 0.10, 0.20, 0.40],
        consequence_blend_verify=0.70,
        consequence_mode="conditional",
        objective_clean_penalty=0.60,
        objective_delay_penalty=0.15,
        objective_queue_penalty=0.10,
        objective_cost_penalty=0.05,
        vq_v_grid=[1.0, 2.0, 4.0],
        vq_age_grid=[0.0, 0.10, 0.20],
        vq_urgency_grid=[0.0, 0.10, 0.20],
        vq_fail_grid=[0.0, 0.05],
        vq_busy_grid=[0.5, 1.0, 2.0],
        vq_cost_grid=[0.0, 0.5, 1.0],
        vq_clean_grid=[0.0, 0.20, 0.50],
        vq_admission_threshold_grid=[-0.10, 0.0, 0.10],
        rng_seed=int(args.rng_seed),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential runner for phase-3 holdout protocol.")
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--clean_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_clean_alarm_scores_full.npy")
    p.add_argument("--attack_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_attack_alarm_scores_400.npy")
    p.add_argument("--train_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_fit.npy")
    p.add_argument("--val_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_eval.npy")
    p.add_argument("--test_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_test_holdout.npy")
    p.add_argument("--ranking_output", type=str, default=f"metric/{DEFAULT_CASE_NAME}/phase3_val_regime_ranking_holdout.json")
    p.add_argument("--holdout_output", type=str, default=f"metric/{DEFAULT_CASE_NAME}/budget_scheduler_phase3_holdout_auto.npy")
    p.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    p.add_argument("--decision_step_group_list", type=int, nargs="*", default=[1, 2])
    p.add_argument("--busy_time_quantile_list", type=float, nargs="*", default=[0.35, 0.50, 0.65])
    p.add_argument("--use_cost_budget_modes", type=str, nargs="*", default=["off", "on"])
    p.add_argument("--cost_budget_quantile_list", type=float, nargs="*", default=[0.50, 0.60])
    p.add_argument("--cost_budget_window_steps", type=int, default=20)
    p.add_argument("--max_wait_steps", type=int, default=10)
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--rng_seed", type=int, default=20260402)
    p.add_argument("--reuse_ranking", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--prefer_no_cost", action="store_true", help="If best no-cost regime is close, prefer it for cleaner main result.")
    p.add_argument("--prefer_no_cost_margin", type=float, default=0.01)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    os.chdir(workdir)
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    from phase3_holdout_core import run_train_tune_eval

    _ensure_parent(args.ranking_output)
    _ensure_parent(args.holdout_output)

    ranking_path = Path(args.ranking_output)
    holdout_path = Path(args.holdout_output)
    holdout_summary_path = Path(str(holdout_path).replace('.npy', '.summary.json'))

    print(f"[INFO] Workdir: {workdir}")
    print(f"[INFO] Ranking output: {ranking_path}")
    print(f"[INFO] Holdout output: {holdout_path}")

    regimes: List[Dict[str, Any]] = []
    if ranking_path.exists() and args.reuse_ranking and not args.force:
        with ranking_path.open('r', encoding='utf-8') as f:
            regimes = json.load(f)
        print(f"[SKIP] Reusing ranking file: {ranking_path}")
    else:
        total = 0
        for g in args.decision_step_group_list:
            for btq in args.busy_time_quantile_list:
                for cost_mode in args.use_cost_budget_modes:
                    use_cost = str(cost_mode).lower() in {"1", "true", "yes", "on"}
                    q_list = args.cost_budget_quantile_list if use_cost else [0.0]
                    total += len(q_list)

        t0 = time.time()
        idx = 0
        for g in args.decision_step_group_list:
            for btq in args.busy_time_quantile_list:
                for cost_mode in args.use_cost_budget_modes:
                    use_cost = str(cost_mode).lower() in {"1", "true", "yes", "on"}
                    q_list = args.cost_budget_quantile_list if use_cost else [0.0]
                    for cbq in q_list:
                        idx += 1
                        print("\n" + "=" * 88)
                        print(
                            f"[VAL {idx}/{total}] decision_step_group={g}, busy_time_quantile={btq}, "
                            f"use_cost_budget={use_cost}, cost_budget_quantile={None if not use_cost else cbq}"
                        )
                        print("=" * 88)
                        ns = _build_ns(
                            args,
                            decision_step_group=int(g),
                            busy_time_quantile=float(btq),
                            use_cost_budget=use_cost,
                            cost_budget_quantile=float(cbq),
                            eval_bank=args.val_bank,
                        )
                        t_regime = time.time()
                        result = run_train_tune_eval(ns)
                        entry: Dict[str, Any] = {
                            "decision_step_group": int(g),
                            "busy_time_quantile": float(btq),
                            "use_cost_budget": bool(use_cost),
                            "cost_budget_quantile": None if not use_cost else float(cbq),
                            "slot_budgets": {},
                        }
                        slot_scores: List[float] = []
                        for slot_budget, payload in result["slot_budget_results"].items():
                            compact = payload["eval_compact"]
                            proposed = compact["proposed_ca_vq_hard"]
                            compare_pool = {k: v for k, v in compact.items() if k != "proposed_ca_vq_hard"}
                            threshold_names = [k for k in compare_pool if "threshold" in k]
                            best_threshold_name = max(
                                threshold_names,
                                key=lambda k: _score_one(compare_pool[k]),
                            )
                            entry["slot_budgets"][str(slot_budget)] = {
                                "proposed": proposed,
                                "best_threshold_name": best_threshold_name,
                                "best_threshold": compare_pool[best_threshold_name],
                                "delta_vs_best_threshold_recall": round(
                                    float(proposed["weighted_attack_recall_no_backend_fail"]) - float(compare_pool[best_threshold_name]["weighted_attack_recall_no_backend_fail"]),
                                    4,
                                ),
                                "delta_vs_best_threshold_unnecessary": int(
                                    int(proposed["unnecessary_mtd_count"]) - int(compare_pool[best_threshold_name]["unnecessary_mtd_count"])
                                ),
                                "delta_vs_best_threshold_delay_p95": round(
                                    float(proposed["queue_delay_p95"]) - float(compare_pool[best_threshold_name]["queue_delay_p95"]),
                                    4,
                                ),
                                "selection_score": round(_score_one(proposed), 6),
                            }
                            slot_scores.append(_score_one(proposed))
                        entry["avg_selection_score"] = round(float(np.mean(slot_scores)), 6)
                        regimes.append(entry)

                        with ranking_path.open('w', encoding='utf-8') as f:
                            json.dump(_jsonable(regimes), f, ensure_ascii=False, indent=2)

                        elapsed = time.time() - t_regime
                        print(f"[DONE] validation regime {idx}/{total} in {elapsed:.1f}s")
                        for sb in args.slot_budget_list:
                            s = entry["slot_budgets"][str(sb)]
                            print(
                                f"  slot_budget={sb}: proposed_recall={s['proposed']['weighted_attack_recall_no_backend_fail']:.4f}, "
                                f"unnecessary={s['proposed']['unnecessary_mtd_count']}, delay_p95={s['proposed']['queue_delay_p95']:.2f}, "
                                f"best_threshold={s['best_threshold_name']}"
                            )
        print(f"\n[VAL DONE] {len(regimes)} regimes finished in {(time.time() - t0) / 60:.2f} min")
        print(f"[SAVED] {ranking_path}")

    if not regimes:
        with ranking_path.open('r', encoding='utf-8') as f:
            regimes = json.load(f)

    ranked = sorted(regimes, key=lambda e: float(e.get("avg_selection_score", -1e9)), reverse=True)
    chosen = ranked[0]

    if args.prefer_no_cost:
        best_nocost = None
        for e in ranked:
            if not bool(e["use_cost_budget"]):
                best_nocost = e
                break
        if best_nocost is not None and float(best_nocost["avg_selection_score"]) >= float(chosen["avg_selection_score"]) - float(args.prefer_no_cost_margin):
            chosen = best_nocost

    print("\n" + "#" * 88)
    print("[SELECTED REGIME FOR HOLDOUT]")
    print(json.dumps(chosen, ensure_ascii=False, indent=2))
    print("#" * 88)

    if holdout_path.exists() and holdout_summary_path.exists() and not args.force:
        print(f"[SKIP] Holdout outputs already exist: {holdout_path}")
        return 0

    ns = _build_ns(
        args,
        decision_step_group=int(chosen["decision_step_group"]),
        busy_time_quantile=float(chosen["busy_time_quantile"]),
        use_cost_budget=bool(chosen["use_cost_budget"]),
        cost_budget_quantile=0.0 if chosen["cost_budget_quantile"] is None else float(chosen["cost_budget_quantile"]),
        eval_bank=args.test_bank,
    )

    print("\n" + "=" * 88)
    print("[RUN HOLDOUT TEST]")
    print(
        f"decision_step_group={ns.decision_step_group}, busy_time_quantile={ns.busy_time_quantile}, "
        f"use_cost_budget={ns.use_cost_budget}, cost_budget_quantile={None if not ns.use_cost_budget else ns.cost_budget_quantile}"
    )
    print("=" * 88)
    t_hold = time.time()
    holdout = run_train_tune_eval(ns)
    np.save(holdout_path, holdout, allow_pickle=True)
    with holdout_summary_path.open('w', encoding='utf-8') as f:
        json.dump(_jsonable(holdout), f, ensure_ascii=False, indent=2)
    print(f"[DONE] Holdout finished in {(time.time() - t_hold) / 60:.2f} min")
    print(f"[SAVED] {holdout_path}")
    print(f"[SAVED] {holdout_summary_path}")

    print("\n==== Holdout TEST compact summaries ====\n")
    for slot_budget, payload in holdout["slot_budget_results"].items():
        print(f"-- slot_budget={slot_budget} --")
        print(f"busy_time_unit {payload['busy_time_unit']}")
        print(f"window_cost_budget {payload['window_cost_budget']}")
        for policy, compact in payload['eval_compact'].items():
            print(policy, compact)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List

from evaluation_budget_scheduler_phase3 import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime sweep for phase-3 consequence-aware budget scheduler.")
    p.add_argument("--clean_bank", type=str, required=True)
    p.add_argument("--attack_bank", type=str, required=True)
    p.add_argument("--fit_bank", type=str, required=True)
    p.add_argument("--eval_bank", type=str, required=True)
    p.add_argument("--output", type=str, default="metric/case14/phase3_regime_sweep.json")
    p.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    p.add_argument("--decision_step_group_list", type=int, nargs="*", default=[1, 2])
    p.add_argument("--busy_time_quantile_list", type=float, nargs="*", default=[0.35, 0.50, 0.65])
    p.add_argument("--use_cost_budget_modes", type=str, nargs="*", default=["off", "on"])
    p.add_argument("--cost_budget_quantile_list", type=float, nargs="*", default=[0.50, 0.60])
    p.add_argument("--cost_budget_window_steps", type=int, default=20)
    p.add_argument("--max_wait_steps", type=int, default=10)
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--rng_seed", type=int, default=20260402)
    return p.parse_args()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_namespace(base: argparse.Namespace, *, decision_step_group: int, busy_time_quantile: float, use_cost_budget: bool, cost_budget_quantile: float) -> SimpleNamespace:
    # Keep the tuning grids from evaluation script defaults; we only sweep regimes here.
    return SimpleNamespace(
        clean_bank=base.clean_bank,
        attack_bank=base.attack_bank,
        fit_bank=base.fit_bank,
        eval_bank=base.eval_bank,
        output="",
        n_bins=base.n_bins,
        slot_budget_list=list(base.slot_budget_list),
        max_wait_steps=base.max_wait_steps,
        decision_step_group=int(decision_step_group),
        busy_time_quantile=float(busy_time_quantile),
        use_cost_budget=bool(use_cost_budget),
        cost_budget_window_steps=int(base.cost_budget_window_steps),
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
        rng_seed=int(base.rng_seed),
    )


def main() -> None:
    args = parse_args()
    _ensure_parent(args.output)

    regimes: List[Dict[str, object]] = []
    for g in args.decision_step_group_list:
        for btq in args.busy_time_quantile_list:
            for cost_mode in args.use_cost_budget_modes:
                use_cost = str(cost_mode).lower() in {"1", "true", "yes", "on"}
                q_list = args.cost_budget_quantile_list if use_cost else [0.0]
                for cbq in q_list:
                    ns = _build_namespace(
                        args,
                        decision_step_group=int(g),
                        busy_time_quantile=float(btq),
                        use_cost_budget=use_cost,
                        cost_budget_quantile=float(cbq),
                    )
                    result = run_experiment(ns)
                    entry = {
                        "decision_step_group": int(g),
                        "busy_time_quantile": float(btq),
                        "use_cost_budget": bool(use_cost),
                        "cost_budget_quantile": None if not use_cost else float(cbq),
                        "slot_budgets": {},
                    }
                    for slot_budget, payload in result["slot_budget_results"].items():
                        compact = payload["compact"]
                        proposed = compact["proposed_ca_vq_hard"]
                        compare_pool = {
                            k: v for k, v in compact.items()
                            if k != "proposed_ca_vq_hard"
                        }
                        best_non_proposed_name = max(
                            compare_pool,
                            key=lambda k: (
                                compare_pool[k]["weighted_attack_recall_no_backend_fail"]
                                - 0.005 * compare_pool[k]["unnecessary_mtd_count"]
                            ),
                        )
                        best_threshold_name = max(
                            [k for k in compare_pool if "threshold" in k],
                            key=lambda k: (
                                compare_pool[k]["weighted_attack_recall_no_backend_fail"]
                                - 0.005 * compare_pool[k]["unnecessary_mtd_count"]
                            ),
                        )
                        entry["slot_budgets"][str(slot_budget)] = {
                            "proposed": proposed,
                            "best_non_proposed_name": best_non_proposed_name,
                            "best_non_proposed": compare_pool[best_non_proposed_name],
                            "best_threshold_name": best_threshold_name,
                            "best_threshold": compare_pool[best_threshold_name],
                            "delta_vs_best_threshold_recall": round(
                                proposed["weighted_attack_recall_no_backend_fail"] - compare_pool[best_threshold_name]["weighted_attack_recall_no_backend_fail"], 4
                            ),
                            "delta_vs_best_threshold_unnecessary": int(
                                proposed["unnecessary_mtd_count"] - compare_pool[best_threshold_name]["unnecessary_mtd_count"]
                            ),
                            "delta_vs_best_threshold_delay_p95": round(
                                proposed["queue_delay_p95"] - compare_pool[best_threshold_name]["queue_delay_p95"], 4
                            ),
                        }
                    regimes.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(regimes, f, ensure_ascii=False, indent=2)

    print(f"Saved regime sweep json: {args.output}")
    print("\n=== Compact regime ranking by slot budget ===")
    for slot_budget in [str(x) for x in args.slot_budget_list]:
        ranked = sorted(
            regimes,
            key=lambda e: (
                e["slot_budgets"][slot_budget]["proposed"]["weighted_attack_recall_no_backend_fail"]
                - 0.005 * e["slot_budgets"][slot_budget]["proposed"]["unnecessary_mtd_count"]
                - 0.01 * e["slot_budgets"][slot_budget]["proposed"]["queue_delay_p95"]
            ),
            reverse=True,
        )
        top = ranked[:5]
        print(f"\n-- slot_budget={slot_budget} top regimes --")
        for r in top:
            s = r["slot_budgets"][slot_budget]
            print({
                "decision_step_group": r["decision_step_group"],
                "busy_time_quantile": r["busy_time_quantile"],
                "use_cost_budget": r["use_cost_budget"],
                "cost_budget_quantile": r["cost_budget_quantile"],
                "proposed": s["proposed"],
                "best_threshold_name": s["best_threshold_name"],
                "best_threshold": s["best_threshold"],
                "delta_vs_best_threshold_recall": s["delta_vs_best_threshold_recall"],
                "delta_vs_best_threshold_unnecessary": s["delta_vs_best_threshold_unnecessary"],
                "delta_vs_best_threshold_delay_p95": s["delta_vs_best_threshold_delay_p95"],
            })


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from lag_adm_phase3_core import (
    aggregate_multi_holdout_results,
    evaluate_on_holdout_bank,
    prepare_train_val_context,
    save_json,
    tune_lag_admission_plus_phase3_dispatch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lagrangian-admission + phase3-dispatch on a frozen multi-holdout manifest."
    )
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True, help="Path to phase3 multi-holdout manifest.json")
    parser.add_argument("--output", type=str, default="metric/case14/lagadm_phase3_multi_holdout/aggregate_summary.json")
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--threshold_quantiles", type=float, nargs="*", default=[0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--adaptive_gain_scale_list", type=float, nargs="*", default=[0.0, 0.10, 0.20, 0.40])
    parser.add_argument("--consequence_blend_verify", type=float, default=0.70)
    parser.add_argument("--consequence_mode", type=str, default="conditional", choices=["conditional", "expected"])
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


def _load_json(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_threshold_from_eval_compact(eval_compact: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    threshold_names = [
        "threshold_verify_fifo",
        "threshold_ddd_fifo",
        "threshold_expected_consequence_fifo",
        "adaptive_threshold_verify_fifo",
    ]
    best_name = threshold_names[0]
    best = eval_compact[best_name]
    best_key = (
        float(best["weighted_attack_recall_no_backend_fail"]),
        -float(best["unnecessary_mtd_count"]),
        -float(best["queue_delay_p95"]),
    )
    for name in threshold_names[1:]:
        row = eval_compact[name]
        key = (
            float(row["weighted_attack_recall_no_backend_fail"]),
            -float(row["unnecessary_mtd_count"]),
            -float(row["queue_delay_p95"]),
        )
        if key > best_key:
            best_name = name
            best = row
            best_key = key
    return best_name, best


def main() -> int:
    ns = parse_args()
    workdir = Path(os.path.expanduser(ns.workdir)).resolve()
    manifest_path = workdir / ns.manifest if not os.path.isabs(ns.manifest) else Path(ns.manifest)
    manifest = _load_json(manifest_path)

    frozen = manifest["frozen_regime"]
    args = SimpleNamespace(
        clean_bank=str(workdir / manifest["clean_bank"]),
        attack_bank=str(workdir / manifest["attack_bank"]),
        train_bank=str(workdir / manifest["train_bank"]),
        val_bank=str(workdir / manifest["val_bank"]),
        n_bins=int(ns.n_bins),
        slot_budget_list=[int(x) for x in frozen["slot_budget_list"]],
        max_wait_steps=int(frozen["max_wait_steps"]),
        decision_step_group=int(frozen["decision_step_group"]),
        busy_time_quantile=float(frozen["busy_time_quantile"]),
        use_cost_budget=bool(frozen["use_cost_budget"]),
        cost_budget_window_steps=int(frozen.get("cost_budget_window_steps", 20)),
        cost_budget_quantile=float(frozen.get("cost_budget_quantile", 0.6)),
        threshold_quantiles=list(ns.threshold_quantiles),
        adaptive_gain_scale_list=list(ns.adaptive_gain_scale_list),
        consequence_blend_verify=float(ns.consequence_blend_verify),
        consequence_mode=str(ns.consequence_mode),
        vq_v_grid=list(ns.vq_v_grid),
        vq_age_grid=list(ns.vq_age_grid),
        vq_urgency_grid=list(ns.vq_urgency_grid),
        vq_fail_grid=list(ns.vq_fail_grid),
        vq_busy_grid=list(ns.vq_busy_grid),
        vq_cost_grid=list(ns.vq_cost_grid),
        vq_clean_grid=list(ns.vq_clean_grid),
        vq_admission_threshold_grid=list(ns.vq_admission_threshold_grid),
        rng_seed=int(ns.rng_seed),
    )

    ctx = prepare_train_val_context(args)
    tuned_by_slot = tune_lag_admission_plus_phase3_dispatch(args, ctx)

    per_holdout: List[Dict[str, object]] = []
    for holdout in manifest["holdouts"]:
        tag = str(holdout["tag"])
        test_bank = workdir / holdout["test_bank"]
        lag_eval = evaluate_on_holdout_bank(
            test_bank=str(test_bank),
            ctx=ctx,
            tuned_by_slot=tuned_by_slot,
            decision_step_group=int(args.decision_step_group),
            consequence_blend_verify=float(args.consequence_blend_verify),
            consequence_mode=str(args.consequence_mode),
        )
        phase3_summary_path = workdir / holdout["result_summary"]
        phase3_summary = _load_json(phase3_summary_path)

        rec = {
            "tag": tag,
            "seed_base": int(holdout["seed_base"]),
            "start_offset": int(holdout["start_offset"]),
            "test_bank": str(holdout["test_bank"]),
            "slot_budget_results": {},
        }
        for slot_key, slot_res in lag_eval["slot_budget_results"].items():
            slot_eval_compact = phase3_summary["slot_budget_results"][slot_key]["eval_compact"]
            best_thr_name, best_thr = _best_threshold_from_eval_compact(slot_eval_compact)
            rec["slot_budget_results"][slot_key] = {
                "lag_adm_phase3_dispatch": slot_res["lag_adm_phase3_dispatch"],
                "best_threshold_name": best_thr_name,
                "best_threshold": {
                    "weighted_attack_recall_no_backend_fail": float(best_thr["weighted_attack_recall_no_backend_fail"]),
                    "unnecessary_mtd_count": int(best_thr["unnecessary_mtd_count"]),
                    "queue_delay_p95": float(best_thr["queue_delay_p95"]),
                    "average_service_cost_per_step": float(best_thr["average_service_cost_per_step"]),
                    "pred_expected_consequence_served_ratio": float(best_thr["pred_expected_consequence_served_ratio"]),
                },
                "phase3_proposed": {
                    "weighted_attack_recall_no_backend_fail": float(slot_eval_compact["proposed_ca_vq_hard"]["weighted_attack_recall_no_backend_fail"]),
                    "unnecessary_mtd_count": int(slot_eval_compact["proposed_ca_vq_hard"]["unnecessary_mtd_count"]),
                    "queue_delay_p95": float(slot_eval_compact["proposed_ca_vq_hard"]["queue_delay_p95"]),
                    "average_service_cost_per_step": float(slot_eval_compact["proposed_ca_vq_hard"]["average_service_cost_per_step"]),
                    "pred_expected_consequence_served_ratio": float(slot_eval_compact["proposed_ca_vq_hard"]["pred_expected_consequence_served_ratio"]),
                },
                "topk_expected_consequence": {
                    "weighted_attack_recall_no_backend_fail": float(slot_eval_compact["topk_expected_consequence"]["weighted_attack_recall_no_backend_fail"]),
                    "unnecessary_mtd_count": int(slot_eval_compact["topk_expected_consequence"]["unnecessary_mtd_count"]),
                    "queue_delay_p95": float(slot_eval_compact["topk_expected_consequence"]["queue_delay_p95"]),
                    "average_service_cost_per_step": float(slot_eval_compact["topk_expected_consequence"]["average_service_cost_per_step"]),
                    "pred_expected_consequence_served_ratio": float(slot_eval_compact["topk_expected_consequence"]["pred_expected_consequence_served_ratio"]),
                },
            }
        per_holdout.append(rec)

    aggregate = {
        "method": "lagrangian_admission_plus_phase3_dispatch",
        "manifest": manifest,
        "config": {
            "decision_step_group": int(args.decision_step_group),
            "busy_time_quantile": float(args.busy_time_quantile),
            "use_cost_budget": bool(args.use_cost_budget),
            "cost_budget_window_steps": int(args.cost_budget_window_steps),
            "cost_budget_quantile": float(args.cost_budget_quantile),
            "slot_budget_list": [int(x) for x in args.slot_budget_list],
            "max_wait_steps": int(args.max_wait_steps),
            "consequence_blend_verify": float(args.consequence_blend_verify),
            "consequence_mode": str(args.consequence_mode),
        },
        "environment": {
            "busy_time_unit": float(ctx["busy_time_unit"]),
            "train_job_stats": ctx["train_job_stats"],
            "val_job_stats": ctx["val_job_stats"],
            "train_arrival_diagnostics": ctx["train_arrival_diag"],
            "val_arrival_diagnostics": ctx["val_arrival_diag"],
        },
        "tuned_by_slot": tuned_by_slot,
        "n_holdouts": len(per_holdout),
        "per_holdout_results": per_holdout,
        "slot_budget_aggregates": aggregate_multi_holdout_results(per_holdout),
    }
    output_path = workdir / ns.output if not os.path.isabs(ns.output) else Path(ns.output)
    save_json(output_path, aggregate)
    print(f"Saved lag-admission + phase3-dispatch aggregate summary: {output_path}")
    print(json.dumps({"output": str(output_path), "slot_budget_aggregates_keys": sorted(aggregate['slot_budget_aggregates'].keys(), key=int)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

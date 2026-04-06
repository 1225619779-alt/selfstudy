from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from carkm_core import (
    CARKMConfig,
    compact_eval_from_phase3_summary,
    ensure_parent,
    prepare_jobs,
    simulate_carkm,
    summarize_policy_stats,
    tune_carkm,
    tune_threshold_reference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CA-RK-M on existing multi-holdout manifest and aggregate against phase3 baselines.")
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--manifest", type=str, default="metric/case14/phase3_multi_holdout/manifest.json")
    p.add_argument("--output", type=str, default="metric/case14/carkm_multi_holdout/aggregate_summary.json")
    p.add_argument("--decision_step_group", type=int, default=None, help="Override manifest frozen_regime decision_step_group")
    p.add_argument("--busy_time_quantile", type=float, default=None, help="Override manifest frozen_regime busy_time_quantile")
    p.add_argument("--use_cost_budget", action="store_true", help="Force cost budget on even if frozen_regime says off")
    p.add_argument("--consequence_blend_verify", type=float, default=0.7)
    p.add_argument("--consequence_mode", type=str, default="conditional", choices=["conditional", "expected"])
    p.add_argument("--max_wait_steps", type=int, default=None)
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_threshold(compact: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    names = [n for n in compact.keys() if n.startswith("threshold_") or n == "adaptive_threshold_verify_fifo"]
    best_name = max(
        names,
        key=lambda n: (
            compact[n]["weighted_attack_recall_no_backend_fail"],
            -compact[n]["unnecessary_mtd_count"],
            -compact[n]["average_service_cost_per_step"],
        ),
    )
    return best_name, compact[best_name]


def main() -> int:
    ns = parse_args()
    workdir = Path(os.path.expanduser(ns.workdir)).resolve()
    manifest_path = (workdir / ns.manifest).resolve() if not os.path.isabs(ns.manifest) else Path(ns.manifest)
    output_path = (workdir / ns.output).resolve() if not os.path.isabs(ns.output) else Path(ns.output)
    ensure_parent(str(output_path))

    manifest = _load_json(manifest_path)
    frozen = dict(manifest.get("frozen_regime", {}))
    decision_step_group = int(ns.decision_step_group if ns.decision_step_group is not None else frozen.get("decision_step_group", 1))
    busy_time_quantile = float(ns.busy_time_quantile if ns.busy_time_quantile is not None else frozen.get("busy_time_quantile", 0.65))
    use_cost_budget = bool(ns.use_cost_budget or frozen.get("use_cost_budget", False))
    cost_budget_window_steps = int(frozen.get("cost_budget_window_steps", 20))
    cost_budget_quantile = float(frozen.get("cost_budget_quantile", 0.6))
    slot_budget_list = [int(x) for x in frozen.get("slot_budget_list", [1, 2])]
    max_wait_steps = int(ns.max_wait_steps if ns.max_wait_steps is not None else frozen.get("max_wait_steps", 10))

    # absolute data paths from manifest root
    clean_bank = workdir / str(manifest["clean_bank"])
    attack_bank = workdir / str(manifest["attack_bank"])
    train_bank = workdir / str(manifest["train_bank"])
    tune_bank = workdir / str(manifest["val_bank"])
    holdouts = list(manifest["holdouts"])
    if not holdouts:
        raise ValueError("manifest holdouts empty")

    # tune once using the first holdout bank just to construct the shared fitted models and tune-set jobs
    first_test_bank = workdir / str(holdouts[0]["test_bank"])
    prep0 = prepare_jobs(
        clean_bank=str(clean_bank),
        attack_bank=str(attack_bank),
        train_bank=str(train_bank),
        tune_bank=str(tune_bank),
        eval_bank=str(first_test_bank),
        n_bins=int(ns.n_bins),
        decision_step_group=decision_step_group,
        busy_time_quantile=busy_time_quantile,
        consequence_blend_verify=float(ns.consequence_blend_verify),
        consequence_mode=str(ns.consequence_mode),
    )

    tuned_by_slot: Dict[str, Dict[str, object]] = {}
    for slot_budget in slot_budget_list:
        window_cost_budget = None
        if use_cost_budget:
            per_step_cost = np.zeros(int(prep0["train_steps"]), dtype=float)
            for job in prep0["train_jobs"]:
                step = int(job.arrival_step)
                if 0 <= step < len(per_step_cost):
                    per_step_cost[step] += float(job.actual_service_cost)
            # rolling quantile over training bank
            c = np.concatenate([[0.0], np.cumsum(per_step_cost)])
            rolling = np.empty_like(per_step_cost)
            for i in range(len(per_step_cost)):
                j0 = max(0, i - int(cost_budget_window_steps) + 1)
                rolling[i] = c[i + 1] - c[j0]
            window_cost_budget = float(np.quantile(rolling, cost_budget_quantile))

        threshold_ref = tune_threshold_reference(
            prep0["tune_jobs"],
            total_steps_tune=int(prep0["tune_steps"]),
            slot_budget=int(slot_budget),
            max_wait_steps=max_wait_steps,
            window_cost_budget=window_cost_budget,
            cost_budget_window_steps=cost_budget_window_steps if use_cost_budget else 0,
        )
        best_cfg, tuning_payload = tune_carkm(
            prep0["tune_jobs"],
            total_steps_tune=int(prep0["tune_steps"]),
            slot_budget=int(slot_budget),
            max_wait_steps=max_wait_steps,
            threshold_reference=threshold_ref,
            mean_pred_busy_steps=float(np.mean([j.pred_busy_steps for j in prep0["train_jobs"]])) if prep0["train_jobs"] else 1.0,
            mean_pred_service_cost=float(np.mean([j.pred_service_cost for j in prep0["train_jobs"]])) if prep0["train_jobs"] else 1.0,
            mean_pred_expected_consequence=float(np.mean([j.pred_expected_consequence for j in prep0["train_jobs"]])) if prep0["train_jobs"] else 1.0,
            use_cost_budget=use_cost_budget,
            window_cost_budget=window_cost_budget,
            cost_budget_window_steps=cost_budget_window_steps,
        )
        tuned_by_slot[str(slot_budget)] = {
            "config": best_cfg,
            "threshold_reference": threshold_ref,
            "tuning_payload": tuning_payload,
            "window_cost_budget": window_cost_budget,
        }

    per_holdout_results: List[Dict[str, object]] = []
    for h in holdouts:
        tag = str(h["tag"])
        test_bank = workdir / str(h["test_bank"])
        baseline_summary = workdir / str(h["result_summary"])
        prep = prepare_jobs(
            clean_bank=str(clean_bank),
            attack_bank=str(attack_bank),
            train_bank=str(train_bank),
            tune_bank=str(tune_bank),
            eval_bank=str(test_bank),
            n_bins=int(ns.n_bins),
            decision_step_group=decision_step_group,
            busy_time_quantile=busy_time_quantile,
            consequence_blend_verify=float(ns.consequence_blend_verify),
            consequence_mode=str(ns.consequence_mode),
        )
        slot_payload: Dict[str, object] = {}
        for slot_budget in slot_budget_list:
            slot = str(slot_budget)
            tuned = tuned_by_slot[slot]
            cfg: CARKMConfig = tuned["config"]
            # refresh normalization with current training jobs from this prep (same in practice)
            cfg_eval = CARKMConfig(**cfg.__dict__)
            cfg_eval.slot_budget = int(slot_budget)
            cfg_eval.max_wait_steps = int(max_wait_steps)
            cfg_eval.mean_pred_busy_steps = float(np.mean([j.pred_busy_steps for j in prep["train_jobs"]])) if prep["train_jobs"] else 1.0
            cfg_eval.mean_pred_service_cost = float(np.mean([j.pred_service_cost for j in prep["train_jobs"]])) if prep["train_jobs"] else 1.0
            cfg_eval.mean_pred_expected_consequence = float(np.mean([j.pred_expected_consequence for j in prep["train_jobs"]])) if prep["train_jobs"] else 1.0
            cfg_eval.use_cost_budget = bool(use_cost_budget)
            cfg_eval.window_cost_budget = tuned["window_cost_budget"]
            cfg_eval.cost_budget_window_steps = int(cost_budget_window_steps if use_cost_budget else 0)
            res = simulate_carkm(prep["eval_jobs"], total_steps=int(prep["eval_steps"]), cfg=cfg_eval)
            base_compact = compact_eval_from_phase3_summary(str(baseline_summary), int(slot_budget))
            best_thr_name, best_thr = _best_threshold(base_compact)
            proposed_old = dict(base_compact["proposed_ca_vq_hard"])
            topk_ec = dict(base_compact["topk_expected_consequence"])
            slot_payload[slot] = {
                "carkm_eval_compact": dict(res["summary"]),
                "carkm_tuning": dict(res["tuning"]),
                "best_threshold_name": best_thr_name,
                "best_threshold": best_thr,
                "phase3_proposed": proposed_old,
                "topk_expected_consequence": topk_ec,
            }
        per_holdout_results.append({
            "tag": tag,
            "seed_base": h.get("seed_base"),
            "start_offset": h.get("start_offset"),
            "test_bank": str(h["test_bank"]),
            "slot_budget_results": slot_payload,
        })

    # aggregate
    aggregate: Dict[str, object] = {
        "method": "CA-RK-M",
        "manifest": manifest,
        "config": {
            "decision_step_group": decision_step_group,
            "busy_time_quantile": busy_time_quantile,
            "use_cost_budget": use_cost_budget,
            "cost_budget_window_steps": cost_budget_window_steps,
            "cost_budget_quantile": cost_budget_quantile if use_cost_budget else None,
            "slot_budget_list": slot_budget_list,
            "max_wait_steps": max_wait_steps,
            "consequence_blend_verify": float(ns.consequence_blend_verify),
            "consequence_mode": str(ns.consequence_mode),
        },
        "tuned_by_slot": {},
        "n_holdouts": len(per_holdout_results),
        "per_holdout_results": per_holdout_results,
        "slot_budget_aggregates": {},
    }
    for slot_budget in slot_budget_list:
        slot = str(slot_budget)
        tuned = tuned_by_slot[slot]
        aggregate["tuned_by_slot"][slot] = {
            "window_cost_budget": tuned["window_cost_budget"],
            "config": dict(tuned["config"].__dict__),
            "threshold_reference": tuned["threshold_reference"],
            "tuning_payload": tuned["tuning_payload"],
        }
        carkm_vals = [r["slot_budget_results"][slot]["carkm_eval_compact"] for r in per_holdout_results]
        phase3_vals = [r["slot_budget_results"][slot]["phase3_proposed"] for r in per_holdout_results]
        topk_vals = [r["slot_budget_results"][slot]["topk_expected_consequence"] for r in per_holdout_results]
        best_thr_vals = [r["slot_budget_results"][slot]["best_threshold"] for r in per_holdout_results]
        best_thr_names = [r["slot_budget_results"][slot]["best_threshold_name"] for r in per_holdout_results]

        def paired_delta(a_key: str, b_key: str, arr_a: List[Dict[str, float]], arr_b: List[Dict[str, float]]) -> Dict[str, float]:
            x = np.asarray([float(a[a_key]) - float(b[b_key]) for a, b in zip(arr_a, arr_b)], dtype=float)
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        aggregate["slot_budget_aggregates"][slot] = {
            "policy_stats": {
                "carkm": summarize_policy_stats(carkm_vals),
                "phase3_proposed": summarize_policy_stats(phase3_vals),
                "topk_expected_consequence": summarize_policy_stats(topk_vals),
            },
            "paired_stats": {
                "carkm_vs_best_threshold": {
                    "delta_recall": paired_delta("weighted_attack_recall_no_backend_fail", "weighted_attack_recall_no_backend_fail", carkm_vals, best_thr_vals),
                    "delta_unnecessary": paired_delta("unnecessary_mtd_count", "unnecessary_mtd_count", carkm_vals, best_thr_vals),
                    "delta_delay_p95": paired_delta("queue_delay_p95", "queue_delay_p95", carkm_vals, best_thr_vals),
                    "delta_cost_per_step": paired_delta("average_service_cost_per_step", "average_service_cost_per_step", carkm_vals, best_thr_vals),
                    "carkm_wins_on_recall": int(sum(float(a["weighted_attack_recall_no_backend_fail"]) > float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(carkm_vals, best_thr_vals))),
                    "carkm_no_worse_unnecessary": int(sum(float(a["unnecessary_mtd_count"]) <= float(b["unnecessary_mtd_count"]) for a, b in zip(carkm_vals, best_thr_vals))),
                },
                "carkm_vs_phase3_proposed": {
                    "delta_recall": paired_delta("weighted_attack_recall_no_backend_fail", "weighted_attack_recall_no_backend_fail", carkm_vals, phase3_vals),
                    "delta_unnecessary": paired_delta("unnecessary_mtd_count", "unnecessary_mtd_count", carkm_vals, phase3_vals),
                    "delta_delay_p95": paired_delta("queue_delay_p95", "queue_delay_p95", carkm_vals, phase3_vals),
                    "delta_cost_per_step": paired_delta("average_service_cost_per_step", "average_service_cost_per_step", carkm_vals, phase3_vals),
                    "carkm_wins_on_recall": int(sum(float(a["weighted_attack_recall_no_backend_fail"]) > float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(carkm_vals, phase3_vals))),
                    "carkm_lower_unnecessary": int(sum(float(a["unnecessary_mtd_count"]) < float(b["unnecessary_mtd_count"]) for a, b in zip(carkm_vals, phase3_vals))),
                },
                "carkm_vs_topk_expected": {
                    "delta_recall": paired_delta("weighted_attack_recall_no_backend_fail", "weighted_attack_recall_no_backend_fail", carkm_vals, topk_vals),
                    "delta_unnecessary": paired_delta("unnecessary_mtd_count", "unnecessary_mtd_count", carkm_vals, topk_vals),
                    "delta_delay_p95": paired_delta("queue_delay_p95", "queue_delay_p95", carkm_vals, topk_vals),
                    "delta_cost_per_step": paired_delta("average_service_cost_per_step", "average_service_cost_per_step", carkm_vals, topk_vals),
                    "carkm_lower_unnecessary": int(sum(float(a["unnecessary_mtd_count"]) < float(b["unnecessary_mtd_count"]) for a, b in zip(carkm_vals, topk_vals))),
                    "carkm_lower_cost": int(sum(float(a["average_service_cost_per_step"]) < float(b["average_service_cost_per_step"]) for a, b in zip(carkm_vals, topk_vals))),
                },
            },
            "best_threshold_frequency": {name: int(best_thr_names.count(name)) for name in sorted(set(best_thr_names))},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved CA-RK-M aggregate summary: {output_path}")
    print(json.dumps({
        "output": str(output_path),
        "slot_budget_aggregates_keys": list(aggregate["slot_budget_aggregates"].keys()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

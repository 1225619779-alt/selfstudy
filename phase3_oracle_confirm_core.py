
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from evaluation_budget_scheduler_phase3 import (
    _aggregate_arrival_steps,
    _busy_time_unit_from_fit,
    _job_stats,
)
from phase3_oracle_family_core import (
    DEFAULT_VARIANTS,
    _build_jobs_for_variant,
    _extract_baselines,
    _load_json,
    _policy_compact,
    _simulate_with_tuned_phase3,
    _to_jsonable,
)


def _variant_payload(variant) -> Dict[str, object]:
    return {
        "name": str(variant.name),
        "mode": str(variant.mode),
        "description": str(variant.description),
        "clean_scale": float(getattr(variant, "clean_scale", 0.0)),
        "protected_blend": float(getattr(variant, "protected_blend", 0.0)),
    }


def _formal_rule_summary(slot_budget_aggregates: Dict[str, object]) -> Dict[str, object]:
    per_slot = {}
    threshold_ok = True
    efficiency_ok = True
    recall_ok = True

    for slot, payload in slot_budget_aggregates.items():
        wins_vs_best_threshold = int(payload["paired_stats"]["oracle_vs_best_threshold"]["wins_on_recall"])
        oracle_stats = payload["policy_stats"]["phase3_oracle_upgrade"]
        phase3_stats = payload["policy_stats"]["phase3_proposed"]

        mean_recall_oracle = float(oracle_stats["weighted_attack_recall_no_backend_fail"]["mean"])
        mean_recall_phase3 = float(phase3_stats["weighted_attack_recall_no_backend_fail"]["mean"])
        mean_unnecessary_oracle = float(oracle_stats["unnecessary_mtd_count"]["mean"])
        mean_unnecessary_phase3 = float(phase3_stats["unnecessary_mtd_count"]["mean"])
        mean_cost_oracle = float(oracle_stats["average_service_cost_per_step"]["mean"])
        mean_cost_phase3 = float(phase3_stats["average_service_cost_per_step"]["mean"])

        slot_threshold_ok = wins_vs_best_threshold >= 7
        slot_efficiency_ok = (
            mean_recall_oracle >= mean_recall_phase3 - 0.01
            and mean_unnecessary_oracle < mean_unnecessary_phase3
            and mean_cost_oracle <= mean_cost_phase3
        )
        slot_recall_ok = (
            mean_recall_oracle > mean_recall_phase3
            and mean_unnecessary_oracle <= mean_unnecessary_phase3 + 1.0
        )

        per_slot[str(slot)] = {
            "wins_vs_best_threshold": wins_vs_best_threshold,
            "passes_threshold_baseline": bool(slot_threshold_ok),
            "passes_efficiency_type": bool(slot_efficiency_ok),
            "passes_recall_type": bool(slot_recall_ok),
            "mean_recall_oracle": mean_recall_oracle,
            "mean_recall_phase3": mean_recall_phase3,
            "mean_unnecessary_oracle": mean_unnecessary_oracle,
            "mean_unnecessary_phase3": mean_unnecessary_phase3,
            "mean_cost_oracle": mean_cost_oracle,
            "mean_cost_phase3": mean_cost_phase3,
        }

        threshold_ok = threshold_ok and slot_threshold_ok
        efficiency_ok = efficiency_ok and slot_efficiency_ok
        recall_ok = recall_ok and slot_recall_ok

    overall = threshold_ok and (efficiency_ok or recall_ok)
    overall_mode = None
    if overall:
        if recall_ok:
            overall_mode = "recall_type"
        elif efficiency_ok:
            overall_mode = "efficiency_type"

    return {
        "per_slot": per_slot,
        "passes_threshold_baseline_all_slots": bool(threshold_ok),
        "passes_efficiency_type_all_slots": bool(efficiency_ok),
        "passes_recall_type_all_slots": bool(recall_ok),
        "passes_overall_rule": bool(overall),
        "overall_mode": overall_mode,
    }


def run_phase3_oracle_confirm(
    confirm_manifest_path: str,
    dev_screen_summary_path: str,
    output_dir: str,
) -> Dict[str, object]:
    manifest = _load_json(confirm_manifest_path)
    dev_screen = _load_json(dev_screen_summary_path)

    winner_name = str(dev_screen["selection"]["winner_variant"])
    winner_variant = next(v for v in DEFAULT_VARIANTS if v.name == winner_name)
    winner_payload = dev_screen["selection"]["winner_payload"]

    workdir = Path(manifest["workdir"])
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        clean_bank=str(workdir / manifest["clean_bank"]),
        attack_bank=str(workdir / manifest["attack_bank"]),
        train_bank=str(workdir / manifest["train_bank"]),
        n_bins=20,
        max_wait_steps=int(manifest["frozen_regime"]["max_wait_steps"]),
        decision_step_group=int(manifest["frozen_regime"]["decision_step_group"]),
        busy_time_quantile=float(manifest["frozen_regime"]["busy_time_quantile"]),
        consequence_blend_verify=0.70,
        consequence_mode="conditional",
        rng_seed=20260402,
    )

    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.train_bank), int(args.decision_step_group))
    posterior_verify = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins)
    posterior_ddd = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="ddd_loss_alarm", n_bins=args.n_bins)
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))

    # winner variant in our current winning screen is not help_gain; keep generic anyway
    gain_bundle_by_variant = {}
    if winner_variant.mode == "help_gain":
        from phase3_oracle_family_core import _fit_net_gain_models  # local import to avoid unused dependency if not needed

        gain_bundle_by_variant[winner_variant.name] = _fit_net_gain_models(
            arrays_train,
            clean_scale=float(winner_variant.clean_scale),
            n_bins=int(args.n_bins),
        )

    jobs_train_winner, _, train_oracle_diag = _build_jobs_for_variant(
        arrays_bank=arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=winner_variant,
        gain_bundle_by_variant=gain_bundle_by_variant,
        severity_blend_verify=float(args.consequence_blend_verify),
        busy_time_unit=busy_time_unit,
    )
    train_stats = _job_stats(jobs_train_winner)

    results: Dict[str, object] = {
        "method": "phase3_oracle_upgrade_confirm",
        "confirm_manifest": manifest,
        "dev_screen_summary_path": str(Path(dev_screen_summary_path).resolve()),
        "winner_variant": _variant_payload(winner_variant),
        "winner_dev_selection": {
            "winner_variant": winner_name,
            "winner_joint_val_delta_objective": float(dev_screen["selection"]["winner_joint_val_delta_objective"]),
            "winner_joint_val_delta_recall": float(dev_screen["selection"]["winner_joint_val_delta_recall"]),
        },
        "winner_train_oracle_stats": train_oracle_diag,
        "winner_train_job_stats": train_stats,
        "n_holdouts": int(len(manifest["holdouts"])),
        "per_holdout_results": [],
        "slot_budget_aggregates": {},
        "formal_decision_summary": {},
    }

    per_slot_records: Dict[str, List[Dict[str, object]]] = {
        str(slot): [] for slot in manifest["frozen_regime"]["slot_budget_list"]
    }

    for hold in manifest["holdouts"]:
        row = {
            "tag": hold["tag"],
            "family_tag": hold.get("family_tag"),
            "schedule": hold.get("schedule"),
            "seed_base": hold["seed_base"],
            "start_offset": hold["start_offset"],
            "test_bank": hold["test_bank"],
            "slot_budget_results": {},
        }

        summary_json = _load_json(workdir / hold["result_summary"])
        arrays_test = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(str(workdir / hold["test_bank"])), int(args.decision_step_group))
        jobs_test, total_steps_test, test_oracle_diag = _build_jobs_for_variant(
            arrays_bank=arrays_test,
            arrays_train=arrays_train,
            posterior_verify=posterior_verify,
            posterior_ddd=posterior_ddd,
            service_models=service_models,
            severity_models=severity_models,
            variant=winner_variant,
            gain_bundle_by_variant=gain_bundle_by_variant,
            severity_blend_verify=float(args.consequence_blend_verify),
            busy_time_unit=busy_time_unit,
        )

        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            tuned = winner_payload["by_slot"][str(slot_budget)]["tuned_config"]
            eval_res = _simulate_with_tuned_phase3(
                jobs_test,
                total_steps=total_steps_test,
                slot_budget=slot_budget,
                tuned_config=tuned,
                train_stats=train_stats,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
            )
            baselines = _extract_baselines(summary_json, slot_budget)
            compact = _policy_compact(eval_res["summary"])
            payload = {
                "phase3_oracle_upgrade": compact,
                "winner_tuned_config": tuned,
                "winner_oracle_stats": test_oracle_diag,
                **baselines,
            }
            row["slot_budget_results"][str(slot_budget)] = payload
            per_slot_records[str(slot_budget)].append(payload)

        results["per_holdout_results"].append(row)

    for slot, rows in per_slot_records.items():
        policies = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]
        slot_payload: Dict[str, object] = {"policy_stats": {}, "paired_stats": {}, "best_threshold_frequency": {}}

        for pol in policies:
            slot_payload["policy_stats"][pol] = {}
            for metric in [
                "weighted_attack_recall_no_backend_fail",
                "unnecessary_mtd_count",
                "queue_delay_p95",
                "average_service_cost_per_step",
                "pred_expected_consequence_served_ratio",
            ]:
                vals = [float(r[pol][metric]) for r in rows]
                slot_payload["policy_stats"][pol][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

        best_names = [str(r["best_threshold_name"]) for r in rows]
        for name in best_names:
            slot_payload["best_threshold_frequency"][name] = slot_payload["best_threshold_frequency"].get(name, 0) + 1

        def _paired(a: str, b: str, tag: str) -> None:
            delta_rec = [float(r[a]["weighted_attack_recall_no_backend_fail"]) - float(r[b]["weighted_attack_recall_no_backend_fail"]) for r in rows]
            delta_un = [float(r[a]["unnecessary_mtd_count"]) - float(r[b]["unnecessary_mtd_count"]) for r in rows]
            delta_del = [float(r[a]["queue_delay_p95"]) - float(r[b]["queue_delay_p95"]) for r in rows]
            delta_cost = [float(r[a]["average_service_cost_per_step"]) - float(r[b]["average_service_cost_per_step"]) for r in rows]
            slot_payload["paired_stats"][tag] = {
                "delta_recall": {
                    "mean": float(np.mean(delta_rec)),
                    "std": float(np.std(delta_rec)),
                    "min": float(np.min(delta_rec)),
                    "max": float(np.max(delta_rec)),
                },
                "delta_unnecessary": {
                    "mean": float(np.mean(delta_un)),
                    "std": float(np.std(delta_un)),
                    "min": float(np.min(delta_un)),
                    "max": float(np.max(delta_un)),
                },
                "delta_delay_p95": {
                    "mean": float(np.mean(delta_del)),
                    "std": float(np.std(delta_del)),
                    "min": float(np.min(delta_del)),
                    "max": float(np.max(delta_del)),
                },
                "delta_cost_per_step": {
                    "mean": float(np.mean(delta_cost)),
                    "std": float(np.std(delta_cost)),
                    "min": float(np.min(delta_cost)),
                    "max": float(np.max(delta_cost)),
                },
                "wins_on_recall": int(np.sum(np.asarray(delta_rec) > 0)),
                "lower_unnecessary": int(np.sum(np.asarray(delta_un) < 0)),
            }

        _paired("phase3_oracle_upgrade", "phase3_proposed", "oracle_vs_phase3")
        _paired("phase3_oracle_upgrade", "best_threshold", "oracle_vs_best_threshold")
        _paired("phase3_oracle_upgrade", "topk_expected_consequence", "oracle_vs_topk_expected")
        results["slot_budget_aggregates"][str(slot)] = slot_payload

    results["formal_decision_summary"] = _formal_rule_summary(results["slot_budget_aggregates"])

    aggregate_path = out_root / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    return {"aggregate_summary_path": str(aggregate_path.resolve())}

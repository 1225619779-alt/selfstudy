from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

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
    _fit_net_gain_models,
    _load_json,
    _policy_compact,
    _simulate_with_tuned_phase3,
    _to_jsonable,
)

METRICS: Sequence[str] = (
    "weighted_attack_recall_no_backend_fail",
    "unnecessary_mtd_count",
    "queue_delay_p95",
    "average_service_cost_per_step",
    "pred_expected_consequence_served_ratio",
)


def _variant_payload(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if name == "phase3_reference":
        return {
            "name": "phase3_reference",
            "mode": "baseline_phase3",
            "description": "original phase3 consequence / dispatch shell",
        }
    variant = payload.get("variant")
    if variant is not None:
        return {
            "name": str(variant["name"]),
            "mode": str(variant["mode"]),
            "description": str(variant["description"]),
            "clean_scale": float(variant.get("clean_scale", 0.0)),
            "protected_blend": float(variant.get("protected_blend", 0.0)),
        }
    return {"name": name}


def _stats(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _policy_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    return {metric: _stats([float(r[key][metric]) for r in rows]) for metric in METRICS}


def _paired(rows: List[Dict[str, Any]], a_key: str, b_key: str) -> Dict[str, Any]:
    def _delta(metric: str) -> List[float]:
        return [float(r[a_key][metric]) - float(r[b_key][metric]) for r in rows]

    delta_rec = np.asarray(_delta("weighted_attack_recall_no_backend_fail"), dtype=float)
    delta_un = np.asarray(_delta("unnecessary_mtd_count"), dtype=float)
    delta_del = np.asarray(_delta("queue_delay_p95"), dtype=float)
    delta_cost = np.asarray(_delta("average_service_cost_per_step"), dtype=float)
    delta_ratio = np.asarray(_delta("pred_expected_consequence_served_ratio"), dtype=float)

    return {
        "delta_recall": _stats(delta_rec.tolist()),
        "delta_unnecessary": _stats(delta_un.tolist()),
        "delta_delay_p95": _stats(delta_del.tolist()),
        "delta_cost_per_step": _stats(delta_cost.tolist()),
        "delta_pred_expected_consequence_served_ratio": _stats(delta_ratio.tolist()),
        "wins_on_recall": int(np.sum(delta_rec > 0)),
        "ties_on_recall": int(np.sum(delta_rec == 0)),
        "lower_unnecessary": int(np.sum(delta_un < 0)),
        "lower_delay": int(np.sum(delta_del < 0)),
        "lower_cost": int(np.sum(delta_cost < 0)),
        "higher_served_ratio": int(np.sum(delta_ratio > 0)),
    }


def _resolve_variant_payloads(dev_screen: Dict[str, Any], variant_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in variant_names:
        if name == "phase3_reference":
            out[name] = {
                "variant": {"name": "phase3_reference", "mode": "baseline_phase3", "description": "original phase3 consequence / dispatch shell"},
                "by_slot": {
                    str(slot): {"tuned_config": payload["config"]}
                    for slot, payload in dev_screen["phase3_reference_by_slot"].items()
                },
            }
        else:
            if name not in dev_screen["variants"]:
                raise KeyError(f"Variant {name!r} not found in dev_screen['variants']")
            out[name] = dev_screen["variants"][name]
    return out


def run_phase3_oracle_ablation(
    manifest_path: str,
    dev_screen_summary_path: str,
    output_dir: str,
    variant_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    dev_screen = _load_json(dev_screen_summary_path)
    variant_names = list(variant_names or ["phase3_reference", "oracle_fused_ec", "oracle_protected_ec"])
    variant_payloads = _resolve_variant_payloads(dev_screen, variant_names)

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

    gain_bundle_by_variant: Dict[str, Dict[str, Any]] = {}
    for spec in DEFAULT_VARIANTS:
        if spec.mode == "help_gain" and spec.name in variant_names:
            gain_bundle_by_variant[spec.name] = _fit_net_gain_models(
                arrays_train,
                clean_scale=float(spec.clean_scale),
                n_bins=int(args.n_bins),
            )

    spec_map = {spec.name: spec for spec in DEFAULT_VARIANTS}
    train_diag_by_variant: Dict[str, Dict[str, Any]] = {}
    train_stats_by_variant: Dict[str, Dict[str, Any]] = {}
    jobs_train_by_variant: Dict[str, List[Any]] = {}

    for name in variant_names:
        spec = None if name == "phase3_reference" else spec_map[name]
        jobs_train, _total_steps_train, train_diag = _build_jobs_for_variant(
            arrays_bank=arrays_train,
            arrays_train=arrays_train,
            posterior_verify=posterior_verify,
            posterior_ddd=posterior_ddd,
            service_models=service_models,
            severity_models=severity_models,
            variant=spec,
            gain_bundle_by_variant=gain_bundle_by_variant,
            severity_blend_verify=float(args.consequence_blend_verify),
            busy_time_unit=busy_time_unit,
        )
        jobs_train_by_variant[name] = jobs_train
        train_diag_by_variant[name] = train_diag
        train_stats_by_variant[name] = _job_stats(jobs_train)

    results: Dict[str, Any] = {
        "method": "phase3_oracle_fixed_ablation",
        "manifest": manifest,
        "dev_screen_summary_path": str(Path(dev_screen_summary_path).resolve()),
        "variant_order": list(variant_names),
        "variant_payloads": {name: _variant_payload(name, variant_payloads[name]) for name in variant_names},
        "train_oracle_stats": {name: train_diag_by_variant[name] for name in variant_names},
        "train_job_stats": {name: train_stats_by_variant[name] for name in variant_names},
        "n_holdouts": int(len(manifest["holdouts"])),
        "per_holdout_results": [],
        "slot_budget_aggregates": {},
        "paired_focus": {},
    }

    per_slot_records: Dict[str, List[Dict[str, Any]]] = {str(slot): [] for slot in manifest["frozen_regime"]["slot_budget_list"]}

    for hold in manifest["holdouts"]:
        summary_json = _load_json(workdir / hold["result_summary"])
        arrays_test = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(str(workdir / hold["test_bank"])), int(args.decision_step_group))

        row: Dict[str, Any] = {
            "tag": hold["tag"],
            "family_tag": hold.get("family_tag"),
            "schedule": hold.get("schedule"),
            "seed_base": hold["seed_base"],
            "start_offset": hold["start_offset"],
            "test_bank": hold["test_bank"],
            "slot_budget_results": {},
        }

        jobs_test_by_variant: Dict[str, List[Any]] = {}
        oracle_diag_by_variant: Dict[str, Dict[str, Any]] = {}
        total_steps_test = None
        for name in variant_names:
            spec = None if name == "phase3_reference" else spec_map[name]
            jobs_test, total_steps_variant, test_oracle_diag = _build_jobs_for_variant(
                arrays_bank=arrays_test,
                arrays_train=arrays_train,
                posterior_verify=posterior_verify,
                posterior_ddd=posterior_ddd,
                service_models=service_models,
                severity_models=severity_models,
                variant=spec,
                gain_bundle_by_variant=gain_bundle_by_variant,
                severity_blend_verify=float(args.consequence_blend_verify),
                busy_time_unit=busy_time_unit,
            )
            jobs_test_by_variant[name] = jobs_test
            oracle_diag_by_variant[name] = test_oracle_diag
            total_steps_test = int(total_steps_variant)

        assert total_steps_test is not None

        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            slot_payload: Dict[str, Any] = {}
            for name in variant_names:
                tuned_config = variant_payloads[name]["by_slot"][str(slot_budget)]["tuned_config"]
                sim = _simulate_with_tuned_phase3(
                    jobs_test_by_variant[name],
                    total_steps=total_steps_test,
                    slot_budget=slot_budget,
                    tuned_config=tuned_config,
                    train_stats=train_stats_by_variant[name],
                    max_wait_steps=int(args.max_wait_steps),
                    rng_seed=int(args.rng_seed),
                )
                compact = _policy_compact(sim["summary"])
                slot_payload[name] = compact
                slot_payload[f"{name}_tuned_config"] = tuned_config
                slot_payload[f"{name}_oracle_stats"] = oracle_diag_by_variant[name]

            baselines = _extract_baselines(summary_json, slot_budget)
            slot_payload.update(baselines)
            row["slot_budget_results"][str(slot_budget)] = slot_payload
            per_slot_records[str(slot_budget)].append(slot_payload)

        results["per_holdout_results"].append(row)

    for slot, rows in per_slot_records.items():
        policy_stats = {name: _policy_stats(rows, name) for name in variant_names}
        policy_stats["best_threshold"] = _policy_stats(rows, "best_threshold")
        policy_stats["phase3_proposed"] = _policy_stats(rows, "phase3_proposed")
        paired = {}
        if "oracle_fused_ec" in variant_names:
            paired["fused_vs_phase3"] = _paired(rows, "oracle_fused_ec", "phase3_reference")
        if "oracle_protected_ec" in variant_names:
            paired["protected_vs_phase3"] = _paired(rows, "oracle_protected_ec", "phase3_reference")
        if "oracle_fused_ec" in variant_names and "oracle_protected_ec" in variant_names:
            paired["protected_vs_fused"] = _paired(rows, "oracle_protected_ec", "oracle_fused_ec")
        if "oracle_protected_ec" in variant_names:
            paired["protected_vs_best_threshold"] = _paired(rows, "oracle_protected_ec", "best_threshold")
            paired["protected_vs_phase3_proposed"] = _paired(rows, "oracle_protected_ec", "phase3_proposed")

        results["slot_budget_aggregates"][str(slot)] = {
            "policy_stats": policy_stats,
            "paired_stats": paired,
        }
        results["paired_focus"][str(slot)] = paired

    out_path = out_root / "aggregate_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    return {"aggregate_summary_path": str(out_path.resolve())}

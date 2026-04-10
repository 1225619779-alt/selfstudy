#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np

from phase3_oracle_family_core import (  # type: ignore
    OracleVariantSpec,
    _aggregate_arrival_steps,
    _arrival_diagnostics,
    _build_jobs_for_variant,
    _busy_time_unit_from_fit,
    _job_stats,
    _objective,
    _policy_compact,
    _score_kwargs,
    _tune_proposed_ca_policy,
    _variant_payload,
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)


def _load(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_protected_cfgs(source_screen: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    # Prefer explicit protected payload, otherwise selection if it already is protected.
    payload = None
    variants = source_screen.get("variants")
    if isinstance(variants, dict) and "oracle_protected_ec" in variants:
        payload = variants["oracle_protected_ec"]
    if payload is None:
        sel = source_screen.get("selection", {})
        if isinstance(sel, dict) and sel.get("winner_variant") == "oracle_protected_ec":
            payload = sel.get("winner_payload")
    if not isinstance(payload, dict):
        raise ValueError("Could not locate oracle_protected_ec payload in source screen summary")

    by_slot = payload.get("by_slot")
    if not isinstance(by_slot, dict):
        raise ValueError("Source oracle_protected_ec payload missing by_slot")
    out: Dict[str, Dict[str, float]] = {}
    for slot, slot_payload in by_slot.items():
        if not isinstance(slot_payload, dict) or "tuned_config" not in slot_payload:
            raise ValueError(f"Source by_slot[{slot}] missing tuned_config")
        out[str(slot)] = {k: float(v) for k, v in slot_payload["tuned_config"].items()}
    return out


def _anchored_grids(source_cfgs: Dict[str, Dict[str, float]]) -> Dict[str, List[float]]:
    # Collect base values from source winner configs across slots.
    v_vals = sorted({float(cfg["v_weight"]) for cfg in source_cfgs.values()})
    age_vals = sorted({float(cfg["age_bonus"]) for cfg in source_cfgs.values()})
    urgency_vals = sorted({float(cfg["urgency_bonus"]) for cfg in source_cfgs.values()})
    fail_vals = sorted({float(cfg["fail_penalty"]) for cfg in source_cfgs.values()})
    busy_vals = sorted({float(cfg["busy_penalty"]) for cfg in source_cfgs.values()})
    cost_vals = sorted({float(cfg["cost_penalty"]) for cfg in source_cfgs.values()})
    clean_vals = sorted({float(cfg["clean_penalty"]) for cfg in source_cfgs.values()})
    thr_vals = sorted({float(cfg["admission_score_threshold"]) for cfg in source_cfgs.values()})

    def expand(base_vals: List[float], *, deltas: List[float], lo: float | None = None) -> List[float]:
        vals = set()
        for b in base_vals:
            for d in deltas:
                x = b + d
                if lo is not None:
                    x = max(lo, x)
                vals.add(round(float(x), 6))
        return sorted(vals)

    # Conservative neighborhood around source; keep family and regime fixed.
    grids = {
        "vq_v_grid": sorted(set(v_vals + [max(1.0, v / 2.0) for v in v_vals])),
        "vq_age_grid": sorted(set(age_vals + [0.0, 0.1])),
        "vq_urgency_grid": sorted(set(urgency_vals + [0.0])),
        "vq_fail_grid": sorted(set(fail_vals + [0.0])),
        "vq_busy_grid": expand(busy_vals, deltas=[0.0, 0.5], lo=0.0),
        "vq_cost_grid": sorted(set(cost_vals + [0.0])),
        "vq_clean_grid": expand(clean_vals, deltas=[-0.25, 0.0, 0.25], lo=0.0),
        "vq_admission_threshold_grid": expand(thr_vals, deltas=[-0.1, 0.0, 0.1]),
    }
    return grids


def _build_anchored_screen(local_screen: Dict[str, Any], source_screen: Dict[str, Any]) -> Dict[str, Any]:
    manifest = local_screen["manifest"]
    workdir = Path(manifest["workdir"])

    source_cfgs = _source_protected_cfgs(source_screen)
    grids = _anchored_grids(source_cfgs)

    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(str(workdir / manifest["train_bank"])), int(manifest["frozen_regime"]["decision_step_group"]))
    arrays_tune = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(str(workdir / manifest["val_bank"])), int(manifest["frozen_regime"]["decision_step_group"]))

    clean_bank = str(workdir / manifest["clean_bank"])
    attack_bank = str(workdir / manifest["attack_bank"])

    posterior_verify = fit_attack_posterior_from_banks(clean_bank, attack_bank, signal_key="score_phys_l2", n_bins=20)
    posterior_ddd = fit_attack_posterior_from_banks(clean_bank, attack_bank, signal_key="ddd_loss_alarm", n_bins=20)
    service_models = fit_service_models_from_mixed_bank(str(workdir / manifest["train_bank"]), signal_key="verify_score", n_bins=20)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=20)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=20)
    consequence_mode = str(local_screen.get("config", {}).get("consequence_mode", "conditional"))
    severity_models = severity_models_cond if consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(manifest["frozen_regime"]["busy_time_quantile"]))

    protected = OracleVariantSpec(
        name="oracle_protected_ec",
        mode="protected_ec",
        description="deliverable/protected consequence = fused consequence × (1-fail risk)",
        clean_scale=0.0,
        protected_blend=0.0,
    )

    jobs_train, total_steps_train, train_oracle_diag = _build_jobs_for_variant(
        arrays_bank=arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=protected,
        gain_bundle_by_variant={},
        severity_blend_verify=float(local_screen.get("config", {}).get("consequence_blend_verify", 0.70)),
        busy_time_unit=busy_time_unit,
    )
    jobs_tune, total_steps_tune, val_oracle_diag = _build_jobs_for_variant(
        arrays_bank=arrays_tune,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=protected,
        gain_bundle_by_variant={},
        severity_blend_verify=float(local_screen.get("config", {}).get("consequence_blend_verify", 0.70)),
        busy_time_unit=busy_time_unit,
    )
    train_stats = _job_stats(jobs_train)
    tune_stats = _job_stats(jobs_tune)

    # Start from local screen, then overwrite just the protected-EC branch and selection.
    screen = copy.deepcopy(local_screen)
    screen["method"] = "phase3_oracle_upgrade_family_source_anchored"
    screen["config"] = dict(screen.get("config", {}))
    screen["config"]["variants"] = [_variant_payload(protected)]
    screen["config"]["anchor_strategy"] = "source_constrained_local_search"
    screen["config"]["anchor_source_case"] = str(source_screen.get("manifest", {}).get("case_name", "case14"))
    screen["config"]["anchored_grids"] = grids
    screen["environment"] = dict(screen.get("environment", {}))
    screen["environment"]["busy_time_unit"] = float(busy_time_unit)
    screen["environment"]["train_arrival_diagnostics"] = _arrival_diagnostics(jobs_train, total_steps_train)
    screen["environment"]["val_arrival_diagnostics"] = _arrival_diagnostics(jobs_tune, total_steps_tune)

    phase3_by_slot = screen["phase3_reference_by_slot"]
    variant_payload: Dict[str, Any] = {
        "variant": _variant_payload(protected),
        "oracle_train_stats": train_oracle_diag,
        "oracle_val_stats": val_oracle_diag,
        "train_job_stats": train_stats,
        "val_job_stats": tune_stats,
        "by_slot": {},
    }
    joint_delta_obj: List[float] = []
    joint_delta_recall: List[float] = []

    for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
        score_kwargs = _score_kwargs(
            argparse.Namespace(
                max_wait_steps=int(manifest["frozen_regime"]["max_wait_steps"]),
                objective_clean_penalty=0.60,
                objective_delay_penalty=0.15,
                objective_queue_penalty=0.10,
                objective_cost_penalty=0.05,
            ),
            cost_budget_per_step=None,
        )
        best_cfg, val_res = _tune_proposed_ca_policy(
            jobs_tune,
            total_steps_tune,
            slot_budget=slot_budget,
            max_wait_steps=int(manifest["frozen_regime"]["max_wait_steps"]),
            rng_seed=20260402,
            cost_budget_window_steps=0,
            window_cost_budget=None,
            mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            v_grid=grids["vq_v_grid"],
            clean_grid=grids["vq_clean_grid"],
            age_grid=grids["vq_age_grid"],
            urgency_grid=grids["vq_urgency_grid"],
            fail_grid=grids["vq_fail_grid"],
            busy_grid=grids["vq_busy_grid"],
            cost_grid=grids["vq_cost_grid"],
            admission_threshold_grid=grids["vq_admission_threshold_grid"],
            score_kwargs=score_kwargs,
        )
        phase3_ref = phase3_by_slot[str(slot_budget)]
        val_obj = float(_objective(val_res["summary"], slot_budget=int(slot_budget), **score_kwargs))
        delta_obj = float(val_obj - phase3_ref["val_objective"])
        delta_recall = float(val_res["summary"]["weighted_attack_recall_no_backend_fail"] - phase3_ref["val_summary"]["weighted_attack_recall_no_backend_fail"])
        joint_delta_obj.append(delta_obj)
        joint_delta_recall.append(delta_recall)
        variant_payload["by_slot"][str(slot_budget)] = {
            "tuned_config": best_cfg,
            "phase3_reference": phase3_ref,
            "train_oracle_stats": train_oracle_diag,
            "val_oracle_stats": val_oracle_diag,
            "train_job_stats": train_stats,
            "val_job_stats": tune_stats,
            "val_summary": _policy_compact(val_res["summary"]),
            "val_objective": val_obj,
            "val_delta_objective_vs_phase3": delta_obj,
            "val_delta_recall_vs_phase3": round(delta_recall, 6),
            "val_delta_unnecessary_vs_phase3": int(int(val_res["summary"]["unnecessary_mtd_count"]) - int(phase3_ref["val_summary"]["unnecessary_mtd_count"])),
            "val_delta_delay_vs_phase3": round(float(val_res["summary"]["queue_delay_p95"] - phase3_ref["val_summary"]["queue_delay_p95"]), 6),
            "val_delta_cost_vs_phase3": round(float(val_res["summary"]["average_service_cost_per_step"] - phase3_ref["val_summary"]["average_service_cost_per_step"]), 6),
            "anchor_source_tuned_config": source_cfgs[str(slot_budget)],
        }
    variant_payload["joint_val_delta_objective"] = float(np.mean(joint_delta_obj))
    variant_payload["joint_val_delta_recall"] = float(np.mean(joint_delta_recall))

    screen["variants"] = {"oracle_protected_ec": variant_payload}
    screen["selection"] = {
        "winner_variant": "oracle_protected_ec",
        "winner_joint_val_delta_objective": variant_payload["joint_val_delta_objective"],
        "winner_joint_val_delta_recall": variant_payload["joint_val_delta_recall"],
        "winner_payload": variant_payload,
        "selection_note": "Protocol-aware source-anchored local retune: local native train/val, protected-EC only, constrained grids near source winner.",
    }
    screen["anchored_from_source_screen"] = str(source_screen.get("manifest", {}).get("case_name", "case14"))
    return screen


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a protocol-aware source-anchored local-retune screen summary for case39.")
    ap.add_argument("--source_screen", required=True, help="case14 source oracle-family screen summary")
    ap.add_argument("--local_screen", required=True, help="case39 local protected-EC screen summary used as manifest/native-stats source")
    ap.add_argument("--output", required=True, help="output json path")
    args = ap.parse_args()

    source = _load(args.source_screen)
    local = _load(args.local_screen)
    screen = _build_anchored_screen(local, source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(screen, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {
        slot: {
            "tuned_config": screen["variants"]["oracle_protected_ec"]["by_slot"][slot]["tuned_config"],
            "val_summary": screen["variants"]["oracle_protected_ec"]["by_slot"][slot]["val_summary"],
        }
        for slot in sorted(screen["variants"]["oracle_protected_ec"]["by_slot"].keys())
    }
    print(json.dumps({
        "output": str(out),
        "stage": "case39_source_anchored_localretune",
        "joint_val_delta_objective": screen["selection"]["winner_joint_val_delta_objective"],
        "joint_val_delta_recall": screen["selection"]["winner_joint_val_delta_recall"],
        "by_slot": compact,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

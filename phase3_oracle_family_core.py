from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_binned_mean,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import (
    AlarmJob,
    SimulationConfig,
    build_jobs_from_arrays,
    simulate_policy,
)
from evaluation_budget_scheduler_phase3 import (
    _aggregate_arrival_steps,
    _arrival_diagnostics,
    _busy_time_unit_from_fit,
    _job_stats,
    _objective,
    _tune_proposed_ca_policy,
)

EPS = 1e-12


@dataclass(frozen=True)
class OracleVariantSpec:
    name: str
    mode: str
    description: str
    clean_scale: float = 0.0
    protected_blend: float = 0.0


DEFAULT_VARIANTS: Tuple[OracleVariantSpec, ...] = (
    OracleVariantSpec(
        name="oracle_fused_ec",
        mode="fused_ec",
        description="dual-signal fused attack posterior + conditional consequence",
    ),
    OracleVariantSpec(
        name="oracle_protected_ec",
        mode="protected_ec",
        description="deliverable/protected consequence = fused consequence × (1-fail risk)",
    ),
    OracleVariantSpec(
        name="oracle_help_lite",
        mode="help_gain",
        description="L2H-style light net-help oracle blended with protected consequence",
        clean_scale=0.35,
        protected_blend=0.50,
    ),
    OracleVariantSpec(
        name="oracle_help_strong",
        mode="help_gain",
        description="L2H-style stronger net-help oracle blended with protected consequence",
        clean_scale=0.70,
        protected_blend=0.35,
    ),
)


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


def _load_json(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_baselines(summary_json: Dict[str, object], slot_budget: int) -> Dict[str, Dict[str, float]]:
    payload = summary_json["slot_budget_results"][str(slot_budget)]["eval_compact"]
    best_name = None
    best = None
    for name in [
        "threshold_verify_fifo",
        "threshold_ddd_fifo",
        "threshold_expected_consequence_fifo",
        "adaptive_threshold_verify_fifo",
    ]:
        comp = payload[name]
        if best is None or float(comp["weighted_attack_recall_no_backend_fail"]) > float(best["weighted_attack_recall_no_backend_fail"]):
            best_name = name
            best = comp
    return {
        "best_threshold_name": best_name,
        "best_threshold": best,
        "phase3_proposed": payload["proposed_ca_vq_hard"],
        "topk_expected_consequence": payload["topk_expected_consequence"],
    }


def _policy_compact(summary: Dict[str, float]) -> Dict[str, float | int]:
    return {
        "weighted_attack_recall_no_backend_fail": round(float(summary["weighted_attack_recall_no_backend_fail"]), 4),
        "unnecessary_mtd_count": int(summary["unnecessary_mtd_count"]),
        "queue_delay_p95": round(float(summary["queue_delay_p95"]), 4),
        "average_service_cost_per_step": round(float(summary["average_service_cost_per_step"]), 6),
        "pred_expected_consequence_served_ratio": round(float(summary["pred_expected_consequence_served_ratio"]), 4),
    }


def _summarize_oracle(x: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _score_kwargs(args: SimpleNamespace, *, cost_budget_per_step: float | None) -> Dict[str, float]:
    return {
        "max_wait_steps": int(args.max_wait_steps),
        "clean_penalty": float(args.objective_clean_penalty),
        "delay_penalty": float(args.objective_delay_penalty),
        "queue_penalty": float(args.objective_queue_penalty),
        "cost_penalty": float(args.objective_cost_penalty),
        "cost_budget_per_step": cost_budget_per_step,
    }


def _clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _fuse_posteriors(p_verify: np.ndarray, p_ddd: np.ndarray, *, verify_weight: float = 0.65) -> np.ndarray:
    wv = float(np.clip(verify_weight, 0.0, 1.0))
    wd = 1.0 - wv
    fused = _sigmoid(wv * _logit(p_verify) + wd * _logit(p_ddd))
    return np.clip(fused, 0.0, 1.0)


def _make_value_proxy(arrays: Dict[str, np.ndarray], fit_verify_score: np.ndarray) -> np.ndarray:
    if "consequence_proxy" in arrays:
        x = np.asarray(arrays["consequence_proxy"], dtype=float)
        fit_x = np.asarray(arrays["consequence_proxy"], dtype=float)
    elif "value_proxy" in arrays:
        x = np.asarray(arrays["value_proxy"], dtype=float)
        fit_x = np.asarray(arrays["value_proxy"], dtype=float)
    else:
        x = np.asarray(arrays["verify_score"], dtype=float)
        fit_x = np.asarray(fit_verify_score, dtype=float)
    fit_x = fit_x[np.isfinite(fit_x)]
    scale = float(np.quantile(fit_x, 0.90)) if fit_x.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(np.asarray(x, dtype=float) / scale, 0.0, 5.0)


def _variant_payload(variant: OracleVariantSpec) -> Dict[str, object]:
    return {
        "name": str(variant.name),
        "mode": str(variant.mode),
        "description": str(variant.description),
        "clean_scale": float(variant.clean_scale),
        "protected_blend": float(variant.protected_blend),
    }


def _available_signal_keys(arrays: Mapping[str, np.ndarray]) -> List[str]:
    order = ["verify_score", "ddd_loss_recons", "consequence_proxy", "value_proxy", "score_phys_l2"]
    return [k for k in order if k in arrays]


def _signal_blend_weights(signal_keys: Sequence[str]) -> Dict[str, float]:
    raw = {
        "verify_score": 0.35,
        "ddd_loss_recons": 0.25,
        "consequence_proxy": 0.25,
        "value_proxy": 0.15,
        "score_phys_l2": 0.15,
    }
    weights = {k: raw.get(k, 1.0) for k in signal_keys}
    total = float(sum(weights.values()))
    if total <= 0:
        n = max(len(signal_keys), 1)
        return {k: 1.0 / n for k in signal_keys}
    return {k: float(v / total) for k, v in weights.items()}


def _fit_net_gain_models(
    arrays_train: Dict[str, np.ndarray],
    *,
    clean_scale: float,
    n_bins: int,
) -> Dict[str, object]:
    attack_gain = np.asarray(arrays_train["severity_true"], dtype=float) * (1.0 - np.asarray(arrays_train["backend_fail"], dtype=float))
    attack_mask = np.asarray(arrays_train["is_attack"], dtype=int) == 1
    attack_gain_mean = float(np.mean(attack_gain[attack_mask])) if np.any(attack_mask) else 0.0
    clean_offset = max(float(clean_scale) * attack_gain_mean, 0.05)
    target = attack_gain - clean_offset * (1.0 - np.asarray(arrays_train["is_attack"], dtype=float))

    signal_keys = _available_signal_keys(arrays_train)
    weights = _signal_blend_weights(signal_keys)
    models = {}
    default_value = float(np.mean(target)) if target.size else 0.0
    for key in signal_keys:
        x = np.asarray(arrays_train[key], dtype=float)
        models[key] = fit_binned_mean(
            x,
            target,
            n_bins=n_bins,
            default_value=default_value,
            name=f"net_help_given_{key}",
        )
    return {
        "signal_keys": signal_keys,
        "signal_weights": weights,
        "clean_offset": float(clean_offset),
        "target_stats": _summarize_oracle(target),
        "models": models,
    }


def _predict_net_gain(
    arrays_bank: Dict[str, np.ndarray],
    gain_bundle: Dict[str, object],
) -> np.ndarray:
    signal_keys = gain_bundle["signal_keys"]
    weights = gain_bundle["signal_weights"]
    models = gain_bundle["models"]
    out = np.zeros(len(np.asarray(arrays_bank["arrival_step"])), dtype=float)
    if not signal_keys:
        return out
    for key in signal_keys:
        out += float(weights[key]) * np.asarray(models[key].predict(np.asarray(arrays_bank[key], dtype=float)), dtype=float)
    return out


def _prepare_common_predictions(
    *,
    arrays_bank: Dict[str, np.ndarray],
    arrays_train: Dict[str, np.ndarray],
    posterior_verify,
    posterior_ddd,
    service_models,
    severity_models,
    severity_blend_verify: float,
) -> Dict[str, np.ndarray]:
    x_verify = np.asarray(arrays_bank["verify_score"], dtype=float)
    x_ddd = np.asarray(arrays_bank["ddd_loss_recons"], dtype=float)
    p_verify = np.asarray(posterior_verify.predict(x_verify), dtype=float)
    p_ddd = np.asarray(posterior_ddd.predict(x_ddd), dtype=float)
    p_fused = _fuse_posteriors(p_verify, p_ddd, verify_weight=float(severity_blend_verify))

    tau_hat = np.asarray(service_models["service_time"].predict(x_verify), dtype=float)
    cost_hat = np.asarray(service_models["service_cost"].predict(x_verify), dtype=float)
    fail_hat = np.asarray(service_models["backend_fail"].predict(x_verify), dtype=float)

    sev_verify = severity_models["verify_score"].predict(x_verify) if "verify_score" in severity_models else np.zeros_like(x_verify)
    sev_ddd = severity_models["ddd_loss_recons"].predict(x_ddd) if "ddd_loss_recons" in severity_models else np.zeros_like(x_ddd)
    wv = float(np.clip(severity_blend_verify, 0.0, 1.0))
    wd = 1.0 - wv
    attack_severity_hat = wv * np.asarray(sev_verify, dtype=float) + wd * np.asarray(sev_ddd, dtype=float)

    baseline_ec = np.clip(p_verify, 0.0, 1.0) * np.maximum(attack_severity_hat, 0.0)
    fused_ec = np.clip(p_fused, 0.0, 1.0) * np.maximum(attack_severity_hat, 0.0)
    protected_ec = fused_ec * (1.0 - np.clip(fail_hat, 0.0, 1.0))
    value_proxy = _make_value_proxy(arrays_bank, np.asarray(arrays_train["verify_score"], dtype=float))
    return {
        "p_verify": p_verify,
        "p_ddd": p_ddd,
        "p_fused": p_fused,
        "tau_hat": tau_hat,
        "cost_hat": cost_hat,
        "fail_hat": fail_hat,
        "attack_severity_hat": attack_severity_hat,
        "baseline_ec": baseline_ec,
        "fused_ec": fused_ec,
        "protected_ec": protected_ec,
        "value_proxy": value_proxy,
    }


def _build_jobs_for_variant(
    *,
    arrays_bank: Dict[str, np.ndarray],
    arrays_train: Dict[str, np.ndarray],
    posterior_verify,
    posterior_ddd,
    service_models,
    severity_models,
    variant: OracleVariantSpec | None,
    gain_bundle_by_variant: Dict[str, Dict[str, object]],
    severity_blend_verify: float,
    busy_time_unit: float,
) -> Tuple[List[AlarmJob], int, Dict[str, object]]:
    common = _prepare_common_predictions(
        arrays_bank=arrays_bank,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        severity_blend_verify=severity_blend_verify,
    )
    if variant is None:
        p_hat = common["p_verify"]
        ec_hat = common["baseline_ec"]
        diag = {
            "pred_attack_prob": _summarize_oracle(p_hat),
            "pred_expected_consequence": _summarize_oracle(ec_hat),
            "signal_keys": ["verify_score", "ddd_loss_recons"],
            "oracle_mode": "baseline_phase3",
        }
    elif variant.mode == "fused_ec":
        p_hat = common["p_fused"]
        ec_hat = common["fused_ec"]
        diag = {
            "pred_attack_prob": _summarize_oracle(p_hat),
            "pred_expected_consequence": _summarize_oracle(ec_hat),
            "signal_keys": ["verify_score", "ddd_loss_recons"],
            "oracle_mode": "fused_ec",
        }
    elif variant.mode == "protected_ec":
        p_hat = common["p_fused"]
        ec_hat = common["protected_ec"]
        diag = {
            "pred_attack_prob": _summarize_oracle(p_hat),
            "pred_expected_consequence": _summarize_oracle(ec_hat),
            "pred_fail_prob": _summarize_oracle(common["fail_hat"]),
            "signal_keys": ["verify_score", "ddd_loss_recons"],
            "oracle_mode": "protected_ec",
        }
    elif variant.mode == "help_gain":
        p_hat = common["p_fused"]
        gain_bundle = gain_bundle_by_variant[variant.name]
        net_gain = _predict_net_gain(arrays_bank, gain_bundle)
        ec_hat = float(variant.protected_blend) * common["protected_ec"] + (1.0 - float(variant.protected_blend)) * np.maximum(net_gain, 0.0)
        diag = {
            "pred_attack_prob": _summarize_oracle(p_hat),
            "pred_expected_consequence": _summarize_oracle(ec_hat),
            "direct_net_gain": _summarize_oracle(net_gain),
            "signal_keys": list(gain_bundle["signal_keys"]),
            "signal_weights": dict(gain_bundle["signal_weights"]),
            "clean_offset": float(gain_bundle["clean_offset"]),
            "target_stats": dict(gain_bundle["target_stats"]),
            "oracle_mode": "help_gain",
        }
    else:
        raise KeyError(f"Unknown variant mode={variant.mode!r}")

    jobs, total_steps = build_jobs_from_arrays(
        arrays_bank,
        p_hat=p_hat,
        tau_hat=common["tau_hat"],
        cost_hat=common["cost_hat"],
        fail_hat=common["fail_hat"],
        attack_severity_hat=common["attack_severity_hat"],
        expected_consequence_hat=ec_hat,
        value_proxy=common["value_proxy"],
        busy_time_unit=busy_time_unit,
    )
    return jobs, total_steps, diag


def _simulate_with_tuned_phase3(
    jobs: Sequence[AlarmJob],
    *,
    total_steps: int,
    slot_budget: int,
    tuned_config: Dict[str, float],
    train_stats: Dict[str, float],
    max_wait_steps: int,
    rng_seed: int,
) -> Dict[str, object]:
    cfg = SimulationConfig(
        policy_name="proposed_ca_vq_hard",
        slot_budget=int(slot_budget),
        max_wait_steps=int(max_wait_steps),
        rng_seed=int(rng_seed),
        cost_budget_window_steps=0,
        window_cost_budget=None,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
        v_weight=float(tuned_config["v_weight"]),
        clean_penalty=float(tuned_config["clean_penalty"]),
        age_bonus=float(tuned_config["age_bonus"]),
        urgency_bonus=float(tuned_config["urgency_bonus"]),
        fail_penalty=float(tuned_config["fail_penalty"]),
        busy_penalty=float(tuned_config["busy_penalty"]),
        cost_penalty=float(tuned_config["cost_penalty"]),
        admission_score_threshold=float(tuned_config["admission_score_threshold"]),
    )
    return simulate_policy(jobs, total_steps=total_steps, cfg=cfg)


def _select_joint_winner(screen_payload: Dict[str, object]) -> Dict[str, object]:
    best_name = None
    best_score = -1e18
    best_record = None
    for name, payload in screen_payload["variants"].items():
        joint = float(payload["joint_val_delta_objective"])
        tie_break = float(payload.get("joint_val_delta_recall", 0.0))
        if (joint > best_score + 1e-12) or (
            abs(joint - best_score) <= 1e-12 and best_record is not None and tie_break > float(best_record.get("joint_val_delta_recall", 0.0))
        ) or best_record is None:
            best_score = joint
            best_name = name
            best_record = payload
    assert best_name is not None and best_record is not None
    return {
        "winner_variant": str(best_name),
        "winner_joint_val_delta_objective": float(best_score),
        "winner_joint_val_delta_recall": float(best_record.get("joint_val_delta_recall", 0.0)),
        "winner_payload": best_record,
    }


def _screen_variants(*, manifest: Dict[str, object]) -> Dict[str, object]:
    workdir = Path(manifest["workdir"])
    args = SimpleNamespace(
        clean_bank=str(workdir / manifest["clean_bank"]),
        attack_bank=str(workdir / manifest["attack_bank"]),
        train_bank=str(workdir / manifest["train_bank"]),
        tune_bank=str(workdir / manifest["val_bank"]),
        n_bins=20,
        max_wait_steps=int(manifest["frozen_regime"]["max_wait_steps"]),
        decision_step_group=int(manifest["frozen_regime"]["decision_step_group"]),
        busy_time_quantile=float(manifest["frozen_regime"]["busy_time_quantile"]),
        use_cost_budget=bool(manifest["frozen_regime"]["use_cost_budget"]),
        cost_budget_window_steps=int(manifest["frozen_regime"]["cost_budget_window_steps"]),
        cost_budget_quantile=float(manifest["frozen_regime"]["cost_budget_quantile"]),
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
        rng_seed=20260402,
    )

    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.train_bank), int(args.decision_step_group))
    arrays_tune = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.tune_bank), int(args.decision_step_group))

    posterior_verify = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins)
    posterior_ddd = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="ddd_loss_alarm", n_bins=args.n_bins)
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))

    gain_bundle_by_variant = {}
    for variant in DEFAULT_VARIANTS:
        if variant.mode == "help_gain":
            gain_bundle_by_variant[variant.name] = _fit_net_gain_models(
                arrays_train,
                clean_scale=float(variant.clean_scale),
                n_bins=int(args.n_bins),
            )

    jobs_train_phase3, total_steps_train, train_diag_phase3 = _build_jobs_for_variant(
        arrays_bank=arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=None,
        gain_bundle_by_variant=gain_bundle_by_variant,
        severity_blend_verify=float(args.consequence_blend_verify),
        busy_time_unit=busy_time_unit,
    )
    jobs_tune_phase3, total_steps_tune, tune_diag_phase3 = _build_jobs_for_variant(
        arrays_bank=arrays_tune,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=None,
        gain_bundle_by_variant=gain_bundle_by_variant,
        severity_blend_verify=float(args.consequence_blend_verify),
        busy_time_unit=busy_time_unit,
    )

    train_stats_phase3 = _job_stats(jobs_train_phase3)
    tune_stats_phase3 = _job_stats(jobs_tune_phase3)

    screen: Dict[str, object] = {
        "method": "phase3_oracle_upgrade_family",
        "manifest": manifest,
        "config": {
            "decision_step_group": int(args.decision_step_group),
            "busy_time_quantile": float(args.busy_time_quantile),
            "use_cost_budget": bool(args.use_cost_budget),
            "slot_budget_list": manifest["frozen_regime"]["slot_budget_list"],
            "max_wait_steps": int(args.max_wait_steps),
            "consequence_blend_verify": float(args.consequence_blend_verify),
            "consequence_mode": str(args.consequence_mode),
            "variants": [_variant_payload(v) for v in DEFAULT_VARIANTS],
        },
        "environment": {
            "busy_time_unit": float(busy_time_unit),
            "train_arrival_diagnostics": _arrival_diagnostics(jobs_train_phase3, total_steps_train),
            "val_arrival_diagnostics": _arrival_diagnostics(jobs_tune_phase3, total_steps_tune),
            "phase3_train_job_stats": train_stats_phase3,
            "phase3_val_job_stats": tune_stats_phase3,
            "available_signal_keys": _available_signal_keys(arrays_train),
            "phase3_train_oracle_stats": train_diag_phase3,
            "phase3_val_oracle_stats": tune_diag_phase3,
        },
        "phase3_reference_by_slot": {},
        "variants": {},
    }

    phase3_by_slot: Dict[str, Dict[str, object]] = {}
    phase3_val_objective_by_slot: Dict[str, float] = {}
    for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
        score_kwargs = _score_kwargs(args, cost_budget_per_step=None)
        phase3_best, phase3_res = _tune_proposed_ca_policy(
            jobs_tune_phase3,
            total_steps_tune,
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=0,
            window_cost_budget=None,
            mean_pred_busy_steps=train_stats_phase3["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats_phase3["mean_pred_service_cost"],
            mean_pred_expected_consequence=train_stats_phase3["mean_pred_expected_consequence"],
            v_grid=args.vq_v_grid,
            clean_grid=args.vq_clean_grid,
            age_grid=args.vq_age_grid,
            urgency_grid=args.vq_urgency_grid,
            fail_grid=args.vq_fail_grid,
            busy_grid=args.vq_busy_grid,
            cost_grid=args.vq_cost_grid,
            admission_threshold_grid=args.vq_admission_threshold_grid,
            score_kwargs=score_kwargs,
        )
        phase3_obj = float(_objective(phase3_res["summary"], slot_budget=int(slot_budget), **score_kwargs))
        phase3_val_objective_by_slot[str(slot_budget)] = phase3_obj
        phase3_by_slot[str(slot_budget)] = {
            "config": phase3_best,
            "val_summary": _policy_compact(phase3_res["summary"]),
            "val_objective": phase3_obj,
        }
        screen["phase3_reference_by_slot"][str(slot_budget)] = phase3_by_slot[str(slot_budget)]

    for variant in DEFAULT_VARIANTS:
        jobs_train, _, train_oracle_diag = _build_jobs_for_variant(
            arrays_bank=arrays_train,
            arrays_train=arrays_train,
            posterior_verify=posterior_verify,
            posterior_ddd=posterior_ddd,
            service_models=service_models,
            severity_models=severity_models,
            variant=variant,
            gain_bundle_by_variant=gain_bundle_by_variant,
            severity_blend_verify=float(args.consequence_blend_verify),
            busy_time_unit=busy_time_unit,
        )
        jobs_tune, _, val_oracle_diag = _build_jobs_for_variant(
            arrays_bank=arrays_tune,
            arrays_train=arrays_train,
            posterior_verify=posterior_verify,
            posterior_ddd=posterior_ddd,
            service_models=service_models,
            severity_models=severity_models,
            variant=variant,
            gain_bundle_by_variant=gain_bundle_by_variant,
            severity_blend_verify=float(args.consequence_blend_verify),
            busy_time_unit=busy_time_unit,
        )
        train_stats = _job_stats(jobs_train)
        tune_stats = _job_stats(jobs_tune)

        variant_payload: Dict[str, object] = {
            "variant": _variant_payload(variant),
            "oracle_train_stats": train_oracle_diag,
            "oracle_val_stats": val_oracle_diag,
            "train_job_stats": train_stats,
            "val_job_stats": tune_stats,
            "by_slot": {},
        }
        joint_delta_obj: List[float] = []
        joint_delta_recall: List[float] = []
        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            score_kwargs = _score_kwargs(args, cost_budget_per_step=None)
            best_cfg, val_res = _tune_proposed_ca_policy(
                jobs_tune,
                total_steps_tune,
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=0,
                window_cost_budget=None,
                mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=train_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
                v_grid=args.vq_v_grid,
                clean_grid=args.vq_clean_grid,
                age_grid=args.vq_age_grid,
                urgency_grid=args.vq_urgency_grid,
                fail_grid=args.vq_fail_grid,
                busy_grid=args.vq_busy_grid,
                cost_grid=args.vq_cost_grid,
                admission_threshold_grid=args.vq_admission_threshold_grid,
                score_kwargs=score_kwargs,
            )
            val_obj = float(_objective(val_res["summary"], slot_budget=int(slot_budget), **score_kwargs))
            delta_obj = float(val_obj - phase3_val_objective_by_slot[str(slot_budget)])
            delta_recall = float(
                val_res["summary"]["weighted_attack_recall_no_backend_fail"]
                - phase3_by_slot[str(slot_budget)]["val_summary"]["weighted_attack_recall_no_backend_fail"]
            )
            joint_delta_obj.append(delta_obj)
            joint_delta_recall.append(delta_recall)
            variant_payload["by_slot"][str(slot_budget)] = {
                "tuned_config": best_cfg,
                "phase3_reference": phase3_by_slot[str(slot_budget)],
                "train_oracle_stats": train_oracle_diag,
                "val_oracle_stats": val_oracle_diag,
                "train_job_stats": train_stats,
                "val_job_stats": tune_stats,
                "val_summary": _policy_compact(val_res["summary"]),
                "val_objective": val_obj,
                "val_delta_objective_vs_phase3": delta_obj,
                "val_delta_recall_vs_phase3": round(delta_recall, 6),
                "val_delta_unnecessary_vs_phase3": int(
                    int(val_res["summary"]["unnecessary_mtd_count"]) - int(phase3_by_slot[str(slot_budget)]["val_summary"]["unnecessary_mtd_count"])
                ),
                "val_delta_delay_vs_phase3": round(
                    float(val_res["summary"]["queue_delay_p95"] - phase3_by_slot[str(slot_budget)]["val_summary"]["queue_delay_p95"]),
                    6,
                ),
                "val_delta_cost_vs_phase3": round(
                    float(val_res["summary"]["average_service_cost_per_step"] - phase3_by_slot[str(slot_budget)]["val_summary"]["average_service_cost_per_step"]),
                    6,
                ),
            }
        variant_payload["joint_val_delta_objective"] = float(np.mean(joint_delta_obj))
        variant_payload["joint_val_delta_recall"] = float(np.mean(joint_delta_recall))
        screen["variants"][variant.name] = variant_payload

    screen["selection"] = _select_joint_winner(screen)
    return screen


def run_phase3_oracle_family_experiment(manifest_path: str, output_dir: str, *, screen_only: bool = False) -> Dict[str, object]:
    manifest = _load_json(manifest_path)
    workdir = Path(manifest["workdir"])
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    screen = _screen_variants(manifest=manifest)
    screen_path = out_root / "screen_train_val_summary.json"
    with open(screen_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(screen), f, ensure_ascii=False, indent=2)

    if screen_only:
        return {"screen_summary_path": str(screen_path.resolve()), "screen_only": True}

    winner_name = str(screen["selection"]["winner_variant"])
    winner_variant = next(v for v in DEFAULT_VARIANTS if v.name == winner_name)

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

    gain_bundle_by_variant = {}
    for variant in DEFAULT_VARIANTS:
        if variant.mode == "help_gain":
            gain_bundle_by_variant[variant.name] = _fit_net_gain_models(
                arrays_train,
                clean_scale=float(variant.clean_scale),
                n_bins=int(args.n_bins),
            )

    jobs_train_winner, total_steps_train, train_oracle_diag = _build_jobs_for_variant(
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
        "method": "phase3_oracle_upgrade_family",
        "winner_variant": _variant_payload(winner_variant),
        "screen_summary_path": str(screen_path.resolve()),
        "manifest": manifest,
        "phase3_reference_by_slot": screen["phase3_reference_by_slot"],
        "screen_selection": screen["selection"],
        "winner_train_oracle_stats": train_oracle_diag,
        "winner_train_job_stats": train_stats,
        "n_holdouts": int(len(manifest["holdouts"])),
        "per_holdout_results": [],
        "slot_budget_aggregates": {},
    }

    per_slot_records: Dict[str, List[Dict[str, object]]] = {
        str(slot): [] for slot in manifest["frozen_regime"]["slot_budget_list"]
    }

    for hold in manifest["holdouts"]:
        holdout_row = {
            "tag": hold["tag"],
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
            tuned = screen["variants"][winner_name]["by_slot"][str(slot_budget)]["tuned_config"]
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
            holdout_row["slot_budget_results"][str(slot_budget)] = payload
            per_slot_records[str(slot_budget)].append(payload)
        results["per_holdout_results"].append(holdout_row)

    for slot, rows in per_slot_records.items():
        policies = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]
        slot_payload = {"policy_stats": {}, "paired_stats": {}, "best_threshold_frequency": {}}
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

        def _paired(a: str, b: str, tag: str):
            delta_rec = [float(r[a]["weighted_attack_recall_no_backend_fail"]) - float(r[b]["weighted_attack_recall_no_backend_fail"]) for r in rows]
            delta_un = [float(r[a]["unnecessary_mtd_count"]) - float(r[b]["unnecessary_mtd_count"]) for r in rows]
            delta_del = [float(r[a]["queue_delay_p95"]) - float(r[b]["queue_delay_p95"]) for r in rows]
            delta_cost = [float(r[a]["average_service_cost_per_step"]) - float(r[b]["average_service_cost_per_step"]) for r in rows]
            slot_payload["paired_stats"][tag] = {
                "delta_recall": {"mean": float(np.mean(delta_rec)), "std": float(np.std(delta_rec)), "min": float(np.min(delta_rec)), "max": float(np.max(delta_rec))},
                "delta_unnecessary": {"mean": float(np.mean(delta_un)), "std": float(np.std(delta_un)), "min": float(np.min(delta_un)), "max": float(np.max(delta_un))},
                "delta_delay_p95": {"mean": float(np.mean(delta_del)), "std": float(np.std(delta_del)), "min": float(np.min(delta_del)), "max": float(np.max(delta_del))},
                "delta_cost_per_step": {"mean": float(np.mean(delta_cost)), "std": float(np.std(delta_cost)), "min": float(np.min(delta_cost)), "max": float(np.max(delta_cost))},
                "wins_on_recall": int(np.sum(np.asarray(delta_rec) > 0)),
                "lower_unnecessary": int(np.sum(np.asarray(delta_un) < 0)),
            }

        _paired("phase3_oracle_upgrade", "phase3_proposed", "oracle_vs_phase3")
        _paired("phase3_oracle_upgrade", "best_threshold", "oracle_vs_best_threshold")
        _paired("phase3_oracle_upgrade", "topk_expected_consequence", "oracle_vs_topk_expected")
        results["slot_budget_aggregates"][slot] = slot_payload

    multi_dir = out_root / "multi_holdout"
    multi_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = multi_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    return {
        "screen_summary_path": str(screen_path.resolve()),
        "screen_only": False,
        "aggregate_summary_path": str(aggregate_path.resolve()),
    }


from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scheduler.calibration import (
    BinnedStatisticModel,
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
    summarize_array,
)
from scheduler.policies_phase3 import AlarmJob, SimulationConfig, build_jobs_from_arrays, simulate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-3 consequence-aware hard-constrained offline evaluation for budget-aware backend MTD scheduling."
    )
    parser.add_argument("--clean_bank", type=str, required=True)
    parser.add_argument("--attack_bank", type=str, required=True)
    parser.add_argument("--fit_bank", type=str, required=True)
    parser.add_argument("--eval_bank", type=str, required=True)
    parser.add_argument("--output", type=str, default="metric/case14/budget_scheduler_phase3_ca.npy")
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
    parser.add_argument("--consequence_blend_verify", type=float, default=0.70, help="Blend weight for verify-score-based severity predictor; ddd receives the rest.")
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


def _aggregate_arrival_steps(arrays: Dict[str, np.ndarray], group_size: int) -> Dict[str, np.ndarray]:
    if int(group_size) <= 1:
        return {k: np.asarray(v).copy() for k, v in arrays.items()}
    out = {k: np.asarray(v).copy() for k, v in arrays.items()}
    out["arrival_step"] = np.floor_divide(np.asarray(out["arrival_step"], dtype=int), int(group_size))
    total_steps = int(np.asarray(out["total_steps"]).reshape(-1)[0])
    total_steps = int(math.ceil(total_steps / int(group_size)))
    out["total_steps"] = np.asarray([total_steps], dtype=int)
    return out


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


def _threshold_candidates(x: np.ndarray, quantiles: Sequence[float]) -> List[float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0]
    return sorted({float(np.quantile(arr, q)) for q in quantiles})


def _busy_time_unit_from_fit(arrays_fit: Dict[str, np.ndarray], q: float) -> float:
    x = np.asarray(arrays_fit["service_time"], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    unit = float(np.quantile(x, q))
    return max(unit, 1e-6)


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if int(window) <= 1:
        return arr.copy()
    c = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.empty_like(arr)
    for i in range(arr.size):
        j0 = max(0, i - int(window) + 1)
        out[i] = c[i + 1] - c[j0]
    return out


def _derive_cost_budget_from_fit(jobs_fit: List[AlarmJob], total_steps_fit: int, *, window_steps: int, q: float) -> float | None:
    if int(window_steps) <= 0:
        return None
    per_step_cost = np.zeros(int(total_steps_fit), dtype=float)
    for job in jobs_fit:
        step = int(job.arrival_step)
        if 0 <= step < len(per_step_cost):
            per_step_cost[step] += float(job.actual_service_cost)
    rolling = _rolling_sum(per_step_cost, int(window_steps))
    rolling = rolling[np.isfinite(rolling)]
    if rolling.size == 0:
        return None
    return float(np.quantile(rolling, q))


def _job_stats(jobs: List[AlarmJob]) -> Dict[str, float]:
    pred_busy = np.asarray([j.pred_busy_steps for j in jobs], dtype=float)
    pred_cost = np.asarray([j.pred_service_cost for j in jobs], dtype=float)
    pred_ec = np.asarray([j.pred_expected_consequence for j in jobs], dtype=float)
    actual_busy = np.asarray([j.actual_busy_steps for j in jobs], dtype=float)
    return {
        "mean_pred_busy_steps": float(np.mean(pred_busy)) if pred_busy.size else 1.0,
        "mean_pred_service_cost": float(np.mean(pred_cost)) if pred_cost.size else 1.0,
        "mean_pred_expected_consequence": float(np.mean(pred_ec)) if pred_ec.size else 1.0,
        "busy_step_stats": summarize_array(actual_busy),
        "pred_expected_consequence_stats": summarize_array(pred_ec),
    }


def _arrival_diagnostics(jobs: List[AlarmJob], total_steps: int) -> Dict[str, float]:
    if total_steps <= 0:
        return {"total_steps": 0.0}
    counts = np.zeros(int(total_steps), dtype=int)
    for job in jobs:
        if 0 <= int(job.arrival_step) < len(counts):
            counts[int(job.arrival_step)] += 1
    nonzero = counts[counts > 0]
    return {
        "total_steps": float(total_steps),
        "total_jobs": float(len(jobs)),
        "steps_with_jobs": float(np.sum(counts > 0)),
        "share_steps_with_jobs": float(np.mean(counts > 0)),
        "max_jobs_same_step": float(np.max(counts)) if counts.size else 0.0,
        "mean_jobs_given_nonzero_step": float(np.mean(nonzero)) if nonzero.size else 0.0,
    }


def _objective(
    summary: Dict[str, float],
    *,
    max_wait_steps: int,
    slot_budget: int,
    clean_penalty: float,
    delay_penalty: float,
    queue_penalty: float,
    cost_penalty: float,
    cost_budget_per_step: float | None,
) -> float:
    score = float(summary["weighted_attack_recall_no_backend_fail"])
    score -= float(clean_penalty) * float(summary["clean_service_ratio"])
    score -= float(delay_penalty) * float(summary["attack_delay_p95"] / max(max_wait_steps, 1))
    score -= float(queue_penalty) * float(summary["mean_queue_len"] / max(slot_budget, 1))
    if cost_budget_per_step is not None and cost_budget_per_step > 0:
        score -= float(cost_penalty) * float(summary["average_service_cost_per_step"] / max(cost_budget_per_step, 1e-9))
    return float(score)


def _predict_jobs(
    arrays: Dict[str, np.ndarray],
    *,
    posterior_model: BinnedStatisticModel,
    posterior_signal_key: str,
    service_models: Dict[str, BinnedStatisticModel],
    service_signal_key: str,
    severity_models: Dict[str, BinnedStatisticModel],
    severity_blend_verify: float,
    consequence_mode: str,
    fit_verify_score: np.ndarray,
    busy_time_unit: float,
) -> Tuple[List[AlarmJob], int]:
    x_post = np.asarray(arrays[posterior_signal_key], dtype=float)
    x_srv = np.asarray(arrays[service_signal_key], dtype=float)
    p_hat = posterior_model.predict(x_post)
    tau_hat = service_models["service_time"].predict(x_srv)
    cost_hat = service_models["service_cost"].predict(x_srv)
    fail_hat = service_models["backend_fail"].predict(x_srv)

    sev_verify = severity_models["verify_score"].predict(np.asarray(arrays["verify_score"], dtype=float)) if "verify_score" in severity_models else np.zeros_like(x_post, dtype=float)
    sev_ddd = severity_models["ddd_loss_recons"].predict(np.asarray(arrays["ddd_loss_recons"], dtype=float)) if "ddd_loss_recons" in severity_models else np.zeros_like(x_post, dtype=float)
    wv = float(np.clip(severity_blend_verify, 0.0, 1.0))
    wd = 1.0 - wv
    attack_severity_hat = wv * sev_verify + wd * sev_ddd
    if consequence_mode == "conditional":
        expected_consequence_hat = np.clip(p_hat, 0.0, 1.0) * np.maximum(attack_severity_hat, 0.0)
    elif consequence_mode == "expected":
        expected_consequence_hat = np.maximum(attack_severity_hat, 0.0)
    else:
        raise KeyError(f"Unknown consequence_mode={consequence_mode!r}")

    value_proxy = _make_value_proxy(arrays, fit_verify_score)
    return build_jobs_from_arrays(
        arrays,
        p_hat=p_hat,
        tau_hat=tau_hat,
        cost_hat=cost_hat,
        fail_hat=fail_hat,
        attack_severity_hat=attack_severity_hat,
        expected_consequence_hat=expected_consequence_hat,
        value_proxy=value_proxy,
        busy_time_unit=busy_time_unit,
    )


def _run_one_policy(jobs: List[AlarmJob], total_steps: int, cfg: SimulationConfig) -> Dict[str, object]:
    return simulate_policy(jobs, total_steps=total_steps, cfg=cfg)


def _tune_threshold_policy(
    jobs_fit: List[AlarmJob],
    total_steps_fit: int,
    *,
    threshold_candidates: List[float],
    policy_name: str,
    slot_budget: int,
    max_wait_steps: int,
    rng_seed: int,
    cost_budget_window_steps: int,
    window_cost_budget: float | None,
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    score_kwargs: Dict[str, float],
) -> Tuple[float, Dict[str, object]]:
    best_thr = threshold_candidates[0]
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for thr in threshold_candidates:
        cfg = SimulationConfig(
            policy_name=policy_name,
            slot_budget=slot_budget,
            max_wait_steps=max_wait_steps,
            threshold=float(thr),
            rng_seed=rng_seed,
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=mean_pred_busy_steps,
            mean_pred_service_cost=mean_pred_service_cost,
            mean_pred_expected_consequence=mean_pred_expected_consequence,
        )
        res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
        obj = _objective(res["summary"], slot_budget=slot_budget, **score_kwargs)
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
            best_res = res
    assert best_res is not None
    return best_thr, best_res


def _tune_adaptive_threshold_policy(
    jobs_fit: List[AlarmJob],
    total_steps_fit: int,
    *,
    threshold_candidates: List[float],
    gain_candidates: List[float],
    slot_budget: int,
    max_wait_steps: int,
    rng_seed: int,
    cost_budget_window_steps: int,
    window_cost_budget: float | None,
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    score_kwargs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    best = {"base_threshold": threshold_candidates[0], "adaptive_gain": gain_candidates[0]}
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for thr in threshold_candidates:
        for gain in gain_candidates:
            cfg = SimulationConfig(
                policy_name="adaptive_threshold_verify_fifo",
                slot_budget=slot_budget,
                max_wait_steps=max_wait_steps,
                threshold=float(thr),
                adaptive_gain=float(gain),
                rng_seed=rng_seed,
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=mean_pred_busy_steps,
                mean_pred_service_cost=mean_pred_service_cost,
                mean_pred_expected_consequence=mean_pred_expected_consequence,
            )
            res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
            obj = _objective(res["summary"], slot_budget=slot_budget, **score_kwargs)
            if obj > best_obj:
                best_obj = obj
                best = {"base_threshold": float(thr), "adaptive_gain": float(gain)}
                best_res = res
    assert best_res is not None
    return best, best_res


def _tune_proposed_ca_policy(
    jobs_fit: List[AlarmJob],
    total_steps_fit: int,
    *,
    slot_budget: int,
    max_wait_steps: int,
    rng_seed: int,
    cost_budget_window_steps: int,
    window_cost_budget: float | None,
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    v_grid: Sequence[float],
    clean_grid: Sequence[float],
    age_grid: Sequence[float],
    urgency_grid: Sequence[float],
    fail_grid: Sequence[float],
    busy_grid: Sequence[float],
    cost_grid: Sequence[float],
    admission_threshold_grid: Sequence[float],
    score_kwargs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    best_params = {}
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for v_weight in v_grid:
        for clean_penalty in clean_grid:
            for age_bonus in age_grid:
                for urgency_bonus in urgency_grid:
                    for fail_penalty in fail_grid:
                        for busy_penalty in busy_grid:
                            for cost_penalty in cost_grid:
                                for admission_thr in admission_threshold_grid:
                                    cfg = SimulationConfig(
                                        policy_name="proposed_ca_vq_hard",
                                        slot_budget=slot_budget,
                                        max_wait_steps=max_wait_steps,
                                        rng_seed=rng_seed,
                                        cost_budget_window_steps=cost_budget_window_steps,
                                        window_cost_budget=window_cost_budget,
                                        mean_pred_busy_steps=mean_pred_busy_steps,
                                        mean_pred_service_cost=mean_pred_service_cost,
                                        mean_pred_expected_consequence=mean_pred_expected_consequence,
                                        v_weight=float(v_weight),
                                        clean_penalty=float(clean_penalty),
                                        age_bonus=float(age_bonus),
                                        urgency_bonus=float(urgency_bonus),
                                        fail_penalty=float(fail_penalty),
                                        busy_penalty=float(busy_penalty),
                                        cost_penalty=float(cost_penalty),
                                        admission_score_threshold=float(admission_thr),
                                    )
                                    res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
                                    obj = _objective(res["summary"], slot_budget=slot_budget, **score_kwargs)
                                    if obj > best_obj:
                                        best_obj = obj
                                        best_params = {
                                            "v_weight": float(v_weight),
                                            "clean_penalty": float(clean_penalty),
                                            "age_bonus": float(age_bonus),
                                            "urgency_bonus": float(urgency_bonus),
                                            "fail_penalty": float(fail_penalty),
                                            "busy_penalty": float(busy_penalty),
                                            "cost_penalty": float(cost_penalty),
                                            "admission_score_threshold": float(admission_thr),
                                        }
                                        best_res = res
    assert best_res is not None
    return best_params, best_res


def _summary_row(policy_label: str, policy_res: Dict[str, object], tune_payload: Dict[str, object] | None = None) -> Dict[str, object]:
    row = {"policy": policy_label}
    row.update(policy_res["summary"])
    if tune_payload is not None:
        row["tuned"] = tune_payload
    return row


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    arrays_fit = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.fit_bank), int(args.decision_step_group))
    arrays_eval = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.eval_bank), int(args.decision_step_group))

    posterior_verify = fit_attack_posterior_from_banks(
        args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins
    )
    posterior_ddd = fit_attack_posterior_from_banks(
        args.clean_bank, args.attack_bank, signal_key="ddd_loss_alarm", n_bins=args.n_bins
    )
    service_models = fit_service_models_from_mixed_bank(args.fit_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_fit, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_fit, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)

    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_fit, float(args.busy_time_quantile))

    jobs_fit, total_steps_fit = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_eval, total_steps_eval = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_fit_ddd, _ = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_eval_ddd, _ = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )

    fit_stats = _job_stats(jobs_fit)
    eval_stats = _job_stats(jobs_eval)
    fit_arrival_diag = _arrival_diagnostics(jobs_fit, total_steps_fit)
    eval_arrival_diag = _arrival_diagnostics(jobs_eval, total_steps_eval)

    verify_threshold_candidates = _threshold_candidates(arrays_fit["verify_score"], list(args.threshold_quantiles))
    ddd_threshold_candidates = _threshold_candidates(arrays_fit["ddd_loss_recons"], list(args.threshold_quantiles))
    ec_fit = np.asarray([j.pred_expected_consequence for j in jobs_fit], dtype=float)
    ec_threshold_candidates = _threshold_candidates(ec_fit, list(args.threshold_quantiles))

    verify_signal = np.asarray(arrays_fit["verify_score"], dtype=float)
    verify_signal = verify_signal[np.isfinite(verify_signal)]
    verify_iqr = float(np.quantile(verify_signal, 0.80) - np.quantile(verify_signal, 0.20)) if verify_signal.size else 1.0
    verify_iqr = max(verify_iqr, 1e-6)
    adaptive_gain_candidates = [float(x) * verify_iqr for x in list(args.adaptive_gain_scale_list)]

    all_results: Dict[str, object] = {
        "config": vars(args),
        "environment": {
            "busy_time_unit": float(busy_time_unit),
            "fit_job_stats": fit_stats,
            "eval_job_stats": eval_stats,
            "fit_arrival_diagnostics": fit_arrival_diag,
            "eval_arrival_diagnostics": eval_arrival_diag,
            "severity_model_mode": str(args.consequence_mode),
            "consequence_blend_verify": float(args.consequence_blend_verify),
        },
        "slot_budget_results": {},
        "notes": {
            "phase": "phase3_consequence_aware_hard_constrained_scheduler_validation",
            "fit_bank": args.fit_bank,
            "eval_bank": args.eval_bank,
        },
    }

    for slot_budget in [int(x) for x in args.slot_budget_list]:
        window_cost_budget = None
        cost_budget_window_steps = 0
        if args.use_cost_budget:
            cost_budget_window_steps = int(args.cost_budget_window_steps)
            window_cost_budget = _derive_cost_budget_from_fit(
                jobs_fit,
                total_steps_fit,
                window_steps=cost_budget_window_steps,
                q=float(args.cost_budget_quantile),
            )

        cost_budget_per_step = None
        if window_cost_budget is not None and cost_budget_window_steps > 0:
            cost_budget_per_step = float(window_cost_budget) / max(int(cost_budget_window_steps), 1)

        score_kwargs = {
            "max_wait_steps": int(args.max_wait_steps),
            "clean_penalty": float(args.objective_clean_penalty),
            "delay_penalty": float(args.objective_delay_penalty),
            "queue_penalty": float(args.objective_queue_penalty),
            "cost_penalty": float(args.objective_cost_penalty),
            "cost_budget_per_step": cost_budget_per_step,
        }

        per_budget: Dict[str, object] = {
            "busy_time_unit": float(busy_time_unit),
            "window_cost_budget": window_cost_budget,
            "cost_budget_window_steps": int(cost_budget_window_steps),
            "policies": {},
            "tuning": {},
            "summary_rows": [],
        }

        thr_verify, _ = _tune_threshold_policy(
            jobs_fit, total_steps_fit, threshold_candidates=verify_threshold_candidates,
            policy_name="threshold_verify_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        thr_ddd, _ = _tune_threshold_policy(
            jobs_fit_ddd, total_steps_fit, threshold_candidates=ddd_threshold_candidates,
            policy_name="threshold_ddd_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        thr_ec, _ = _tune_threshold_policy(
            jobs_fit, total_steps_fit, threshold_candidates=ec_threshold_candidates,
            policy_name="threshold_expected_consequence_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        adaptive_best, _ = _tune_adaptive_threshold_policy(
            jobs_fit, total_steps_fit, threshold_candidates=verify_threshold_candidates, gain_candidates=adaptive_gain_candidates,
            slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"], score_kwargs=score_kwargs,
        )
        proposed_ca_best, _ = _tune_proposed_ca_policy(
            jobs_fit, total_steps_fit, slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            v_grid=args.vq_v_grid, clean_grid=args.vq_clean_grid, age_grid=args.vq_age_grid, urgency_grid=args.vq_urgency_grid,
            fail_grid=args.vq_fail_grid, busy_grid=args.vq_busy_grid, cost_grid=args.vq_cost_grid,
            admission_threshold_grid=args.vq_admission_threshold_grid, score_kwargs=score_kwargs,
        )

        policy_cfgs = {
            "always_fifo": SimulationConfig(
                policy_name="fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "random": SimulationConfig(
                policy_name="random", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "topk_verify": SimulationConfig(
                policy_name="topk_verify", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "topk_expected_consequence": SimulationConfig(
                policy_name="topk_expected_consequence", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "static_value_cost": SimulationConfig(
                policy_name="static_value_cost", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "static_expected_consequence_cost": SimulationConfig(
                policy_name="static_expected_consequence_cost", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget, mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"], mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "threshold_verify_fifo": SimulationConfig(
                policy_name="threshold_verify_fifo", threshold=float(thr_verify), slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "threshold_ddd_fifo": SimulationConfig(
                policy_name="threshold_ddd_fifo", threshold=float(thr_ddd), slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "threshold_expected_consequence_fifo": SimulationConfig(
                policy_name="threshold_expected_consequence_fifo", threshold=float(thr_ec), slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "adaptive_threshold_verify_fifo": SimulationConfig(
                policy_name="adaptive_threshold_verify_fifo", threshold=float(adaptive_best["base_threshold"]),
                adaptive_gain=float(adaptive_best["adaptive_gain"]), slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
            ),
            "proposed_ca_vq_hard": SimulationConfig(
                policy_name="proposed_ca_vq_hard", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"], mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                mean_pred_expected_consequence=fit_stats["mean_pred_expected_consequence"],
                v_weight=float(proposed_ca_best["v_weight"]), clean_penalty=float(proposed_ca_best["clean_penalty"]),
                age_bonus=float(proposed_ca_best["age_bonus"]), urgency_bonus=float(proposed_ca_best["urgency_bonus"]),
                fail_penalty=float(proposed_ca_best["fail_penalty"]), busy_penalty=float(proposed_ca_best["busy_penalty"]),
                cost_penalty=float(proposed_ca_best["cost_penalty"]), admission_score_threshold=float(proposed_ca_best["admission_score_threshold"]),
            ),
        }

        policy_jobs = {
            "always_fifo": jobs_eval,
            "random": jobs_eval,
            "topk_verify": jobs_eval,
            "topk_expected_consequence": jobs_eval,
            "static_value_cost": jobs_eval,
            "static_expected_consequence_cost": jobs_eval,
            "threshold_verify_fifo": jobs_eval,
            "threshold_ddd_fifo": jobs_eval_ddd,
            "threshold_expected_consequence_fifo": jobs_eval,
            "adaptive_threshold_verify_fifo": jobs_eval,
            "proposed_ca_vq_hard": jobs_eval,
        }

        per_budget["tuning"] = {
            "threshold_verify_fifo": {"threshold": float(thr_verify)},
            "threshold_ddd_fifo": {"threshold": float(thr_ddd)},
            "threshold_expected_consequence_fifo": {"threshold": float(thr_ec)},
            "adaptive_threshold_verify_fifo": adaptive_best,
            "proposed_ca_vq_hard": proposed_ca_best,
        }

        for label, cfg in policy_cfgs.items():
            res = _run_one_policy(policy_jobs[label], total_steps_eval, cfg)
            per_budget["policies"][label] = res
            tune_payload = per_budget["tuning"].get(label, None)
            per_budget["summary_rows"].append(_summary_row(label, res, tune_payload))

        # Useful compact comparisons for later paper writing.
        compact = {}
        for row in per_budget["summary_rows"]:
            compact[row["policy"]] = {
                "weighted_attack_recall_no_backend_fail": round(float(row["weighted_attack_recall_no_backend_fail"]), 4),
                "unnecessary_mtd_count": int(row["unnecessary_mtd_count"]),
                "queue_delay_p95": round(float(row["queue_delay_p95"]), 4),
                "average_service_cost_per_step": round(float(row["average_service_cost_per_step"]), 6),
                "pred_expected_consequence_served_ratio": round(float(row["pred_expected_consequence_served_ratio"]), 4),
            }
        per_budget["compact"] = compact
        all_results["slot_budget_results"][str(slot_budget)] = per_budget

    return all_results


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
    results = run_experiment(args)

    np.save(args.output, results, allow_pickle=True)
    summary_path = args.output.replace(".npy", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    print(f"Saved result npy: {args.output}")
    print(f"Saved compact summary json: {summary_path}")
    print("\n==== Environment diagnostics ====")
    print(json.dumps(_to_jsonable(results["environment"]), ensure_ascii=False, indent=2))
    print("\n==== Compact summaries ====")
    for slot_budget, payload in results["slot_budget_results"].items():
        print(f"\n-- slot_budget={slot_budget} --")
        print("busy_time_unit", payload["busy_time_unit"])
        print("window_cost_budget", payload["window_cost_budget"])
        for policy, compact in payload["compact"].items():
            print(policy, compact)


if __name__ == "__main__":
    main()

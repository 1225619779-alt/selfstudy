
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from scheduler.calibration import (
    BinnedStatisticModel,
    fit_attack_posterior_from_banks,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
    summarize_array,
)
from scheduler.policies_hard import AlarmJob, SimulationConfig, build_jobs_from_arrays, simulate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-2 hard-constrained offline evaluation for budget-aware backend MTD scheduling."
    )
    parser.add_argument("--clean_bank", type=str, required=True)
    parser.add_argument("--attack_bank", type=str, required=True)
    parser.add_argument("--fit_bank", type=str, required=True)
    parser.add_argument("--eval_bank", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="metric/case14/budget_scheduler_phase2_hard.npy",
    )
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--max_wait_steps", type=int, default=10)
    parser.add_argument("--decision_step_group", type=int, default=1, help="Aggregate original steps into larger decision epochs.")
    parser.add_argument("--busy_time_quantile", type=float, default=0.50, help="Quantile of fit-bank service_time used as one backend busy-time unit.")
    parser.add_argument("--use_cost_budget", action="store_true", help="Enable rolling hard cost budget.")
    parser.add_argument("--cost_budget_window_steps", type=int, default=20)
    parser.add_argument("--cost_budget_quantile", type=float, default=0.60, help="Quantile of rolling arrival-cost demand used as hard budget.")
    parser.add_argument("--threshold_quantiles", type=float, nargs="*", default=[0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--adaptive_gain_scale_list", type=float, nargs="*", default=[0.0, 0.10, 0.20, 0.40])
    parser.add_argument("--vq_v_grid", type=float, nargs="*", default=[1.0, 2.0, 4.0])
    parser.add_argument("--vq_age_grid", type=float, nargs="*", default=[0.0, 0.10, 0.20])
    parser.add_argument("--vq_fail_grid", type=float, nargs="*", default=[0.0, 0.05])
    parser.add_argument("--vq_busy_grid", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--vq_cost_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    parser.add_argument("--clean_penalty", type=float, default=0.60)
    parser.add_argument("--delay_penalty", type=float, default=0.15)
    parser.add_argument("--queue_penalty", type=float, default=0.10)
    parser.add_argument("--cost_penalty", type=float, default=0.05)
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


def _normalize_value_proxy(
    arrays: Dict[str, np.ndarray],
    fit_verify_score: np.ndarray,
) -> np.ndarray:
    if "consequence_proxy" in arrays:
        x = np.asarray(arrays["consequence_proxy"], dtype=float)
        fit_x = x[np.isfinite(x)]
    elif "value_proxy" in arrays:
        x = np.asarray(arrays["value_proxy"], dtype=float)
        fit_x = x[np.isfinite(x)]
    else:
        x = np.asarray(arrays["verify_score"], dtype=float)
        fit_x = np.asarray(fit_verify_score, dtype=float)
        fit_x = fit_x[np.isfinite(fit_x)]
    scale = float(np.quantile(fit_x, 0.90)) if fit_x.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(np.asarray(x, dtype=float) / scale, 0.0, 5.0)


def _predict_jobs(
    arrays: Dict[str, np.ndarray],
    *,
    posterior_model: BinnedStatisticModel,
    posterior_signal_key: str,
    service_models: Dict[str, BinnedStatisticModel],
    service_signal_key: str,
    fit_verify_score: np.ndarray,
    busy_time_unit: float,
) -> Tuple[List[AlarmJob], int]:
    if posterior_signal_key not in arrays:
        raise KeyError(f"posterior_signal_key={posterior_signal_key!r} not found in arrays")
    if service_signal_key not in arrays:
        raise KeyError(f"service_signal_key={service_signal_key!r} not found in arrays")

    x_post = np.asarray(arrays[posterior_signal_key], dtype=float)
    x_srv = np.asarray(arrays[service_signal_key], dtype=float)
    p_hat = posterior_model.predict(x_post)
    tau_hat = service_models["service_time"].predict(x_srv)
    cost_hat = service_models["service_cost"].predict(x_srv)
    fail_hat = service_models["backend_fail"].predict(x_srv)
    value_proxy = _normalize_value_proxy(arrays, fit_verify_score)
    return build_jobs_from_arrays(
        arrays,
        p_hat=p_hat,
        tau_hat=tau_hat,
        cost_hat=cost_hat,
        fail_hat=fail_hat,
        value_proxy=value_proxy,
        busy_time_unit=busy_time_unit,
    )


def _threshold_candidates(x: np.ndarray, quantiles: List[float]) -> List[float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0]
    candidates = sorted({float(np.quantile(arr, q)) for q in quantiles})
    return candidates


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
    actual_busy = np.asarray([j.actual_busy_steps for j in jobs], dtype=float)
    return {
        "mean_pred_busy_steps": float(np.mean(pred_busy)) if pred_busy.size else 1.0,
        "mean_pred_service_cost": float(np.mean(pred_cost)) if pred_cost.size else 1.0,
        "busy_step_stats": summarize_array(actual_busy),
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
            )
            res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
            obj = _objective(res["summary"], slot_budget=slot_budget, **score_kwargs)
            if obj > best_obj:
                best_obj = obj
                best = {"base_threshold": float(thr), "adaptive_gain": float(gain)}
                best_res = res
    assert best_res is not None
    return best, best_res


def _tune_proposed_policy(
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
    v_grid: List[float],
    age_grid: List[float],
    fail_grid: List[float],
    busy_grid: List[float],
    cost_grid: List[float],
    score_kwargs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    best_params = {
        "v_weight": v_grid[0],
        "age_bonus": age_grid[0],
        "fail_penalty": fail_grid[0],
        "busy_penalty": busy_grid[0],
        "cost_penalty": cost_grid[0],
    }
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for v_weight in v_grid:
        for age_bonus in age_grid:
            for fail_penalty in fail_grid:
                for busy_penalty in busy_grid:
                    for cost_penalty in cost_grid:
                        cfg = SimulationConfig(
                            policy_name="proposed_vq_hard",
                            slot_budget=slot_budget,
                            max_wait_steps=max_wait_steps,
                            rng_seed=rng_seed,
                            cost_budget_window_steps=cost_budget_window_steps,
                            window_cost_budget=window_cost_budget,
                            mean_pred_busy_steps=mean_pred_busy_steps,
                            mean_pred_service_cost=mean_pred_service_cost,
                            v_weight=float(v_weight),
                            age_bonus=float(age_bonus),
                            fail_penalty=float(fail_penalty),
                            busy_penalty=float(busy_penalty),
                            cost_penalty=float(cost_penalty),
                        )
                        res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
                        obj = _objective(res["summary"], slot_budget=slot_budget, **score_kwargs)
                        if obj > best_obj:
                            best_obj = obj
                            best_params = {
                                "v_weight": float(v_weight),
                                "age_bonus": float(age_bonus),
                                "fail_penalty": float(fail_penalty),
                                "busy_penalty": float(busy_penalty),
                                "cost_penalty": float(cost_penalty),
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


def main() -> None:
    args = parse_args()
    ensure_parent(args.output)

    arrays_fit = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.fit_bank), int(args.decision_step_group))
    arrays_eval = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.eval_bank), int(args.decision_step_group))

    posterior_verify = fit_attack_posterior_from_banks(
        args.clean_bank,
        args.attack_bank,
        signal_key="score_phys_l2",
        n_bins=args.n_bins,
    )
    posterior_ddd = fit_attack_posterior_from_banks(
        args.clean_bank,
        args.attack_bank,
        signal_key="ddd_loss_alarm",
        n_bins=args.n_bins,
    )
    service_models = fit_service_models_from_mixed_bank(args.fit_bank, signal_key="verify_score", n_bins=args.n_bins)

    busy_time_unit = _busy_time_unit_from_fit(arrays_fit, float(args.busy_time_quantile))

    jobs_fit, total_steps_fit = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_eval, total_steps_eval = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )

    jobs_fit_ddd, _ = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_eval_ddd, _ = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )

    fit_stats = _job_stats(jobs_fit)
    eval_stats = _job_stats(jobs_eval)
    fit_arrival_diag = _arrival_diagnostics(jobs_fit, total_steps_fit)
    eval_arrival_diag = _arrival_diagnostics(jobs_eval, total_steps_eval)

    verify_threshold_candidates = _threshold_candidates(arrays_fit["verify_score"], list(args.threshold_quantiles))
    ddd_threshold_candidates = _threshold_candidates(arrays_fit["ddd_loss_recons"], list(args.threshold_quantiles))
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
        },
        "slot_budget_results": {},
        "notes": {
            "phase": "phase2_hard_constrained_scheduler_validation",
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
            "clean_penalty": float(args.clean_penalty),
            "delay_penalty": float(args.delay_penalty),
            "queue_penalty": float(args.queue_penalty),
            "cost_penalty": float(args.cost_penalty),
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

        thr_verify, thr_verify_fit_res = _tune_threshold_policy(
            jobs_fit,
            total_steps_fit,
            threshold_candidates=verify_threshold_candidates,
            policy_name="threshold_verify_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            score_kwargs=score_kwargs,
        )
        thr_ddd, thr_ddd_fit_res = _tune_threshold_policy(
            jobs_fit_ddd,
            total_steps_fit,
            threshold_candidates=ddd_threshold_candidates,
            policy_name="threshold_ddd_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            score_kwargs=score_kwargs,
        )
        adaptive_params, adaptive_fit_res = _tune_adaptive_threshold_policy(
            jobs_fit,
            total_steps_fit,
            threshold_candidates=verify_threshold_candidates,
            gain_candidates=adaptive_gain_candidates,
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            score_kwargs=score_kwargs,
        )
        proposed_params, proposed_fit_res = _tune_proposed_policy(
            jobs_fit,
            total_steps_fit,
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            v_grid=list(args.vq_v_grid),
            age_grid=list(args.vq_age_grid),
            fail_grid=list(args.vq_fail_grid),
            busy_grid=list(args.vq_busy_grid),
            cost_grid=list(args.vq_cost_grid),
            score_kwargs=score_kwargs,
        )

        per_budget["tuning"] = {
            "threshold_verify": {"best_threshold": thr_verify, "fit_summary": thr_verify_fit_res["summary"]},
            "threshold_ddd": {"best_threshold": thr_ddd, "fit_summary": thr_ddd_fit_res["summary"]},
            "adaptive_threshold_verify": {"best_params": adaptive_params, "fit_summary": adaptive_fit_res["summary"]},
            "proposed_vq_hard": {"best_params": proposed_params, "fit_summary": proposed_fit_res["summary"]},
        }

        eval_policy_payloads = {
            "always_fifo": SimulationConfig(
                policy_name="fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "random": SimulationConfig(
                policy_name="random",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "topk_verify": SimulationConfig(
                policy_name="topk_verify",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "static_value_cost": SimulationConfig(
                policy_name="static_value_cost",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "threshold_verify_fifo": SimulationConfig(
                policy_name="threshold_verify_fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                threshold=thr_verify,
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "threshold_ddd_fifo": SimulationConfig(
                policy_name="threshold_ddd_fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                threshold=thr_ddd,
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "adaptive_threshold_verify_fifo": SimulationConfig(
                policy_name="adaptive_threshold_verify_fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                threshold=adaptive_params["base_threshold"],
                adaptive_gain=adaptive_params["adaptive_gain"],
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
            ),
            "proposed_vq_hard": SimulationConfig(
                policy_name="proposed_vq_hard",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                rng_seed=int(args.rng_seed),
                cost_budget_window_steps=cost_budget_window_steps,
                window_cost_budget=window_cost_budget,
                mean_pred_busy_steps=fit_stats["mean_pred_busy_steps"],
                mean_pred_service_cost=fit_stats["mean_pred_service_cost"],
                v_weight=proposed_params["v_weight"],
                age_bonus=proposed_params["age_bonus"],
                fail_penalty=proposed_params["fail_penalty"],
                busy_penalty=proposed_params["busy_penalty"],
                cost_penalty=proposed_params["cost_penalty"],
            ),
        }

        policy_jobs_lookup = {
            "always_fifo": jobs_eval,
            "random": jobs_eval,
            "topk_verify": jobs_eval,
            "static_value_cost": jobs_eval,
            "threshold_verify_fifo": jobs_eval,
            "threshold_ddd_fifo": jobs_eval_ddd,
            "adaptive_threshold_verify_fifo": jobs_eval,
            "proposed_vq_hard": jobs_eval,
        }

        for label, cfg in eval_policy_payloads.items():
            jobs_this = policy_jobs_lookup[label]
            res = _run_one_policy(jobs_this, total_steps_eval, cfg)
            per_budget["policies"][label] = res
            tune_payload = None
            if label == "threshold_verify_fifo":
                tune_payload = {"best_threshold": thr_verify}
            elif label == "threshold_ddd_fifo":
                tune_payload = {"best_threshold": thr_ddd}
            elif label == "adaptive_threshold_verify_fifo":
                tune_payload = adaptive_params
            elif label == "proposed_vq_hard":
                tune_payload = proposed_params
            per_budget["summary_rows"].append(_summary_row(label, res, tune_payload=tune_payload))

        all_results["slot_budget_results"][str(slot_budget)] = per_budget

    np.save(args.output, all_results, allow_pickle=True)

    summary_json_path = os.path.splitext(args.output)[0] + ".summary.json"
    compact = {
        "config": all_results["config"],
        "environment": all_results["environment"],
        "slot_budget_results": {
            key: {
                "busy_time_unit": value["busy_time_unit"],
                "window_cost_budget": value["window_cost_budget"],
                "cost_budget_window_steps": value["cost_budget_window_steps"],
                "tuning": value["tuning"],
                "summary_rows": value["summary_rows"],
            }
            for key, value in all_results["slot_budget_results"].items()
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    print(f"Saved result npy: {args.output}")
    print(f"Saved compact summary json: {summary_json_path}")
    print("\n==== Environment diagnostics ====")
    print(json.dumps(all_results["environment"], indent=2, ensure_ascii=False))
    print("\n==== Compact summaries ====")
    for slot_budget, payload in all_results["slot_budget_results"].items():
        print(f"\n-- slot_budget={slot_budget} --")
        print("busy_time_unit", payload["busy_time_unit"])
        print("window_cost_budget", payload["window_cost_budget"])
        for row in payload["summary_rows"]:
            metric_view = {
                "weighted_attack_recall_no_backend_fail": round(float(row["weighted_attack_recall_no_backend_fail"]), 4),
                "unnecessary_mtd_count": int(row["unnecessary_mtd_count"]),
                "queue_delay_p95": round(float(row["queue_delay_p95"]), 4),
                "max_queue_len": int(row["max_queue_len"]),
                "server_utilization": round(float(row["server_utilization"]), 4),
                "average_service_cost_per_step": round(float(row["average_service_cost_per_step"]), 6),
            }
            print(row["policy"], metric_view)


if __name__ == "__main__":
    main()

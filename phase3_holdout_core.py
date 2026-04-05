from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import SimulationConfig

from evaluation_budget_scheduler_phase3 import (
    _aggregate_arrival_steps,
    _arrival_diagnostics,
    _busy_time_unit_from_fit,
    _derive_cost_budget_from_fit,
    _job_stats,
    _objective,
    _predict_jobs,
    _run_one_policy,
    _threshold_candidates,
    _tune_adaptive_threshold_policy,
    _tune_proposed_ca_policy,
    _tune_threshold_policy,
)


def _policy_compact(summary: Dict[str, float]) -> Dict[str, float | int]:
    return {
        "weighted_attack_recall_no_backend_fail": round(float(summary["weighted_attack_recall_no_backend_fail"]), 4),
        "unnecessary_mtd_count": int(summary["unnecessary_mtd_count"]),
        "queue_delay_p95": round(float(summary["queue_delay_p95"]), 4),
        "average_service_cost_per_step": round(float(summary["average_service_cost_per_step"]), 6),
        "pred_expected_consequence_served_ratio": round(float(summary["pred_expected_consequence_served_ratio"]), 4),
    }


def _build_policy_cfgs(
    *,
    slot_budget: int,
    max_wait_steps: int,
    rng_seed: int,
    cost_budget_window_steps: int,
    window_cost_budget: float | None,
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    thr_verify: float,
    thr_ddd: float,
    thr_ec: float,
    adaptive_best: Dict[str, float],
    proposed_best: Dict[str, float],
) -> Dict[str, SimulationConfig]:
    common = dict(
        slot_budget=slot_budget,
        max_wait_steps=max_wait_steps,
        rng_seed=rng_seed,
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=mean_pred_busy_steps,
        mean_pred_service_cost=mean_pred_service_cost,
        mean_pred_expected_consequence=mean_pred_expected_consequence,
    )
    return {
        "always_fifo": SimulationConfig(policy_name="fifo", **common),
        "random": SimulationConfig(policy_name="random", **common),
        "topk_verify": SimulationConfig(policy_name="topk_verify", **common),
        "topk_expected_consequence": SimulationConfig(policy_name="topk_expected_consequence", **common),
        "static_value_cost": SimulationConfig(policy_name="static_value_cost", **common),
        "static_expected_consequence_cost": SimulationConfig(policy_name="static_expected_consequence_cost", **common),
        "threshold_verify_fifo": SimulationConfig(policy_name="threshold_verify_fifo", threshold=float(thr_verify), **common),
        "threshold_ddd_fifo": SimulationConfig(policy_name="threshold_ddd_fifo", threshold=float(thr_ddd), **common),
        "threshold_expected_consequence_fifo": SimulationConfig(policy_name="threshold_expected_consequence_fifo", threshold=float(thr_ec), **common),
        "adaptive_threshold_verify_fifo": SimulationConfig(
            policy_name="adaptive_threshold_verify_fifo",
            threshold=float(adaptive_best["base_threshold"]),
            adaptive_gain=float(adaptive_best["adaptive_gain"]),
            **common,
        ),
        "proposed_ca_vq_hard": SimulationConfig(
            policy_name="proposed_ca_vq_hard",
            v_weight=float(proposed_best["v_weight"]),
            clean_penalty=float(proposed_best["clean_penalty"]),
            age_bonus=float(proposed_best["age_bonus"]),
            urgency_bonus=float(proposed_best["urgency_bonus"]),
            fail_penalty=float(proposed_best["fail_penalty"]),
            busy_penalty=float(proposed_best["busy_penalty"]),
            cost_penalty=float(proposed_best["cost_penalty"]),
            admission_score_threshold=float(proposed_best["admission_score_threshold"]),
            **common,
        ),
    }


def _policy_job_map(jobs_default: List, jobs_ddd: List) -> Dict[str, List]:
    return {
        "always_fifo": jobs_default,
        "random": jobs_default,
        "topk_verify": jobs_default,
        "topk_expected_consequence": jobs_default,
        "static_value_cost": jobs_default,
        "static_expected_consequence_cost": jobs_default,
        "threshold_verify_fifo": jobs_default,
        "threshold_ddd_fifo": jobs_ddd,
        "threshold_expected_consequence_fifo": jobs_default,
        "adaptive_threshold_verify_fifo": jobs_default,
        "proposed_ca_vq_hard": jobs_default,
    }


def run_train_tune_eval(args: SimpleNamespace) -> Dict[str, object]:
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.train_bank), int(args.decision_step_group))
    arrays_tune = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.tune_bank), int(args.decision_step_group))
    arrays_eval = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.eval_bank), int(args.decision_step_group))

    posterior_verify = fit_attack_posterior_from_banks(
        args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins
    )
    posterior_ddd = fit_attack_posterior_from_banks(
        args.clean_bank, args.attack_bank, signal_key="ddd_loss_alarm", n_bins=args.n_bins
    )
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp

    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))

    jobs_train, total_steps_train = _predict_jobs(
        arrays_train,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_tune, total_steps_tune = _predict_jobs(
        arrays_tune,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
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
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_tune_ddd, _ = _predict_jobs(
        arrays_tune,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
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
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )

    train_stats = _job_stats(jobs_train)
    tune_stats = _job_stats(jobs_tune)
    eval_stats = _job_stats(jobs_eval)
    train_arrival_diag = _arrival_diagnostics(jobs_train, total_steps_train)
    tune_arrival_diag = _arrival_diagnostics(jobs_tune, total_steps_tune)
    eval_arrival_diag = _arrival_diagnostics(jobs_eval, total_steps_eval)

    verify_threshold_candidates = _threshold_candidates(arrays_train["verify_score"], list(args.threshold_quantiles))
    ddd_threshold_candidates = _threshold_candidates(arrays_train["ddd_loss_recons"], list(args.threshold_quantiles))
    ec_train = np.asarray([j.pred_expected_consequence for j in jobs_train], dtype=float)
    ec_threshold_candidates = _threshold_candidates(ec_train, list(args.threshold_quantiles))

    verify_signal = np.asarray(arrays_train["verify_score"], dtype=float)
    verify_signal = verify_signal[np.isfinite(verify_signal)]
    verify_iqr = float(np.quantile(verify_signal, 0.80) - np.quantile(verify_signal, 0.20)) if verify_signal.size else 1.0
    verify_iqr = max(verify_iqr, 1e-6)
    adaptive_gain_candidates = [float(x) * verify_iqr for x in list(args.adaptive_gain_scale_list)]

    all_results: Dict[str, object] = {
        "config": vars(args),
        "environment": {
            "busy_time_unit": float(busy_time_unit),
            "train_job_stats": train_stats,
            "tune_job_stats": tune_stats,
            "eval_job_stats": eval_stats,
            "train_arrival_diagnostics": train_arrival_diag,
            "tune_arrival_diagnostics": tune_arrival_diag,
            "eval_arrival_diagnostics": eval_arrival_diag,
            "severity_model_mode": str(args.consequence_mode),
            "consequence_blend_verify": float(args.consequence_blend_verify),
        },
        "slot_budget_results": {},
        "notes": {
            "phase": "phase3_clean_train_val_test_protocol",
            "train_bank": args.train_bank,
            "tune_bank": args.tune_bank,
            "eval_bank": args.eval_bank,
        },
    }

    for slot_budget in [int(x) for x in args.slot_budget_list]:
        window_cost_budget = None
        cost_budget_window_steps = 0
        if args.use_cost_budget:
            cost_budget_window_steps = int(args.cost_budget_window_steps)
            window_cost_budget = _derive_cost_budget_from_fit(
                jobs_train,
                total_steps_train,
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
            "tuning": {},
            "tune_compact": {},
            "eval_compact": {},
        }

        thr_verify, tune_thr_verify_res = _tune_threshold_policy(
            jobs_tune, total_steps_tune, threshold_candidates=verify_threshold_candidates,
            policy_name="threshold_verify_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats["mean_pred_service_cost"], mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        thr_ddd, tune_thr_ddd_res = _tune_threshold_policy(
            jobs_tune_ddd, total_steps_tune, threshold_candidates=ddd_threshold_candidates,
            policy_name="threshold_ddd_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats["mean_pred_service_cost"], mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        thr_ec, tune_thr_ec_res = _tune_threshold_policy(
            jobs_tune, total_steps_tune, threshold_candidates=ec_threshold_candidates,
            policy_name="threshold_expected_consequence_fifo", slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed), cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget, mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats["mean_pred_service_cost"], mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            score_kwargs=score_kwargs,
        )
        adaptive_best, tune_adaptive_res = _tune_adaptive_threshold_policy(
            jobs_tune, total_steps_tune, threshold_candidates=verify_threshold_candidates, gain_candidates=adaptive_gain_candidates,
            slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=train_stats["mean_pred_busy_steps"], mean_pred_service_cost=train_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"], score_kwargs=score_kwargs,
        )
        proposed_best, tune_proposed_res = _tune_proposed_ca_policy(
            jobs_tune, total_steps_tune, slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=train_stats["mean_pred_busy_steps"], mean_pred_service_cost=train_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            v_grid=args.vq_v_grid, clean_grid=args.vq_clean_grid, age_grid=args.vq_age_grid, urgency_grid=args.vq_urgency_grid,
            fail_grid=args.vq_fail_grid, busy_grid=args.vq_busy_grid, cost_grid=args.vq_cost_grid,
            admission_threshold_grid=args.vq_admission_threshold_grid, score_kwargs=score_kwargs,
        )

        per_budget["tuning"] = {
            "threshold_verify_fifo": {"threshold": float(thr_verify)},
            "threshold_ddd_fifo": {"threshold": float(thr_ddd)},
            "threshold_expected_consequence_fifo": {"threshold": float(thr_ec)},
            "adaptive_threshold_verify_fifo": adaptive_best,
            "proposed_ca_vq_hard": proposed_best,
        }
        per_budget["tune_compact"] = {
            "threshold_verify_fifo": _policy_compact(tune_thr_verify_res["summary"]),
            "threshold_ddd_fifo": _policy_compact(tune_thr_ddd_res["summary"]),
            "threshold_expected_consequence_fifo": _policy_compact(tune_thr_ec_res["summary"]),
            "adaptive_threshold_verify_fifo": _policy_compact(tune_adaptive_res["summary"]),
            "proposed_ca_vq_hard": _policy_compact(tune_proposed_res["summary"]),
        }

        policy_cfgs = _build_policy_cfgs(
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
            mean_pred_service_cost=train_stats["mean_pred_service_cost"],
            mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            thr_verify=thr_verify,
            thr_ddd=thr_ddd,
            thr_ec=thr_ec,
            adaptive_best=adaptive_best,
            proposed_best=proposed_best,
        )
        eval_jobs = _policy_job_map(jobs_eval, jobs_eval_ddd)
        for label, cfg in policy_cfgs.items():
            res = _run_one_policy(eval_jobs[label], total_steps_eval, cfg)
            per_budget["eval_compact"][label] = _policy_compact(res["summary"])

        all_results["slot_budget_results"][str(slot_budget)] = per_budget

    return all_results

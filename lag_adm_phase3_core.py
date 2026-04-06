from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import AlarmJob

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

EPS = 1e-12


@dataclass
class LagAdmissionPhase3DispatchConfig:
    slot_budget: int
    max_wait_steps: int
    rng_seed: int = 20260402
    window_cost_budget: Optional[float] = None
    cost_budget_window_steps: int = 0
    mean_pred_busy_steps: float = 1.0
    mean_pred_service_cost: float = 1.0
    mean_pred_expected_consequence: float = 1.0
    # Lagrangian admission
    adm_reward_weight: float = 1.0
    adm_clean_lambda: float = 2.0
    adm_fail_lambda: float = 0.0
    adm_busy_lambda: float = 0.5
    adm_queue_lambda: float = 0.5
    adm_cost_lambda: float = 0.0
    adm_threshold: float = 0.0
    target_clean_ratio: float = 0.0
    # phase3 dispatch parameters
    dsp_v_weight: float = 1.0
    dsp_clean_penalty: float = 0.0
    dsp_age_bonus: float = 0.0
    dsp_urgency_bonus: float = 0.0
    dsp_fail_penalty: float = 0.0
    dsp_busy_penalty: float = 0.0
    dsp_cost_penalty: float = 0.0
    dsp_min_total_score: float = -1e18


@dataclass
class _QueuedJob:
    job: AlarmJob
    enqueue_step: int


@dataclass
class _ActiveServer:
    job_id: int
    busy_until_step: int


@dataclass
class _TuneRecord:
    cfg: LagAdmissionPhase3DispatchConfig
    objective: float
    feasible: bool
    clean_violation: float
    summary: Dict[str, float]


def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: LagAdmissionPhase3DispatchConfig) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))


def _queue_pressure(queue_len: int, cfg: LagAdmissionPhase3DispatchConfig) -> float:
    return float(queue_len / max(int(cfg.slot_budget), 1))


def _cost_pressure(cost_spent_window: float, cfg: LagAdmissionPhase3DispatchConfig) -> float:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))


def _job_fits_cost_budget(job: AlarmJob, *, cost_spent_window: float, cfg: LagAdmissionPhase3DispatchConfig) -> bool:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0 or cfg.cost_budget_window_steps <= 0:
        return True
    projected = float(cost_spent_window) + float(job.actual_service_cost)
    return bool(projected <= float(cfg.window_cost_budget) + 1e-9)


def _admission_score(
    job: AlarmJob,
    *,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: LagAdmissionPhase3DispatchConfig,
) -> float:
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    server_pressure = _server_pressure(active_servers, cfg)
    queue_pressure = _queue_pressure(queue_len, cfg)
    cost_pressure = _cost_pressure(cost_spent_window, cfg)
    reward = float(cfg.adm_reward_weight) * norm_ec
    clean = float(cfg.adm_clean_lambda) * float(1.0 - job.pred_attack_prob)
    fail = float(cfg.adm_fail_lambda) * float(job.pred_fail_prob)
    busy = float(cfg.adm_busy_lambda) * server_pressure * norm_busy
    qpen = float(cfg.adm_queue_lambda) * queue_pressure * norm_busy
    cpen = float(cfg.adm_cost_lambda) * cost_pressure * norm_cost
    return float(reward - clean - fail - busy - qpen - cpen)


def _dispatch_score(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: LagAdmissionPhase3DispatchConfig,
) -> float:
    job = item.job
    age = max(0, int(step - item.enqueue_step))
    ttl_left = max(int(cfg.max_wait_steps) - age, 0)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    pressure = _server_pressure(active_servers, cfg) + _queue_pressure(queue_len, cfg)
    cost_pressure = _cost_pressure(cost_spent_window, cfg)
    value_term = float(cfg.dsp_v_weight) * norm_ec
    clean_term = float(cfg.dsp_clean_penalty) * float(1.0 - job.pred_attack_prob)
    age_term = float(cfg.dsp_age_bonus) * float(age / max(cfg.max_wait_steps, 1))
    urgency_term = float(cfg.dsp_urgency_bonus) * float(1.0 / (ttl_left + 1.0))
    fail_term = float(cfg.dsp_fail_penalty) * float(job.pred_fail_prob)
    busy_term = float(cfg.dsp_busy_penalty) * pressure * norm_busy
    cost_term = float(cfg.dsp_cost_penalty) * cost_pressure * norm_cost
    return float(value_term + age_term + urgency_term - clean_term - fail_term - busy_term - cost_term)


def simulate_lag_admission_phase3_dispatch(
    jobs: Sequence[AlarmJob], *, total_steps: int, cfg: LagAdmissionPhase3DispatchConfig
) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    rng = np.random.default_rng(int(cfg.rng_seed))
    active_servers: List[_ActiveServer] = []
    rolling_cost: Deque[Tuple[int, float]] = deque()

    total_true_severity = float(np.sum([job.severity_true for job in jobs if job.is_attack == 1]))
    served_true_severity = 0.0
    served_true_severity_no_fail = 0.0
    total_pred_expected_consequence = float(np.sum([job.pred_expected_consequence for job in jobs]))
    served_pred_expected_consequence = 0.0

    served_jobs: List[int] = []
    served_attack_jobs: List[int] = []
    served_clean_jobs: List[int] = []
    dropped_jobs_admission: List[int] = []
    dropped_jobs_ttl: List[int] = []
    dropped_jobs_budget_blocked: List[int] = []
    queue_delays_served: List[int] = []
    attack_delays_served: List[int] = []
    clean_delays_served: List[int] = []
    step_trace: List[Dict[str, float]] = []

    total_service_time = 0.0
    total_service_cost = 0.0
    total_backend_fail = 0
    occupied_server_steps = 0.0
    queue_len_trace: List[int] = []
    active_server_trace: List[int] = []
    budget_blocked_starts = 0

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]

        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        arrivals_this_step = arrivals.get(step, [])
        for job in arrivals_this_step:
            score = _admission_score(
                job,
                queue_len=len(queue),
                active_servers=active_servers,
                cost_spent_window=current_cost_window,
                cfg=cfg,
            )
            if score >= float(cfg.adm_threshold):
                queue.append(_QueuedJob(job=job, enqueue_step=step))
            else:
                dropped_jobs_admission.append(job.job_id)

        new_queue: List[_QueuedJob] = []
        for item in queue:
            age = int(step - item.enqueue_step)
            if age > int(cfg.max_wait_steps):
                dropped_jobs_ttl.append(item.job.job_id)
            else:
                new_queue.append(item)
        queue = new_queue

        available_servers = max(int(cfg.slot_budget) - len(active_servers), 0)
        selected_items: List[_QueuedJob] = []
        residual_queue: List[_QueuedJob] = []

        if available_servers > 0 and queue:
            scored = [
                (
                    _dispatch_score(
                        item,
                        step=step,
                        queue_len=len(queue),
                        active_servers=active_servers,
                        cost_spent_window=current_cost_window,
                        cfg=cfg,
                    ),
                    item,
                )
                for item in queue
            ]
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)

            running_cost_window = float(current_cost_window)
            selected_ids: set[int] = set()
            residual_ids: set[int] = set()
            for score, item in scored:
                jid = int(item.job.job_id)
                if len(selected_items) >= available_servers:
                    residual_queue.append(item)
                    residual_ids.add(jid)
                    continue
                if score < float(cfg.dsp_min_total_score):
                    residual_queue.append(item)
                    residual_ids.add(jid)
                    continue
                if not _job_fits_cost_budget(item.job, cost_spent_window=running_cost_window, cfg=cfg):
                    residual_queue.append(item)
                    residual_ids.add(jid)
                    budget_blocked_starts += 1
                    dropped_jobs_budget_blocked.append(jid)
                    continue
                selected_items.append(item)
                selected_ids.add(jid)
                running_cost_window += float(item.job.actual_service_cost)

            for _, item in scored:
                jid = int(item.job.job_id)
                if jid not in selected_ids and jid not in residual_ids:
                    residual_queue.append(item)
                    residual_ids.add(jid)
            queue = residual_queue

        used_time = 0.0
        used_cost = 0.0
        for item in selected_items:
            job = item.job
            delay = int(step - item.enqueue_step)
            served_jobs.append(job.job_id)
            queue_delays_served.append(delay)
            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
            used_time += float(job.actual_service_time)
            used_cost += float(job.actual_service_cost)
            total_backend_fail += int(job.actual_backend_fail)
            served_pred_expected_consequence += float(job.pred_expected_consequence)
            active_servers.append(_ActiveServer(job_id=job.job_id, busy_until_step=int(step + max(int(job.actual_busy_steps), 1))))
            if cfg.cost_budget_window_steps > 0 and cfg.window_cost_budget is not None and cfg.window_cost_budget > 0:
                rolling_cost.append((int(step), float(job.actual_service_cost)))
                current_cost_window += float(job.actual_service_cost)
            if job.is_attack == 1:
                served_attack_jobs.append(job.job_id)
                attack_delays_served.append(delay)
                served_true_severity += float(job.severity_true)
                if int(job.actual_backend_fail) == 0:
                    served_true_severity_no_fail += float(job.severity_true)
            else:
                served_clean_jobs.append(job.job_id)
                clean_delays_served.append(delay)

        occupied_server_steps += float(len(active_servers))
        queue_len_trace.append(int(len(queue)))
        active_server_trace.append(int(len(active_servers)))
        step_trace.append(
            {
                "step": float(step),
                "arrivals_this_step": float(len(arrivals_this_step)),
                "queue_len_after_action": float(len(queue)),
                "active_servers_after_action": float(len(active_servers)),
                "selected_count": float(len(selected_items)),
                "available_servers_before_selection": float(available_servers),
                "used_time": float(used_time),
                "used_cost": float(used_cost),
                "current_cost_window_after_action": float(current_cost_window),
            }
        )

    dropped_jobs_horizon = [item.job.job_id for item in queue]
    all_attack_jobs = [job.job_id for job in jobs if job.is_attack == 1]
    all_clean_jobs = [job.job_id for job in jobs if job.is_attack == 0]
    dropped_all = set(dropped_jobs_admission) | set(dropped_jobs_ttl) | set(dropped_jobs_horizon)
    dropped_attack_jobs = [job_id for job_id in all_attack_jobs if job_id in dropped_all]
    dropped_clean_jobs = [job_id for job_id in all_clean_jobs if job_id in dropped_all]

    served_attack_count = len(served_attack_jobs)
    served_clean_count = len(served_clean_jobs)
    total_attack_count = len(all_attack_jobs)
    total_clean_count = len(all_clean_jobs)

    return {
        "summary": {
            "total_steps": int(total_steps),
            "total_jobs": int(len(jobs)),
            "total_attack_jobs": int(total_attack_count),
            "total_clean_jobs": int(total_clean_count),
            "served_jobs": int(len(served_jobs)),
            "served_attack_jobs": int(served_attack_count),
            "served_clean_jobs": int(served_clean_count),
            "dropped_admission": int(len(dropped_jobs_admission)),
            "dropped_ttl": int(len(dropped_jobs_ttl)),
            "dropped_horizon": int(len(dropped_jobs_horizon)),
            "budget_blocked_starts": int(budget_blocked_starts),
            "dropped_attack_jobs": int(len(dropped_attack_jobs)),
            "dropped_clean_jobs": int(len(dropped_clean_jobs)),
            "attack_recall": float(served_attack_count / max(total_attack_count, 1)),
            "weighted_attack_recall": float(served_true_severity / max(total_true_severity, EPS)),
            "weighted_attack_recall_no_backend_fail": float(served_true_severity_no_fail / max(total_true_severity, EPS)),
            "served_attack_precision": float(served_attack_count / max(served_attack_count + served_clean_count, 1)),
            "unnecessary_mtd_count": int(served_clean_count),
            "clean_service_ratio": float(served_clean_count / max(total_clean_count, 1)),
            "total_service_time": float(total_service_time),
            "total_service_cost": float(total_service_cost),
            "average_service_time_per_step": float(total_service_time / max(total_steps, 1)),
            "average_service_cost_per_step": float(total_service_cost / max(total_steps, 1)),
            "total_backend_fail": int(total_backend_fail),
            "queue_delay_mean": float(np.mean(queue_delays_served)) if queue_delays_served else 0.0,
            "queue_delay_p95": float(np.quantile(queue_delays_served, 0.95)) if queue_delays_served else 0.0,
            "attack_delay_mean": float(np.mean(attack_delays_served)) if attack_delays_served else 0.0,
            "attack_delay_p95": float(np.quantile(attack_delays_served, 0.95)) if attack_delays_served else 0.0,
            "clean_delay_mean": float(np.mean(clean_delays_served)) if clean_delays_served else 0.0,
            "clean_delay_p95": float(np.quantile(clean_delays_served, 0.95)) if clean_delays_served else 0.0,
            "mean_queue_len": float(np.mean(queue_len_trace)) if queue_len_trace else 0.0,
            "queue_delay_max": float(np.max(queue_delays_served)) if queue_delays_served else 0.0,
            "max_queue_len": int(np.max(queue_len_trace)) if queue_len_trace else 0,
            "mean_active_servers": float(np.mean(active_server_trace)) if active_server_trace else 0.0,
            "server_utilization": float(occupied_server_steps / max(int(total_steps) * max(int(cfg.slot_budget), 1), 1)),
            "pred_expected_consequence_served_ratio": float(served_pred_expected_consequence / max(total_pred_expected_consequence, EPS)),
        },
        "step_trace": step_trace,
    }


def _policy_compact(summary: Dict[str, float]) -> Dict[str, float | int]:
    return {
        "weighted_attack_recall_no_backend_fail": round(float(summary["weighted_attack_recall_no_backend_fail"]), 4),
        "unnecessary_mtd_count": int(summary["unnecessary_mtd_count"]),
        "queue_delay_p95": round(float(summary["queue_delay_p95"]), 4),
        "average_service_cost_per_step": round(float(summary["average_service_cost_per_step"]), 6),
        "pred_expected_consequence_served_ratio": round(float(summary["pred_expected_consequence_served_ratio"]), 4),
        "clean_service_ratio": round(float(summary["clean_service_ratio"]), 6),
    }


def _best_threshold_from_family(family: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    best_name = None
    best_summary = None
    best_key = None
    for name, summary in family.items():
        key = (
            float(summary["weighted_attack_recall_no_backend_fail"]),
            -float(summary["unnecessary_mtd_count"]),
            -float(summary["queue_delay_p95"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
            best_summary = summary
    assert best_name is not None and best_summary is not None
    return best_name, best_summary


def _make_score_kwargs(slot_budget: int, max_wait_steps: int, cost_budget_per_step: Optional[float]) -> Dict[str, float]:
    return {
        "max_wait_steps": int(max_wait_steps),
        "clean_penalty": 0.60,
        "delay_penalty": 0.15,
        "queue_penalty": 0.10,
        "cost_penalty": 0.05,
        "cost_budget_per_step": cost_budget_per_step,
    }


def prepare_train_val_context(args: SimpleNamespace) -> Dict[str, object]:
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.train_bank), int(args.decision_step_group))
    arrays_val = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.val_bank), int(args.decision_step_group))

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
    fit_verify_score = np.asarray(arrays_train["verify_score"], dtype=float)

    jobs_train, total_steps_train = _predict_jobs(
        arrays_train,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )
    jobs_val, total_steps_val = _predict_jobs(
        arrays_val,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )
    jobs_val_ddd, _ = _predict_jobs(
        arrays_val,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )

    verify_threshold_candidates = _threshold_candidates(arrays_train["verify_score"], list(args.threshold_quantiles))
    ddd_threshold_candidates = _threshold_candidates(arrays_train["ddd_loss_recons"], list(args.threshold_quantiles))
    ec_train = np.asarray([j.pred_expected_consequence for j in jobs_train], dtype=float)
    ec_threshold_candidates = _threshold_candidates(ec_train, list(args.threshold_quantiles))
    verify_signal = np.asarray(arrays_train["verify_score"], dtype=float)
    verify_signal = verify_signal[np.isfinite(verify_signal)]
    verify_iqr = float(np.quantile(verify_signal, 0.80) - np.quantile(verify_signal, 0.20)) if verify_signal.size else 1.0
    verify_iqr = max(verify_iqr, 1e-6)
    adaptive_gain_candidates = [float(x) * verify_iqr for x in list(args.adaptive_gain_scale_list)]

    return {
        "arrays_train": arrays_train,
        "fit_verify_score": fit_verify_score,
        "posterior_verify": posterior_verify,
        "posterior_ddd": posterior_ddd,
        "service_models": service_models,
        "severity_models": severity_models,
        "busy_time_unit": busy_time_unit,
        "jobs_train": jobs_train,
        "jobs_val": jobs_val,
        "jobs_val_ddd": jobs_val_ddd,
        "total_steps_train": total_steps_train,
        "total_steps_val": total_steps_val,
        "train_job_stats": _job_stats(jobs_train),
        "val_job_stats": _job_stats(jobs_val),
        "train_arrival_diag": _arrival_diagnostics(jobs_train, total_steps_train),
        "val_arrival_diag": _arrival_diagnostics(jobs_val, total_steps_val),
        "verify_threshold_candidates": verify_threshold_candidates,
        "ddd_threshold_candidates": ddd_threshold_candidates,
        "ec_threshold_candidates": ec_threshold_candidates,
        "adaptive_gain_candidates": adaptive_gain_candidates,
    }


def build_jobs_for_bank(test_bank: str, *, ctx: Dict[str, object], decision_step_group: int, consequence_blend_verify: float, consequence_mode: str) -> Tuple[List[AlarmJob], int]:
    arrays_test = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(test_bank), int(decision_step_group))
    jobs_test, total_steps_test = _predict_jobs(
        arrays_test,
        posterior_model=ctx["posterior_verify"],
        posterior_signal_key="verify_score",
        service_models=ctx["service_models"],
        service_signal_key="verify_score",
        severity_models=ctx["severity_models"],
        severity_blend_verify=float(consequence_blend_verify),
        consequence_mode=str(consequence_mode),
        fit_verify_score=np.asarray(ctx["fit_verify_score"], dtype=float),
        busy_time_unit=float(ctx["busy_time_unit"]),
    )
    return jobs_test, total_steps_test


def tune_lag_admission_plus_phase3_dispatch(args: SimpleNamespace, ctx: Dict[str, object]) -> Dict[str, object]:
    results: Dict[str, object] = {}
    jobs_val = ctx["jobs_val"]
    jobs_val_ddd = ctx["jobs_val_ddd"]
    total_steps_val = int(ctx["total_steps_val"])
    train_stats = ctx["train_job_stats"]

    for slot_budget in [int(x) for x in args.slot_budget_list]:
        window_cost_budget = None
        cost_budget_window_steps = 0
        if bool(args.use_cost_budget):
            cost_budget_window_steps = int(args.cost_budget_window_steps)
            window_cost_budget = _derive_cost_budget_from_fit(
                ctx["jobs_train"],
                int(ctx["total_steps_train"]),
                window_steps=cost_budget_window_steps,
                q=float(args.cost_budget_quantile),
            )
        cost_budget_per_step = None
        if window_cost_budget is not None and cost_budget_window_steps > 0:
            cost_budget_per_step = float(window_cost_budget) / max(int(cost_budget_window_steps), 1)
        score_kwargs = _make_score_kwargs(slot_budget, int(args.max_wait_steps), cost_budget_per_step)

        thr_verify, thr_verify_res = _tune_threshold_policy(
            jobs_val,
            total_steps_val,
            threshold_candidates=list(ctx["verify_threshold_candidates"]),
            policy_name="threshold_verify_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
            score_kwargs=score_kwargs,
        )
        thr_ddd, thr_ddd_res = _tune_threshold_policy(
            jobs_val_ddd,
            total_steps_val,
            threshold_candidates=list(ctx["ddd_threshold_candidates"]),
            policy_name="threshold_ddd_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
            score_kwargs=score_kwargs,
        )
        thr_ec, thr_ec_res = _tune_threshold_policy(
            jobs_val,
            total_steps_val,
            threshold_candidates=list(ctx["ec_threshold_candidates"]),
            policy_name="threshold_expected_consequence_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
            score_kwargs=score_kwargs,
        )
        adaptive_best, adaptive_res = _tune_adaptive_threshold_policy(
            jobs_val,
            total_steps_val,
            threshold_candidates=list(ctx["verify_threshold_candidates"]),
            gain_candidates=list(ctx["adaptive_gain_candidates"]),
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
            score_kwargs=score_kwargs,
        )
        threshold_family = {
            "threshold_verify_fifo": thr_verify_res["summary"],
            "threshold_ddd_fifo": thr_ddd_res["summary"],
            "threshold_expected_consequence_fifo": thr_ec_res["summary"],
            "adaptive_threshold_verify_fifo": adaptive_res["summary"],
        }
        best_thr_name, best_thr_summary = _best_threshold_from_family(threshold_family)

        phase3_best, phase3_res = _tune_proposed_ca_policy(
            jobs_val,
            total_steps_val,
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps,
            window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
            v_grid=list(args.vq_v_grid),
            clean_grid=list(args.vq_clean_grid),
            age_grid=list(args.vq_age_grid),
            urgency_grid=list(args.vq_urgency_grid),
            fail_grid=list(args.vq_fail_grid),
            busy_grid=list(args.vq_busy_grid),
            cost_grid=list(args.vq_cost_grid),
            admission_threshold_grid=list(args.vq_admission_threshold_grid),
            score_kwargs=score_kwargs,
        )

        best_record: Optional[_TuneRecord] = None
        slack_grid = [0.0, 0.01, 0.03, 0.05]
        reward_grid = [1.0, 2.0, 4.0]
        clean_lambda_grid = [1.0, 2.0, 4.0, 8.0]
        fail_lambda_grid = [0.0, 0.05]
        busy_lambda_grid = [0.0, 0.5, 1.0]
        queue_lambda_grid = [0.0, 0.5, 1.0]
        cost_lambda_grid = [0.0] if window_cost_budget is None else [0.0, 0.5]
        adm_threshold_grid = [-0.2, -0.1, 0.0, 0.1, 0.2]

        for slack in slack_grid:
            target_clean_ratio = float(best_thr_summary["clean_service_ratio"]) + float(slack)
            for reward_weight in reward_grid:
                for clean_lambda in clean_lambda_grid:
                    for fail_lambda in fail_lambda_grid:
                        for busy_lambda in busy_lambda_grid:
                            for queue_lambda in queue_lambda_grid:
                                for cost_lambda in cost_lambda_grid:
                                    for adm_thr in adm_threshold_grid:
                                        cfg = LagAdmissionPhase3DispatchConfig(
                                            slot_budget=slot_budget,
                                            max_wait_steps=int(args.max_wait_steps),
                                            rng_seed=int(args.rng_seed),
                                            window_cost_budget=window_cost_budget,
                                            cost_budget_window_steps=cost_budget_window_steps,
                                            mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
                                            mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
                                            mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
                                            adm_reward_weight=float(reward_weight),
                                            adm_clean_lambda=float(clean_lambda),
                                            adm_fail_lambda=float(fail_lambda),
                                            adm_busy_lambda=float(busy_lambda),
                                            adm_queue_lambda=float(queue_lambda),
                                            adm_cost_lambda=float(cost_lambda),
                                            adm_threshold=float(adm_thr),
                                            target_clean_ratio=float(target_clean_ratio),
                                            dsp_v_weight=float(phase3_best["v_weight"]),
                                            dsp_clean_penalty=float(phase3_best["clean_penalty"]),
                                            dsp_age_bonus=float(phase3_best["age_bonus"]),
                                            dsp_urgency_bonus=float(phase3_best["urgency_bonus"]),
                                            dsp_fail_penalty=float(phase3_best["fail_penalty"]),
                                            dsp_busy_penalty=float(phase3_best["busy_penalty"]),
                                            dsp_cost_penalty=float(phase3_best["cost_penalty"]),
                                            dsp_min_total_score=float(phase3_best["admission_score_threshold"]),
                                        )
                                        res = simulate_lag_admission_phase3_dispatch(jobs_val, total_steps=total_steps_val, cfg=cfg)
                                        summary = res["summary"]
                                        clean_violation = max(0.0, float(summary["clean_service_ratio"]) - float(target_clean_ratio))
                                        feasible = clean_violation <= 1e-12
                                        obj = float(summary["weighted_attack_recall_no_backend_fail"])
                                        obj -= 0.10 * float(summary["attack_delay_p95"] / max(int(args.max_wait_steps), 1))
                                        obj -= 0.03 * float(summary["mean_queue_len"] / max(slot_budget, 1))
                                        if cost_budget_per_step is not None and cost_budget_per_step > 0:
                                            obj -= 0.03 * float(summary["average_service_cost_per_step"] / max(cost_budget_per_step, 1e-9))
                                        obj -= 6.0 * clean_violation
                                        record = _TuneRecord(cfg=cfg, objective=obj, feasible=feasible, clean_violation=clean_violation, summary=summary)
                                        if best_record is None:
                                            best_record = record
                                        else:
                                            if best_record.feasible != record.feasible:
                                                if record.feasible and not best_record.feasible:
                                                    best_record = record
                                            elif not record.feasible and not best_record.feasible:
                                                if (record.clean_violation, -record.objective) < (best_record.clean_violation, -best_record.objective):
                                                    best_record = record
                                            else:
                                                cur_key = (
                                                    record.objective,
                                                    float(record.summary["weighted_attack_recall_no_backend_fail"]),
                                                    -float(record.summary["average_service_cost_per_step"]),
                                                    -float(record.summary["unnecessary_mtd_count"]),
                                                )
                                                best_key = (
                                                    best_record.objective,
                                                    float(best_record.summary["weighted_attack_recall_no_backend_fail"]),
                                                    -float(best_record.summary["average_service_cost_per_step"]),
                                                    -float(best_record.summary["unnecessary_mtd_count"]),
                                                )
                                                if cur_key > best_key:
                                                    best_record = record
        assert best_record is not None
        results[str(slot_budget)] = {
            "threshold_reference": {
                "family_tuning": {
                    "threshold_verify_fifo": {"threshold": float(thr_verify)},
                    "threshold_ddd_fifo": {"threshold": float(thr_ddd)},
                    "threshold_expected_consequence_fifo": {"threshold": float(thr_ec)},
                    "adaptive_threshold_verify_fifo": adaptive_best,
                },
                "family_compact": {name: _policy_compact(s) for name, s in threshold_family.items()},
                "best_threshold_name": best_thr_name,
                "best_threshold_summary": _policy_compact(best_thr_summary),
            },
            "phase3_dispatch_reference": {
                "config": phase3_best,
                "val_summary": _policy_compact(phase3_res["summary"]),
            },
            "lag_admission_best": {
                **asdict(best_record.cfg),
                "val_summary": _policy_compact(best_record.summary),
                "feasible": bool(best_record.feasible),
                "clean_violation": float(best_record.clean_violation),
                "objective": float(best_record.objective),
            },
            "window_cost_budget": None if window_cost_budget is None else float(window_cost_budget),
            "cost_budget_window_steps": int(cost_budget_window_steps),
        }
    return results


def evaluate_on_holdout_bank(
    *,
    test_bank: str,
    ctx: Dict[str, object],
    tuned_by_slot: Dict[str, object],
    decision_step_group: int,
    consequence_blend_verify: float,
    consequence_mode: str,
) -> Dict[str, object]:
    jobs_test, total_steps_test = build_jobs_for_bank(
        test_bank,
        ctx=ctx,
        decision_step_group=decision_step_group,
        consequence_blend_verify=consequence_blend_verify,
        consequence_mode=consequence_mode,
    )
    out: Dict[str, object] = {
        "job_stats": _job_stats(jobs_test),
        "arrival_diagnostics": _arrival_diagnostics(jobs_test, total_steps_test),
        "slot_budget_results": {},
    }
    for slot_key, slot_tuned in tuned_by_slot.items():
        cfg_dict = dict(slot_tuned["lag_admission_best"])
        cfg_dict.pop("val_summary", None)
        cfg_dict.pop("feasible", None)
        cfg_dict.pop("clean_violation", None)
        cfg_dict.pop("objective", None)
        cfg = LagAdmissionPhase3DispatchConfig(**cfg_dict)
        res = simulate_lag_admission_phase3_dispatch(jobs_test, total_steps=total_steps_test, cfg=cfg)
        out["slot_budget_results"][str(slot_key)] = {
            "lag_adm_phase3_dispatch": _policy_compact(res["summary"]),
            "full_summary": res["summary"],
        }
    return out


def _metric_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr, ddof=0)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def aggregate_multi_holdout_results(per_holdout: List[Dict[str, object]]) -> Dict[str, object]:
    slot_keys = sorted({str(k) for rec in per_holdout for k in rec["slot_budget_results"].keys()}, key=int)
    agg: Dict[str, object] = {}
    for slot_key in slot_keys:
        lag_rows = [rec["slot_budget_results"][slot_key]["lag_adm_phase3_dispatch"] for rec in per_holdout]
        phase3_rows = [rec["slot_budget_results"][slot_key]["phase3_proposed"] for rec in per_holdout]
        topk_rows = [rec["slot_budget_results"][slot_key]["topk_expected_consequence"] for rec in per_holdout]
        best_thr_rows = [rec["slot_budget_results"][slot_key]["best_threshold"] for rec in per_holdout]

        def collect(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
            return _metric_stats([float(r[key]) for r in rows])

        policy_stats = {
            "lag_adm_phase3_dispatch": {
                "weighted_attack_recall_no_backend_fail": collect(lag_rows, "weighted_attack_recall_no_backend_fail"),
                "unnecessary_mtd_count": collect(lag_rows, "unnecessary_mtd_count"),
                "queue_delay_p95": collect(lag_rows, "queue_delay_p95"),
                "average_service_cost_per_step": collect(lag_rows, "average_service_cost_per_step"),
                "pred_expected_consequence_served_ratio": collect(lag_rows, "pred_expected_consequence_served_ratio"),
            },
            "phase3_proposed": {
                "weighted_attack_recall_no_backend_fail": collect(phase3_rows, "weighted_attack_recall_no_backend_fail"),
                "unnecessary_mtd_count": collect(phase3_rows, "unnecessary_mtd_count"),
                "queue_delay_p95": collect(phase3_rows, "queue_delay_p95"),
                "average_service_cost_per_step": collect(phase3_rows, "average_service_cost_per_step"),
                "pred_expected_consequence_served_ratio": collect(phase3_rows, "pred_expected_consequence_served_ratio"),
            },
            "topk_expected_consequence": {
                "weighted_attack_recall_no_backend_fail": collect(topk_rows, "weighted_attack_recall_no_backend_fail"),
                "unnecessary_mtd_count": collect(topk_rows, "unnecessary_mtd_count"),
                "queue_delay_p95": collect(topk_rows, "queue_delay_p95"),
                "average_service_cost_per_step": collect(topk_rows, "average_service_cost_per_step"),
                "pred_expected_consequence_served_ratio": collect(topk_rows, "pred_expected_consequence_served_ratio"),
            },
        }
        deltas_thr_recall = [float(a["weighted_attack_recall_no_backend_fail"]) - float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(lag_rows, best_thr_rows)]
        deltas_thr_unnec = [float(a["unnecessary_mtd_count"]) - float(b["unnecessary_mtd_count"]) for a, b in zip(lag_rows, best_thr_rows)]
        deltas_thr_delay = [float(a["queue_delay_p95"]) - float(b["queue_delay_p95"]) for a, b in zip(lag_rows, best_thr_rows)]
        deltas_thr_cost = [float(a["average_service_cost_per_step"]) - float(b["average_service_cost_per_step"]) for a, b in zip(lag_rows, best_thr_rows)]
        deltas_p3_recall = [float(a["weighted_attack_recall_no_backend_fail"]) - float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(lag_rows, phase3_rows)]
        deltas_p3_unnec = [float(a["unnecessary_mtd_count"]) - float(b["unnecessary_mtd_count"]) for a, b in zip(lag_rows, phase3_rows)]
        deltas_p3_delay = [float(a["queue_delay_p95"]) - float(b["queue_delay_p95"]) for a, b in zip(lag_rows, phase3_rows)]
        deltas_p3_cost = [float(a["average_service_cost_per_step"]) - float(b["average_service_cost_per_step"]) for a, b in zip(lag_rows, phase3_rows)]
        deltas_topk_recall = [float(a["weighted_attack_recall_no_backend_fail"]) - float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(lag_rows, topk_rows)]
        deltas_topk_unnec = [float(a["unnecessary_mtd_count"]) - float(b["unnecessary_mtd_count"]) for a, b in zip(lag_rows, topk_rows)]
        deltas_topk_delay = [float(a["queue_delay_p95"]) - float(b["queue_delay_p95"]) for a, b in zip(lag_rows, topk_rows)]
        deltas_topk_cost = [float(a["average_service_cost_per_step"]) - float(b["average_service_cost_per_step"]) for a, b in zip(lag_rows, topk_rows)]

        best_threshold_frequency: Dict[str, int] = {}
        for rec in per_holdout:
            name = rec["slot_budget_results"][slot_key]["best_threshold_name"]
            best_threshold_frequency[name] = best_threshold_frequency.get(name, 0) + 1

        agg[slot_key] = {
            "policy_stats": policy_stats,
            "paired_stats": {
                "lagadm_vs_best_threshold": {
                    "delta_recall": _metric_stats(deltas_thr_recall),
                    "delta_unnecessary": _metric_stats(deltas_thr_unnec),
                    "delta_delay_p95": _metric_stats(deltas_thr_delay),
                    "delta_cost_per_step": _metric_stats(deltas_thr_cost),
                    "lagadm_wins_on_recall": int(sum(1 for x in deltas_thr_recall if x > 0)),
                    "lagadm_no_worse_unnecessary": int(sum(1 for x in deltas_thr_unnec if x <= 0)),
                },
                "lagadm_vs_phase3_proposed": {
                    "delta_recall": _metric_stats(deltas_p3_recall),
                    "delta_unnecessary": _metric_stats(deltas_p3_unnec),
                    "delta_delay_p95": _metric_stats(deltas_p3_delay),
                    "delta_cost_per_step": _metric_stats(deltas_p3_cost),
                    "lagadm_wins_on_recall": int(sum(1 for x in deltas_p3_recall if x > 0)),
                    "lagadm_lower_unnecessary": int(sum(1 for x in deltas_p3_unnec if x < 0)),
                },
                "lagadm_vs_topk_expected": {
                    "delta_recall": _metric_stats(deltas_topk_recall),
                    "delta_unnecessary": _metric_stats(deltas_topk_unnec),
                    "delta_delay_p95": _metric_stats(deltas_topk_delay),
                    "delta_cost_per_step": _metric_stats(deltas_topk_cost),
                    "lagadm_lower_unnecessary": int(sum(1 for x in deltas_topk_unnec if x < 0)),
                    "lagadm_lower_cost": int(sum(1 for x in deltas_topk_cost if x < 0)),
                },
            },
            "best_threshold_frequency": best_threshold_frequency,
        }
    return agg


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

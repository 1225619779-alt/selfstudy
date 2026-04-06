from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scheduler.policies_phase3 import AlarmJob

from evaluation_budget_scheduler_phase3 import (
    _derive_cost_budget_from_fit,
    _objective,
    _tune_adaptive_threshold_policy,
    _tune_proposed_ca_policy,
    _tune_threshold_policy,
)
from lag_adm_phase3_core import (
    _best_threshold_from_family,
    _policy_compact,
    _arrival_diagnostics,
    _job_stats,
    build_jobs_for_bank,
    prepare_train_val_context,
    save_json,
)

EPS = 1e-12


@dataclass
class DualHelpPhase3Config:
    slot_budget: int
    max_wait_steps: int
    rng_seed: int = 20260402
    window_cost_budget: Optional[float] = None
    cost_budget_window_steps: int = 0
    mean_pred_busy_steps: float = 1.0
    mean_pred_service_cost: float = 1.0
    mean_pred_expected_consequence: float = 1.0
    # dual-updated help admission
    adm_reward_weight: float = 1.0
    adm_help_threshold: float = 0.0
    adm_clean_lambda_init: float = 1.0
    adm_busy_lambda_init: float = 0.0
    adm_queue_lambda_init: float = 0.0
    adm_cost_lambda_init: float = 0.0
    adm_clean_eta: float = 0.1
    adm_busy_eta: float = 0.05
    adm_queue_eta: float = 0.05
    adm_cost_eta: float = 0.05
    target_clean_ratio: float = 0.02
    target_server_util: float = 0.8
    target_queue_ratio: float = 1.0
    target_cost_pressure: float = 0.9
    # fixed phase3 dispatch reference
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
    cfg: DualHelpPhase3Config
    objective: float
    violation: float
    summary: Dict[str, float]


def _metric_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr, ddof=0)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def _make_score_kwargs(slot_budget: int, max_wait_steps: int, cost_budget_per_step: Optional[float]) -> Dict[str, float]:
    return {
        "max_wait_steps": int(max_wait_steps),
        "clean_penalty": 0.60,
        "delay_penalty": 0.15,
        "queue_penalty": 0.10,
        "cost_penalty": 0.05,
        "cost_budget_per_step": cost_budget_per_step,
    }


def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: DualHelpPhase3Config) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))


def _queue_pressure(queue_len: int, cfg: DualHelpPhase3Config) -> float:
    return float(queue_len / max(int(cfg.slot_budget), 1))


def _cost_pressure(cost_spent_window: float, cfg: DualHelpPhase3Config) -> float:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))


def _job_fits_cost_budget(job: AlarmJob, *, cost_spent_window: float, cfg: DualHelpPhase3Config) -> bool:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0 or cfg.cost_budget_window_steps <= 0:
        return True
    projected = float(cost_spent_window) + float(job.actual_service_cost)
    return bool(projected <= float(cfg.window_cost_budget) + 1e-9)


def _help_score(
    job: AlarmJob,
    *,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    lam_clean: float,
    lam_busy: float,
    lam_queue: float,
    lam_cost: float,
    cfg: DualHelpPhase3Config,
) -> float:
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    # "help" advantage: predicted backend-protected consequence discounted by backend fail risk.
    reward = float(cfg.adm_reward_weight) * norm_ec * float(max(1.0 - job.pred_fail_prob, 0.0))
    clean_pen = float(lam_clean) * float(1.0 - job.pred_attack_prob)
    busy_pen = float(lam_busy) * norm_busy * (1.0 + _server_pressure(active_servers, cfg))
    queue_pen = float(lam_queue) * (1.0 + _queue_pressure(queue_len + 1, cfg)) * norm_busy
    cost_pen = float(lam_cost) * norm_cost * (1.0 + _cost_pressure(cost_spent_window, cfg))
    return float(reward - clean_pen - busy_pen - queue_pen - cost_pen)


def _dispatch_score(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: DualHelpPhase3Config,
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


def simulate_dual_help_phase3_dispatch(
    jobs: Sequence[AlarmJob], *, total_steps: int, cfg: DualHelpPhase3Config
) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    active_servers: List[_ActiveServer] = []
    rolling_cost: Deque[Tuple[int, float]] = deque()

    lam_clean = float(cfg.adm_clean_lambda_init)
    lam_busy = float(cfg.adm_busy_lambda_init)
    lam_queue = float(cfg.adm_queue_lambda_init)
    lam_cost = float(cfg.adm_cost_lambda_init)

    total_true_severity = float(np.sum([job.severity_true for job in jobs if job.is_attack == 1]))
    served_true_severity = 0.0
    served_true_severity_no_fail = 0.0
    total_pred_expected_consequence = float(np.sum([job.pred_expected_consequence for job in jobs]))
    served_pred_expected_consequence = 0.0

    seen_clean_jobs = 0
    seen_jobs = 0
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
    lambda_trace: List[Dict[str, float]] = []

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]
        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        arrivals_this_step = arrivals.get(step, [])
        for job in arrivals_this_step:
            seen_jobs += 1
            if int(job.is_attack) == 0:
                seen_clean_jobs += 1
            score = _help_score(
                job,
                queue_len=len(queue),
                active_servers=active_servers,
                cost_spent_window=current_cost_window,
                lam_clean=lam_clean,
                lam_busy=lam_busy,
                lam_queue=lam_queue,
                lam_cost=lam_cost,
                cfg=cfg,
            )
            if score >= float(cfg.adm_help_threshold):
                queue.append(_QueuedJob(job=job, enqueue_step=step))
            else:
                dropped_jobs_admission.append(job.job_id)

        kept_queue: List[_QueuedJob] = []
        for item in queue:
            age = int(step - item.enqueue_step)
            if age > int(cfg.max_wait_steps):
                dropped_jobs_ttl.append(item.job.job_id)
            else:
                kept_queue.append(item)
        queue = kept_queue

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
            selected_ids: set[int] = set()
            residual_ids: set[int] = set()
            running_cost_window = float(current_cost_window)
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

        # Online dual updates: this is the main "borrowed core" from constrained defer/online-memory ideas.
        clean_ratio = float(len(served_clean_jobs) / max(seen_clean_jobs, 1)) if seen_clean_jobs > 0 else 0.0
        server_util = _server_pressure(active_servers, cfg)
        queue_ratio = _queue_pressure(len(queue), cfg)
        cost_press = _cost_pressure(current_cost_window, cfg)
        lam_clean = max(0.0, lam_clean + float(cfg.adm_clean_eta) * (clean_ratio - float(cfg.target_clean_ratio)))
        lam_busy = max(0.0, lam_busy + float(cfg.adm_busy_eta) * (server_util - float(cfg.target_server_util)))
        lam_queue = max(0.0, lam_queue + float(cfg.adm_queue_eta) * (queue_ratio - float(cfg.target_queue_ratio)))
        if cfg.window_cost_budget is not None and cfg.window_cost_budget > 0:
            lam_cost = max(0.0, lam_cost + float(cfg.adm_cost_eta) * (cost_press - float(cfg.target_cost_pressure)))

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
        lambda_trace.append(
            {
                "step": float(step),
                "lambda_clean": float(lam_clean),
                "lambda_busy": float(lam_busy),
                "lambda_queue": float(lam_queue),
                "lambda_cost": float(lam_cost),
                "clean_ratio": float(clean_ratio),
                "server_util": float(server_util),
                "queue_ratio": float(queue_ratio),
                "cost_pressure": float(cost_press),
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

    summary = {
        "total_steps": int(total_steps),
        "total_jobs": int(len(jobs)),
        "total_attack_jobs": int(total_attack_count),
        "total_clean_jobs": int(total_clean_count),
        "seen_clean_jobs": int(seen_clean_jobs),
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
        "final_lambda_clean": float(lam_clean),
        "final_lambda_busy": float(lam_busy),
        "final_lambda_queue": float(lam_queue),
        "final_lambda_cost": float(lam_cost),
    }
    return {"summary": summary, "step_trace": step_trace, "lambda_trace": lambda_trace}


def tune_dual_help_admission_plus_phase3_dispatch(args: SimpleNamespace, ctx: Dict[str, object]) -> Dict[str, object]:
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
        target_clean_base = float(best_thr_summary["clean_service_ratio"]) if "clean_service_ratio" in best_thr_summary else 0.0
        target_clean_grid = [float(max(0.0, target_clean_base + s)) for s in [0.0, 0.01, 0.03]]
        reward_grid = [1.0, 1.5, 2.0]
        help_threshold_grid = [-0.10, 0.0]
        clean_init_grid = [0.5, 1.0, 2.0]
        busy_init_grid = [0.0, 0.5]
        queue_init_grid = [0.0, 0.5]
        clean_eta_grid = [0.05, 0.10]
        busy_eta_grid = [0.05]
        queue_eta_grid = [0.05]
        target_server_util_grid = [0.70, 0.85]
        target_queue_ratio_grid = [1.0, 2.0]
        cost_init_grid = [0.0] if window_cost_budget is None else [0.0, 0.25]
        cost_eta_grid = [0.05]
        target_cost_pressure_grid = [0.9]

        for reward_weight in reward_grid:
            for help_thr in help_threshold_grid:
                for target_clean_ratio in target_clean_grid:
                    for clean_init in clean_init_grid:
                        for clean_eta in clean_eta_grid:
                            for busy_init in busy_init_grid:
                                for busy_eta in busy_eta_grid:
                                    for target_server_util in target_server_util_grid:
                                        for queue_init in queue_init_grid:
                                            for queue_eta in queue_eta_grid:
                                                for target_queue_ratio in target_queue_ratio_grid:
                                                    for cost_init in cost_init_grid:
                                                        for cost_eta in cost_eta_grid:
                                                            for target_cost_pressure in target_cost_pressure_grid:
                                                                cfg = DualHelpPhase3Config(
                                                                    slot_budget=slot_budget,
                                                                    max_wait_steps=int(args.max_wait_steps),
                                                                    rng_seed=int(args.rng_seed),
                                                                    window_cost_budget=window_cost_budget,
                                                                    cost_budget_window_steps=cost_budget_window_steps,
                                                                    mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
                                                                    mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
                                                                    mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
                                                                    adm_reward_weight=float(reward_weight),
                                                                    adm_help_threshold=float(help_thr),
                                                                    adm_clean_lambda_init=float(clean_init),
                                                                    adm_busy_lambda_init=float(busy_init),
                                                                    adm_queue_lambda_init=float(queue_init),
                                                                    adm_cost_lambda_init=float(cost_init),
                                                                    adm_clean_eta=float(clean_eta),
                                                                    adm_busy_eta=float(busy_eta),
                                                                    adm_queue_eta=float(queue_eta),
                                                                    adm_cost_eta=float(cost_eta),
                                                                    target_clean_ratio=float(target_clean_ratio),
                                                                    target_server_util=float(target_server_util),
                                                                    target_queue_ratio=float(target_queue_ratio),
                                                                    target_cost_pressure=float(target_cost_pressure),
                                                                    dsp_v_weight=float(phase3_best["v_weight"]),
                                                                    dsp_clean_penalty=float(phase3_best["clean_penalty"]),
                                                                    dsp_age_bonus=float(phase3_best["age_bonus"]),
                                                                    dsp_urgency_bonus=float(phase3_best["urgency_bonus"]),
                                                                    dsp_fail_penalty=float(phase3_best["fail_penalty"]),
                                                                    dsp_busy_penalty=float(phase3_best["busy_penalty"]),
                                                                    dsp_cost_penalty=float(phase3_best["cost_penalty"]),
                                                                    dsp_min_total_score=float(phase3_best["admission_score_threshold"]),
                                                                )
                                                                res = simulate_dual_help_phase3_dispatch(jobs_val, total_steps=total_steps_val, cfg=cfg)
                                                                summary = res["summary"]
                                                                clean_violation = max(0.0, float(summary["clean_service_ratio"]) - float(target_clean_ratio))
                                                                util_violation = max(0.0, float(summary["server_utilization"]) - float(target_server_util))
                                                                queue_violation = max(0.0, float(summary["mean_queue_len"] / max(slot_budget, 1)) - float(target_queue_ratio))
                                                                cost_violation = 0.0
                                                                if cost_budget_per_step is not None and cost_budget_per_step > 0:
                                                                    cost_violation = max(0.0, float(summary["average_service_cost_per_step"] / max(cost_budget_per_step, 1e-9)) - float(target_cost_pressure))
                                                                violation = float(clean_violation + 0.5 * util_violation + 0.5 * queue_violation + 0.5 * cost_violation)
                                                                obj = _objective(
                                                                    summary,
                                                                    max_wait_steps=int(args.max_wait_steps),
                                                                    slot_budget=slot_budget,
                                                                    clean_penalty=0.60,
                                                                    delay_penalty=0.15,
                                                                    queue_penalty=0.10,
                                                                    cost_penalty=0.05,
                                                                    cost_budget_per_step=cost_budget_per_step,
                                                                )
                                                                obj -= 2.0 * violation
                                                                rec = _TuneRecord(cfg=cfg, objective=float(obj), violation=float(violation), summary=summary)
                                                                if best_record is None:
                                                                    best_record = rec
                                                                else:
                                                                    cur_key = (
                                                                        rec.objective,
                                                                        -rec.violation,
                                                                        float(rec.summary["weighted_attack_recall_no_backend_fail"]),
                                                                        -float(rec.summary["unnecessary_mtd_count"]),
                                                                    )
                                                                    best_key = (
                                                                        best_record.objective,
                                                                        -best_record.violation,
                                                                        float(best_record.summary["weighted_attack_recall_no_backend_fail"]),
                                                                        -float(best_record.summary["unnecessary_mtd_count"]),
                                                                    )
                                                                    if cur_key > best_key:
                                                                        best_record = rec
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
            "dual_help_best": {
                **asdict(best_record.cfg),
                "val_summary": _policy_compact(best_record.summary),
                "violation": float(best_record.violation),
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
        cfg_dict = dict(slot_tuned["dual_help_best"])
        cfg_dict.pop("val_summary", None)
        cfg_dict.pop("violation", None)
        cfg_dict.pop("objective", None)
        cfg = DualHelpPhase3Config(**cfg_dict)
        res = simulate_dual_help_phase3_dispatch(jobs_test, total_steps=total_steps_test, cfg=cfg)
        out["slot_budget_results"][str(slot_key)] = {
            "dual_help_phase3_dispatch": _policy_compact(res["summary"]),
            "full_summary": res["summary"],
        }
    return out


def aggregate_multi_holdout_results(per_holdout: List[Dict[str, object]]) -> Dict[str, object]:
    slot_keys = sorted({str(k) for rec in per_holdout for k in rec["slot_budget_results"].keys()}, key=int)
    agg: Dict[str, object] = {}
    for slot_key in slot_keys:
        dh_rows = [rec["slot_budget_results"][slot_key]["dual_help_phase3_dispatch"] for rec in per_holdout]
        phase3_rows = [rec["slot_budget_results"][slot_key]["phase3_proposed"] for rec in per_holdout]
        topk_rows = [rec["slot_budget_results"][slot_key]["topk_expected_consequence"] for rec in per_holdout]
        lagadm_rows = [rec["slot_budget_results"][slot_key].get("lag_adm_phase3_dispatch") for rec in per_holdout if "lag_adm_phase3_dispatch" in rec["slot_budget_results"][slot_key]]
        best_thr_rows = [rec["slot_budget_results"][slot_key]["best_threshold"] for rec in per_holdout]

        def collect(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
            return _metric_stats([float(r[key]) for r in rows])

        policy_stats = {
            "dual_help_phase3_dispatch": {
                "weighted_attack_recall_no_backend_fail": collect(dh_rows, "weighted_attack_recall_no_backend_fail"),
                "unnecessary_mtd_count": collect(dh_rows, "unnecessary_mtd_count"),
                "queue_delay_p95": collect(dh_rows, "queue_delay_p95"),
                "average_service_cost_per_step": collect(dh_rows, "average_service_cost_per_step"),
                "pred_expected_consequence_served_ratio": collect(dh_rows, "pred_expected_consequence_served_ratio"),
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
        if lagadm_rows:
            policy_stats["lag_adm_phase3_dispatch"] = {
                "weighted_attack_recall_no_backend_fail": collect(lagadm_rows, "weighted_attack_recall_no_backend_fail"),
                "unnecessary_mtd_count": collect(lagadm_rows, "unnecessary_mtd_count"),
                "queue_delay_p95": collect(lagadm_rows, "queue_delay_p95"),
                "average_service_cost_per_step": collect(lagadm_rows, "average_service_cost_per_step"),
                "pred_expected_consequence_served_ratio": collect(lagadm_rows, "pred_expected_consequence_served_ratio"),
            }

        def paired(a_rows: List[Dict[str, float]], b_rows: List[Dict[str, float]], win_label: str, low_un_label: str) -> Dict[str, object]:
            d_recall = [float(a["weighted_attack_recall_no_backend_fail"]) - float(b["weighted_attack_recall_no_backend_fail"]) for a, b in zip(a_rows, b_rows)]
            d_un = [float(a["unnecessary_mtd_count"]) - float(b["unnecessary_mtd_count"]) for a, b in zip(a_rows, b_rows)]
            d_delay = [float(a["queue_delay_p95"]) - float(b["queue_delay_p95"]) for a, b in zip(a_rows, b_rows)]
            d_cost = [float(a["average_service_cost_per_step"]) - float(b["average_service_cost_per_step"]) for a, b in zip(a_rows, b_rows)]
            return {
                "delta_recall": _metric_stats(d_recall),
                "delta_unnecessary": _metric_stats(d_un),
                "delta_delay_p95": _metric_stats(d_delay),
                "delta_cost_per_step": _metric_stats(d_cost),
                win_label: int(sum(1 for x in d_recall if x > 0)),
                low_un_label: int(sum(1 for x in d_un if x < 0)),
            }

        best_threshold_frequency: Dict[str, int] = {}
        for rec in per_holdout:
            name = rec["slot_budget_results"][slot_key]["best_threshold_name"]
            best_threshold_frequency[name] = best_threshold_frequency.get(name, 0) + 1

        paired_stats = {
            "dualhelp_vs_best_threshold": paired(dh_rows, best_thr_rows, "dualhelp_wins_on_recall", "dualhelp_lower_unnecessary"),
            "dualhelp_vs_phase3_proposed": paired(dh_rows, phase3_rows, "dualhelp_wins_on_recall", "dualhelp_lower_unnecessary"),
            "dualhelp_vs_topk_expected": paired(dh_rows, topk_rows, "dualhelp_wins_on_recall", "dualhelp_lower_unnecessary"),
        }
        if lagadm_rows:
            paired_stats["dualhelp_vs_lagadm"] = paired(dh_rows, lagadm_rows, "dualhelp_wins_on_recall", "dualhelp_lower_unnecessary")

        agg[slot_key] = {
            "policy_stats": policy_stats,
            "paired_stats": paired_stats,
            "best_threshold_frequency": best_threshold_frequency,
        }
    return agg

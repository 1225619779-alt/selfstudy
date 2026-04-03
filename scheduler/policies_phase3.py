
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class AlarmJob:
    job_id: int
    arrival_step: int
    verify_score: float
    ddd_loss: float
    is_attack: int
    severity_true: float
    actual_service_time: float
    actual_service_cost: float
    actual_backend_fail: int
    actual_busy_steps: int
    pred_attack_prob: float
    pred_service_time: float
    pred_service_cost: float
    pred_busy_steps: int
    pred_fail_prob: float
    pred_attack_severity: float
    pred_expected_consequence: float
    value_proxy: float
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    policy_name: str
    slot_budget: int = 1
    max_wait_steps: int = 10
    threshold: Optional[float] = None
    adaptive_gain: float = 0.0
    v_weight: float = 1.0
    clean_penalty: float = 0.0
    age_bonus: float = 0.0
    urgency_bonus: float = 0.0
    fail_penalty: float = 0.0
    busy_penalty: float = 1.0
    cost_penalty: float = 0.0
    admission_score_threshold: float = 0.0
    mean_pred_busy_steps: float = 1.0
    mean_pred_service_cost: float = 1.0
    mean_pred_expected_consequence: float = 1.0
    window_cost_budget: Optional[float] = None
    cost_budget_window_steps: int = 0
    rng_seed: int = 20260402


@dataclass
class _QueuedJob:
    job: AlarmJob
    enqueue_step: int


@dataclass
class _ActiveServer:
    job_id: int
    busy_until_step: int


def quantize_busy_steps(x: float | np.ndarray, *, time_unit: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    unit = max(float(time_unit), 1e-6)
    out = np.ceil(np.maximum(arr, 0.0) / unit).astype(int)
    out = np.maximum(out, 1)
    return out


def build_jobs_from_arrays(
    arrays: Dict[str, np.ndarray],
    *,
    p_hat: np.ndarray,
    tau_hat: np.ndarray,
    cost_hat: np.ndarray,
    fail_hat: np.ndarray,
    attack_severity_hat: np.ndarray,
    expected_consequence_hat: np.ndarray,
    value_proxy: np.ndarray,
    busy_time_unit: float,
) -> Tuple[List[AlarmJob], int]:
    n = int(len(arrays["arrival_step"]))
    required_pred = [p_hat, tau_hat, cost_hat, fail_hat, attack_severity_hat, expected_consequence_hat, value_proxy]
    if not all(len(np.asarray(x)) == n for x in required_pred):
        raise ValueError("Predicted arrays must have the same length as alarm arrays")

    actual_busy = quantize_busy_steps(np.asarray(arrays["service_time"], dtype=float), time_unit=busy_time_unit)
    pred_busy = quantize_busy_steps(np.asarray(tau_hat, dtype=float), time_unit=busy_time_unit)

    jobs: List[AlarmJob] = []
    for i in range(n):
        jobs.append(
            AlarmJob(
                job_id=i,
                arrival_step=int(arrays["arrival_step"][i]),
                verify_score=float(arrays["verify_score"][i]),
                ddd_loss=float(arrays["ddd_loss_recons"][i]),
                is_attack=int(arrays["is_attack"][i]),
                severity_true=float(arrays["severity_true"][i]),
                actual_service_time=float(arrays["service_time"][i]),
                actual_service_cost=float(arrays["service_cost"][i]),
                actual_backend_fail=int(arrays["backend_fail"][i]),
                actual_busy_steps=int(actual_busy[i]),
                pred_attack_prob=float(np.clip(p_hat[i], 0.0, 1.0)),
                pred_service_time=float(max(0.0, tau_hat[i])),
                pred_service_cost=float(max(0.0, cost_hat[i])),
                pred_busy_steps=int(pred_busy[i]),
                pred_fail_prob=float(np.clip(fail_hat[i], 0.0, 1.0)),
                pred_attack_severity=float(max(0.0, attack_severity_hat[i])),
                pred_expected_consequence=float(max(0.0, expected_consequence_hat[i])),
                value_proxy=float(max(0.0, value_proxy[i])),
                meta={
                    "recover_fail": float(arrays["recover_fail"][i]),
                    "ang_no": float(arrays["ang_no"][i]),
                    "ang_str": float(arrays["ang_str"][i]),
                },
            )
        )
    total_steps = int(np.asarray(arrays["total_steps"]).reshape(-1)[0])
    return jobs, total_steps


def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: SimulationConfig) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))


def _queue_pressure(queue_len: int, cfg: SimulationConfig) -> float:
    return float(queue_len / max(int(cfg.slot_budget), 1))


def _cost_pressure(cost_spent_window: float, cfg: SimulationConfig) -> float:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))


def _dynamic_threshold(step: int, queue_len: int, active_servers: Sequence[_ActiveServer], cost_spent_window: float, cfg: SimulationConfig) -> float:
    if cfg.threshold is None:
        raise ValueError("adaptive threshold policy requires cfg.threshold as base threshold")
    pressure = _queue_pressure(queue_len, cfg) + _server_pressure(active_servers, cfg) + _cost_pressure(cost_spent_window, cfg)
    return float(cfg.threshold + cfg.adaptive_gain * pressure)


def _admission_accept(
    job: AlarmJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: SimulationConfig,
) -> Tuple[bool, float | None]:
    if cfg.policy_name == "threshold_verify_fifo":
        if cfg.threshold is None:
            raise ValueError("threshold_verify_fifo requires cfg.threshold")
        thr = float(cfg.threshold)
        return bool(job.verify_score >= thr), thr

    if cfg.policy_name == "threshold_ddd_fifo":
        if cfg.threshold is None:
            raise ValueError("threshold_ddd_fifo requires cfg.threshold")
        thr = float(cfg.threshold)
        return bool(job.ddd_loss >= thr), thr

    if cfg.policy_name == "threshold_expected_consequence_fifo":
        if cfg.threshold is None:
            raise ValueError("threshold_expected_consequence_fifo requires cfg.threshold")
        thr = float(cfg.threshold)
        return bool(job.pred_expected_consequence >= thr), thr

    if cfg.policy_name == "adaptive_threshold_verify_fifo":
        thr = _dynamic_threshold(step, queue_len, active_servers, cost_spent_window, cfg)
        return bool(job.verify_score >= thr), thr

    return True, None


def _policy_score(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    job = item.job
    age = max(0, int(step - item.enqueue_step))
    ttl_left = max(int(cfg.max_wait_steps) - age, 0)

    if cfg.policy_name == "fifo":
        return -(item.enqueue_step + job.job_id * 1e-6)

    if cfg.policy_name == "random":
        return float(rng.random())

    if cfg.policy_name == "topk_verify":
        return float(job.verify_score)

    if cfg.policy_name == "topk_ddd":
        return float(job.ddd_loss)

    if cfg.policy_name == "topk_expected_consequence":
        return float(job.pred_expected_consequence)

    if cfg.policy_name == "static_value_cost":
        denom = (
            1.0
            + float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
            + float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
        )
        return float(job.pred_attack_prob * max(job.value_proxy, 0.05) / denom)

    if cfg.policy_name == "static_expected_consequence_cost":
        denom = (
            1.0
            + float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
            + float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
        )
        return float(job.pred_expected_consequence / denom)

    if cfg.policy_name in {
        "threshold_verify_fifo",
        "threshold_ddd_fifo",
        "threshold_expected_consequence_fifo",
        "adaptive_threshold_verify_fifo",
    }:
        return -(item.enqueue_step + job.job_id * 1e-6)

    if cfg.policy_name == "proposed_vq_hard":
        server_pressure = _server_pressure(active_servers, cfg)
        queue_pressure = _queue_pressure(queue_len, cfg)
        cost_pressure = _cost_pressure(cost_spent_window, cfg)
        norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
        norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
        value_term = float(cfg.v_weight) * float(job.pred_attack_prob) * max(float(job.value_proxy), 0.05)
        age_term = float(cfg.age_bonus) * float(age / max(cfg.max_wait_steps, 1))
        fail_term = float(cfg.fail_penalty) * float(job.pred_fail_prob)
        busy_term = float(cfg.busy_penalty) * (server_pressure + queue_pressure) * norm_busy
        cost_term = float(cfg.cost_penalty) * cost_pressure * norm_cost
        return float(value_term + age_term - fail_term - busy_term - cost_term)

    if cfg.policy_name == "proposed_ca_vq_hard":
        server_pressure = _server_pressure(active_servers, cfg)
        queue_pressure = _queue_pressure(queue_len, cfg)
        cost_pressure = _cost_pressure(cost_spent_window, cfg)
        norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
        norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
        norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
        value_term = float(cfg.v_weight) * norm_ec
        clean_term = float(cfg.clean_penalty) * float(1.0 - job.pred_attack_prob)
        age_term = float(cfg.age_bonus) * float(age / max(cfg.max_wait_steps, 1))
        urgency_term = float(cfg.urgency_bonus) * float(1.0 / (ttl_left + 1.0))
        fail_term = float(cfg.fail_penalty) * float(job.pred_fail_prob)
        busy_term = float(cfg.busy_penalty) * (server_pressure + queue_pressure) * norm_busy
        cost_term = float(cfg.cost_penalty) * cost_pressure * norm_cost
        return float(value_term + age_term + urgency_term - clean_term - fail_term - busy_term - cost_term)

    raise KeyError(f"Unknown policy_name={cfg.policy_name!r}")


def _job_fits_cost_budget(job: AlarmJob, *, cost_spent_window: float, cfg: SimulationConfig) -> bool:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0 or cfg.cost_budget_window_steps <= 0:
        return True
    projected = float(cost_spent_window) + float(job.actual_service_cost)
    return bool(projected <= float(cfg.window_cost_budget) + 1e-9)


def simulate_policy(jobs: Sequence[AlarmJob], *, total_steps: int, cfg: SimulationConfig) -> Dict[str, object]:
    if cfg.slot_budget < 0:
        raise ValueError("slot_budget must be non-negative")

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
    dropped_jobs_threshold: List[int] = []
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
    threshold_trace: List[float] = []
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
        last_thr_used: float | None = None
        for job in arrivals_this_step:
            accept, thr_used = _admission_accept(
                job, step=step, queue_len=len(queue), active_servers=active_servers,
                cost_spent_window=current_cost_window, cfg=cfg
            )
            if thr_used is not None:
                last_thr_used = float(thr_used)
            if accept:
                queue.append(_QueuedJob(job=job, enqueue_step=step))
            else:
                dropped_jobs_threshold.append(job.job_id)

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
                    _policy_score(
                        item, step=step, queue_len=len(queue), active_servers=active_servers,
                        cost_spent_window=current_cost_window, cfg=cfg, rng=rng
                    ),
                    item,
                )
                for item in queue
            ]
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)

            running_cost_window = float(current_cost_window)
            for score, item in scored:
                if len(selected_items) >= available_servers:
                    residual_queue.append(item)
                    continue

                if cfg.policy_name in {"proposed_vq_hard", "proposed_ca_vq_hard"} and score < float(cfg.admission_score_threshold):
                    residual_queue.append(item)
                    continue

                if not _job_fits_cost_budget(item.job, cost_spent_window=running_cost_window, cfg=cfg):
                    residual_queue.append(item)
                    budget_blocked_starts += 1
                    dropped_jobs_budget_blocked.append(item.job.job_id)
                    continue

                selected_items.append(item)
                running_cost_window += float(item.job.actual_service_cost)

            selected_ids = {x.job.job_id for x in selected_items}
            for _, item in scored:
                if item.job.job_id not in selected_ids and item not in residual_queue:
                    residual_queue.append(item)
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
        if last_thr_used is not None:
            threshold_trace.append(float(last_thr_used))

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
                "threshold_used": float(last_thr_used) if last_thr_used is not None else np.nan,
            }
        )

    dropped_jobs_horizon = [item.job.job_id for item in queue]
    all_attack_jobs = [job.job_id for job in jobs if job.is_attack == 1]
    all_clean_jobs = [job.job_id for job in jobs if job.is_attack == 0]
    dropped_all = set(dropped_jobs_threshold) | set(dropped_jobs_ttl) | set(dropped_jobs_horizon)
    dropped_attack_jobs = [job_id for job_id in all_attack_jobs if job_id in dropped_all]
    dropped_clean_jobs = [job_id for job_id in all_clean_jobs if job_id in dropped_all]

    served_attack_count = len(served_attack_jobs)
    served_clean_count = len(served_clean_jobs)
    total_attack_count = len(all_attack_jobs)
    total_clean_count = len(all_clean_jobs)

    result = {
        "policy_name": cfg.policy_name,
        "slot_budget": int(cfg.slot_budget),
        "max_wait_steps": int(cfg.max_wait_steps),
        "threshold": None if cfg.threshold is None else float(cfg.threshold),
        "adaptive_gain": float(cfg.adaptive_gain),
        "window_cost_budget": None if cfg.window_cost_budget is None else float(cfg.window_cost_budget),
        "cost_budget_window_steps": int(cfg.cost_budget_window_steps),
        "summary": {
            "total_steps": int(total_steps),
            "total_jobs": int(len(jobs)),
            "total_attack_jobs": int(total_attack_count),
            "total_clean_jobs": int(total_clean_count),
            "served_jobs": int(len(served_jobs)),
            "served_attack_jobs": int(served_attack_count),
            "served_clean_jobs": int(served_clean_count),
            "dropped_threshold": int(len(dropped_jobs_threshold)),
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
            "threshold_trace_mean": float(np.mean(threshold_trace)) if threshold_trace else np.nan,
            "pred_expected_consequence_served_ratio": float(served_pred_expected_consequence / max(total_pred_expected_consequence, EPS)),
        },
        "served_jobs": served_jobs,
        "served_attack_jobs": served_attack_jobs,
        "served_clean_jobs": served_clean_jobs,
        "dropped_jobs_threshold": dropped_jobs_threshold,
        "dropped_jobs_ttl": dropped_jobs_ttl,
        "dropped_jobs_horizon": dropped_jobs_horizon,
        "step_trace": step_trace,
    }
    return result

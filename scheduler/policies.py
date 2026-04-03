from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    pred_attack_prob: float
    pred_service_time: float
    pred_service_cost: float
    pred_fail_prob: float
    value_proxy: float
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    policy_name: str
    slot_budget: int = 1
    max_wait_steps: int = 10
    signal_key_for_threshold: str = "verify"
    threshold: Optional[float] = None
    average_time_budget: Optional[float] = None
    average_cost_budget: Optional[float] = None
    v_weight: float = 1.0
    age_bonus: float = 0.0
    fail_penalty: float = 0.0
    rng_seed: int = 20260402


@dataclass
class _QueuedJob:
    job: AlarmJob
    enqueue_step: int



def build_jobs_from_arrays(
    arrays: Dict[str, np.ndarray],
    *,
    p_hat: np.ndarray,
    tau_hat: np.ndarray,
    cost_hat: np.ndarray,
    fail_hat: np.ndarray,
    value_proxy: np.ndarray,
) -> Tuple[List[AlarmJob], int]:
    n = int(len(arrays["arrival_step"]))
    if not all(len(np.asarray(x)) == n for x in [p_hat, tau_hat, cost_hat, fail_hat, value_proxy]):
        raise ValueError("Predicted arrays must have the same length as alarm arrays")

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
                pred_attack_prob=float(p_hat[i]),
                pred_service_time=float(max(0.0, tau_hat[i])),
                pred_service_cost=float(max(0.0, cost_hat[i])),
                pred_fail_prob=float(np.clip(fail_hat[i], 0.0, 1.0)),
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



def _normalized_time_cost_penalty(job: AlarmJob, cfg: SimulationConfig, z_time: float, z_cost: float) -> float:
    penalty = 0.0
    if cfg.average_time_budget is not None and cfg.average_time_budget > 0:
        penalty += z_time * (job.pred_service_time / max(cfg.average_time_budget, EPS))
    if cfg.average_cost_budget is not None and cfg.average_cost_budget > 0:
        penalty += z_cost * (job.pred_service_cost / max(cfg.average_cost_budget, EPS))
    return penalty



def _policy_score(item: _QueuedJob, step: int, cfg: SimulationConfig, z_time: float, z_cost: float, rng: np.random.Generator) -> float:
    job = item.job
    age = max(0, int(step - item.enqueue_step))

    if cfg.policy_name == "fifo":
        return -(job.arrival_step + job.job_id * 1e-6)
    if cfg.policy_name == "random":
        return float(rng.random())
    if cfg.policy_name == "topk_verify":
        return float(job.verify_score)
    if cfg.policy_name == "topk_ddd":
        return float(job.ddd_loss)
    if cfg.policy_name == "static_value_cost":
        denom = 1.0 + job.pred_service_time + job.pred_service_cost
        return float(job.pred_attack_prob * max(job.value_proxy, 0.05) / denom)
    if cfg.policy_name == "proposed_vq":
        value_term = cfg.v_weight * job.pred_attack_prob * max(job.value_proxy, 0.05)
        age_term = cfg.age_bonus * age
        fail_term = cfg.fail_penalty * job.pred_fail_prob
        budget_pen = _normalized_time_cost_penalty(job, cfg, z_time=z_time, z_cost=z_cost)
        return float(value_term + age_term - fail_term - budget_pen)
    if cfg.policy_name in {"threshold_verify_fifo", "threshold_ddd_fifo"}:
        return -(job.arrival_step + job.job_id * 1e-6)
    raise KeyError(f"Unknown policy_name={cfg.policy_name!r}")



def _admission_accept(job: AlarmJob, cfg: SimulationConfig) -> bool:
    if cfg.policy_name == "threshold_verify_fifo":
        if cfg.threshold is None:
            raise ValueError("threshold_verify_fifo requires cfg.threshold")
        return bool(job.verify_score >= cfg.threshold)
    if cfg.policy_name == "threshold_ddd_fifo":
        if cfg.threshold is None:
            raise ValueError("threshold_ddd_fifo requires cfg.threshold")
        return bool(job.ddd_loss >= cfg.threshold)
    return True



def _update_virtual_queues(z_time: float, z_cost: float, used_time: float, used_cost: float, cfg: SimulationConfig) -> Tuple[float, float]:
    next_z_time = z_time
    next_z_cost = z_cost
    if cfg.average_time_budget is not None and cfg.average_time_budget > 0:
        next_z_time = max(0.0, z_time + used_time / max(cfg.average_time_budget, EPS) - 1.0)
    if cfg.average_cost_budget is not None and cfg.average_cost_budget > 0:
        next_z_cost = max(0.0, z_cost + used_cost / max(cfg.average_cost_budget, EPS) - 1.0)
    return next_z_time, next_z_cost



def simulate_policy(
    jobs: Sequence[AlarmJob],
    *,
    total_steps: int,
    cfg: SimulationConfig,
) -> Dict[str, object]:
    if cfg.slot_budget < 0:
        raise ValueError("slot_budget must be non-negative")
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    rng = np.random.default_rng(int(cfg.rng_seed))
    z_time = 0.0
    z_cost = 0.0

    total_true_severity = float(np.sum([job.severity_true for job in jobs if job.is_attack == 1]))
    served_true_severity = 0.0
    served_true_severity_no_fail = 0.0

    served_jobs: List[int] = []
    served_attack_jobs: List[int] = []
    served_clean_jobs: List[int] = []
    dropped_jobs_threshold: List[int] = []
    dropped_jobs_ttl: List[int] = []
    queue_delays_served: List[int] = []
    attack_delays_served: List[int] = []
    clean_delays_served: List[int] = []
    step_trace: List[Dict[str, float]] = []

    total_service_time = 0.0
    total_service_cost = 0.0
    total_backend_fail = 0

    for step in range(int(total_steps)):
        for job in arrivals.get(step, []):
            if _admission_accept(job, cfg):
                queue.append(_QueuedJob(job=job, enqueue_step=step))
            else:
                dropped_jobs_threshold.append(job.job_id)

        # Drop expired jobs before making the step decision.
        new_queue: List[_QueuedJob] = []
        for item in queue:
            age = int(step - item.enqueue_step)
            if age > int(cfg.max_wait_steps):
                dropped_jobs_ttl.append(item.job.job_id)
            else:
                new_queue.append(item)
        queue = new_queue

        if queue and cfg.slot_budget > 0:
            scored = [(_policy_score(item, step, cfg, z_time=z_time, z_cost=z_cost, rng=rng), item) for item in queue]
            scored.sort(key=lambda x: (x[0], -x[1].job.arrival_step, -x[1].job.job_id), reverse=True)
        else:
            scored = []

        selected_items: List[_QueuedJob] = []
        chosen_ids: set[int] = set()
        for score, item in scored:
            if len(selected_items) >= int(cfg.slot_budget):
                break
            if cfg.policy_name == "proposed_vq" and score <= 0.0:
                continue
            selected_items.append(item)
            chosen_ids.add(item.job.job_id)

        residual_queue = [item for _, item in scored if item.job.job_id not in chosen_ids]
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

            if job.is_attack == 1:
                served_attack_jobs.append(job.job_id)
                attack_delays_served.append(delay)
                served_true_severity += float(job.severity_true)
                if int(job.actual_backend_fail) == 0:
                    served_true_severity_no_fail += float(job.severity_true)
            else:
                served_clean_jobs.append(job.job_id)
                clean_delays_served.append(delay)

        z_time, z_cost = _update_virtual_queues(z_time, z_cost, used_time, used_cost, cfg)
        step_trace.append(
            {
                "step": float(step),
                "queue_len_after_action": float(len(queue)),
                "selected_count": float(len(selected_items)),
                "used_time": float(used_time),
                "used_cost": float(used_cost),
                "z_time": float(z_time),
                "z_cost": float(z_cost),
            }
        )

    # Horizon end: remaining jobs count as dropped due horizon exhaustion.
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
        "average_time_budget": None if cfg.average_time_budget is None else float(cfg.average_time_budget),
        "average_cost_budget": None if cfg.average_cost_budget is None else float(cfg.average_cost_budget),
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
            "slot_utilization": float(len(served_jobs) / max(total_steps * max(cfg.slot_budget, 1), 1)),
            "final_queue_len": int(len(queue)),
            "final_z_time": float(z_time),
            "final_z_cost": float(z_cost),
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

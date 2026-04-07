from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, List, Sequence, Tuple

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
    _job_stats,
    _objective,
    _predict_jobs,
    _tune_proposed_ca_policy,
)

EPS = 1e-12


@dataclass(frozen=True)
class OverloadVariantSpec:
    name: str
    tau_gain: float
    clean_gain: float
    use_urgency_mask: bool
    description: str


@dataclass
class OverloadConfig:
    slot_budget: int
    max_wait_steps: int
    rng_seed: int
    window_cost_budget: float | None
    cost_budget_window_steps: int
    mean_pred_busy_steps: float
    mean_pred_service_cost: float
    mean_pred_expected_consequence: float
    # frozen phase3 weights
    v_weight: float
    clean_penalty: float
    age_bonus: float
    urgency_bonus: float
    fail_penalty: float
    busy_penalty: float
    cost_penalty: float
    admission_score_threshold: float
    # overload-only correction
    tau_gain: float
    clean_gain: float
    use_urgency_mask: bool
    overload_queue_onset: float = 0.25
    overload_server_onset: float = 0.85
    overload_server_weight: float = 0.50


@dataclass
class _QueuedJob:
    job: AlarmJob
    enqueue_step: int


@dataclass
class _ActiveServer:
    job_id: int
    busy_until_step: int


DEFAULT_VARIANTS: Tuple[OverloadVariantSpec, ...] = (
    OverloadVariantSpec(
        name="olc_t",
        tau_gain=0.06,
        clean_gain=0.0,
        use_urgency_mask=False,
        description="threshold-only overload correction",
    ),
    OverloadVariantSpec(
        name="olc_tc",
        tau_gain=0.03,
        clean_gain=0.10,
        use_urgency_mask=False,
        description="threshold + clean-penalty overload correction",
    ),
    OverloadVariantSpec(
        name="olc_u",
        tau_gain=0.08,
        clean_gain=0.0,
        use_urgency_mask=True,
        description="urgency-masked threshold overload correction",
    ),
)


def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: OverloadConfig) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))



def _available_servers(active_servers: Sequence[_ActiveServer], cfg: OverloadConfig) -> int:
    return max(int(cfg.slot_budget) - len(active_servers), 0)



def _queue_pressure(queue_len: int, cfg: OverloadConfig) -> float:
    return float(int(queue_len) / max(int(cfg.slot_budget), 1))


def _backlog_pressure(queue_len: int, active_servers: Sequence[_ActiveServer], cfg: OverloadConfig) -> float:
    excess = max(int(queue_len) - _available_servers(active_servers, cfg), 0)
    return float(excess / max(int(cfg.slot_budget), 1))



def _cost_pressure(cost_spent_window: float, cfg: OverloadConfig) -> float:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))



def _base_phase3_score(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    clean_penalty_override: float,
    cfg: OverloadConfig,
) -> float:
    job = item.job
    age = max(0, int(step - item.enqueue_step))
    ttl_left = max(int(cfg.max_wait_steps) - age, 0)

    server_pressure = _server_pressure(active_servers, cfg)
    queue_pressure = _queue_pressure(queue_len, cfg)
    cost_pressure = _cost_pressure(cost_spent_window, cfg)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    value_term = float(cfg.v_weight) * norm_ec
    clean_term = float(clean_penalty_override) * float(1.0 - job.pred_attack_prob)
    age_term = float(cfg.age_bonus) * float(age / max(cfg.max_wait_steps, 1))
    urgency_term = float(cfg.urgency_bonus) * float(1.0 / (ttl_left + 1.0))
    fail_term = float(cfg.fail_penalty) * float(job.pred_fail_prob)
    busy_term = float(cfg.busy_penalty) * (server_pressure + queue_pressure) * norm_busy
    cost_term = float(cfg.cost_penalty) * cost_pressure * norm_cost
    return float(value_term + age_term + urgency_term - clean_term - fail_term - busy_term - cost_term)



def _overload_severity(
    *,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cfg: OverloadConfig,
) -> float:
    backlog_pressure = _backlog_pressure(queue_len, active_servers, cfg)
    server_pressure = _server_pressure(active_servers, cfg)
    raw = (
        max(0.0, backlog_pressure - float(cfg.overload_queue_onset))
        + float(cfg.overload_server_weight) * max(0.0, server_pressure - float(cfg.overload_server_onset))
    )
    return float(np.clip(raw, 0.0, 1.0))



def _urgency_gate(item: _QueuedJob, *, step: int, cfg: OverloadConfig) -> float:
    if not cfg.use_urgency_mask:
        return 1.0
    job = item.job
    age = max(0, int(step - item.enqueue_step))
    ttl_left = max(int(cfg.max_wait_steps) - age, 0)
    urgency = min(1.0, float(job.pred_fail_prob) + float(1.0 / (ttl_left + 1.0)))
    return float(max(0.0, 1.0 - urgency))



def _effective_adjustments(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cfg: OverloadConfig,
) -> Tuple[float, float, float, float]:
    ov = _overload_severity(queue_len=queue_len, active_servers=active_servers, cfg=cfg)
    mask = _urgency_gate(item, step=step, cfg=cfg)
    strength = float(ov * mask)
    tau_eff = float(cfg.admission_score_threshold + cfg.tau_gain * strength)
    clean_eff = float(cfg.clean_penalty + cfg.clean_gain * strength)
    return tau_eff, clean_eff, ov, strength



def _job_fits_cost_budget(job: AlarmJob, *, cost_spent_window: float, cfg: OverloadConfig) -> bool:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0 or cfg.cost_budget_window_steps <= 0:
        return True
    projected = float(cost_spent_window) + float(job.actual_service_cost)
    return bool(projected <= float(cfg.window_cost_budget) + 1e-9)



def simulate_overload_phase3(jobs: Sequence[AlarmJob], *, total_steps: int, cfg: OverloadConfig) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    active_servers: List[_ActiveServer] = []
    rolling_cost: Deque[Tuple[int, float]] = deque()

    total_true_severity = float(np.sum([job.severity_true for job in jobs if job.is_attack == 1]))
    served_true_severity = 0.0
    served_true_severity_no_fail = 0.0
    total_pred_expected_consequence = float(np.sum([job.pred_expected_consequence for job in jobs]))
    served_pred_expected_consequence = 0.0

    served_attack_jobs: List[int] = []
    served_clean_jobs: List[int] = []
    dropped_jobs_ttl: List[int] = []
    queue_delays_served: List[int] = []
    attack_delays_served: List[int] = []
    clean_delays_served: List[int] = []

    total_service_time = 0.0
    total_service_cost = 0.0
    total_backend_fail = 0
    occupied_server_steps = 0.0
    queue_len_trace: List[int] = []
    active_server_trace: List[int] = []
    threshold_trace: List[float] = []
    clean_penalty_trace: List[float] = []
    budget_blocked_starts = 0

    # overload diagnostics
    total_step_count = 0
    arrival_step_count = 0
    decision_step_count = 0
    overload_step_count = 0
    overload_arrival_step_count = 0
    overload_decision_step_count = 0
    effective_correction_step_count = 0
    effective_arrival_step_count = 0
    effective_decision_step_count = 0
    ov_on_trigger: List[float] = []
    max_strength_trace: List[float] = []

    for step in range(int(total_steps)):
        total_step_count += 1
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]

        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        arrivals_this_step = arrivals.get(step, [])
        if arrivals_this_step:
            arrival_step_count += 1
            for job in arrivals_this_step:
                queue.append(_QueuedJob(job=job, enqueue_step=step))

        new_queue: List[_QueuedJob] = []
        for item in queue:
            age = int(step - item.enqueue_step)
            if age > int(cfg.max_wait_steps):
                dropped_jobs_ttl.append(item.job.job_id)
            else:
                new_queue.append(item)
        queue = new_queue

        step_ov = _overload_severity(queue_len=len(queue), active_servers=active_servers, cfg=cfg)
        if step_ov > 0.0:
            overload_step_count += 1
            ov_on_trigger.append(float(step_ov))
            if arrivals_this_step:
                overload_arrival_step_count += 1

        available_servers = _available_servers(active_servers, cfg)
        selected_items: List[_QueuedJob] = []
        residual_queue: List[_QueuedJob] = []
        last_tau = float(cfg.admission_score_threshold)
        last_clean = float(cfg.clean_penalty)
        any_effective_correction = False
        max_strength = 0.0

        if queue:
            decision_step_count += 1
            if step_ov > 0.0:
                overload_decision_step_count += 1

        if available_servers > 0 and queue:
            scored: List[Tuple[float, _QueuedJob, float, float, float]] = []
            for item in queue:
                tau_eff, clean_eff, _ov, strength = _effective_adjustments(
                    item,
                    step=step,
                    queue_len=len(queue),
                    active_servers=active_servers,
                    cfg=cfg,
                )
                score = _base_phase3_score(
                    item,
                    step=step,
                    queue_len=len(queue),
                    active_servers=active_servers,
                    cost_spent_window=current_cost_window,
                    clean_penalty_override=clean_eff,
                    cfg=cfg,
                )
                scored.append((score, item, tau_eff, clean_eff, strength))
                if abs(tau_eff - float(cfg.admission_score_threshold)) > 1e-12 or abs(clean_eff - float(cfg.clean_penalty)) > 1e-12:
                    any_effective_correction = True
                max_strength = max(max_strength, float(strength))
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)

            running_cost_window = float(current_cost_window)
            for score, item, tau_eff, clean_eff, _strength in scored:
                if len(selected_items) >= available_servers:
                    residual_queue.append(item)
                    continue
                last_tau = float(tau_eff)
                last_clean = float(clean_eff)
                if score < tau_eff:
                    residual_queue.append(item)
                    continue
                if not _job_fits_cost_budget(item.job, cost_spent_window=running_cost_window, cfg=cfg):
                    residual_queue.append(item)
                    budget_blocked_starts += 1
                    continue
                selected_items.append(item)
                running_cost_window += float(item.job.actual_service_cost)

            selected_ids = {x.job.job_id for x in selected_items}
            residual_ids = {x.job.job_id for x in residual_queue}
            for _, item, _, _, _ in scored:
                if item.job.job_id not in selected_ids and item.job.job_id not in residual_ids:
                    residual_queue.append(item)
            queue = residual_queue

        if any_effective_correction:
            effective_correction_step_count += 1
            if arrivals_this_step:
                effective_arrival_step_count += 1
            if queue or selected_items:
                effective_decision_step_count += 1

        for item in selected_items:
            job = item.job
            delay = int(step - item.enqueue_step)
            queue_delays_served.append(delay)
            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
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
        threshold_trace.append(float(last_tau))
        clean_penalty_trace.append(float(last_clean))
        max_strength_trace.append(float(max_strength))

    total_attack_jobs = int(np.sum([job.is_attack for job in jobs]))
    total_clean_jobs = int(len(jobs) - total_attack_jobs)

    attack_recall = float(len(served_attack_jobs) / max(total_attack_jobs, 1))
    weighted_attack_recall = float(served_true_severity / max(total_true_severity, EPS))
    weighted_attack_recall_no_backend_fail = float(served_true_severity_no_fail / max(total_true_severity, EPS))
    served_attack_precision = float(len(served_attack_jobs) / max(len(served_attack_jobs) + len(served_clean_jobs), 1))
    clean_service_ratio = float(len(served_clean_jobs) / max(total_clean_jobs, 1))

    mean_active_servers = float(np.mean(active_server_trace)) if active_server_trace else 0.0
    server_utilization = float(mean_active_servers / max(int(cfg.slot_budget), 1))

    summary = {
        "total_steps": int(total_steps),
        "total_jobs": int(len(jobs)),
        "total_attack_jobs": int(total_attack_jobs),
        "total_clean_jobs": int(total_clean_jobs),
        "served_attack_jobs": int(len(served_attack_jobs)),
        "served_clean_jobs": int(len(served_clean_jobs)),
        "dropped_ttl": int(len(dropped_jobs_ttl)),
        "budget_blocked_starts": int(budget_blocked_starts),
        "attack_recall": float(attack_recall),
        "weighted_attack_recall": float(weighted_attack_recall),
        "weighted_attack_recall_no_backend_fail": float(weighted_attack_recall_no_backend_fail),
        "served_attack_precision": float(served_attack_precision),
        "unnecessary_mtd_count": int(len(served_clean_jobs)),
        "clean_service_ratio": float(clean_service_ratio),
        "total_service_time": float(total_service_time),
        "total_service_cost": float(total_service_cost),
        "average_service_time_per_step": float(total_service_time / max(int(total_steps), 1)),
        "average_service_cost_per_step": float(total_service_cost / max(int(total_steps), 1)),
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
        "mean_active_servers": float(mean_active_servers),
        "server_utilization": float(server_utilization),
        "threshold_trace_mean": float(np.mean(threshold_trace)) if threshold_trace else np.nan,
        "clean_penalty_trace_mean": float(np.mean(clean_penalty_trace)) if clean_penalty_trace else np.nan,
        "overload_strength_trace_mean": float(np.mean(max_strength_trace)) if max_strength_trace else 0.0,
        "pred_expected_consequence_served_ratio": float(served_pred_expected_consequence / max(total_pred_expected_consequence, EPS)),
    }
    diagnostics = {
        "all_steps_rate": float(overload_step_count / max(total_step_count, 1)),
        "arrival_steps_rate": float(overload_arrival_step_count / max(arrival_step_count, 1)),
        "active_decision_steps_rate": float(overload_decision_step_count / max(decision_step_count, 1)),
        "effective_all_steps_rate": float(effective_correction_step_count / max(total_step_count, 1)),
        "effective_arrival_steps_rate": float(effective_arrival_step_count / max(arrival_step_count, 1)),
        "effective_active_decision_steps_rate": float(effective_decision_step_count / max(decision_step_count, 1)),
        "mean_ov_given_trigger": float(np.mean(ov_on_trigger)) if ov_on_trigger else 0.0,
        "p95_ov_given_trigger": float(np.quantile(ov_on_trigger, 0.95)) if ov_on_trigger else 0.0,
        "n_total_steps": int(total_step_count),
        "n_arrival_steps": int(arrival_step_count),
        "n_active_decision_steps": int(decision_step_count),
        "n_overload_steps": int(overload_step_count),
        "n_overload_arrival_steps": int(overload_arrival_step_count),
        "n_overload_active_decision_steps": int(overload_decision_step_count),
        "n_effective_correction_steps": int(effective_correction_step_count),
    }
    return {"summary": summary, "diagnostics": diagnostics}



def _policy_compact(summary: Dict[str, float]) -> Dict[str, float | int]:
    return {
        "weighted_attack_recall_no_backend_fail": round(float(summary["weighted_attack_recall_no_backend_fail"]), 4),
        "unnecessary_mtd_count": int(summary["unnecessary_mtd_count"]),
        "queue_delay_p95": round(float(summary["queue_delay_p95"]), 4),
        "average_service_cost_per_step": round(float(summary["average_service_cost_per_step"]), 6),
        "pred_expected_consequence_served_ratio": round(float(summary["pred_expected_consequence_served_ratio"]), 4),
    }



def _diag_compact(diag: Dict[str, float]) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for key, value in diag.items():
        if key.startswith("n_"):
            out[key] = int(value)
        else:
            out[key] = round(float(value), 6)
    return out



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



def _score_kwargs(args: SimpleNamespace, *, cost_budget_per_step: float | None) -> Dict[str, float]:
    return {
        "max_wait_steps": int(args.max_wait_steps),
        "clean_penalty": float(args.objective_clean_penalty),
        "delay_penalty": float(args.objective_delay_penalty),
        "queue_penalty": float(args.objective_queue_penalty),
        "cost_penalty": float(args.objective_cost_penalty),
        "cost_budget_per_step": cost_budget_per_step,
    }



def _build_overload_cfg(
    *,
    slot_budget: int,
    phase3_best: Dict[str, float],
    variant: OverloadVariantSpec,
    args: SimpleNamespace,
    train_stats: Dict[str, float],
) -> OverloadConfig:
    return OverloadConfig(
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        window_cost_budget=None,
        cost_budget_window_steps=0,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
        v_weight=float(phase3_best["v_weight"]),
        clean_penalty=float(phase3_best["clean_penalty"]),
        age_bonus=float(phase3_best["age_bonus"]),
        urgency_bonus=float(phase3_best["urgency_bonus"]),
        fail_penalty=float(phase3_best["fail_penalty"]),
        busy_penalty=float(phase3_best["busy_penalty"]),
        cost_penalty=float(phase3_best["cost_penalty"]),
        admission_score_threshold=float(phase3_best["admission_score_threshold"]),
        tau_gain=float(variant.tau_gain),
        clean_gain=float(variant.clean_gain),
        use_urgency_mask=bool(variant.use_urgency_mask),
    )



def _variant_payload(variant: OverloadVariantSpec) -> Dict[str, object]:
    return {
        "name": str(variant.name),
        "tau_gain": float(variant.tau_gain),
        "clean_gain": float(variant.clean_gain),
        "use_urgency_mask": bool(variant.use_urgency_mask),
        "description": str(variant.description),
    }



def _select_joint_winner(screen_payload: Dict[str, object]) -> Dict[str, object]:
    variants = screen_payload["variants"]
    best_name = None
    best_score = -1e18
    best_record = None
    for name, payload in variants.items():
        joint = float(payload["joint_val_delta_objective"])
        if joint > best_score:
            best_score = joint
            best_name = name
            best_record = payload
    assert best_name is not None and best_record is not None
    return {
        "winner_variant": str(best_name),
        "winner_joint_val_delta_objective": float(best_score),
        "winner_payload": best_record,
    }



def _build_jobs_for_bank(
    arrays_bank: Dict[str, np.ndarray],
    *,
    arrays_train: Dict[str, np.ndarray],
    posterior_verify,
    service_models,
    severity_models,
    args: SimpleNamespace,
    busy_time_unit: float,
) -> Tuple[List[AlarmJob], int]:
    return _predict_jobs(
        arrays_bank,
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



def _screen_variants(
    *,
    manifest: Dict[str, object],
) -> Dict[str, object]:
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
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp

    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))
    jobs_train, total_steps_train = _build_jobs_for_bank(
        arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        service_models=service_models,
        severity_models=severity_models,
        args=args,
        busy_time_unit=busy_time_unit,
    )
    jobs_tune, total_steps_tune = _build_jobs_for_bank(
        arrays_tune,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        service_models=service_models,
        severity_models=severity_models,
        args=args,
        busy_time_unit=busy_time_unit,
    )

    train_stats = _job_stats(jobs_train)
    tune_stats = _job_stats(jobs_tune)
    train_arrival_diag = _arrival_diagnostics(jobs_train, total_steps_train)
    tune_arrival_diag = _arrival_diagnostics(jobs_tune, total_steps_tune)

    screen: Dict[str, object] = {
        "method": "phase3_overload_only_correction_family",
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
            "train_job_stats": train_stats,
            "val_job_stats": tune_stats,
            "train_arrival_diagnostics": train_arrival_diag,
            "val_arrival_diagnostics": tune_arrival_diag,
        },
        "phase3_reference_by_slot": {},
        "variants": {},
    }

    phase3_by_slot: Dict[str, Dict[str, object]] = {}
    phase3_val_objective_by_slot: Dict[str, float] = {}

    for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
        score_kwargs = _score_kwargs(args, cost_budget_per_step=None)
        phase3_best, phase3_res = _tune_proposed_ca_policy(
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
        phase3_obj = float(_objective(phase3_res["summary"], slot_budget=int(slot_budget), **score_kwargs))
        phase3_val_objective_by_slot[str(slot_budget)] = phase3_obj
        phase3_by_slot[str(slot_budget)] = {
            "config": phase3_best,
            "val_summary": _policy_compact(phase3_res["summary"]),
            "val_objective": phase3_obj,
        }
        screen["phase3_reference_by_slot"][str(slot_budget)] = phase3_by_slot[str(slot_budget)]

    for variant in DEFAULT_VARIANTS:
        variant_payload: Dict[str, object] = {
            "variant": _variant_payload(variant),
            "by_slot": {},
        }
        joint_delta = []
        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            phase3_best = phase3_by_slot[str(slot_budget)]["config"]
            cfg = _build_overload_cfg(
                slot_budget=slot_budget,
                phase3_best=phase3_best,
                variant=variant,
                args=args,
                train_stats=train_stats,
            )
            train_res = simulate_overload_phase3(jobs_train, total_steps=total_steps_train, cfg=cfg)
            val_res = simulate_overload_phase3(jobs_tune, total_steps=total_steps_tune, cfg=cfg)
            score_kwargs = _score_kwargs(args, cost_budget_per_step=None)
            val_obj = float(_objective(val_res["summary"], slot_budget=int(slot_budget), **score_kwargs))
            delta_obj = float(val_obj - phase3_val_objective_by_slot[str(slot_budget)])
            joint_delta.append(delta_obj)
            variant_payload["by_slot"][str(slot_budget)] = {
                "phase3_reference": phase3_by_slot[str(slot_budget)],
                "variant_config": {
                    **_variant_payload(variant),
                    "base_phase3_clean_penalty": float(phase3_best["clean_penalty"]),
                    "base_phase3_admission_threshold": float(phase3_best["admission_score_threshold"]),
                },
                "train_trigger_diagnostics": _diag_compact(train_res["diagnostics"]),
                "val_trigger_diagnostics": _diag_compact(val_res["diagnostics"]),
                "train_summary": _policy_compact(train_res["summary"]),
                "val_summary": _policy_compact(val_res["summary"]),
                "val_objective": val_obj,
                "val_delta_objective_vs_phase3": delta_obj,
                "val_delta_recall_vs_phase3": round(
                    float(val_res["summary"]["weighted_attack_recall_no_backend_fail"] - phase3_by_slot[str(slot_budget)]["val_summary"]["weighted_attack_recall_no_backend_fail"]),
                    6,
                ),
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
        variant_payload["joint_val_delta_objective"] = float(np.mean(joint_delta))
        screen["variants"][variant.name] = variant_payload

    screen["selection"] = _select_joint_winner(screen)
    return screen



def run_phase3_overload_experiment(manifest_path: str, output_dir: str, *, screen_only: bool = False) -> Dict[str, object]:
    manifest = _load_json(manifest_path)
    workdir = Path(manifest["workdir"])
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    screen = _screen_variants(manifest=manifest)
    screen_path = out_root / "screen_train_val_summary.json"
    with open(screen_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(screen), f, ensure_ascii=False, indent=2)

    if screen_only:
        return {"screen_summary_path": str(screen_path.resolve()), "screen_summary": screen}

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
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))
    jobs_train, total_steps_train = _build_jobs_for_bank(
        arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        service_models=service_models,
        severity_models=severity_models,
        args=args,
        busy_time_unit=busy_time_unit,
    )
    train_stats = _job_stats(jobs_train)

    results: Dict[str, object] = {
        "method": "phase3_overload_only_correction_family",
        "winner_variant": _variant_payload(winner_variant),
        "screen_summary_path": str(screen_path.resolve()),
        "manifest": manifest,
        "phase3_reference_by_slot": screen["phase3_reference_by_slot"],
        "screen_selection": screen["selection"],
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
        jobs_test, total_steps_test = _build_jobs_for_bank(
            arrays_test,
            arrays_train=arrays_train,
            posterior_verify=posterior_verify,
            service_models=service_models,
            severity_models=severity_models,
            args=args,
            busy_time_unit=busy_time_unit,
        )

        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            phase3_best = screen["phase3_reference_by_slot"][str(slot_budget)]["config"]
            cfg = _build_overload_cfg(
                slot_budget=slot_budget,
                phase3_best=phase3_best,
                variant=winner_variant,
                args=args,
                train_stats=train_stats,
            )
            eval_res = simulate_overload_phase3(jobs_test, total_steps=total_steps_test, cfg=cfg)
            baselines = _extract_baselines(summary_json, slot_budget)
            compact = _policy_compact(eval_res["summary"])
            payload = {
                "phase3_overload": compact,
                "winner_trigger_diagnostics": _diag_compact(eval_res["diagnostics"]),
                **baselines,
            }
            holdout_row["slot_budget_results"][str(slot_budget)] = payload
            per_slot_records[str(slot_budget)].append(payload)
        results["per_holdout_results"].append(holdout_row)

    for slot, rows in per_slot_records.items():
        policies = ["phase3_overload", "phase3_proposed", "topk_expected_consequence"]
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

        def paired(name_a: str, name_b: str, key: str):
            return [float(r[name_a][key]) - float(r[name_b][key]) for r in rows]

        for pair_name, a, b in [
            ("overload_vs_phase3", "phase3_overload", "phase3_proposed"),
            ("overload_vs_best_threshold", "phase3_overload", "best_threshold"),
            ("overload_vs_topk_expected", "phase3_overload", "topk_expected_consequence"),
        ]:
            dre = paired(a, b, "weighted_attack_recall_no_backend_fail")
            dun = paired(a, b, "unnecessary_mtd_count")
            ddl = paired(a, b, "queue_delay_p95")
            dco = paired(a, b, "average_service_cost_per_step")
            slot_payload["paired_stats"][pair_name] = {
                "delta_recall": {"mean": float(np.mean(dre)), "std": float(np.std(dre)), "min": float(np.min(dre)), "max": float(np.max(dre))},
                "delta_unnecessary": {"mean": float(np.mean(dun)), "std": float(np.std(dun)), "min": float(np.min(dun)), "max": float(np.max(dun))},
                "delta_delay_p95": {"mean": float(np.mean(ddl)), "std": float(np.std(ddl)), "min": float(np.min(ddl)), "max": float(np.max(ddl))},
                "delta_cost_per_step": {"mean": float(np.mean(dco)), "std": float(np.std(dco)), "min": float(np.min(dco)), "max": float(np.max(dco))},
                "wins_on_recall": int(sum(x > 0 for x in dre)),
                "lower_unnecessary": int(sum(x < 0 for x in dun)),
            }
        slot_payload["trigger_diagnostics_mean"] = {
            key: float(np.mean([float(r["winner_trigger_diagnostics"][key]) for r in rows]))
            for key in [
                "all_steps_rate",
                "arrival_steps_rate",
                "active_decision_steps_rate",
                "effective_all_steps_rate",
                "effective_arrival_steps_rate",
                "effective_active_decision_steps_rate",
                "mean_ov_given_trigger",
                "p95_ov_given_trigger",
            ]
        }
        results["slot_budget_aggregates"][str(slot)] = slot_payload

    aggregate_path = out_root / "multi_holdout" / "aggregate_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
    return {
        "screen_summary_path": str(screen_path.resolve()),
        "aggregate_summary_path": str(aggregate_path.resolve()),
        "aggregate_summary": results,
    }

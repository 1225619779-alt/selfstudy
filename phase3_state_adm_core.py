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
from scheduler.policies_phase3 import AlarmJob, quantize_busy_steps
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


@dataclass
class StateAdmConfig:
    slot_budget: int
    max_wait_steps: int
    rng_seed: int
    window_cost_budget: float | None
    cost_budget_window_steps: int
    mean_pred_busy_steps: float
    mean_pred_service_cost: float
    mean_pred_expected_consequence: float
    # phase3 score weights
    v_weight: float
    clean_penalty: float
    age_bonus: float
    urgency_bonus: float
    fail_penalty: float
    busy_penalty: float
    cost_penalty: float
    # state-conditioned admission threshold
    base_threshold: float
    queue_lambda: float
    server_lambda: float
    cost_lambda: float


@dataclass
class _QueuedJob:
    job: AlarmJob
    enqueue_step: int


@dataclass
class _ActiveServer:
    job_id: int
    busy_until_step: int


def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: StateAdmConfig) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))


def _queue_pressure(queue_len: int, cfg: StateAdmConfig) -> float:
    return float(queue_len / max(int(cfg.slot_budget), 1))


def _cost_pressure(cost_spent_window: float, cfg: StateAdmConfig) -> float:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))


def _phase3_score(
    item: _QueuedJob,
    *,
    step: int,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: StateAdmConfig,
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
    clean_term = float(cfg.clean_penalty) * float(1.0 - job.pred_attack_prob)
    age_term = float(cfg.age_bonus) * float(age / max(cfg.max_wait_steps, 1))
    urgency_term = float(cfg.urgency_bonus) * float(1.0 / (ttl_left + 1.0))
    fail_term = float(cfg.fail_penalty) * float(job.pred_fail_prob)
    busy_term = float(cfg.busy_penalty) * (server_pressure + queue_pressure) * norm_busy
    cost_term = float(cfg.cost_penalty) * cost_pressure * norm_cost
    return float(value_term + age_term + urgency_term - clean_term - fail_term - busy_term - cost_term)


def _state_threshold(
    *,
    queue_len: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: StateAdmConfig,
) -> float:
    return float(
        cfg.base_threshold
        + cfg.queue_lambda * _queue_pressure(queue_len, cfg)
        + cfg.server_lambda * _server_pressure(active_servers, cfg)
        + cfg.cost_lambda * _cost_pressure(cost_spent_window, cfg)
    )


def _job_fits_cost_budget(job: AlarmJob, *, cost_spent_window: float, cfg: StateAdmConfig) -> bool:
    if cfg.window_cost_budget is None or cfg.window_cost_budget <= 0 or cfg.cost_budget_window_steps <= 0:
        return True
    projected = float(cost_spent_window) + float(job.actual_service_cost)
    return bool(projected <= float(cfg.window_cost_budget) + 1e-9)


def simulate_state_adm_phase3(jobs: Sequence[AlarmJob], *, total_steps: int, cfg: StateAdmConfig) -> Dict[str, object]:
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
    budget_blocked_starts = 0

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]

        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        for job in arrivals.get(step, []):
            queue.append(_QueuedJob(job=job, enqueue_step=step))

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
        last_thr = _state_threshold(queue_len=len(queue), active_servers=active_servers, cost_spent_window=current_cost_window, cfg=cfg)

        if available_servers > 0 and queue:
            scored = [
                (_phase3_score(item, step=step, queue_len=len(queue), active_servers=active_servers, cost_spent_window=current_cost_window, cfg=cfg), item)
                for item in queue
            ]
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)
            running_cost_window = float(current_cost_window)
            for score, item in scored:
                if len(selected_items) >= available_servers:
                    residual_queue.append(item)
                    continue
                thr = _state_threshold(queue_len=len(queue), active_servers=active_servers, cost_spent_window=running_cost_window, cfg=cfg)
                last_thr = thr
                if score < thr:
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
            for _, item in scored:
                if item.job.job_id not in selected_ids and item.job.job_id not in residual_ids:
                    residual_queue.append(item)
            queue = residual_queue

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
        threshold_trace.append(float(last_thr))

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
        "pred_expected_consequence_served_ratio": float(served_pred_expected_consequence / max(total_pred_expected_consequence, EPS)),
    }
    return {"summary": summary}


def _policy_compact(summary: Dict[str, float]) -> Dict[str, float | int]:
    return {
        "weighted_attack_recall_no_backend_fail": round(float(summary["weighted_attack_recall_no_backend_fail"]), 4),
        "unnecessary_mtd_count": int(summary["unnecessary_mtd_count"]),
        "queue_delay_p95": round(float(summary["queue_delay_p95"]), 4),
        "average_service_cost_per_step": round(float(summary["average_service_cost_per_step"]), 6),
        "pred_expected_consequence_served_ratio": round(float(summary["pred_expected_consequence_served_ratio"]), 4),
    }


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
    for name in ["threshold_verify_fifo", "threshold_ddd_fifo", "threshold_expected_consequence_fifo", "adaptive_threshold_verify_fifo"]:
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


def _tune_state_adm(
    *,
    jobs_tune: List[AlarmJob],
    total_steps_tune: int,
    phase3_best: Dict[str, float],
    slot_budget: int,
    args: SimpleNamespace,
    window_cost_budget: float | None,
    cost_budget_window_steps: int,
    train_stats: Dict[str, float],
    score_kwargs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    base_candidates = [float(phase3_best["admission_score_threshold"]) + x for x in [-0.2, -0.1, 0.0, 0.1, 0.2]]
    queue_candidates = [0.0, 0.25, 0.5, 1.0]
    server_candidates = [0.0, 0.25, 0.5, 1.0]
    cost_candidates = [0.0] if not args.use_cost_budget else [0.0, 0.25, 0.5]

    best_cfg = None
    best_res = None
    best_obj = -1e18
    for base in base_candidates:
        for ql in queue_candidates:
            for sl in server_candidates:
                for cl in cost_candidates:
                    cfg = StateAdmConfig(
                        slot_budget=slot_budget,
                        max_wait_steps=int(args.max_wait_steps),
                        rng_seed=int(args.rng_seed),
                        window_cost_budget=window_cost_budget,
                        cost_budget_window_steps=cost_budget_window_steps,
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
                        base_threshold=float(base),
                        queue_lambda=float(ql),
                        server_lambda=float(sl),
                        cost_lambda=float(cl),
                    )
                    res = simulate_state_adm_phase3(jobs_tune, total_steps=total_steps_tune, cfg=cfg)
                    obj = _objective(res["summary"], slot_budget=int(slot_budget), **score_kwargs)
                    if obj > best_obj:
                        best_obj = float(obj)
                        best_cfg = {
                            "base_threshold": float(base),
                            "queue_lambda": float(ql),
                            "server_lambda": float(sl),
                            "cost_lambda": float(cl),
                        }
                        best_res = res
    assert best_cfg is not None and best_res is not None
    return best_cfg, best_res


def run_phase3_state_adm_experiment(manifest_path: str, output_path: str) -> Dict[str, object]:
    manifest = _load_json(manifest_path)
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
        threshold_quantiles=[0.5, 0.6, 0.7, 0.8, 0.9],
        adaptive_gain_scale_list=[0.0, 0.1, 0.2, 0.4],
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
    jobs_train, total_steps_train = _predict_jobs(arrays_train, posterior_model=posterior_verify, posterior_signal_key="verify_score", service_models=service_models, service_signal_key="verify_score", severity_models=severity_models, severity_blend_verify=float(args.consequence_blend_verify), consequence_mode=str(args.consequence_mode), fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float), busy_time_unit=busy_time_unit)
    jobs_tune, total_steps_tune = _predict_jobs(arrays_tune, posterior_model=posterior_verify, posterior_signal_key="verify_score", service_models=service_models, service_signal_key="verify_score", severity_models=severity_models, severity_blend_verify=float(args.consequence_blend_verify), consequence_mode=str(args.consequence_mode), fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float), busy_time_unit=busy_time_unit)
    jobs_tune_ddd, _ = _predict_jobs(arrays_tune, posterior_model=posterior_ddd, posterior_signal_key="ddd_loss_recons", service_models=service_models, service_signal_key="verify_score", severity_models=severity_models, severity_blend_verify=float(args.consequence_blend_verify), consequence_mode=str(args.consequence_mode), fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float), busy_time_unit=busy_time_unit)

    train_stats = _job_stats(jobs_train)
    tune_stats = _job_stats(jobs_tune)
    train_arrival_diag = _arrival_diagnostics(jobs_train, total_steps_train)
    tune_arrival_diag = _arrival_diagnostics(jobs_tune, total_steps_tune)

    results: Dict[str, object] = {
        "method": "phase3_state_conditional_admission_plus_phase3_dispatch",
        "manifest": manifest,
        "config": {
            "decision_step_group": int(args.decision_step_group),
            "busy_time_quantile": float(args.busy_time_quantile),
            "use_cost_budget": bool(args.use_cost_budget),
            "slot_budget_list": manifest["frozen_regime"]["slot_budget_list"],
            "max_wait_steps": int(args.max_wait_steps),
            "consequence_blend_verify": float(args.consequence_blend_verify),
            "consequence_mode": str(args.consequence_mode),
        },
        "environment": {
            "busy_time_unit": float(busy_time_unit),
            "train_job_stats": train_stats,
            "val_job_stats": tune_stats,
            "train_arrival_diagnostics": train_arrival_diag,
            "val_arrival_diagnostics": tune_arrival_diag,
        },
        "tuned_by_slot": {},
        "n_holdouts": int(len(manifest["holdouts"])),
        "per_holdout_results": [],
        "slot_budget_aggregates": {},
    }

    per_slot_records: Dict[str, List[Dict[str, object]]] = {}

    for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
        window_cost_budget = None
        cost_budget_window_steps = 0
        cost_budget_per_step = None
        score_kwargs = _score_kwargs(args, cost_budget_per_step=cost_budget_per_step)

        phase3_best, phase3_res = _tune_proposed_ca_policy(
            jobs_tune, total_steps_tune,
            slot_budget=slot_budget, max_wait_steps=int(args.max_wait_steps), rng_seed=int(args.rng_seed),
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=train_stats["mean_pred_busy_steps"], mean_pred_service_cost=train_stats["mean_pred_service_cost"], mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
            v_grid=args.vq_v_grid, clean_grid=args.vq_clean_grid, age_grid=args.vq_age_grid, urgency_grid=args.vq_urgency_grid,
            fail_grid=args.vq_fail_grid, busy_grid=args.vq_busy_grid, cost_grid=args.vq_cost_grid,
            admission_threshold_grid=args.vq_admission_threshold_grid, score_kwargs=score_kwargs,
        )
        state_best, state_res = _tune_state_adm(
            jobs_tune=jobs_tune,
            total_steps_tune=total_steps_tune,
            phase3_best=phase3_best,
            slot_budget=slot_budget,
            args=args,
            window_cost_budget=window_cost_budget,
            cost_budget_window_steps=cost_budget_window_steps,
            train_stats=train_stats,
            score_kwargs=score_kwargs,
        )
        results["tuned_by_slot"][str(slot_budget)] = {
            "phase3_reference": {
                "config": phase3_best,
                "val_summary": _policy_compact(phase3_res["summary"]),
            },
            "state_adm_best": {
                **state_best,
                "val_summary": _policy_compact(state_res["summary"]),
            },
        }
        per_slot_records[str(slot_budget)] = []

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
        posterior_verify = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins)
        service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
        severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
        severity_models = severity_models_cond
        jobs_test, total_steps_test = _predict_jobs(arrays_test, posterior_model=posterior_verify, posterior_signal_key="verify_score", service_models=service_models, service_signal_key="verify_score", severity_models=severity_models, severity_blend_verify=float(args.consequence_blend_verify), consequence_mode=str(args.consequence_mode), fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float), busy_time_unit=busy_time_unit)

        for slot_budget in [int(x) for x in manifest["frozen_regime"]["slot_budget_list"]]:
            base = results["tuned_by_slot"][str(slot_budget)]
            phase3_best = base["phase3_reference"]["config"]
            state_best = base["state_adm_best"]
            cfg = StateAdmConfig(
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
                base_threshold=float(state_best["base_threshold"]),
                queue_lambda=float(state_best["queue_lambda"]),
                server_lambda=float(state_best["server_lambda"]),
                cost_lambda=float(state_best["cost_lambda"]),
            )
            eval_res = simulate_state_adm_phase3(jobs_test, total_steps=total_steps_test, cfg=cfg)
            baselines = _extract_baselines(summary_json, slot_budget)
            compact = _policy_compact(eval_res["summary"])
            payload = {
                "state_adm_phase3_dispatch": compact,
                **baselines,
            }
            holdout_row["slot_budget_results"][str(slot_budget)] = payload
            per_slot_records[str(slot_budget)].append(payload)
        results["per_holdout_results"].append(holdout_row)

    for slot, rows in per_slot_records.items():
        policies = ["state_adm_phase3_dispatch", "phase3_proposed", "topk_expected_consequence"]
        slot_payload = {"policy_stats": {}, "paired_stats": {}, "best_threshold_frequency": {}}
        for pol in policies:
            slot_payload["policy_stats"][pol] = {}
            for metric in ["weighted_attack_recall_no_backend_fail", "unnecessary_mtd_count", "queue_delay_p95", "average_service_cost_per_step", "pred_expected_consequence_served_ratio"]:
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
            ("stateadm_vs_phase3", "state_adm_phase3_dispatch", "phase3_proposed"),
            ("stateadm_vs_best_threshold", "state_adm_phase3_dispatch", "best_threshold"),
            ("stateadm_vs_topk_expected", "state_adm_phase3_dispatch", "topk_expected_consequence"),
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
        results["slot_budget_aggregates"][str(slot)] = slot_payload

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
    return results

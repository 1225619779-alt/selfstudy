from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Reuse the existing repo modules at runtime.
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
    _tune_threshold_policy,
    _tune_proposed_ca_policy,
)
from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import AlarmJob, SimulationConfig, _QueuedJob, _ActiveServer

EPS = 1e-12


def _tune_fixed_threshold_policy(
    jobs_fit,
    total_steps_fit,
    *,
    signal,
    policy_name,
    threshold_candidates,
    slot_budget,
    max_wait_steps,
    rng_seed,
    cost_budget_window_steps,
    window_cost_budget,
    mean_pred_busy_steps,
    mean_pred_service_cost,
    mean_pred_expected_consequence,
    score_kwargs,
):
    # Compatibility wrapper for repos where evaluation_budget_scheduler_phase3
    # exposes _tune_threshold_policy instead of _tune_fixed_threshold_policy.
    tune_score_kwargs = dict(score_kwargs)
    tune_score_kwargs.setdefault('max_wait_steps', int(max_wait_steps))
    best_thr, best_res = _tune_threshold_policy(
        jobs_fit,
        total_steps_fit,
        threshold_candidates=list(threshold_candidates),
        policy_name=str(policy_name),
        slot_budget=int(slot_budget),
        max_wait_steps=int(max_wait_steps),
        rng_seed=int(rng_seed),
        cost_budget_window_steps=int(cost_budget_window_steps),
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=float(mean_pred_busy_steps),
        mean_pred_service_cost=float(mean_pred_service_cost),
        mean_pred_expected_consequence=float(mean_pred_expected_consequence),
        score_kwargs=tune_score_kwargs,
    )
    return ({"threshold": float(best_thr)}, best_res)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_manifest(path: str) -> Dict[str, object]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _feature_vector(job: AlarmJob, queue_len: float, active_servers: float, slot_budget: int,
                    mean_busy: float, mean_cost: float, mean_ec: float) -> np.ndarray:
    q_pressure = float(queue_len) / max(float(slot_budget), 1.0)
    s_pressure = float(active_servers) / max(float(slot_budget), 1.0)
    norm_busy = float(job.pred_busy_steps) / max(float(mean_busy), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(mean_cost), EPS)
    norm_ec = float(job.pred_expected_consequence) / max(float(mean_ec), EPS)
    return np.asarray([
        1.0,
        norm_ec,
        float(job.pred_attack_prob),
        float(1.0 - job.pred_attack_prob),
        float(job.pred_fail_prob),
        norm_busy,
        norm_cost,
        q_pressure,
        s_pressure,
        float(job.verify_score),
        float(job.ddd_loss),
    ], dtype=float)


@dataclass
class LinearModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = (X - self.mean) / self.std
        return Z @ self.coef


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> LinearModel:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = (X - mean) / std
    A = Z.T @ Z + float(ridge) * np.eye(Z.shape[1])
    b = Z.T @ y
    coef = np.linalg.solve(A, b)
    return LinearModel(mean=mean, std=std, coef=coef)


def _build_predictions_and_jobs(args_like) -> Dict[str, object]:
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args_like.train_bank), int(args_like.decision_step_group))
    arrays_val = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args_like.val_bank), int(args_like.decision_step_group))

    posterior_verify = fit_attack_posterior_from_banks(
        args_like.clean_bank, args_like.attack_bank, signal_key='score_phys_l2', n_bins=args_like.n_bins
    )
    service_models = fit_service_models_from_mixed_bank(args_like.train_bank, signal_key='verify_score', n_bins=args_like.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=('verify_score', 'ddd_loss_recons'), n_bins=args_like.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=('verify_score', 'ddd_loss_recons'), n_bins=args_like.n_bins)
    severity_models = severity_models_cond if args_like.consequence_mode == 'conditional' else severity_models_exp

    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args_like.busy_time_quantile))
    jobs_train, total_steps_train = _predict_jobs(
        arrays_train,
        posterior_model=posterior_verify,
        posterior_signal_key='verify_score',
        service_models=service_models,
        service_signal_key='verify_score',
        severity_models=severity_models,
        severity_blend_verify=float(args_like.consequence_blend_verify),
        consequence_mode=str(args_like.consequence_mode),
        fit_verify_score=np.asarray(arrays_train['verify_score'], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_val, total_steps_val = _predict_jobs(
        arrays_val,
        posterior_model=posterior_verify,
        posterior_signal_key='verify_score',
        service_models=service_models,
        service_signal_key='verify_score',
        severity_models=severity_models,
        severity_blend_verify=float(args_like.consequence_blend_verify),
        consequence_mode=str(args_like.consequence_mode),
        fit_verify_score=np.asarray(arrays_train['verify_score'], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    return {
        'arrays_train': arrays_train,
        'arrays_val': arrays_val,
        'jobs_train': jobs_train,
        'jobs_val': jobs_val,
        'total_steps_train': total_steps_train,
        'total_steps_val': total_steps_val,
        'busy_time_unit': busy_time_unit,
        'job_stats_train': _job_stats(jobs_train),
        'job_stats_val': _job_stats(jobs_val),
        'arrival_diag_train': _arrival_diagnostics(jobs_train, total_steps_train),
        'arrival_diag_val': _arrival_diagnostics(jobs_val, total_steps_val),
    }


@dataclass
class AdmissionConfig:
    base_threshold: float
    lam_clean_init: float
    lam_busy_init: float
    lam_queue_init: float
    lam_cost_init: float
    eta_clean: float
    eta_busy: float
    eta_queue: float
    eta_cost: float
    target_clean_ratio: float
    target_server_util: float
    target_queue_ratio: float
    target_cost_pressure: float


@dataclass
class DispatchConfig:
    v_weight: float
    clean_penalty: float
    age_bonus: float
    urgency_bonus: float
    fail_penalty: float
    busy_penalty: float
    cost_penalty: float
    admission_score_threshold: float


def _phase3_score(job: AlarmJob, *, age: int, queue_len: int, active_servers: Sequence[_ActiveServer],
                  cost_spent_window: float, slot_budget: int, max_wait_steps: int,
                  mean_pred_busy_steps: float, mean_pred_service_cost: float,
                  mean_pred_expected_consequence: float, dsp: DispatchConfig) -> float:
    queue_pressure = float(queue_len) / max(float(slot_budget), 1.0)
    server_pressure = float(len(active_servers)) / max(float(slot_budget), 1.0)
    cost_pressure = 0.0
    norm_busy = float(job.pred_busy_steps) / max(float(mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(mean_pred_service_cost), EPS)
    norm_ec = float(job.pred_expected_consequence) / max(float(mean_pred_expected_consequence), EPS)
    ttl_left = max(int(max_wait_steps) - age, 0)
    return float(
        dsp.v_weight * norm_ec
        + dsp.age_bonus * float(age / max(max_wait_steps, 1))
        + dsp.urgency_bonus * float(1.0 / (ttl_left + 1.0))
        - dsp.clean_penalty * float(1.0 - job.pred_attack_prob)
        - dsp.fail_penalty * float(job.pred_fail_prob)
        - dsp.busy_penalty * (server_pressure + queue_pressure) * norm_busy
        - dsp.cost_penalty * cost_pressure * norm_cost
    )


def _run_phase3_policy(jobs: Sequence[AlarmJob], total_steps: int, *, slot_budget: int, max_wait_steps: int,
                       cost_budget_window_steps: int, window_cost_budget: float | None,
                       mean_pred_busy_steps: float, mean_pred_service_cost: float,
                       mean_pred_expected_consequence: float, dsp: DispatchConfig, rng_seed: int) -> Dict[str, object]:
    cfg = SimulationConfig(
        policy_name='proposed_ca_vq_hard',
        slot_budget=int(slot_budget),
        max_wait_steps=int(max_wait_steps),
        rng_seed=int(rng_seed),
        cost_budget_window_steps=int(cost_budget_window_steps),
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=float(mean_pred_busy_steps),
        mean_pred_service_cost=float(mean_pred_service_cost),
        mean_pred_expected_consequence=float(mean_pred_expected_consequence),
        v_weight=float(dsp.v_weight),
        clean_penalty=float(dsp.clean_penalty),
        age_bonus=float(dsp.age_bonus),
        urgency_bonus=float(dsp.urgency_bonus),
        fail_penalty=float(dsp.fail_penalty),
        busy_penalty=float(dsp.busy_penalty),
        cost_penalty=float(dsp.cost_penalty),
        admission_score_threshold=float(dsp.admission_score_threshold),
    )
    return _run_one_policy(list(jobs), int(total_steps), cfg)


def _state_features_from_baseline(jobs: Sequence[AlarmJob], total_steps: int, baseline_res: Dict[str, object]) -> Dict[int, Tuple[float, float, float]]:
    # state before arrivals at step t = state after action at step t-1
    trace = baseline_res['step_trace']
    state_by_step: Dict[int, Tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
    last_cost = 0.0
    for t in range(1, int(total_steps)+1):
        prev = trace[t-1]
        state_by_step[t] = (
            float(prev['queue_len_after_action']),
            float(prev['active_servers_after_action']),
            float(prev['current_cost_window_after_action']) if 'current_cost_window_after_action' in prev else last_cost,
        )
        last_cost = state_by_step[t][2]
    return {job.job_id: state_by_step.get(int(job.arrival_step), (0.0, 0.0, 0.0)) for job in jobs}


def _fit_helpgain_model(jobs_train: List[AlarmJob], total_steps_train: int, *, slot_budget: int,
                        max_wait_steps: int, cost_budget_window_steps: int,
                        window_cost_budget: float | None, mean_pred_busy_steps: float,
                        mean_pred_service_cost: float, mean_pred_expected_consequence: float,
                        dsp: DispatchConfig, objective_kwargs: Dict[str, float], rng_seed: int) -> Tuple[LinearModel, Dict[str, object]]:
    base_res = _run_phase3_policy(
        jobs_train, total_steps_train,
        slot_budget=slot_budget, max_wait_steps=max_wait_steps,
        cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
        mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp, rng_seed=rng_seed,
    )
    base_obj = _objective(base_res['summary'], max_wait_steps=max_wait_steps, slot_budget=slot_budget, **objective_kwargs)
    state_map = _state_features_from_baseline(jobs_train, total_steps_train, base_res)
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    # Leave-one-out simulations. Heavy but manageable for the current scale.
    jobs_cache = list(jobs_train)
    for idx, job in enumerate(jobs_cache):
        jobs_wo = jobs_cache[:idx] + jobs_cache[idx+1:]
        res_wo = _run_phase3_policy(
            jobs_wo, total_steps_train,
            slot_budget=slot_budget, max_wait_steps=max_wait_steps,
            cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
            mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
            mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp, rng_seed=rng_seed,
        )
        obj_wo = _objective(res_wo['summary'], max_wait_steps=max_wait_steps, slot_budget=slot_budget, **objective_kwargs)
        gain = float(base_obj - obj_wo)
        q_len, a_srv, _cost_win = state_map[job.job_id]
        X_rows.append(_feature_vector(job, q_len, a_srv, slot_budget, mean_pred_busy_steps, mean_pred_service_cost, mean_pred_expected_consequence))
        y_rows.append(gain)
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=float)
    model = _fit_ridge(X, y, ridge=1e-2)
    return model, {
        'baseline_summary': base_res['summary'],
        'baseline_objective': float(base_obj),
        'gain_stats': {
            'mean': float(np.mean(y)), 'std': float(np.std(y)), 'min': float(np.min(y)), 'max': float(np.max(y)),
            'q25': float(np.quantile(y, 0.25)), 'q50': float(np.quantile(y, 0.5)), 'q75': float(np.quantile(y, 0.75)),
        },
    }


def _cost_fits(actual_cost: float, current_cost_window: float, budget: float | None) -> bool:
    if budget is None or budget <= 0:
        return True
    return bool(float(current_cost_window) + float(actual_cost) <= float(budget) + 1e-9)


def simulate_counterfactual_help(
    jobs: Sequence[AlarmJob],
    *,
    total_steps: int,
    slot_budget: int,
    max_wait_steps: int,
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    model: LinearModel,
    adm: AdmissionConfig,
    dsp: DispatchConfig,
    window_cost_budget: float | None,
    cost_budget_window_steps: int,
    rng_seed: int,
) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)
    queue: List[_QueuedJob] = []
    active_servers: List[_ActiveServer] = []
    rolling_cost: List[Tuple[int, float]] = []

    lam_clean = float(adm.lam_clean_init)
    lam_busy = float(adm.lam_busy_init)
    lam_queue = float(adm.lam_queue_init)
    lam_cost = float(adm.lam_cost_init)

    total_true_severity = float(np.sum([j.severity_true for j in jobs if j.is_attack == 1]))
    served_true_severity = 0.0
    served_true_severity_nf = 0.0
    total_pred_ec = float(np.sum([j.pred_expected_consequence for j in jobs]))
    served_pred_ec = 0.0

    served_attack = 0
    served_clean = 0
    total_service_time = 0.0
    total_service_cost = 0.0
    total_backend_fail = 0
    queue_delays: List[int] = []
    attack_delays: List[int] = []
    clean_delays: List[int] = []
    queue_len_trace: List[int] = []
    active_trace: List[int] = []
    occupied_server_steps = 0.0

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]
        if int(cost_budget_window_steps) > 0:
            rolling_cost = [(t, c) for (t, c) in rolling_cost if int(t) > int(step) - int(cost_budget_window_steps)]
        current_cost_window = float(sum(c for _, c in rolling_cost))

        # Admission at arrivals.
        for job in arrivals.get(step, []):
            q_pressure = float(len(queue)) / max(float(slot_budget), 1.0)
            s_pressure = float(len(active_servers)) / max(float(slot_budget), 1.0)
            cost_pressure = 0.0 if not window_cost_budget else float(current_cost_window / max(float(window_cost_budget), EPS))
            feat = _feature_vector(job, len(queue), len(active_servers), slot_budget, mean_pred_busy_steps, mean_pred_service_cost, mean_pred_expected_consequence)
            base_gain = float(model.predict(feat)[0])
            norm_busy = float(job.pred_busy_steps) / max(float(mean_pred_busy_steps), EPS)
            norm_cost = float(job.pred_service_cost) / max(float(mean_pred_service_cost), EPS)
            help_score = (
                base_gain
                - lam_clean * float(1.0 - job.pred_attack_prob)
                - lam_busy * s_pressure * norm_busy
                - lam_queue * q_pressure
                - lam_cost * cost_pressure * norm_cost
            )
            if help_score >= float(adm.base_threshold):
                queue.append(_QueuedJob(job=job, enqueue_step=step))

        # TTL drop
        new_queue: List[_QueuedJob] = []
        for item in queue:
            if int(step - item.enqueue_step) > int(max_wait_steps):
                continue
            new_queue.append(item)
        queue = new_queue

        available = max(int(slot_budget) - len(active_servers), 0)
        selected: List[_QueuedJob] = []
        if available > 0 and queue:
            scored: List[Tuple[float, _QueuedJob]] = []
            for item in queue:
                age = int(step - item.enqueue_step)
                score = _phase3_score(
                    item.job, age=age, queue_len=len(queue), active_servers=active_servers, cost_spent_window=current_cost_window,
                    slot_budget=slot_budget, max_wait_steps=max_wait_steps,
                    mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
                    mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp,
                )
                scored.append((score, item))
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)
            residual: List[_QueuedJob] = []
            running_cost_window = float(current_cost_window)
            for score, item in scored:
                if len(selected) >= available:
                    residual.append(item)
                    continue
                if float(score) < float(dsp.admission_score_threshold):
                    residual.append(item)
                    continue
                if not _cost_fits(item.job.actual_service_cost, running_cost_window, window_cost_budget):
                    residual.append(item)
                    continue
                selected.append(item)
                running_cost_window += float(item.job.actual_service_cost)
            selected_ids = {x.job.job_id for x in selected}
            residual_ids = {x.job.job_id for x in residual}
            for _score, item in scored:
                if item.job.job_id not in selected_ids and item.job.job_id not in residual_ids:
                    residual.append(item)
            queue = residual

        # serve selected
        for item in selected:
            j = item.job
            delay = int(step - item.enqueue_step)
            queue_delays.append(delay)
            total_service_time += float(j.actual_service_time)
            total_service_cost += float(j.actual_service_cost)
            total_backend_fail += int(j.actual_backend_fail)
            served_pred_ec += float(j.pred_expected_consequence)
            active_servers.append(_ActiveServer(job_id=j.job_id, busy_until_step=int(step + max(int(j.actual_busy_steps), 1))))
            if window_cost_budget is not None and cost_budget_window_steps > 0:
                rolling_cost.append((int(step), float(j.actual_service_cost)))
                current_cost_window += float(j.actual_service_cost)
            if int(j.is_attack) == 1:
                served_attack += 1
                attack_delays.append(delay)
                served_true_severity += float(j.severity_true)
                if int(j.actual_backend_fail) == 0:
                    served_true_severity_nf += float(j.severity_true)
            else:
                served_clean += 1
                clean_delays.append(delay)

        # update duals after observing step pressures
        clean_ratio_so_far = float(served_clean) / max(float(sum(1 for jj in jobs if jj.is_attack == 0)), 1.0)
        server_util = float(len(active_servers)) / max(float(slot_budget), 1.0)
        queue_ratio = float(len(queue)) / max(float(slot_budget), 1.0)
        cost_pressure = 0.0 if not window_cost_budget else float(current_cost_window / max(float(window_cost_budget), EPS))
        lam_clean = max(0.0, lam_clean + float(adm.eta_clean) * (clean_ratio_so_far - float(adm.target_clean_ratio)))
        lam_busy = max(0.0, lam_busy + float(adm.eta_busy) * (server_util - float(adm.target_server_util)))
        lam_queue = max(0.0, lam_queue + float(adm.eta_queue) * (queue_ratio - float(adm.target_queue_ratio)))
        lam_cost = max(0.0, lam_cost + float(adm.eta_cost) * (cost_pressure - float(adm.target_cost_pressure)))

        occupied_server_steps += float(len(active_servers))
        queue_len_trace.append(int(len(queue)))
        active_trace.append(int(len(active_servers)))

    total_attack_jobs = int(sum(1 for j in jobs if j.is_attack == 1))
    total_clean_jobs = int(sum(1 for j in jobs if j.is_attack == 0))
    summary = {
        'total_steps': int(total_steps),
        'total_jobs': int(len(jobs)),
        'total_attack_jobs': int(total_attack_jobs),
        'total_clean_jobs': int(total_clean_jobs),
        'served_attack_jobs': int(served_attack),
        'served_clean_jobs': int(served_clean),
        'attack_recall': float(served_attack / max(total_attack_jobs, 1)),
        'weighted_attack_recall': float(served_true_severity / max(total_true_severity, EPS)),
        'weighted_attack_recall_no_backend_fail': float(served_true_severity_nf / max(total_true_severity, EPS)),
        'unnecessary_mtd_count': int(served_clean),
        'clean_service_ratio': float(served_clean / max(total_clean_jobs, 1)),
        'total_service_time': float(total_service_time),
        'total_service_cost': float(total_service_cost),
        'average_service_time_per_step': float(total_service_time / max(int(total_steps), 1)),
        'average_service_cost_per_step': float(total_service_cost / max(int(total_steps), 1)),
        'total_backend_fail': int(total_backend_fail),
        'queue_delay_mean': float(np.mean(queue_delays)) if queue_delays else 0.0,
        'queue_delay_p95': float(np.quantile(queue_delays, 0.95)) if queue_delays else 0.0,
        'attack_delay_mean': float(np.mean(attack_delays)) if attack_delays else 0.0,
        'attack_delay_p95': float(np.quantile(attack_delays, 0.95)) if attack_delays else 0.0,
        'clean_delay_mean': float(np.mean(clean_delays)) if clean_delays else 0.0,
        'clean_delay_p95': float(np.quantile(clean_delays, 0.95)) if clean_delays else 0.0,
        'mean_queue_len': float(np.mean(queue_len_trace)) if queue_len_trace else 0.0,
        'max_queue_len': int(np.max(queue_len_trace)) if queue_len_trace else 0,
        'mean_active_servers': float(np.mean(active_trace)) if active_trace else 0.0,
        'server_utilization': float(occupied_server_steps / max(int(total_steps) * max(int(slot_budget), 1), 1)),
        'pred_expected_consequence_served_ratio': float(served_pred_ec / max(total_pred_ec, EPS)),
    }
    return {'summary': summary}


def _tune_counterfactual_help(
    jobs_train: List[AlarmJob], total_steps_train: int,
    jobs_val: List[AlarmJob], total_steps_val: int,
    *, slot_budget: int, max_wait_steps: int, cost_budget_window_steps: int,
    window_cost_budget: float | None, mean_pred_busy_steps: float,
    mean_pred_service_cost: float, mean_pred_expected_consequence: float,
    dsp: DispatchConfig, objective_kwargs: Dict[str, float], rng_seed: int,
) -> Tuple[LinearModel, AdmissionConfig, Dict[str, object], Dict[str, object]]:
    model, fit_payload = _fit_helpgain_model(
        jobs_train, total_steps_train,
        slot_budget=slot_budget, max_wait_steps=max_wait_steps,
        cost_budget_window_steps=cost_budget_window_steps, window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
        mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp,
        objective_kwargs=objective_kwargs, rng_seed=rng_seed,
    )
    # Tune admission on validation.
    target_clean_candidates = [0.01, 0.02, 0.03, 0.05, 0.08]
    init_clean_candidates = [0.0, 0.5, 1.0, 2.0]
    init_busy_candidates = [0.0, 0.25, 0.5]
    init_queue_candidates = [0.0, 0.25, 0.5]
    thr_candidates = [-0.2, -0.1, 0.0, 0.1]
    eta_candidates = [0.02, 0.05]

    best_cfg = None
    best_res = None
    best_obj = -1e18
    for thr in thr_candidates:
        for lc in init_clean_candidates:
            for lb in init_busy_candidates:
                for lq in init_queue_candidates:
                    for eta in eta_candidates:
                        for tgt_clean in target_clean_candidates:
                            adm = AdmissionConfig(
                                base_threshold=float(thr),
                                lam_clean_init=float(lc),
                                lam_busy_init=float(lb),
                                lam_queue_init=float(lq),
                                lam_cost_init=0.0,
                                eta_clean=float(eta), eta_busy=float(eta), eta_queue=float(eta), eta_cost=0.0,
                                target_clean_ratio=float(tgt_clean),
                                target_server_util=0.75 if slot_budget == 1 else 0.65,
                                target_queue_ratio=1.0 if slot_budget == 1 else 0.75,
                                target_cost_pressure=0.95,
                            )
                            res = simulate_counterfactual_help(
                                jobs_val, total_steps=total_steps_val, slot_budget=slot_budget,
                                max_wait_steps=max_wait_steps, mean_pred_busy_steps=mean_pred_busy_steps,
                                mean_pred_service_cost=mean_pred_service_cost,
                                mean_pred_expected_consequence=mean_pred_expected_consequence,
                                model=model, adm=adm, dsp=dsp, window_cost_budget=window_cost_budget,
                                cost_budget_window_steps=cost_budget_window_steps, rng_seed=rng_seed,
                            )
                            obj = _objective(res['summary'], max_wait_steps=max_wait_steps, slot_budget=slot_budget, **objective_kwargs)
                            if obj > best_obj:
                                best_obj = obj
                                best_cfg = adm
                                best_res = res
    assert best_cfg is not None and best_res is not None
    return model, best_cfg, fit_payload, best_res


def run_counterfactual_help_experiment(manifest_path: str, output_path: str) -> Dict[str, object]:
    manifest = _load_manifest(manifest_path)
    cfg0 = manifest['frozen_regime']
    args_like = type('Args', (), {
        'clean_bank': manifest['clean_bank'],
        'attack_bank': manifest['attack_bank'],
        'train_bank': manifest['train_bank'],
        'val_bank': manifest['val_bank'],
        'n_bins': 20,
        'decision_step_group': cfg0['decision_step_group'],
        'busy_time_quantile': cfg0['busy_time_quantile'],
        'consequence_blend_verify': 0.7,
        'consequence_mode': 'conditional',
    })()
    env = _build_predictions_and_jobs(args_like)
    jobs_train = env['jobs_train']
    jobs_val = env['jobs_val']
    total_steps_train = env['total_steps_train']
    total_steps_val = env['total_steps_val']
    mean_pred_busy_steps = env['job_stats_train']['mean_pred_busy_steps']
    mean_pred_service_cost = env['job_stats_train']['mean_pred_service_cost']
    mean_pred_expected_consequence = env['job_stats_train']['mean_pred_expected_consequence']

    score_kwargs = {
        'clean_penalty': 0.60,
        'delay_penalty': 0.15,
        'queue_penalty': 0.10,
        'cost_penalty': 0.05,
        'cost_budget_per_step': None,
    }
    tune_score_kwargs = dict(score_kwargs)
    tune_score_kwargs['max_wait_steps'] = int(cfg0['max_wait_steps'])

    # Tune phase3 dispatch reference on train.
    tuned_by_slot: Dict[str, object] = {}
    slot_aggregates: Dict[str, object] = {}

    for slot_budget in cfg0['slot_budget_list']:
        phase3_best, phase3_res = _tune_proposed_ca_policy(
            jobs_train, total_steps_train,
            slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']), rng_seed=20260402,
            cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
            mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
            mean_pred_expected_consequence=mean_pred_expected_consequence,
            v_grid=[1.0, 2.0, 4.0], clean_grid=[0.0, 0.2, 0.5], age_grid=[0.0, 0.1, 0.2],
            urgency_grid=[0.0, 0.1, 0.2], fail_grid=[0.0, 0.05], busy_grid=[0.5, 1.0, 2.0],
            cost_grid=[0.0], admission_threshold_grid=[-0.1, 0.0, 0.1], score_kwargs=tune_score_kwargs,
        )
        dsp = DispatchConfig(**phase3_best)
        model, adm_best, fit_payload, val_res = _tune_counterfactual_help(
            jobs_train, total_steps_train, jobs_val, total_steps_val,
            slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
            cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
            mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost,
            mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp,
            objective_kwargs=score_kwargs, rng_seed=20260402,
        )

        per_holdout_results = []
        recalls = []
        unn = []
        delays = []
        costs = []
        recalls_phase3 = []
        unn_phase3 = []
        recalls_topk = []
        unn_topk = []
        delta_vs_best_recall = []
        delta_vs_best_unn = []
        delta_vs_phase3_recall = []
        delta_vs_phase3_unn = []
        delta_vs_topk_recall = []
        delta_vs_topk_unn = []
        threshold_freq: Dict[str, int] = {}

        for hold in manifest['holdouts']:
            arrays_test = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(hold['test_bank']), int(cfg0['decision_step_group']))
            # Rebuild jobs with the same train-fitted models by reusing helper inputs.
            posterior_verify = fit_attack_posterior_from_banks(args_like.clean_bank, args_like.attack_bank, signal_key='score_phys_l2', n_bins=args_like.n_bins)
            service_models = fit_service_models_from_mixed_bank(args_like.train_bank, signal_key='verify_score', n_bins=args_like.n_bins)
            sev_cond = fit_attack_severity_models_from_arrays(env['arrays_train'], signal_keys=('verify_score','ddd_loss_recons'), n_bins=args_like.n_bins)
            sev_exp = fit_expected_consequence_models_from_arrays(env['arrays_train'], signal_keys=('verify_score','ddd_loss_recons'), n_bins=args_like.n_bins)
            severity_models = sev_cond if args_like.consequence_mode == 'conditional' else sev_exp
            jobs_test, total_steps_test = _predict_jobs(
                arrays_test, posterior_model=posterior_verify, posterior_signal_key='verify_score',
                service_models=service_models, service_signal_key='verify_score', severity_models=severity_models,
                severity_blend_verify=float(args_like.consequence_blend_verify), consequence_mode=str(args_like.consequence_mode),
                fit_verify_score=np.asarray(env['arrays_train']['verify_score'], dtype=float), busy_time_unit=float(env['busy_time_unit']),
            )
            # Threshold family on test, tuned on train same as phase3 threshold reference.
            thr_verify = _tune_fixed_threshold_policy(jobs_train, total_steps_train, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
                                                      signal=np.asarray([j.verify_score for j in jobs_train]), policy_name='threshold_verify_fifo', threshold_candidates=_threshold_candidates(np.asarray([j.verify_score for j in jobs_train]), [0.5,0.6,0.7,0.8,0.9]),
                                                      rng_seed=20260402, cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                                      mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence, score_kwargs=score_kwargs)
            thr_ddd = _tune_fixed_threshold_policy(jobs_train, total_steps_train, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
                                                   signal=np.asarray([j.ddd_loss for j in jobs_train]), policy_name='threshold_ddd_fifo', threshold_candidates=_threshold_candidates(np.asarray([j.ddd_loss for j in jobs_train]), [0.5,0.6,0.7,0.8,0.9]),
                                                   rng_seed=20260402, cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                                   mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence, score_kwargs=score_kwargs)
            thr_ec = _tune_fixed_threshold_policy(jobs_train, total_steps_train, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
                                                   signal=np.asarray([j.pred_expected_consequence for j in jobs_train]), policy_name='threshold_expected_consequence_fifo', threshold_candidates=_threshold_candidates(np.asarray([j.pred_expected_consequence for j in jobs_train]), [0.5,0.6,0.7,0.8,0.9]),
                                                   rng_seed=20260402, cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                                   mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence, score_kwargs=score_kwargs)
            # Evaluate thresholds on test
            def eval_thr(name, tuning):
                cfg = SimulationConfig(policy_name=name, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
                                       threshold=float(tuning[0]['threshold']) if 'threshold' in tuning[0] else float(tuning[0]['base_threshold']),
                                       adaptive_gain=float(tuning[0].get('adaptive_gain', 0.0)), rng_seed=20260402,
                                       cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                       mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence)
                return _run_one_policy(jobs_test, total_steps_test, cfg)
            thr_res_map = {
                'threshold_verify_fifo': eval_thr('threshold_verify_fifo', thr_verify),
                'threshold_ddd_fifo': eval_thr('threshold_ddd_fifo', thr_ddd),
                'threshold_expected_consequence_fifo': eval_thr('threshold_expected_consequence_fifo', thr_ec),
            }
            # choose best threshold on test only for paired reporting (same as prior aggregates)
            best_name, best_res = max(thr_res_map.items(), key=lambda kv: kv[1]['summary']['weighted_attack_recall_no_backend_fail'])
            threshold_freq[best_name] = threshold_freq.get(best_name, 0) + 1

            phase3_test = _run_phase3_policy(jobs_test, total_steps_test, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']), cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                             mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence, dsp=dsp, rng_seed=20260402)
            topk_cfg = SimulationConfig(policy_name='topk_expected_consequence', slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']), rng_seed=20260402,
                                        cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), window_cost_budget=None,
                                        mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence)
            topk_test = _run_one_policy(jobs_test, total_steps_test, topk_cfg)
            cfhelp_test = simulate_counterfactual_help(
                jobs_test, total_steps=total_steps_test, slot_budget=int(slot_budget), max_wait_steps=int(cfg0['max_wait_steps']),
                mean_pred_busy_steps=mean_pred_busy_steps, mean_pred_service_cost=mean_pred_service_cost, mean_pred_expected_consequence=mean_pred_expected_consequence,
                model=model, adm=adm_best, dsp=dsp, window_cost_budget=None, cost_budget_window_steps=int(cfg0['cost_budget_window_steps']), rng_seed=20260402,
            )

            s = cfhelp_test['summary']; b = best_res['summary']; p = phase3_test['summary']; t = topk_test['summary']
            recalls.append(float(s['weighted_attack_recall_no_backend_fail']))
            unn.append(float(s['unnecessary_mtd_count']))
            delays.append(float(s['queue_delay_p95']))
            costs.append(float(s['average_service_cost_per_step']))
            recalls_phase3.append(float(p['weighted_attack_recall_no_backend_fail']))
            unn_phase3.append(float(p['unnecessary_mtd_count']))
            recalls_topk.append(float(t['weighted_attack_recall_no_backend_fail']))
            unn_topk.append(float(t['unnecessary_mtd_count']))
            delta_vs_best_recall.append(float(s['weighted_attack_recall_no_backend_fail'] - b['weighted_attack_recall_no_backend_fail']))
            delta_vs_best_unn.append(float(s['unnecessary_mtd_count'] - b['unnecessary_mtd_count']))
            delta_vs_phase3_recall.append(float(s['weighted_attack_recall_no_backend_fail'] - p['weighted_attack_recall_no_backend_fail']))
            delta_vs_phase3_unn.append(float(s['unnecessary_mtd_count'] - p['unnecessary_mtd_count']))
            delta_vs_topk_recall.append(float(s['weighted_attack_recall_no_backend_fail'] - t['weighted_attack_recall_no_backend_fail']))
            delta_vs_topk_unn.append(float(s['unnecessary_mtd_count'] - t['unnecessary_mtd_count']))

        slot_aggregates[str(slot_budget)] = {
            'policy_stats': {
                'cfhelp_phase3_dispatch': {
                    'weighted_attack_recall_no_backend_fail': {'mean': float(np.mean(recalls)), 'std': float(np.std(recalls)), 'min': float(np.min(recalls)), 'max': float(np.max(recalls))},
                    'unnecessary_mtd_count': {'mean': float(np.mean(unn)), 'std': float(np.std(unn)), 'min': float(np.min(unn)), 'max': float(np.max(unn))},
                    'queue_delay_p95': {'mean': float(np.mean(delays)), 'std': float(np.std(delays)), 'min': float(np.min(delays)), 'max': float(np.max(delays))},
                    'average_service_cost_per_step': {'mean': float(np.mean(costs)), 'std': float(np.std(costs)), 'min': float(np.min(costs)), 'max': float(np.max(costs))},
                },
            },
            'paired_stats': {
                'cfhelp_vs_best_threshold': {
                    'delta_recall': {'mean': float(np.mean(delta_vs_best_recall)), 'std': float(np.std(delta_vs_best_recall)), 'min': float(np.min(delta_vs_best_recall)), 'max': float(np.max(delta_vs_best_recall))},
                    'delta_unnecessary': {'mean': float(np.mean(delta_vs_best_unn)), 'std': float(np.std(delta_vs_best_unn)), 'min': float(np.min(delta_vs_best_unn)), 'max': float(np.max(delta_vs_best_unn))},
                    'cfhelp_wins_on_recall': int(np.sum(np.asarray(delta_vs_best_recall) > 0)),
                    'cfhelp_lower_unnecessary': int(np.sum(np.asarray(delta_vs_best_unn) < 0)),
                },
                'cfhelp_vs_phase3_proposed': {
                    'delta_recall': {'mean': float(np.mean(delta_vs_phase3_recall)), 'std': float(np.std(delta_vs_phase3_recall)), 'min': float(np.min(delta_vs_phase3_recall)), 'max': float(np.max(delta_vs_phase3_recall))},
                    'delta_unnecessary': {'mean': float(np.mean(delta_vs_phase3_unn)), 'std': float(np.std(delta_vs_phase3_unn)), 'min': float(np.min(delta_vs_phase3_unn)), 'max': float(np.max(delta_vs_phase3_unn))},
                    'cfhelp_wins_on_recall': int(np.sum(np.asarray(delta_vs_phase3_recall) > 0)),
                    'cfhelp_lower_unnecessary': int(np.sum(np.asarray(delta_vs_phase3_unn) < 0)),
                },
                'cfhelp_vs_topk_expected': {
                    'delta_recall': {'mean': float(np.mean(delta_vs_topk_recall)), 'std': float(np.std(delta_vs_topk_recall)), 'min': float(np.min(delta_vs_topk_recall)), 'max': float(np.max(delta_vs_topk_recall))},
                    'delta_unnecessary': {'mean': float(np.mean(delta_vs_topk_unn)), 'std': float(np.std(delta_vs_topk_unn)), 'min': float(np.min(delta_vs_topk_unn)), 'max': float(np.max(delta_vs_topk_unn))},
                    'cfhelp_lower_unnecessary': int(np.sum(np.asarray(delta_vs_topk_unn) < 0)),
                },
            },
            'best_threshold_frequency': threshold_freq,
        }
        tuned_by_slot[str(slot_budget)] = {
            'phase3_dispatch_reference': {'config': phase3_best, 'val_summary': phase3_res['summary']},
            'counterfactual_gain_fit': fit_payload,
            'cfhelp_best': {
                'admission': adm_best.__dict__,
                'val_summary': val_res['summary'],
            }
        }

    out = {
        'method': 'counterfactual_help_gain_admission_plus_phase3_dispatch',
        'manifest': manifest,
        'config': {
            'decision_step_group': cfg0['decision_step_group'],
            'busy_time_quantile': cfg0['busy_time_quantile'],
            'use_cost_budget': cfg0['use_cost_budget'],
            'slot_budget_list': cfg0['slot_budget_list'],
            'max_wait_steps': cfg0['max_wait_steps'],
            'consequence_blend_verify': args_like.consequence_blend_verify,
            'consequence_mode': args_like.consequence_mode,
        },
        'environment': {
            'busy_time_unit': env['busy_time_unit'],
            'train_job_stats': env['job_stats_train'],
            'val_job_stats': env['job_stats_val'],
            'train_arrival_diagnostics': env['arrival_diag_train'],
            'val_arrival_diagnostics': env['arrival_diag_val'],
        },
        'tuned_by_slot': tuned_by_slot,
        'n_holdouts': len(manifest['holdouts']),
        'slot_budget_aggregates': slot_aggregates,
    }
    _ensure_parent(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out

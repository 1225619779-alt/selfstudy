from __future__ import annotations

import itertools
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scheduler.calibration import (
    BinnedStatisticModel,
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
    summarize_array,
)
from scheduler.policies_phase3 import AlarmJob, build_jobs_from_arrays, quantize_busy_steps


EPS = 1e-12


@dataclass
class CARKMConfig:
    slot_budget: int = 1
    max_wait_steps: int = 10
    use_cost_budget: bool = False
    window_cost_budget: Optional[float] = None
    cost_budget_window_steps: int = 20
    # admission
    adm_reward_weight: float = 2.0
    adm_clean_penalty: float = 1.0
    adm_fail_penalty: float = 0.05
    adm_busy_penalty: float = 0.5
    adm_cost_penalty: float = 0.0
    adm_threshold: float = 0.0
    # dispatch
    dsp_reward_weight: float = 2.0
    dsp_age_bonus: float = 0.0
    dsp_urgency_bonus: float = 0.1
    dsp_fail_penalty: float = 0.05
    dsp_busy_penalty: float = 0.5
    dsp_cost_penalty: float = 0.0
    dsp_min_total_score: float = 0.0
    busy_cap_scale: float = 1.5
    # normalization and memory
    mean_pred_busy_steps: float = 1.0
    mean_pred_service_cost: float = 1.0
    mean_pred_expected_consequence: float = 1.0
    rng_seed: int = 20260402


@dataclass
class _QueuedItem:
    job: AlarmJob
    enqueue_step: int


@dataclass
class _ActiveServer:
    job_id: int
    busy_until_step: int


# ---------------------------
# data preparation utilities
# ---------------------------

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


def _busy_time_unit_from_fit(arrays_fit: Dict[str, np.ndarray], q: float) -> float:
    x = np.asarray(arrays_fit["service_time"], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    unit = float(np.quantile(x, q))
    return max(unit, 1e-6)


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


def prepare_jobs(
    *,
    clean_bank: str,
    attack_bank: str,
    train_bank: str,
    tune_bank: str,
    eval_bank: str,
    n_bins: int = 20,
    decision_step_group: int = 1,
    busy_time_quantile: float = 0.65,
    consequence_blend_verify: float = 0.7,
    consequence_mode: str = "conditional",
) -> Dict[str, object]:
    posterior_verify = fit_attack_posterior_from_banks(clean_bank, attack_bank, signal_key="score_phys_l2", n_bins=n_bins)
    service_models = fit_service_models_from_mixed_bank(train_bank, signal_key="verify_score", n_bins=n_bins)

    train_raw = mixed_bank_to_alarm_arrays(train_bank)
    tune_raw = mixed_bank_to_alarm_arrays(tune_bank)
    eval_raw = mixed_bank_to_alarm_arrays(eval_bank)

    severity_models = fit_attack_severity_models_from_arrays(
        train_raw,
        signal_keys=("verify_score", "ddd_loss_recons"),
        n_bins=n_bins,
    )

    train_arr = _aggregate_arrival_steps(train_raw, decision_step_group)
    tune_arr = _aggregate_arrival_steps(tune_raw, decision_step_group)
    eval_arr = _aggregate_arrival_steps(eval_raw, decision_step_group)

    busy_time_unit = _busy_time_unit_from_fit(train_arr, busy_time_quantile)
    fit_verify_score = np.asarray(train_arr["verify_score"], dtype=float)

    train_jobs, train_steps = _predict_jobs(
        train_arr,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=consequence_blend_verify,
        consequence_mode=consequence_mode,
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )
    tune_jobs, tune_steps = _predict_jobs(
        tune_arr,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=consequence_blend_verify,
        consequence_mode=consequence_mode,
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )
    eval_jobs, eval_steps = _predict_jobs(
        eval_arr,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=consequence_blend_verify,
        consequence_mode=consequence_mode,
        fit_verify_score=fit_verify_score,
        busy_time_unit=busy_time_unit,
    )

    return {
        "busy_time_unit": busy_time_unit,
        "train_jobs": train_jobs,
        "train_steps": train_steps,
        "tune_jobs": tune_jobs,
        "tune_steps": tune_steps,
        "eval_jobs": eval_jobs,
        "eval_steps": eval_steps,
        "train_arr": train_arr,
        "tune_arr": tune_arr,
        "eval_arr": eval_arr,
        "consequence_blend_verify": consequence_blend_verify,
        "consequence_mode": consequence_mode,
    }


# ---------------------------
# baseline threshold reference
# ---------------------------

def _threshold_candidates(jobs: Sequence[AlarmJob], attr: str, quantiles: Sequence[float]) -> List[float]:
    arr = np.asarray([float(getattr(j, attr)) for j in jobs], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0]
    return sorted({float(np.quantile(arr, q)) for q in quantiles})


def _dynamic_threshold(base_thr: float, gain: float, queue_len: int, active_servers: int, slot_budget: int, cost_pressure: float) -> float:
    pressure = float(queue_len / max(slot_budget, 1)) + float(active_servers / max(slot_budget, 1)) + float(cost_pressure)
    return float(base_thr + gain * pressure)


def simulate_threshold_policy(
    jobs: Sequence[AlarmJob],
    *,
    total_steps: int,
    slot_budget: int,
    max_wait_steps: int,
    policy_name: str,
    threshold: float,
    adaptive_gain: float = 0.0,
    busy_time_unit: float = 1.0,
    window_cost_budget: Optional[float] = None,
    cost_budget_window_steps: int = 0,
    rng_seed: int = 20260402,
) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedItem] = []
    active_servers: List[_ActiveServer] = []
    rolling_cost: Deque[Tuple[int, float]] = deque()
    rng = np.random.default_rng(int(rng_seed))

    total_true_severity = float(np.sum([j.severity_true for j in jobs if j.is_attack == 1]))
    served_true_no_fail = 0.0
    served_true = 0.0
    served_pred_ec = 0.0
    total_pred_ec = float(np.sum([j.pred_expected_consequence for j in jobs]))
    served_clean = 0
    queue_delays: List[int] = []
    attack_delays: List[int] = []
    clean_delays: List[int] = []
    occupied_server_steps = 0.0
    total_service_cost = 0.0
    total_service_time = 0.0
    total_backend_fail = 0
    queue_len_trace: List[int] = []
    active_server_trace: List[int] = []
    threshold_trace: List[float] = []

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]
        if cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))
        cost_pressure = 0.0 if not window_cost_budget else current_cost_window / max(window_cost_budget, EPS)

        arrivals_this_step = arrivals.get(step, [])
        for job in arrivals_this_step:
            thr = float(threshold)
            if policy_name == "adaptive_threshold_verify_fifo":
                thr = _dynamic_threshold(threshold, adaptive_gain, len(queue), len(active_servers), slot_budget, cost_pressure)
            threshold_trace.append(thr)
            if policy_name in {"threshold_verify_fifo", "adaptive_threshold_verify_fifo"}:
                accept = bool(job.verify_score >= thr)
            elif policy_name == "threshold_ddd_fifo":
                accept = bool(job.ddd_loss >= thr)
            elif policy_name == "threshold_expected_consequence_fifo":
                accept = bool(job.pred_expected_consequence >= thr)
            else:
                raise KeyError(policy_name)
            if accept:
                queue.append(_QueuedItem(job=job, enqueue_step=step))

        queue = [item for item in queue if int(step - item.enqueue_step) <= int(max_wait_steps)]
        available_servers = max(int(slot_budget) - len(active_servers), 0)
        # FIFO
        started = 0
        new_queue: List[_QueuedItem] = []
        for item in queue:
            if started >= available_servers:
                new_queue.append(item)
                continue
            job = item.job
            if window_cost_budget is not None and window_cost_budget > 0:
                if current_cost_window + float(job.actual_service_cost) > float(window_cost_budget) + 1e-9:
                    new_queue.append(item)
                    continue
            started += 1
            delay = int(step - item.enqueue_step)
            queue_delays.append(delay)
            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
            total_backend_fail += int(job.actual_backend_fail)
            served_pred_ec += float(job.pred_expected_consequence)
            active_servers.append(_ActiveServer(job_id=job.job_id, busy_until_step=int(step + max(int(job.actual_busy_steps), 1))))
            if window_cost_budget is not None and cost_budget_window_steps > 0:
                rolling_cost.append((int(step), float(job.actual_service_cost)))
                current_cost_window += float(job.actual_service_cost)
            if job.is_attack == 1:
                attack_delays.append(delay)
                served_true += float(job.severity_true)
                if int(job.actual_backend_fail) == 0:
                    served_true_no_fail += float(job.severity_true)
            else:
                clean_delays.append(delay)
                served_clean += 1
        queue = new_queue
        occupied_server_steps += float(len(active_servers))
        queue_len_trace.append(int(len(queue)))
        active_server_trace.append(int(len(active_servers)))

    served_attack_recall_no_fail = float(served_true_no_fail / max(total_true_severity, EPS))
    summary = {
        "weighted_attack_recall_no_backend_fail": served_attack_recall_no_fail,
        "unnecessary_mtd_count": int(served_clean),
        "queue_delay_p95": float(np.quantile(queue_delays, 0.95)) if queue_delays else 0.0,
        "average_service_cost_per_step": float(total_service_cost / max(total_steps, 1)),
        "pred_expected_consequence_served_ratio": float(served_pred_ec / max(total_pred_ec, EPS)),
        "clean_service_ratio": float(served_clean / max(sum(1 for j in jobs if j.is_attack == 0), 1)),
        "attack_delay_p95": float(np.quantile(attack_delays, 0.95)) if attack_delays else 0.0,
        "mean_queue_len": float(np.mean(queue_len_trace)) if queue_len_trace else 0.0,
        "max_queue_len": int(np.max(queue_len_trace)) if queue_len_trace else 0,
        "server_utilization": float(occupied_server_steps / max(int(total_steps) * max(int(slot_budget), 1), 1)),
    }
    return {"summary": summary}


def tune_threshold_reference(
    jobs_tune: Sequence[AlarmJob],
    *,
    total_steps_tune: int,
    slot_budget: int,
    max_wait_steps: int,
    window_cost_budget: Optional[float] = None,
    cost_budget_window_steps: int = 0,
    rng_seed: int = 20260402,
) -> Dict[str, object]:
    # candidates
    cand_verify = _threshold_candidates(jobs_tune, "verify_score", [0.5, 0.6, 0.7, 0.8, 0.9])
    cand_ddd = _threshold_candidates(jobs_tune, "ddd_loss", [0.5, 0.6, 0.7, 0.8, 0.9])
    cand_ec = _threshold_candidates(jobs_tune, "pred_expected_consequence", [0.5, 0.6, 0.7, 0.8, 0.9])

    results: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}
    # static threshold families
    for name, cands in [
        ("threshold_verify_fifo", cand_verify),
        ("threshold_ddd_fifo", cand_ddd),
        ("threshold_expected_consequence_fifo", cand_ec),
    ]:
        best_summary = None
        best_cfg = None
        for thr in cands:
            res = simulate_threshold_policy(
                jobs_tune,
                total_steps=total_steps_tune,
                slot_budget=slot_budget,
                max_wait_steps=max_wait_steps,
                policy_name=name,
                threshold=float(thr),
                window_cost_budget=window_cost_budget,
                cost_budget_window_steps=cost_budget_window_steps,
                rng_seed=rng_seed,
            )
            s = dict(res["summary"])
            # threshold family objective: high recall, then lower unnecessary, then lower cost
            key = (s["weighted_attack_recall_no_backend_fail"], -s["unnecessary_mtd_count"], -s["average_service_cost_per_step"])
            if best_summary is None or key > (
                best_summary["weighted_attack_recall_no_backend_fail"],
                -best_summary["unnecessary_mtd_count"],
                -best_summary["average_service_cost_per_step"],
            ):
                best_summary = s
                best_cfg = {"threshold": float(thr)}
        results[name] = (best_cfg, best_summary)

    # adaptive threshold verify
    best_summary = None
    best_cfg = None
    base_candidates = cand_verify
    for thr in base_candidates:
        for gain in [0.0, 0.05, 0.10, 0.20, 0.40]:
            res = simulate_threshold_policy(
                jobs_tune,
                total_steps=total_steps_tune,
                slot_budget=slot_budget,
                max_wait_steps=max_wait_steps,
                policy_name="adaptive_threshold_verify_fifo",
                threshold=float(thr),
                adaptive_gain=float(gain),
                window_cost_budget=window_cost_budget,
                cost_budget_window_steps=cost_budget_window_steps,
                rng_seed=rng_seed,
            )
            s = dict(res["summary"])
            key = (s["weighted_attack_recall_no_backend_fail"], -s["unnecessary_mtd_count"], -s["average_service_cost_per_step"])
            if best_summary is None or key > (
                best_summary["weighted_attack_recall_no_backend_fail"],
                -best_summary["unnecessary_mtd_count"],
                -best_summary["average_service_cost_per_step"],
            ):
                best_summary = s
                best_cfg = {"threshold": float(thr), "adaptive_gain": float(gain)}
    results["adaptive_threshold_verify_fifo"] = (best_cfg, best_summary)

    family_compact = {k: v[1] for k, v in results.items()}
    # best threshold family member by recall; ties break on lower unnecessary then lower cost
    best_name = max(
        family_compact.keys(),
        key=lambda n: (
            family_compact[n]["weighted_attack_recall_no_backend_fail"],
            -family_compact[n]["unnecessary_mtd_count"],
            -family_compact[n]["average_service_cost_per_step"],
        ),
    )
    min_unnecessary = min(v[1]["unnecessary_mtd_count"] for v in results.values())
    return {
        "family_tuning": {k: v[0] for k, v in results.items()},
        "family_compact": family_compact,
        "best_threshold_name": best_name,
        "best_threshold_summary": family_compact[best_name],
        "min_threshold_unnecessary": int(min_unnecessary),
    }


# ---------------------------
# CA-RK-M simulation
# ---------------------------

def _server_pressure(active_servers: Sequence[_ActiveServer], cfg: CARKMConfig) -> float:
    return float(len(active_servers) / max(int(cfg.slot_budget), 1))


def _queue_pressure(queue_len: int, cfg: CARKMConfig) -> float:
    return float(queue_len / max(int(cfg.slot_budget), 1))


def _cost_pressure(cost_spent_window: float, cfg: CARKMConfig) -> float:
    if not cfg.use_cost_budget or cfg.window_cost_budget is None or cfg.window_cost_budget <= 0:
        return 0.0
    return float(cost_spent_window / max(float(cfg.window_cost_budget), EPS))


def _admission_score(job: AlarmJob, *, queue_len: int, active_servers: Sequence[_ActiveServer], cost_spent_window: float, cfg: CARKMConfig) -> float:
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    server_pressure = _server_pressure(active_servers, cfg)
    queue_pressure = _queue_pressure(queue_len, cfg)
    cost_pressure = _cost_pressure(cost_spent_window, cfg)
    reward = float(cfg.adm_reward_weight) * norm_ec
    clean_pen = float(cfg.adm_clean_penalty) * float(1.0 - job.pred_attack_prob)
    fail_pen = float(cfg.adm_fail_penalty) * float(job.pred_fail_prob)
    busy_pen = float(cfg.adm_busy_penalty) * (1.0 + queue_pressure + server_pressure) * norm_busy
    cost_pen = float(cfg.adm_cost_penalty) * (1.0 + cost_pressure) * norm_cost
    return float(reward - clean_pen - fail_pen - busy_pen - cost_pen)


def _dispatch_item_score(item: _QueuedItem, *, step: int, queue_len: int, active_servers: Sequence[_ActiveServer], cost_spent_window: float, cfg: CARKMConfig) -> float:
    job = item.job
    age = max(0, int(step - item.enqueue_step))
    ttl_left = max(int(cfg.max_wait_steps) - age, 0)
    norm_ec = float(job.pred_expected_consequence) / max(float(cfg.mean_pred_expected_consequence), EPS)
    norm_busy = float(job.pred_busy_steps) / max(float(cfg.mean_pred_busy_steps), EPS)
    norm_cost = float(job.pred_service_cost) / max(float(cfg.mean_pred_service_cost), EPS)
    server_pressure = _server_pressure(active_servers, cfg)
    queue_pressure = _queue_pressure(queue_len, cfg)
    cost_pressure = _cost_pressure(cost_spent_window, cfg)
    return float(
        float(cfg.dsp_reward_weight) * norm_ec
        + float(cfg.dsp_age_bonus) * float(age / max(cfg.max_wait_steps, 1))
        + float(cfg.dsp_urgency_bonus) * float(1.0 / (ttl_left + 1.0))
        - float(cfg.dsp_fail_penalty) * float(job.pred_fail_prob)
        - float(cfg.dsp_busy_penalty) * (server_pressure + queue_pressure) * norm_busy
        - float(cfg.dsp_cost_penalty) * cost_pressure * norm_cost
    )


def _subset_dispatch(
    queue: Sequence[_QueuedItem],
    *,
    step: int,
    active_servers: Sequence[_ActiveServer],
    cost_spent_window: float,
    cfg: CARKMConfig,
) -> List[_QueuedItem]:
    available = max(int(cfg.slot_budget) - len(active_servers), 0)
    if available <= 0 or not queue:
        return []
    queue_len = len(queue)
    # precompute per-item scores and simple filter on positive utility
    scored_items: List[Tuple[float, _QueuedItem]] = []
    for item in queue:
        s = _dispatch_item_score(item, step=step, queue_len=queue_len, active_servers=active_servers, cost_spent_window=cost_spent_window, cfg=cfg)
        scored_items.append((s, item))
    scored_items.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)
    # keep top candidates only for speed; exact within this frontier
    frontier = [item for _, item in scored_items[: min(len(scored_items), 12)]]
    base_cap_busy = float(cfg.busy_cap_scale) * max(float(available) * float(cfg.mean_pred_busy_steps), 1.0)
    cost_remaining = None
    if cfg.use_cost_budget and cfg.window_cost_budget is not None and cfg.window_cost_budget > 0:
        cost_remaining = max(float(cfg.window_cost_budget) - float(cost_spent_window), 0.0)

    best_subset: List[_QueuedItem] = []
    best_total = float(cfg.dsp_min_total_score)
    for r in range(1, min(available, len(frontier)) + 1):
        for combo in itertools.combinations(frontier, r):
            pred_busy = float(sum(item.job.pred_busy_steps for item in combo))
            if pred_busy > base_cap_busy + 1e-9:
                continue
            if cost_remaining is not None:
                pred_cost = float(sum(item.job.pred_service_cost for item in combo))
                if pred_cost > cost_remaining + 1e-9:
                    continue
            total_score = 0.0
            for item in combo:
                total_score += _dispatch_item_score(
                    item,
                    step=step,
                    queue_len=queue_len,
                    active_servers=active_servers,
                    cost_spent_window=cost_spent_window,
                    cfg=cfg,
                )
            if total_score > best_total + 1e-12:
                best_total = float(total_score)
                best_subset = list(combo)
    return best_subset


def simulate_carkm(jobs: Sequence[AlarmJob], *, total_steps: int, cfg: CARKMConfig) -> Dict[str, object]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedItem] = []
    active_servers: List[_ActiveServer] = []
    rolling_cost: Deque[Tuple[int, float]] = deque()
    rolling_pred_clean_risk: Deque[Tuple[int, float]] = deque()  # placeholder memory state for diagnostics

    total_true_severity = float(np.sum([j.severity_true for j in jobs if j.is_attack == 1]))
    served_true_no_fail = 0.0
    served_true = 0.0
    served_pred_ec = 0.0
    total_pred_ec = float(np.sum([j.pred_expected_consequence for j in jobs]))
    served_clean = 0
    served_attack = 0
    queue_delays: List[int] = []
    attack_delays: List[int] = []
    clean_delays: List[int] = []
    queue_len_trace: List[int] = []
    active_server_trace: List[int] = []
    occupied_server_steps = 0.0
    total_service_cost = 0.0
    total_service_time = 0.0
    total_backend_fail = 0
    budget_blocked = 0
    admission_score_trace: List[float] = []
    dispatch_score_trace: List[float] = []
    dropped_by_admission = 0
    dropped_by_ttl = 0

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]
        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
            while rolling_pred_clean_risk and rolling_pred_clean_risk[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_pred_clean_risk.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        # admission on arrivals
        for job in arrivals.get(step, []):
            adm_score = _admission_score(job, queue_len=len(queue), active_servers=active_servers, cost_spent_window=current_cost_window, cfg=cfg)
            admission_score_trace.append(float(adm_score))
            if adm_score >= float(cfg.adm_threshold):
                queue.append(_QueuedItem(job=job, enqueue_step=step))
                rolling_pred_clean_risk.append((int(step), float(1.0 - job.pred_attack_prob)))
            else:
                dropped_by_admission += 1

        # ttl pruning
        new_queue: List[_QueuedItem] = []
        for item in queue:
            if int(step - item.enqueue_step) > int(cfg.max_wait_steps):
                dropped_by_ttl += 1
            else:
                new_queue.append(item)
        queue = new_queue

        # dispatch
        selected = _subset_dispatch(queue, step=step, active_servers=active_servers, cost_spent_window=current_cost_window, cfg=cfg)
        selected_ids = {item.job.job_id for item in selected}
        queue = [item for item in queue if item.job.job_id not in selected_ids]

        for item in selected:
            job = item.job
            delay = int(step - item.enqueue_step)
            queue_delays.append(delay)
            dispatch_score_trace.append(
                _dispatch_item_score(item, step=step, queue_len=len(queue) + len(selected), active_servers=active_servers, cost_spent_window=current_cost_window, cfg=cfg)
            )
            if cfg.use_cost_budget and cfg.window_cost_budget is not None and cfg.cost_budget_window_steps > 0:
                # guard against actual overspend; if violated, treat as blocked and send back to queue if ttl remains
                if current_cost_window + float(job.actual_service_cost) > float(cfg.window_cost_budget) + 1e-9:
                    budget_blocked += 1
                    queue.append(item)
                    continue
                rolling_cost.append((int(step), float(job.actual_service_cost)))
                current_cost_window += float(job.actual_service_cost)

            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
            total_backend_fail += int(job.actual_backend_fail)
            served_pred_ec += float(job.pred_expected_consequence)
            active_servers.append(_ActiveServer(job_id=job.job_id, busy_until_step=int(step + max(int(job.actual_busy_steps), 1))))
            if job.is_attack == 1:
                served_attack += 1
                attack_delays.append(delay)
                served_true += float(job.severity_true)
                if int(job.actual_backend_fail) == 0:
                    served_true_no_fail += float(job.severity_true)
            else:
                served_clean += 1
                clean_delays.append(delay)

        occupied_server_steps += float(len(active_servers))
        queue_len_trace.append(int(len(queue)))
        active_server_trace.append(int(len(active_servers)))

    summary = {
        "weighted_attack_recall_no_backend_fail": float(served_true_no_fail / max(total_true_severity, EPS)),
        "weighted_attack_recall": float(served_true / max(total_true_severity, EPS)),
        "unnecessary_mtd_count": int(served_clean),
        "average_service_cost_per_step": float(total_service_cost / max(total_steps, 1)),
        "average_service_time_per_step": float(total_service_time / max(total_steps, 1)),
        "queue_delay_p95": float(np.quantile(queue_delays, 0.95)) if queue_delays else 0.0,
        "attack_delay_p95": float(np.quantile(attack_delays, 0.95)) if attack_delays else 0.0,
        "clean_delay_p95": float(np.quantile(clean_delays, 0.95)) if clean_delays else 0.0,
        "pred_expected_consequence_served_ratio": float(served_pred_ec / max(total_pred_ec, EPS)),
        "clean_service_ratio": float(served_clean / max(sum(1 for j in jobs if j.is_attack == 0), 1)),
        "mean_queue_len": float(np.mean(queue_len_trace)) if queue_len_trace else 0.0,
        "max_queue_len": int(np.max(queue_len_trace)) if queue_len_trace else 0,
        "server_utilization": float(occupied_server_steps / max(int(total_steps) * max(int(cfg.slot_budget), 1), 1)),
        "budget_blocked_starts": int(budget_blocked),
        "dropped_by_admission": int(dropped_by_admission),
        "dropped_by_ttl": int(dropped_by_ttl),
    }
    return {
        "summary": summary,
        "tuning": {
            "adm_reward_weight": float(cfg.adm_reward_weight),
            "adm_clean_penalty": float(cfg.adm_clean_penalty),
            "adm_fail_penalty": float(cfg.adm_fail_penalty),
            "adm_busy_penalty": float(cfg.adm_busy_penalty),
            "adm_cost_penalty": float(cfg.adm_cost_penalty),
            "adm_threshold": float(cfg.adm_threshold),
            "dsp_reward_weight": float(cfg.dsp_reward_weight),
            "dsp_age_bonus": float(cfg.dsp_age_bonus),
            "dsp_urgency_bonus": float(cfg.dsp_urgency_bonus),
            "dsp_fail_penalty": float(cfg.dsp_fail_penalty),
            "dsp_busy_penalty": float(cfg.dsp_busy_penalty),
            "dsp_cost_penalty": float(cfg.dsp_cost_penalty),
            "busy_cap_scale": float(cfg.busy_cap_scale),
            "dsp_min_total_score": float(cfg.dsp_min_total_score),
        },
        "diagnostics": {
            "admission_score_mean": float(np.mean(admission_score_trace)) if admission_score_trace else np.nan,
            "dispatch_score_mean": float(np.mean(dispatch_score_trace)) if dispatch_score_trace else np.nan,
        },
    }


# ---------------------------
# tuning
# ---------------------------

def _selection_score(summary: Dict[str, float], *, max_wait_steps: int) -> float:
    score = float(summary["weighted_attack_recall_no_backend_fail"])
    score -= 0.05 * float(summary["average_service_cost_per_step"])
    score -= 0.02 * float(summary["queue_delay_p95"] / max(max_wait_steps, 1))
    return float(score)


def _constraint_violation(summary: Dict[str, float], *, clean_cap: int, delay_cap: float | None = None, cost_cap: float | None = None) -> float:
    v = max(float(summary["unnecessary_mtd_count"]) - float(clean_cap), 0.0)
    if delay_cap is not None:
        v += max(float(summary["queue_delay_p95"]) - float(delay_cap), 0.0) * 0.1
    if cost_cap is not None:
        v += max(float(summary["average_service_cost_per_step"]) - float(cost_cap), 0.0) * 0.1
    return float(v)


def tune_carkm(
    jobs_tune: Sequence[AlarmJob],
    *,
    total_steps_tune: int,
    slot_budget: int,
    max_wait_steps: int,
    threshold_reference: Dict[str, object],
    mean_pred_busy_steps: float,
    mean_pred_service_cost: float,
    mean_pred_expected_consequence: float,
    use_cost_budget: bool = False,
    window_cost_budget: Optional[float] = None,
    cost_budget_window_steps: int = 20,
    rng_seed: int = 20260402,
) -> Tuple[CARKMConfig, Dict[str, object]]:
    best_threshold_summary = dict(threshold_reference["best_threshold_summary"])
    min_threshold_unnecessary = int(threshold_reference["min_threshold_unnecessary"])

    # staged tuning: admission then dispatch
    fixed_dispatch = dict(
        dsp_reward_weight=2.0,
        dsp_age_bonus=0.0,
        dsp_urgency_bonus=0.1,
        dsp_fail_penalty=0.05,
        dsp_busy_penalty=0.5,
        dsp_cost_penalty=0.0,
        dsp_min_total_score=0.0,
        busy_cap_scale=1.5,
    )

    best_adm_cfg: Optional[CARKMConfig] = None
    best_adm_res: Optional[Dict[str, object]] = None
    best_adm_key = None
    for clean_slack in [0, 1, 2, 4]:
        clean_cap = int(min_threshold_unnecessary + clean_slack)
        for adm_clean_penalty in [0.5, 1.0, 2.0, 4.0]:
            for adm_busy_penalty in [0.0, 0.5, 1.0]:
                for adm_fail_penalty in [0.0, 0.05]:
                    for adm_threshold in [-0.10, 0.0, 0.10, 0.20, 0.30]:
                        cfg = CARKMConfig(
                            slot_budget=slot_budget,
                            max_wait_steps=max_wait_steps,
                            use_cost_budget=bool(use_cost_budget),
                            window_cost_budget=window_cost_budget,
                            cost_budget_window_steps=cost_budget_window_steps,
                            adm_reward_weight=2.0,
                            adm_clean_penalty=float(adm_clean_penalty),
                            adm_fail_penalty=float(adm_fail_penalty),
                            adm_busy_penalty=float(adm_busy_penalty),
                            adm_cost_penalty=0.0,
                            adm_threshold=float(adm_threshold),
                            mean_pred_busy_steps=mean_pred_busy_steps,
                            mean_pred_service_cost=mean_pred_service_cost,
                            mean_pred_expected_consequence=mean_pred_expected_consequence,
                            rng_seed=rng_seed,
                            **fixed_dispatch,
                        )
                        res = simulate_carkm(jobs_tune, total_steps=total_steps_tune, cfg=cfg)
                        s = dict(res["summary"])
                        violation = _constraint_violation(
                            s,
                            clean_cap=clean_cap,
                            delay_cap=float(best_threshold_summary["queue_delay_p95"]) + 5.0,
                        )
                        score = _selection_score(s, max_wait_steps=max_wait_steps)
                        key = (-violation, score, s["weighted_attack_recall_no_backend_fail"], -s["unnecessary_mtd_count"])
                        if best_adm_key is None or key > best_adm_key:
                            best_adm_key = key
                            best_adm_cfg = cfg
                            best_adm_res = {
                                "clean_cap": clean_cap,
                                "summary": s,
                                "violation": violation,
                            }

    assert best_adm_cfg is not None and best_adm_res is not None

    best_cfg: Optional[CARKMConfig] = None
    best_res: Optional[Dict[str, object]] = None
    best_key = None
    clean_cap = int(best_adm_res["clean_cap"])
    for dsp_reward_weight in [1.0, 2.0, 4.0]:
        for dsp_age_bonus in [0.0, 0.05, 0.10]:
            for dsp_urgency_bonus in [0.0, 0.10, 0.20]:
                for dsp_busy_penalty in [0.0, 0.5, 1.0]:
                    for busy_cap_scale in [1.0, 1.5, 2.0]:
                        cfg = CARKMConfig(
                            slot_budget=slot_budget,
                            max_wait_steps=max_wait_steps,
                            use_cost_budget=bool(use_cost_budget),
                            window_cost_budget=window_cost_budget,
                            cost_budget_window_steps=cost_budget_window_steps,
                            adm_reward_weight=best_adm_cfg.adm_reward_weight,
                            adm_clean_penalty=best_adm_cfg.adm_clean_penalty,
                            adm_fail_penalty=best_adm_cfg.adm_fail_penalty,
                            adm_busy_penalty=best_adm_cfg.adm_busy_penalty,
                            adm_cost_penalty=best_adm_cfg.adm_cost_penalty,
                            adm_threshold=best_adm_cfg.adm_threshold,
                            dsp_reward_weight=float(dsp_reward_weight),
                            dsp_age_bonus=float(dsp_age_bonus),
                            dsp_urgency_bonus=float(dsp_urgency_bonus),
                            dsp_fail_penalty=0.05,
                            dsp_busy_penalty=float(dsp_busy_penalty),
                            dsp_cost_penalty=0.0,
                            busy_cap_scale=float(busy_cap_scale),
                            mean_pred_busy_steps=mean_pred_busy_steps,
                            mean_pred_service_cost=mean_pred_service_cost,
                            mean_pred_expected_consequence=mean_pred_expected_consequence,
                            rng_seed=rng_seed,
                        )
                        res = simulate_carkm(jobs_tune, total_steps=total_steps_tune, cfg=cfg)
                        s = dict(res["summary"])
                        violation = _constraint_violation(
                            s,
                            clean_cap=clean_cap,
                            delay_cap=float(best_threshold_summary["queue_delay_p95"]) + 5.0,
                        )
                        score = _selection_score(s, max_wait_steps=max_wait_steps)
                        key = (-violation, score, s["weighted_attack_recall_no_backend_fail"], -s["unnecessary_mtd_count"])
                        if best_key is None or key > best_key:
                            best_key = key
                            best_cfg = cfg
                            best_res = {
                                "clean_cap": clean_cap,
                                "summary": s,
                                "violation": violation,
                            }

    assert best_cfg is not None and best_res is not None
    tuning_payload = {
        "threshold_reference": threshold_reference,
        "admission_stage_best": {
            "clean_cap": best_adm_res["clean_cap"],
            "violation": best_adm_res["violation"],
            "summary": best_adm_res["summary"],
        },
        "dispatch_stage_best": {
            "clean_cap": best_res["clean_cap"],
            "violation": best_res["violation"],
            "summary": best_res["summary"],
        },
    }
    return best_cfg, tuning_payload


# ---------------------------
# aggregation helpers
# ---------------------------

def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_eval_from_phase3_summary(summary_json_path: str, slot_budget: int) -> Dict[str, Dict[str, float]]:
    payload = _load_json(summary_json_path)
    slot = str(int(slot_budget))
    if "slot_budget_results" not in payload or slot not in payload["slot_budget_results"]:
        raise KeyError(f"slot_budget_results[{slot}] missing in {summary_json_path}")
    return dict(payload["slot_budget_results"][slot]["eval_compact"])


def summarize_policy_stats(values: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = [
        "weighted_attack_recall_no_backend_fail",
        "unnecessary_mtd_count",
        "queue_delay_p95",
        "average_service_cost_per_step",
        "pred_expected_consequence_served_ratio",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        arr = np.asarray([float(v[key]) for v in values], dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


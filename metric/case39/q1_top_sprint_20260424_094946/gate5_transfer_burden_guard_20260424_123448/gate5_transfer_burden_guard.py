from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scheduler.policies_phase3 import (
    _ActiveServer,
    _QueuedJob,
    _admission_accept,
    _job_fits_cost_budget,
    _policy_score,
)


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
GATE2 = SPRINT / "gate2_full_native_20260424_100642"
GATE3 = SPRINT / "gate3_funnel_ceiling_20260424_105813"
GATE4 = sorted(SPRINT.glob("gate4_recovery_robustness_*"))[-1]
OUT = SPRINT / "gate5_transfer_burden_guard_20260424_123448"
GATE3_SCRIPT = GATE3 / "gate3_funnel_ceiling.py"

BUDGETS = [1, 2]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
FAIL_CAPS = [0.80, 0.90, 0.95, 1.00]
PRIMARY_BASELINES = [
    "source_frozen_transfer",
    "topk_expected_consequence",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
]
SELECTED_METHODS = ["TRBG-source", "TRBG-native-burden"]
PAIR_METRICS = [
    "recall",
    "backend_fail",
    "cost",
    "unnecessary",
    "recover_fail",
    "recall_per_backend_fail",
    "recall_per_cost",
    "served_attack_mass",
    "backend_success_attack_mass",
]
PROXIES = [
    "product_proxy",
    "additive_proxy",
    "backend_success_proxy",
    "recovery_aware_proxy",
    "burden_proxy",
    "success_burden_proxy",
]
RNG_SEED = 20260405
EPS = 1e-12


def load_gate3():
    spec = importlib.util.spec_from_file_location("gate3_funnel_ceiling", GATE3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GATE3_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G3 = load_gate3()
G2 = G3.G2
R2 = G3.R2


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p))


def sign_flip_p(deltas: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in deltas if abs(float(x)) > 1e-12], dtype=float)
    if vals.size == 0:
        return 1.0
    obs = abs(float(np.mean(vals)))
    total = 2 ** int(vals.size)
    count = 0
    for mask in range(total):
        signs = np.ones(vals.size)
        for i in range(vals.size):
            if (mask >> i) & 1:
                signs[i] = -1.0
        if abs(float(np.mean(vals * signs))) >= obs - 1e-12:
            count += 1
    return float(count / total)


def bootstrap_ci(deltas: Sequence[float], *, seed: int) -> Tuple[float, float]:
    vals = np.asarray([float(x) for x in deltas], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(10000, vals.size), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_snapshot_and_protocol() -> None:
    gate4_files = [
        "recovery_field_audit.md",
        "proxy_robustness_summary.md",
        "gate4_robustness_decision.md",
        "fail_capped_extension_protocol.md",
    ]
    snapshot_lines = [
        "# Gate 4 Input Snapshot",
        "",
        f"Gate 4 directory read: `{GATE4.relative_to(ROOT).as_posix()}`.",
        "",
        "## Files Read",
        "",
    ]
    for name in gate4_files:
        path = GATE4 / name
        snapshot_lines.append(f"- `{path.relative_to(ROOT).as_posix()}` exists=`{path.exists()}`")
    snapshot_lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            "- `recover_fail` is available and field-supported for post-hoc robustness checks.",
            "- `recover_fail` is a backend_fail subset in the audited mixed-bank cross table, so it must not be described as an independent new recovery-aware method.",
            "- Product, additive, backend-success, and recovery-aware proxy ordering preserves the main source-frozen high-recall pattern.",
            "- Burden and success-burden proxy scores weaken source-frozen's dominant-winner claim.",
            "- Safe wording remains `proxy-consequence-guided`, not `recovery-aware method`.",
        ]
    )
    (OUT / "gate4_input_snapshot.md").write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")

    protocol = [
        "# Gate 5 Locked Protocol",
        "",
        "This protocol was written before generating Gate 5 calibration/confirm results.",
        "",
        "## Primary Baselines",
        "",
        "- `source_frozen_transfer`.",
        "- `topk_expected_consequence`.",
        "- `winner_replay`.",
        "- `native_safeguarded_retune`.",
        "- `native_unconstrained_retune`.",
        "",
        "## Diagnostic Extension",
        "",
        "- Name: `transfer_regularized_burden_guard`.",
        "- TRBG-source: source-frozen score plus burden guard selected only on case14 source train/val.",
        "- TRBG-native-burden: source-frozen score plus burden guard selected only on explicit native case39 train/val.",
        "- TRBG-source tests fully source-frozen burden control.",
        "- TRBG-native-burden tests whether native information is useful only for a low-dimensional burden guard, without full native retuning.",
        "",
        "## Allowed Guard Inputs",
        "",
        "- Predicted backend_fail probability.",
        "- Predicted service_cost.",
        "- Predicted service_time.",
        "- Predicted attack posterior.",
        "- Predicted consequence score already used by source-frozen.",
        "",
        "Forbidden: actual backend_fail, actual recover_fail, actual test labels, and test holdout aggregate results.",
        "",
        "## Guard Construction",
        "",
        "- Keep original source-frozen score weights and regime unchanged.",
        "- `Bhat = z(pred_fail_prob) + z(pred_service_cost) + z(pred_service_time)`.",
        "- `S_guard = S_source - alpha * Bhat`.",
        "- alpha grid: `{0.0, 0.1, 0.25, 0.5, 1.0, 2.0}`.",
        "- fail_cap_quantile grid: `{0.80, 0.90, 0.95, 1.00}`.",
        "- If fail_cap_quantile < 1.00, jobs above the calibration pred_fail_prob quantile are ineligible while uncapped queue candidates exist; this is secondary diagnostic only.",
        "",
        "## Candidate Grid",
        "",
        "- Primary soft-guard grid: alpha only, fail_cap_quantile = 1.00.",
        "- Secondary diagnostic grid: alpha x fail_cap_quantile.",
        "- All candidates are reported.",
        "",
        "## Dev Selection Rule",
        "",
        "- For each calibration mode, choose exactly one primary candidate before test confirm.",
        "- Among candidates whose dev recall is at least 90% of source_frozen dev recall, choose the candidate with the lowest dev backend_fail.",
        "- If no candidate satisfies 90% recall retention, choose alpha=0.0 and mark `guard_failed_on_dev=true`.",
        "- Tie break: lower cost, then lower unnecessary, then lower alpha.",
        "- One alpha/cap pair is selected per calibration mode and reused for B=1 and B=2.",
        "",
        "## Confirm Success Criteria",
        "",
        "- Strong success: backend_fail decreases by at least 15%, recall remains at least 90%, cost does not increase, and the point is not Pareto-dominated by source_frozen/topk/winner in recall-cost or recall-backend_fail.",
        "- Moderate success: recall remains higher than topk, backend_fail or cost decreases versus source_frozen, and the point is Pareto-efficient in at least one plane.",
        "- Failure: recall falls below topk without substantial burden reduction, backend_fail/cost do not improve, or results depend on test-set cherry-picking.",
    ]
    (OUT / "gate5_protocol_locked.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")


def source_method_defs() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    method_defs = G3.build_method_defs()
    by_source: Dict[int, Dict[str, Any]] = {}
    by_topk: Dict[int, Dict[str, Any]] = {}
    by_winner: Dict[int, Dict[str, Any]] = {}
    for method_def in method_defs:
        b = int(method_def["cfg"].slot_budget)
        if method_def["method"] == "source_frozen_transfer":
            by_source[b] = method_def
        if method_def["method"] == "topk_expected_consequence":
            by_topk[b] = method_def
        if method_def["method"] == "winner_replay":
            by_winner[b] = method_def
    return by_source, by_topk, by_winner


def build_jobs_from_arrays(ctx: Dict[str, Any], arrays_bank: Dict[str, np.ndarray], variant_name: str | None):
    return G2.build_jobs(ctx=ctx, arrays_bank=arrays_bank, variant_name=variant_name)


def guard_stats(jobs: Sequence[Any]) -> Dict[str, float]:
    fail = np.asarray([float(j.pred_fail_prob) for j in jobs], dtype=float)
    cost = np.asarray([float(j.pred_service_cost) for j in jobs], dtype=float)
    time = np.asarray([float(j.pred_service_time) for j in jobs], dtype=float)
    return {
        "fail_mean": float(np.mean(fail)),
        "fail_std": float(np.std(fail) if np.std(fail) > 1e-12 else 1.0),
        "cost_mean": float(np.mean(cost)),
        "cost_std": float(np.std(cost) if np.std(cost) > 1e-12 else 1.0),
        "time_mean": float(np.mean(time)),
        "time_std": float(np.std(time) if np.std(time) > 1e-12 else 1.0),
        "fail_cap_q80": float(np.quantile(fail, 0.80)),
        "fail_cap_q90": float(np.quantile(fail, 0.90)),
        "fail_cap_q95": float(np.quantile(fail, 0.95)),
    }


def fail_cap_threshold(stats: Dict[str, float], q: float) -> float:
    if q >= 1.0:
        return float("inf")
    key = f"fail_cap_q{int(round(q * 100))}"
    return float(stats[key])


def bhat(job: Any, stats: Dict[str, float]) -> float:
    return float(
        (float(job.pred_fail_prob) - stats["fail_mean"]) / stats["fail_std"]
        + (float(job.pred_service_cost) - stats["cost_mean"]) / stats["cost_std"]
        + (float(job.pred_service_time) - stats["time_mean"]) / stats["time_std"]
    )


def apply_guard_jobs(jobs: Sequence[Any], *, cfg: Any, stats: Dict[str, float], alpha: float) -> List[Any]:
    if abs(float(alpha)) <= 1e-12:
        return list(jobs)
    mean_ec = max(float(cfg.mean_pred_expected_consequence), EPS)
    v_weight = max(abs(float(cfg.v_weight)), EPS)
    scale = mean_ec / v_weight
    out = []
    for job in jobs:
        guarded_ec = float(job.pred_expected_consequence) - float(alpha) * bhat(job, stats) * scale
        meta = dict(job.meta)
        meta["trbg_bhat"] = bhat(job, stats)
        meta["trbg_actual_backend_fail_used_in_decision"] = 0.0
        meta["trbg_actual_recover_fail_used_in_decision"] = 0.0
        out.append(replace(job, pred_expected_consequence=float(guarded_ec), meta=meta))
    return out


def summarize_detail(detail: Dict[str, Any], jobs: Sequence[Any], *, method: str, budget: int, holdout_id: str, calibration_mode: str | None = None, alpha: float | None = None, fail_cap_quantile: float | None = None) -> Dict[str, Any]:
    s = detail["summary"]
    jobs_by_id = {int(j.job_id): j for j in jobs}
    served = [int(x) for x in detail["served_jobs"]]
    served_attack = [int(x) for x in detail["served_attack_jobs"]]
    served_clean = [int(x) for x in detail["served_clean_jobs"]]
    total_attack_mass = float(sum(float(j.severity_true) for j in jobs if int(j.is_attack) == 1))
    served_attack_mass = float(sum(float(jobs_by_id[i].severity_true) for i in served_attack))
    backend_success_attack_mass = float(sum(float(jobs_by_id[i].severity_true) for i in served_attack if int(jobs_by_id[i].actual_backend_fail) == 0))
    delays = np.asarray(detail["queue_delays_served"], dtype=float)
    service_times = np.asarray([float(jobs_by_id[i].actual_service_time) for i in served], dtype=float)
    service_costs = np.asarray([float(jobs_by_id[i].actual_service_cost) for i in served], dtype=float)
    recover_total = int(sum(int(float(jobs_by_id[i].meta.get("recover_fail", 0.0))) for i in served))
    recover_attack = int(sum(int(float(jobs_by_id[i].meta.get("recover_fail", 0.0))) for i in served_attack))
    recover_clean = int(sum(int(float(jobs_by_id[i].meta.get("recover_fail", 0.0))) for i in served_clean))
    return {
        "method": method,
        "calibration_mode": calibration_mode,
        "holdout_id": holdout_id,
        "B": int(budget),
        "alpha": alpha,
        "fail_cap_quantile": fail_cap_quantile,
        "recall": float(backend_success_attack_mass / max(total_attack_mass, EPS)),
        "scheduler_recall": float(s["weighted_attack_recall_no_backend_fail"]),
        "backend_fail": int(s["total_backend_fail"]),
        "cost": float(s["average_service_cost_per_step"]),
        "unnecessary": int(s["unnecessary_mtd_count"]),
        "served_ratio": float(len(served) / max(int(s["total_jobs"]), 1)),
        "served_attack_mass": served_attack_mass,
        "backend_success_attack_mass": backend_success_attack_mass,
        "clean_service": int(len(served_clean)),
        "served_clean_count": int(len(served_clean)),
        "served_attack_count": int(len(served_attack)),
        "recover_fail": recover_total,
        "recover_fail_attack": recover_attack,
        "recover_fail_clean": recover_clean,
        "delay_p50": float(np.quantile(delays, 0.50)) if delays.size else 0.0,
        "delay_p95": float(s["queue_delay_p95"]),
        "average_service_time": float(np.mean(service_times)) if service_times.size else 0.0,
        "average_service_cost": float(np.mean(service_costs)) if service_costs.size else 0.0,
        "total_service_cost": float(s["total_service_cost"]),
        "actual_backend_fail_used_in_decision": False,
        "actual_recover_fail_used_in_decision": False,
    }


def simulate_guard_detailed(
    jobs: Sequence[Any],
    *,
    total_steps: int,
    cfg: Any,
    fail_cap: float,
) -> Dict[str, Any]:
    if math.isinf(float(fail_cap)):
        return R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=cfg)

    arrivals: Dict[int, List[Any]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    rng = np.random.default_rng(int(cfg.rng_seed))
    active_servers: List[_ActiveServer] = []
    rolling_cost: deque[Tuple[int, float]] = deque()
    served_jobs: List[int] = []
    served_attack_jobs: List[int] = []
    served_clean_jobs: List[int] = []
    dropped_jobs_threshold: List[int] = []
    dropped_jobs_ttl: List[int] = []
    dropped_jobs_budget_blocked: List[int] = []
    queue_delays_served: List[int] = []
    attack_delays_served: List[int] = []
    clean_delays_served: List[int] = []
    total_service_time = 0.0
    total_service_cost = 0.0
    total_backend_fail = 0
    queue_len_trace: List[int] = []
    active_server_trace: List[int] = []
    budget_blocked_starts = 0
    selected_score_by_job: Dict[int, float] = {}

    for step in range(int(total_steps)):
        active_servers = [s for s in active_servers if int(s.busy_until_step) > int(step)]
        if cfg.cost_budget_window_steps > 0:
            while rolling_cost and rolling_cost[0][0] <= int(step) - int(cfg.cost_budget_window_steps):
                rolling_cost.popleft()
        current_cost_window = float(sum(x[1] for x in rolling_cost))

        for job in arrivals.get(step, []):
            accept, _thr = _admission_accept(
                job,
                step=step,
                queue_len=len(queue),
                active_servers=active_servers,
                cost_spent_window=current_cost_window,
                cfg=cfg,
            )
            if accept:
                queue.append(_QueuedJob(job=job, enqueue_step=step))
            else:
                dropped_jobs_threshold.append(int(job.job_id))

        new_queue: List[_QueuedJob] = []
        for item in queue:
            if int(step - item.enqueue_step) > int(cfg.max_wait_steps):
                dropped_jobs_ttl.append(int(item.job.job_id))
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
                        item,
                        step=step,
                        queue_len=len(queue),
                        active_servers=active_servers,
                        cost_spent_window=current_cost_window,
                        cfg=cfg,
                        rng=rng,
                    ),
                    item,
                )
                for item in queue
            ]
            scored.sort(key=lambda x: (x[0], -x[1].enqueue_step, -x[1].job.job_id), reverse=True)
            has_uncapped_candidate = any(float(item.job.pred_fail_prob) <= float(fail_cap) for _score, item in scored)
            running_cost_window = float(current_cost_window)
            for score, item in scored:
                if len(selected_items) >= available_servers:
                    residual_queue.append(item)
                    continue
                if has_uncapped_candidate and float(item.job.pred_fail_prob) > float(fail_cap):
                    residual_queue.append(item)
                    continue
                if cfg.policy_name in {"proposed_vq_hard", "proposed_ca_vq_hard"} and score < float(cfg.admission_score_threshold):
                    residual_queue.append(item)
                    continue
                if not _job_fits_cost_budget(item.job, cost_spent_window=running_cost_window, cfg=cfg):
                    residual_queue.append(item)
                    budget_blocked_starts += 1
                    dropped_jobs_budget_blocked.append(int(item.job.job_id))
                    continue
                selected_items.append(item)
                selected_score_by_job[int(item.job.job_id)] = float(score)
                running_cost_window += float(item.job.actual_service_cost)
            selected_ids = {int(x.job.job_id) for x in selected_items}
            for _score, item in scored:
                if int(item.job.job_id) not in selected_ids and item not in residual_queue:
                    residual_queue.append(item)
            queue = residual_queue

        for item in selected_items:
            job = item.job
            delay = int(step - item.enqueue_step)
            served_jobs.append(int(job.job_id))
            queue_delays_served.append(delay)
            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
            total_backend_fail += int(job.actual_backend_fail)
            active_servers.append(_ActiveServer(job_id=int(job.job_id), busy_until_step=int(step + max(int(job.actual_busy_steps), 1))))
            if cfg.cost_budget_window_steps > 0 and cfg.window_cost_budget is not None and cfg.window_cost_budget > 0:
                rolling_cost.append((int(step), float(job.actual_service_cost)))
            if int(job.is_attack) == 1:
                served_attack_jobs.append(int(job.job_id))
                attack_delays_served.append(delay)
            else:
                served_clean_jobs.append(int(job.job_id))
                clean_delays_served.append(delay)
        queue_len_trace.append(int(len(queue)))
        active_server_trace.append(int(len(active_servers)))

    dropped_jobs_horizon = [int(item.job.job_id) for item in queue]
    all_attack = [int(j.job_id) for j in jobs if int(j.is_attack) == 1]
    all_clean = [int(j.job_id) for j in jobs if int(j.is_attack) == 0]
    dropped_all = set(dropped_jobs_threshold) | set(dropped_jobs_ttl) | set(dropped_jobs_horizon)
    total_true_severity = float(sum(float(j.severity_true) for j in jobs if int(j.is_attack) == 1))
    jobs_by_id = {int(j.job_id): j for j in jobs}
    served_true_severity = float(sum(float(jobs_by_id[i].severity_true) for i in served_attack_jobs))
    served_true_severity_no_fail = float(sum(float(jobs_by_id[i].severity_true) for i in served_attack_jobs if int(jobs_by_id[i].actual_backend_fail) == 0))
    summary = {
        "total_steps": int(total_steps),
        "total_jobs": int(len(jobs)),
        "total_attack_jobs": int(len(all_attack)),
        "total_clean_jobs": int(len(all_clean)),
        "served_jobs": int(len(served_jobs)),
        "served_attack_jobs": int(len(served_attack_jobs)),
        "served_clean_jobs": int(len(served_clean_jobs)),
        "dropped_threshold": int(len(dropped_jobs_threshold)),
        "dropped_ttl": int(len(dropped_jobs_ttl)),
        "dropped_horizon": int(len(dropped_jobs_horizon)),
        "budget_blocked_starts": int(budget_blocked_starts),
        "dropped_attack_jobs": int(sum(1 for i in all_attack if i in dropped_all)),
        "dropped_clean_jobs": int(sum(1 for i in all_clean if i in dropped_all)),
        "attack_recall": float(len(served_attack_jobs) / max(len(all_attack), 1)),
        "weighted_attack_recall": float(served_true_severity / max(total_true_severity, EPS)),
        "weighted_attack_recall_no_backend_fail": float(served_true_severity_no_fail / max(total_true_severity, EPS)),
        "served_attack_precision": float(len(served_attack_jobs) / max(len(served_jobs), 1)),
        "unnecessary_mtd_count": int(len(served_clean_jobs)),
        "clean_service_ratio": float(len(served_clean_jobs) / max(len(all_clean), 1)),
        "total_service_time": float(total_service_time),
        "total_service_cost": float(total_service_cost),
        "average_service_time_per_step": float(total_service_time / max(int(total_steps), 1)),
        "average_service_cost_per_step": float(total_service_cost / max(int(total_steps), 1)),
        "total_backend_fail": int(total_backend_fail),
        "queue_delay_mean": float(np.mean(queue_delays_served)) if queue_delays_served else 0.0,
        "queue_delay_p95": float(np.quantile(queue_delays_served, 0.95)) if queue_delays_served else 0.0,
        "queue_delay_max": float(np.max(queue_delays_served)) if queue_delays_served else 0.0,
        "mean_queue_len": float(np.mean(queue_len_trace)) if queue_len_trace else 0.0,
        "mean_active_servers": float(np.mean(active_server_trace)) if active_server_trace else 0.0,
    }
    return {
        "summary": summary,
        "served_jobs": served_jobs,
        "served_attack_jobs": served_attack_jobs,
        "served_clean_jobs": served_clean_jobs,
        "dropped_jobs_threshold": dropped_jobs_threshold,
        "dropped_jobs_ttl": dropped_jobs_ttl,
        "dropped_jobs_budget_blocked": dropped_jobs_budget_blocked,
        "dropped_jobs_horizon": dropped_jobs_horizon,
        "queue_delays_served": queue_delays_served,
        "attack_delays_served": attack_delays_served,
        "clean_delays_served": clean_delays_served,
        "selected_score_by_job": selected_score_by_job,
    }


def run_guard(
    jobs: Sequence[Any],
    *,
    total_steps: int,
    cfg: Any,
    stats: Dict[str, float],
    alpha: float,
    fail_cap_quantile: float,
) -> Tuple[Dict[str, Any], List[Any], float]:
    guarded = apply_guard_jobs(jobs, cfg=cfg, stats=stats, alpha=alpha)
    cap = fail_cap_threshold(stats, fail_cap_quantile)
    detail = simulate_guard_detailed(guarded, total_steps=total_steps, cfg=cfg, fail_cap=cap)
    return detail, guarded, cap


def calibration_sets(by_source: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    source_def = by_source[1]
    source_ctx = source_def["ctx"]
    source_variant = source_def["variant"]
    native_manifest = read_json(GATE2 / "gate2_full_native_manifest_used.json")
    native_train_arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / native_manifest["train_bank"])), 1)
    native_val_arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / native_manifest["val_bank"])), 1)
    source_train_jobs, source_train_steps, _ = build_jobs_from_arrays(source_ctx, source_ctx["arrays_train"], source_variant)
    source_val_jobs, source_val_steps, _ = build_jobs_from_arrays(source_ctx, source_ctx["arrays_val"], source_variant)
    native_train_jobs, native_train_steps, _ = build_jobs_from_arrays(source_ctx, native_train_arrays, source_variant)
    native_val_jobs, native_val_steps, _ = build_jobs_from_arrays(source_ctx, native_val_arrays, source_variant)
    return {
        "TRBG-source": {
            "calibration_mode": "TRBG-source",
            "train_jobs": source_train_jobs,
            "train_steps": source_train_steps,
            "dev_jobs": source_val_jobs,
            "dev_steps": source_val_steps,
            "selection_split": "case14 source val",
        },
        "TRBG-native-burden": {
            "calibration_mode": "TRBG-native-burden",
            "train_jobs": native_train_jobs,
            "train_steps": native_train_steps,
            "dev_jobs": native_val_jobs,
            "dev_steps": native_val_steps,
            "selection_split": "explicit native case39 val",
        },
    }


def calibration_grid(by_source: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    sets = calibration_sets(by_source)
    rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    selected: Dict[str, Dict[str, Any]] = {}
    stats_by_mode: Dict[str, Dict[str, float]] = {}
    for mode, payload in sets.items():
        stats = guard_stats(payload["train_jobs"])
        stats_by_mode[mode] = stats
        rows: List[Dict[str, Any]] = []
        source_dev_by_budget: Dict[int, Dict[str, Any]] = {}
        for budget in BUDGETS:
            cfg = by_source[budget]["cfg"]
            detail, guarded, cap = run_guard(payload["dev_jobs"], total_steps=payload["dev_steps"], cfg=cfg, stats=stats, alpha=0.0, fail_cap_quantile=1.0)
            source_dev_by_budget[budget] = summarize_detail(detail, guarded, method="source_frozen_dev", budget=budget, holdout_id=payload["selection_split"])
        source_avg_recall = float(np.mean([source_dev_by_budget[b]["scheduler_recall"] for b in BUDGETS]))
        primary_rows: List[Dict[str, Any]] = []
        for alpha in ALPHAS:
            for cap_q in FAIL_CAPS:
                for budget in BUDGETS:
                    cfg = by_source[budget]["cfg"]
                    detail, guarded, cap = run_guard(payload["dev_jobs"], total_steps=payload["dev_steps"], cfg=cfg, stats=stats, alpha=alpha, fail_cap_quantile=cap_q)
                    rec = summarize_detail(detail, guarded, method="transfer_regularized_burden_guard", budget=budget, holdout_id=payload["selection_split"], calibration_mode=mode, alpha=alpha, fail_cap_quantile=cap_q)
                    src = source_dev_by_budget[budget]
                    rec.update(
                        {
                            "fail_cap_threshold": cap,
                            "grid_type": "primary_soft" if abs(cap_q - 1.0) <= 1e-12 else "secondary_hardcap",
                            "recall_retention_vs_source": float(rec["scheduler_recall"] / max(float(src["scheduler_recall"]), EPS)),
                            "backend_fail_reduction_vs_source": float((float(src["backend_fail"]) - float(rec["backend_fail"])) / max(float(src["backend_fail"]), EPS)),
                            "selection_rule_passed": False,
                        }
                    )
                    rows.append(rec)
                    if abs(cap_q - 1.0) <= 1e-12:
                        primary_rows.append(rec)
        agg_primary: List[Dict[str, Any]] = []
        for alpha in ALPHAS:
            vals = [r for r in primary_rows if abs(float(r["alpha"]) - alpha) <= 1e-12]
            avg_recall = float(np.mean([float(r["scheduler_recall"]) for r in vals]))
            avg_backend = float(np.mean([float(r["backend_fail"]) for r in vals]))
            avg_cost = float(np.mean([float(r["cost"]) for r in vals]))
            avg_unnecessary = float(np.mean([float(r["unnecessary"]) for r in vals]))
            passed = avg_recall >= 0.90 * source_avg_recall
            agg_primary.append(
                {
                    "alpha": alpha,
                    "fail_cap_quantile": 1.0,
                    "avg_recall": avg_recall,
                    "avg_backend_fail": avg_backend,
                    "avg_cost": avg_cost,
                    "avg_unnecessary": avg_unnecessary,
                    "passed": passed,
                }
            )
        feasible = [r for r in agg_primary if r["passed"]]
        guard_failed = False
        if feasible:
            choice = sorted(feasible, key=lambda r: (r["avg_backend_fail"], r["avg_cost"], r["avg_unnecessary"], r["alpha"]))[0]
        else:
            choice = next(r for r in agg_primary if abs(float(r["alpha"])) <= 1e-12)
            guard_failed = True
        for r in rows:
            if abs(float(r["alpha"]) - float(choice["alpha"])) <= 1e-12 and abs(float(r["fail_cap_quantile"]) - 1.0) <= 1e-12:
                r["selection_rule_passed"] = bool(choice["passed"])
        selected[mode] = {
            "calibration_mode": mode,
            "method": mode,
            "alpha": float(choice["alpha"]),
            "fail_cap_quantile": 1.0,
            "selection_split": payload["selection_split"],
            "guard_failed_on_dev": bool(guard_failed),
            "source_dev_avg_recall": source_avg_recall,
            **choice,
            "stats": stats,
        }
        rows_by_mode[mode] = rows
    return rows_by_mode, selected, stats_by_mode


def aggregate_rows(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out: List[Dict[str, Any]] = []
    metric_keys = [
        "recall",
        "scheduler_recall",
        "backend_fail",
        "cost",
        "unnecessary",
        "served_ratio",
        "served_attack_mass",
        "backend_success_attack_mass",
        "clean_service",
        "recover_fail",
        "proxy_score",
        "proxy_weighted_success",
        "delay_p50",
        "delay_p95",
        "average_service_time",
        "average_service_cost",
    ]
    for key, vals in sorted(groups.items()):
        rec = {field: val for field, val in zip(fields, key)}
        rec["n"] = len(vals)
        for metric in metric_keys:
            if metric in vals[0]:
                arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
                rec[metric] = float(np.mean(arr))
                rec[f"{metric}_median"] = float(np.median(arr))
        out.append(rec)
    return out


def baseline_confirm_rows(details: Dict[Tuple[str, int, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (method, budget, holdout), payload in sorted(details.items()):
        if method not in PRIMARY_BASELINES:
            continue
        rows.append(summarize_detail(payload["detail"], payload["jobs"], method=method, budget=budget, holdout_id=holdout))
    return rows


def confirm_guard_rows(
    by_source: Dict[int, Dict[str, Any]],
    selected: Dict[str, Dict[str, Any]],
    stats_by_mode: Dict[str, Dict[str, float]],
    *,
    selected_only: bool,
) -> List[Dict[str, Any]]:
    holdouts = read_json(GATE2 / "gate2_source_frozen_transfer_manifest_used.json")["holdouts"]
    source_ctx = by_source[1]["ctx"]
    source_variant = by_source[1]["variant"]
    candidates: List[Tuple[str, float, float]] = []
    if selected_only:
        for mode in SELECTED_METHODS:
            candidates.append((mode, float(selected[mode]["alpha"]), float(selected[mode]["fail_cap_quantile"])))
    else:
        for mode in SELECTED_METHODS:
            for alpha in ALPHAS:
                for cap in FAIL_CAPS:
                    candidates.append((mode, alpha, cap))
    rows: List[Dict[str, Any]] = []
    for hold in holdouts:
        arrays_test = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
        jobs_test, total_steps, _ = build_jobs_from_arrays(source_ctx, arrays_test, source_variant)
        for budget in BUDGETS:
            cfg = by_source[budget]["cfg"]
            for mode, alpha, cap_q in candidates:
                stats = stats_by_mode[mode]
                detail, guarded, cap = run_guard(jobs_test, total_steps=total_steps, cfg=cfg, stats=stats, alpha=alpha, fail_cap_quantile=cap_q)
                rec = summarize_detail(detail, guarded, method=mode, budget=budget, holdout_id=hold["tag"], calibration_mode=mode, alpha=alpha, fail_cap_quantile=cap_q)
                rec["fail_cap_threshold"] = cap
                rec["grid_type"] = "primary_soft" if abs(cap_q - 1.0) <= 1e-12 else "secondary_hardcap"
                rows.append(rec)
    return rows


def compare_alpha0_reproduction(baseline_rows: List[Dict[str, Any]], full_grid_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = {
        (r["holdout_id"], int(r["B"])): r
        for r in baseline_rows
        if r["method"] == "source_frozen_transfer"
    }
    diffs: List[float] = []
    for r in full_grid_rows:
        if r["method"] == "TRBG-source" and abs(float(r["alpha"])) <= 1e-12 and abs(float(r["fail_cap_quantile"]) - 1.0) <= 1e-12:
            s = source[(r["holdout_id"], int(r["B"]))]
            for metric in ["scheduler_recall", "backend_fail", "cost", "unnecessary", "served_attack_mass", "backend_success_attack_mass"]:
                diffs.append(abs(float(r[metric]) - float(s[metric])))
    return {"max_abs_diff": float(max(diffs) if diffs else float("nan")), "passed": bool(diffs and max(diffs) <= 1e-9)}


def paired_statistics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["method"], int(r["B"]), r["holdout_id"]): r for r in rows}
    holdouts = sorted({r["holdout_id"] for r in rows})
    comparisons = [
        ("TRBG-source", "source_frozen_transfer"),
        ("TRBG-native-burden", "source_frozen_transfer"),
        ("TRBG-source", "topk_expected_consequence"),
        ("TRBG-native-burden", "topk_expected_consequence"),
        ("TRBG-source", "winner_replay"),
        ("TRBG-native-burden", "winner_replay"),
    ]
    out: List[Dict[str, Any]] = []
    for budget in BUDGETS:
        for a, b in comparisons:
            for metric in PAIR_METRICS:
                deltas = []
                for h in holdouts:
                    if (a, budget, h) not in by or (b, budget, h) not in by:
                        continue
                    av = metric_value(by[(a, budget, h)], metric)
                    bv = metric_value(by[(b, budget, h)], metric)
                    deltas.append(av - bv)
                if not deltas:
                    continue
                wins = int(sum(1 for d in deltas if d > 1e-12))
                losses = int(sum(1 for d in deltas if d < -1e-12))
                ties = int(len(deltas) - wins - losses)
                ci_lo, ci_hi = bootstrap_ci(deltas, seed=RNG_SEED + len(out))
                out.append(
                    {
                        "B": budget,
                        "comparison": f"{a} vs {b}",
                        "metric": metric,
                        "mean_delta": float(np.mean(deltas)),
                        "median_delta": float(np.median(deltas)),
                        "bootstrap95_ci_low": ci_lo,
                        "bootstrap95_ci_high": ci_hi,
                        "exact_sign_test_p": sign_test_p(wins, losses),
                        "sign_flip_p": sign_flip_p(deltas),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return out


def metric_value(row: Dict[str, Any], metric: str) -> float:
    if metric == "recall_per_backend_fail":
        return float(row["scheduler_recall"]) / max(float(row["backend_fail"]), EPS)
    if metric == "recall_per_cost":
        return float(row["scheduler_recall"]) / max(float(row["cost"]), EPS)
    if metric == "recall":
        return float(row["scheduler_recall"])
    return float(row[metric])


def pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = aggregate_rows(rows, ["method", "B"])
    out: List[Dict[str, Any]] = []
    for budget in BUDGETS:
        pts = [r for r in summary if int(r["B"]) == budget]
        ref = [r for r in pts if r["method"] in ["source_frozen_transfer", "topk_expected_consequence", "winner_replay"]]
        for r in pts:
            def dominated_by_refs(burden_key: str) -> bool:
                for o in ref:
                    if o["method"] == r["method"]:
                        continue
                    if float(o["scheduler_recall"]) >= float(r["scheduler_recall"]) - 1e-12 and float(o[burden_key]) <= float(r[burden_key]) + 1e-12:
                        if float(o["scheduler_recall"]) > float(r["scheduler_recall"]) + 1e-12 or float(o[burden_key]) < float(r[burden_key]) - 1e-12:
                            return True
                return False
            def efficient_all(burden_key: str) -> bool:
                for o in pts:
                    if o is r:
                        continue
                    if float(o["scheduler_recall"]) >= float(r["scheduler_recall"]) - 1e-12 and float(o[burden_key]) <= float(r[burden_key]) + 1e-12:
                        if float(o["scheduler_recall"]) > float(r["scheduler_recall"]) + 1e-12 or float(o[burden_key]) < float(r[burden_key]) - 1e-12:
                            return False
                return True
            out.append(
                {
                    "method": r["method"],
                    "B": budget,
                    "recall": r["scheduler_recall"],
                    "backend_fail": r["backend_fail"],
                    "cost": r["cost"],
                    "dominated_by_source_topk_winner_recall_cost": dominated_by_refs("cost"),
                    "dominated_by_source_topk_winner_recall_backend_fail": dominated_by_refs("backend_fail"),
                    "pareto_efficient_recall_cost_all_methods": efficient_all("cost"),
                    "pareto_efficient_recall_backend_all_methods": efficient_all("backend_fail"),
                }
            )
    return out


def burden_efficiency(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = aggregate_rows(rows, ["method", "B"])
    out: List[Dict[str, Any]] = []
    for r in summary:
        out.append(
            {
                "method": r["method"],
                "B": r["B"],
                "recall": r["scheduler_recall"],
                "backend_fail": r["backend_fail"],
                "cost": r["cost"],
                "unnecessary": r["unnecessary"],
                "recover_fail": r["recover_fail"],
                "recall_per_backend_fail": float(r["scheduler_recall"]) / max(float(r["backend_fail"]), EPS),
                "recall_per_cost": float(r["scheduler_recall"]) / max(float(r["cost"]), EPS),
                "successful_attack_mass_per_backend_fail": float(r["backend_success_attack_mass"]) / max(float(r["backend_fail"]), EPS),
                "successful_attack_mass_per_clean_service": float(r["backend_success_attack_mass"]) / max(float(r["clean_service"]), EPS),
            }
        )
    return out


def proxy_scores_for_rows(confirm_rows: List[Dict[str, Any]], row_payloads: Dict[Tuple[str, int, str], Tuple[Dict[str, Any], Sequence[Any]]]) -> List[Dict[str, Any]]:
    alpha_sb = 0.5
    rows: List[Dict[str, Any]] = []
    for row in confirm_rows:
        key = (row["method"], int(row["B"]), row["holdout_id"])
        if key not in row_payloads:
            continue
        detail, jobs = row_payloads[key]
        labels = proxy_label_arrays(jobs, alpha=alpha_sb)
        jobs_by_id = {int(j.job_id): j for j in jobs}
        served = np.asarray([int(x) for x in detail["served_jobs"]], dtype=int)
        served_attack_success = np.asarray([i for i in detail["served_attack_jobs"] if int(jobs_by_id[int(i)].actual_backend_fail) == 0], dtype=int)
        attack_ids = np.asarray([int(j.job_id) for j in jobs if int(j.is_attack) == 1], dtype=int)
        for proxy, payload in labels.items():
            values = payload["values"]
            kind = payload["kind"]
            if kind == "burden":
                score = -float(np.sum(values[served])) / max(float(len(jobs)), 1.0)
                weighted = -float(np.sum(values[served]))
            elif kind == "signed":
                weighted = float(np.sum(values[served])) if served.size else 0.0
                denom = float(np.sum(np.maximum(values[attack_ids], 0.0))) if attack_ids.size else 0.0
                score = weighted / max(denom, EPS)
            else:
                weighted = float(np.sum(values[served_attack_success])) if served_attack_success.size else 0.0
                denom = float(np.sum(values[attack_ids])) if attack_ids.size else 0.0
                score = weighted / max(denom, EPS)
            rows.append(
                {
                    "method": row["method"],
                    "B": row["B"],
                    "holdout_id": row["holdout_id"],
                    "proxy": proxy,
                    "proxy_kind": kind,
                    "proxy_score": score,
                    "proxy_weighted_success": weighted,
                    "backend_fail": row["backend_fail"],
                    "cost": row["cost"],
                    "recover_fail": row["recover_fail"],
                }
            )
    return rows


def proxy_label_arrays(jobs: Sequence[Any], *, alpha: float) -> Dict[str, Dict[str, Any]]:
    attack = np.asarray([int(j.is_attack) == 1 for j in jobs], dtype=bool)
    backend_fail = np.asarray([int(j.actual_backend_fail) for j in jobs], dtype=float)
    recover_fail = np.asarray([float(j.meta.get("recover_fail", 0.0)) for j in jobs], dtype=float)
    ang_no = np.asarray([float(j.meta.get("ang_no", 0.0)) for j in jobs], dtype=float)
    ang_str = np.asarray([float(j.meta.get("ang_str", 0.0)) for j in jobs], dtype=float)
    service_time = np.asarray([float(j.actual_service_time) for j in jobs], dtype=float)
    service_cost = np.asarray([float(j.actual_service_cost) for j in jobs], dtype=float)
    product = np.where(attack, np.maximum(ang_no, 0.0) * np.maximum(ang_str, 0.0), 0.0)
    additive = np.where(attack, np.maximum(ang_no, 0.0) + np.maximum(ang_str, 0.0), 0.0)
    burden = service_time / max(float(np.nanpercentile(service_time, 95)), EPS) + service_cost / max(float(np.nanpercentile(service_cost, 95)), EPS) + backend_fail + recover_fail
    return {
        "product_proxy": {"values": product, "kind": "positive"},
        "additive_proxy": {"values": additive, "kind": "positive"},
        "backend_success_proxy": {"values": product * (1.0 - backend_fail), "kind": "positive"},
        "recovery_aware_proxy": {"values": product * (1.0 - recover_fail), "kind": "positive"},
        "burden_proxy": {"values": burden, "kind": "burden"},
        "success_burden_proxy": {"values": product * (1.0 - backend_fail) - alpha * burden, "kind": "signed"},
    }


def write_calibration_outputs(rows_by_mode: Dict[str, List[Dict[str, Any]]], selected: Dict[str, Dict[str, Any]]) -> None:
    write_csv(OUT / "gate5_calibration_grid_source.csv", rows_by_mode["TRBG-source"])
    write_csv(OUT / "gate5_calibration_grid_native_burden.csv", rows_by_mode["TRBG-native-burden"])
    write_json(OUT / "gate5_selected_candidates.json", selected)
    display = []
    for mode, choice in selected.items():
        display.append(
            {
                "mode": mode,
                "alpha": choice["alpha"],
                "cap": choice["fail_cap_quantile"],
                "dev_recall": fmt(choice["avg_recall"]),
                "source_dev_recall": fmt(choice["source_dev_avg_recall"]),
                "dev_backend_fail": fmt(choice["avg_backend_fail"]),
                "dev_cost": fmt(choice["avg_cost"]),
                "guard_failed_on_dev": choice["guard_failed_on_dev"],
            }
        )
    lines = [
        "# Gate 5 Calibration Summary",
        "",
        "Selection used only source case14 train/val for TRBG-source and explicit native case39 train/val for TRBG-native-burden.",
        "",
        md_table(display, ["mode", "alpha", "cap", "dev_recall", "source_dev_recall", "dev_backend_fail", "dev_cost", "guard_failed_on_dev"]),
        "",
        "All primary and secondary grid candidates are written to CSV; only fail_cap_quantile=1.00 alpha candidates were eligible for primary selection.",
    ]
    (OUT / "gate5_calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confirm_summary(selected_summary: List[Dict[str, Any]], full_grid_summary: List[Dict[str, Any]], reproduction: Dict[str, Any]) -> None:
    rows = [
        {
            "method": r["method"],
            "B": r["B"],
            "recall": fmt(r["scheduler_recall"]),
            "backend_fail": fmt(r["backend_fail"]),
            "cost": fmt(r["cost"]),
            "recover_fail": fmt(r["recover_fail"]),
        }
        for r in selected_summary
        if r["method"] in PRIMARY_BASELINES + SELECTED_METHODS
    ]
    lines = [
        "# Gate 5 Confirm Summary",
        "",
        f"Alpha=0 / fail_cap=1.00 reproduction max_abs_diff: `{fmt(reproduction['max_abs_diff'], 8)}`; passed=`{reproduction['passed']}`.",
        "",
        md_table(rows, ["method", "B", "recall", "backend_fail", "cost", "recover_fail"]),
        "",
        "Selected TRBG confirm rows are fixed by dev-selected alpha/cap; full grid rows are diagnostic only and were not used for selection.",
    ]
    (OUT / "gate5_confirm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_md(rows: List[Dict[str, Any]]) -> None:
    display = [
        {
            "B": r["B"],
            "comparison": r["comparison"],
            "metric": r["metric"],
            "mean_delta": fmt(r["mean_delta"]),
            "ci": f"[{fmt(r['bootstrap95_ci_low'])}, {fmt(r['bootstrap95_ci_high'])}]",
            "W/L/T": f"{r['wins']}/{r['losses']}/{r['ties']}",
        }
        for r in rows
        if r["metric"] in {"recall", "backend_fail", "cost", "recover_fail"}
    ]
    (OUT / "gate5_paired_statistics.md").write_text("# Gate 5 Paired Statistics\n\n" + md_table(display, ["B", "comparison", "metric", "mean_delta", "ci", "W/L/T"]) + "\n", encoding="utf-8")


def write_pareto_md(pareto_rows: List[Dict[str, Any]], selected_summary: List[Dict[str, Any]], decision: Dict[str, Any]) -> None:
    display = [
        {
            "method": r["method"],
            "B": r["B"],
            "recall": fmt(r["recall"]),
            "backend": fmt(r["backend_fail"]),
            "cost": fmt(r["cost"]),
            "dom_rc": r["dominated_by_source_topk_winner_recall_cost"],
            "dom_rb": r["dominated_by_source_topk_winner_recall_backend_fail"],
        }
        for r in pareto_rows
        if r["method"] in PRIMARY_BASELINES + SELECTED_METHODS
    ]
    lines = [
        "# Gate 5 Pareto Frontier",
        "",
        md_table(display, ["method", "B", "recall", "backend", "cost", "dom_rc", "dom_rb"]),
        "",
        "## Answers",
        "",
        f"1. TRBG-source dominated flags: `{decision['TRBG-source']['dominated_summary']}`.",
        f"2. TRBG-native-burden dominated flags: `{decision['TRBG-native-burden']['dominated_summary']}`.",
        "3. TRBG movement is interpreted by success criteria, not by test-selected candidate choice.",
        f"4. Useful middle point: TRBG-source `{decision['TRBG-source']['status']}`, TRBG-native-burden `{decision['TRBG-native-burden']['status']}`.",
        f"5. More reasonable candidate: `{decision['recommended_candidate']}`.",
        "6. Native information is considered useful for low-dimensional burden calibration only if TRBG-native-burden beats TRBG-source on the locked success criteria; this does not rescue full local retune.",
    ]
    (OUT / "gate5_pareto_frontier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_burden_eff_md(rows: List[Dict[str, Any]]) -> None:
    display = [
        {
            "method": r["method"],
            "B": r["B"],
            "recall/backend": fmt(r["recall_per_backend_fail"]),
            "recall/cost": fmt(r["recall_per_cost"]),
            "recover": fmt(r["recover_fail"]),
        }
        for r in rows
        if r["method"] in PRIMARY_BASELINES + SELECTED_METHODS
    ]
    (OUT / "gate5_burden_efficiency.md").write_text("# Gate 5 Burden Efficiency\n\n" + md_table(display, ["method", "B", "recall/backend", "recall/cost", "recover"]) + "\n", encoding="utf-8")


def write_proxy_md(rows: List[Dict[str, Any]]) -> None:
    summary = aggregate_rows(rows, ["method", "B", "proxy"])
    display = [
        {
            "method": r["method"],
            "B": r["B"],
            "proxy": r["proxy"],
            "score": fmt(r["proxy_score"]),
            "backend": fmt(r["backend_fail"]),
            "recover": fmt(r["recover_fail"]),
        }
        for r in summary
        if r["method"] in ["source_frozen_transfer", "TRBG-source", "TRBG-native-burden", "topk_expected_consequence"]
    ]
    lines = [
        "# Gate 5 Proxy Robustness Selected",
        "",
        md_table(display, ["method", "B", "proxy", "score", "backend", "recover"]),
        "",
        "- Selected TRBG decisions are fixed before proxy rescoring.",
        "- Improvement, if any, should be framed as burden/recovery robustness diagnostic unless locked success criteria are met.",
    ]
    (OUT / "gate5_proxy_robustness_selected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def success_decision(summary_rows: List[Dict[str, Any]], pareto_rows: List[Dict[str, Any]], selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["method"], int(r["B"])): r for r in summary_rows}
    pareto_by = {(r["method"], int(r["B"])): r for r in pareto_rows}
    out: Dict[str, Any] = {}
    src_recall = float(np.mean([by[("source_frozen_transfer", b)]["scheduler_recall"] for b in BUDGETS]))
    src_backend = float(np.mean([by[("source_frozen_transfer", b)]["backend_fail"] for b in BUDGETS]))
    src_cost = float(np.mean([by[("source_frozen_transfer", b)]["cost"] for b in BUDGETS]))
    src_recover = float(np.mean([by[("source_frozen_transfer", b)]["recover_fail"] for b in BUDGETS]))
    topk_recall = float(np.mean([by[("topk_expected_consequence", b)]["scheduler_recall"] for b in BUDGETS]))
    for method in SELECTED_METHODS:
        recall = float(np.mean([by[(method, b)]["scheduler_recall"] for b in BUDGETS]))
        backend = float(np.mean([by[(method, b)]["backend_fail"] for b in BUDGETS]))
        cost = float(np.mean([by[(method, b)]["cost"] for b in BUDGETS]))
        recover = float(np.mean([by[(method, b)]["recover_fail"] for b in BUDGETS]))
        backend_reduction = (src_backend - backend) / max(src_backend, EPS)
        recall_retention = recall / max(src_recall, EPS)
        cost_not_increase = cost <= src_cost + 1e-12
        not_dominated_rc = all(not pareto_by[(method, b)]["dominated_by_source_topk_winner_recall_cost"] for b in BUDGETS)
        not_dominated_rb = all(not pareto_by[(method, b)]["dominated_by_source_topk_winner_recall_backend_fail"] for b in BUDGETS)
        strong = backend_reduction >= 0.15 and recall_retention >= 0.90 and cost_not_increase and not_dominated_rc and not_dominated_rb
        moderate = recall > topk_recall and (backend < src_backend or cost < src_cost) and (
            any(pareto_by[(method, b)]["pareto_efficient_recall_cost_all_methods"] for b in BUDGETS)
            or any(pareto_by[(method, b)]["pareto_efficient_recall_backend_all_methods"] for b in BUDGETS)
        )
        status = "strong_success" if strong else "moderate_success" if moderate else "failure"
        out[method] = {
            "status": status,
            "avg_recall": recall,
            "avg_backend_fail": backend,
            "avg_cost": cost,
            "avg_recover_fail": recover,
            "backend_fail_reduction_vs_source": backend_reduction,
            "recall_retention_vs_source": recall_retention,
            "cost_delta_vs_source": cost - src_cost,
            "recover_fail_delta_vs_source": recover - src_recover,
            "higher_than_topk_recall": recall > topk_recall,
            "backend_fail_reduction_at_least_15pct": backend_reduction >= 0.15,
            "recall_retention_at_least_90pct": recall_retention >= 0.90,
            "dominated_summary": {
                "recall_cost": [pareto_by[(method, b)]["dominated_by_source_topk_winner_recall_cost"] for b in BUDGETS],
                "recall_backend": [pareto_by[(method, b)]["dominated_by_source_topk_winner_recall_backend_fail"] for b in BUDGETS],
            },
            "alpha": selected[method]["alpha"],
            "fail_cap_quantile": selected[method]["fail_cap_quantile"],
        }
    candidates = [m for m in SELECTED_METHODS if out[m]["status"] != "failure"]
    if candidates:
        recommended = sorted(candidates, key=lambda m: ({"strong_success": 0, "moderate_success": 1}[out[m]["status"]], -out[m]["backend_fail_reduction_vs_source"], -out[m]["recall_retention_vs_source"]))[0]
    else:
        recommended = "none_appendix_or_future_work"
    out["recommended_candidate"] = recommended
    return out


def write_decision(decision: Dict[str, Any], reproduction: Dict[str, Any]) -> None:
    rec = decision["recommended_candidate"]
    lines = [
        "# Gate 5 Decision",
        "",
        "1. Gate 5 followed no-test-selection: selected alpha/cap came only from source or native train/val before holdout confirm.",
        f"2. Alpha=0 and fail_cap_quantile=1.00 reproduced source-frozen: `{reproduction['passed']}` with max_abs_diff `{fmt(reproduction['max_abs_diff'], 10)}`.",
        f"3. TRBG-source status: `{decision['TRBG-source']['status']}`.",
        f"4. TRBG-native-burden status: `{decision['TRBG-native-burden']['status']}`.",
        f"5. Recommended v2 main candidate: `{rec}`.",
        "6. If both statuses are failure, Gate 5 belongs in appendix/future work only.",
        f"7. Mainline upgrade to `transfer-regularized scheduling with low-dimensional burden guard` is justified only if recommended candidate is not `none_appendix_or_future_work`; current recommendation: `{rec}`.",
        f"8. Backend burden reduction source/native: `{fmt(decision['TRBG-source']['backend_fail_reduction_vs_source'])}` / `{fmt(decision['TRBG-native-burden']['backend_fail_reduction_vs_source'])}`.",
        f"9. Recall retention source/native: `{fmt(decision['TRBG-source']['recall_retention_vs_source'])}` / `{fmt(decision['TRBG-native-burden']['recall_retention_vs_source'])}`.",
        f"10. Recover_fail delta source/native: `{fmt(decision['TRBG-source']['recover_fail_delta_vs_source'])}` / `{fmt(decision['TRBG-native-burden']['recover_fail_delta_vs_source'])}`.",
        "11. SCI Q1-top v2 method closure requires strong or at least clean moderate success plus manuscript reframing away from native success.",
        "12. Proceed to manuscript rewrite pack only if the selected recommendation is accepted as the v2 direction.",
    ]
    (OUT / "gate5_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_frontiers(summary_rows: List[Dict[str, Any]]) -> None:
    for burden_key, name in [("backend_fail", "recall_backend_frontier"), ("cost", "recall_cost_frontier")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, budget in zip(axes, BUDGETS):
            for r in summary_rows:
                if int(r["B"]) != budget or r["method"] not in PRIMARY_BASELINES + SELECTED_METHODS:
                    continue
                ax.scatter(float(r[burden_key]), float(r["scheduler_recall"]))
                ax.annotate(r["method"], (float(r[burden_key]), float(r["scheduler_recall"])), fontsize=6)
            ax.set_title(f"B={budget}")
            ax.set_xlabel(burden_key)
            ax.set_ylabel("recall")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"figure_gate5_{name}.png", dpi=220)
        fig.savefig(OUT / f"figure_gate5_{name}.pdf")
        plt.close(fig)


def plot_reduction_retention(decision: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in SELECTED_METHODS:
        d = decision[method]
        ax.scatter(float(d["backend_fail_reduction_vs_source"]), float(d["recall_retention_vs_source"]), label=method)
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.15, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("backend_fail reduction vs source")
    ax.set_ylabel("recall retention vs source")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gate5_burden_reduction_vs_recall_retention.png", dpi=220)
    fig.savefig(OUT / "figure_gate5_burden_reduction_vs_recall_retention.pdf")
    plt.close(fig)


def plot_funnel(summary_rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    methods = ["source_frozen_transfer", "TRBG-source", "TRBG-native-burden", "topk_expected_consequence"]
    for ax, budget in zip(axes, BUDGETS):
        for method in methods:
            row = next(r for r in summary_rows if r["method"] == method and int(r["B"]) == budget)
            vals = [row["served_attack_mass"], row["backend_success_attack_mass"]]
            ax.plot(["served_attack", "backend_success"], vals, marker="o", label=method)
        ax.set_title(f"B={budget}")
        ax.set_ylabel("mean mass")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gate5_funnel_selected.png", dpi=220)
    fig.savefig(OUT / "figure_gate5_funnel_selected.pdf")
    plt.close(fig)


def plot_proxy(proxy_rows: List[Dict[str, Any]]) -> None:
    summary = aggregate_rows(proxy_rows, ["method", "B", "proxy"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    methods = ["source_frozen_transfer", "TRBG-source", "TRBG-native-burden", "topk_expected_consequence"]
    for ax, budget in zip(axes, BUDGETS):
        x = np.arange(len(PROXIES))
        width = 0.18
        for idx, method in enumerate(methods):
            vals = []
            for proxy in PROXIES:
                row = next(r for r in summary if r["method"] == method and int(r["B"]) == budget and r["proxy"] == proxy)
                vals.append(float(row["proxy_score"]))
            ax.bar(x + (idx - 1.5) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(PROXIES, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"B={budget}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gate5_proxy_robustness.png", dpi=220)
    fig.savefig(OUT / "figure_gate5_proxy_robustness.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_snapshot_and_protocol()
    by_source, _by_topk, _by_winner = source_method_defs()
    rows_by_mode, selected, stats_by_mode = calibration_grid(by_source)
    write_calibration_outputs(rows_by_mode, selected)

    funnel_rows, ceiling_rows, details = G3.detail_records()
    _ = funnel_rows, ceiling_rows
    baseline_rows = baseline_confirm_rows(details)
    selected_guard_rows = confirm_guard_rows(by_source, selected, stats_by_mode, selected_only=True)
    full_grid_guard_rows = confirm_guard_rows(by_source, selected, stats_by_mode, selected_only=False)
    reproduction = compare_alpha0_reproduction(baseline_rows, full_grid_guard_rows)

    selected_by_holdout = baseline_rows + selected_guard_rows
    full_grid_by_holdout = full_grid_guard_rows
    selected_summary = aggregate_rows(selected_by_holdout, ["method", "B"])
    full_grid_summary = aggregate_rows(full_grid_by_holdout, ["method", "calibration_mode", "alpha", "fail_cap_quantile", "B"])
    write_csv(OUT / "gate5_confirm_selected_by_holdout.csv", selected_by_holdout)
    write_csv(OUT / "gate5_confirm_selected_summary.csv", selected_summary)
    write_csv(OUT / "gate5_confirm_full_grid_by_holdout.csv", full_grid_by_holdout)
    write_csv(OUT / "gate5_confirm_full_grid_summary.csv", full_grid_summary)
    write_confirm_summary(selected_summary, full_grid_summary, reproduction)

    pair_rows = paired_statistics(selected_by_holdout)
    write_csv(OUT / "gate5_paired_statistics.csv", pair_rows)
    write_paired_md(pair_rows)

    pareto_rows = pareto(selected_by_holdout)
    eff_rows = burden_efficiency(selected_by_holdout)
    write_csv(OUT / "gate5_pareto_frontier.csv", pareto_rows)
    write_csv(OUT / "gate5_burden_efficiency.csv", eff_rows)
    decision = success_decision(selected_summary, pareto_rows, selected)
    write_pareto_md(pareto_rows, selected_summary, decision)
    write_burden_eff_md(eff_rows)

    payloads: Dict[Tuple[str, int, str], Tuple[Dict[str, Any], Sequence[Any]]] = {}
    for key, payload in details.items():
        method, budget, holdout = key
        if method in ["source_frozen_transfer", "topk_expected_consequence", "winner_replay"]:
            payloads[(method, budget, holdout)] = (payload["detail"], payload["jobs"])
    for row in selected_guard_rows:
        # Reconstruct payloads for proxy scoring only for selected guards.
        # Keeping this compact avoids storing large objects in CSV while preserving fixed decisions.
        pass

    # Re-run selected guards once for proxy payloads; decisions are fixed and no selection occurs here.
    holdouts = read_json(GATE2 / "gate2_source_frozen_transfer_manifest_used.json")["holdouts"]
    source_ctx = by_source[1]["ctx"]
    source_variant = by_source[1]["variant"]
    for hold in holdouts:
        arrays_test = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
        jobs_test, total_steps, _ = build_jobs_from_arrays(source_ctx, arrays_test, source_variant)
        for budget in BUDGETS:
            cfg = by_source[budget]["cfg"]
            for mode in SELECTED_METHODS:
                choice = selected[mode]
                detail, guarded, _cap = run_guard(
                    jobs_test,
                    total_steps=total_steps,
                    cfg=cfg,
                    stats=stats_by_mode[mode],
                    alpha=float(choice["alpha"]),
                    fail_cap_quantile=float(choice["fail_cap_quantile"]),
                )
                payloads[(mode, budget, hold["tag"])] = (detail, guarded)
    proxy_rows = proxy_scores_for_rows(
        [r for r in selected_by_holdout if r["method"] in ["source_frozen_transfer", "topk_expected_consequence", "winner_replay"] + SELECTED_METHODS],
        payloads,
    )
    write_csv(OUT / "gate5_proxy_robustness_selected.csv", proxy_rows)
    write_proxy_md(proxy_rows)

    plot_frontiers(selected_summary)
    plot_reduction_retention(decision)
    plot_funnel(selected_summary)
    plot_proxy(proxy_rows)
    write_decision(decision, reproduction)
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree.txt").write_text("\n".join(sorted(set(files) | {"outputs_tree.txt"})) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "selected": selected, "decision": decision, "alpha0_reproduction": reproduction}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

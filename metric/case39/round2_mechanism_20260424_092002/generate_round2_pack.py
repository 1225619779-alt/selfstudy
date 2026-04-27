from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from evaluation_budget_scheduler_phase3 import _aggregate_arrival_steps, _busy_time_unit_from_fit, _job_stats
from phase3_oracle_family_core import (
    DEFAULT_VARIANTS,
    _build_jobs_for_variant,
    _fit_net_gain_models,
    _load_json,
    _simulate_with_tuned_phase3,
)
from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import (
    AlarmJob,
    SimulationConfig,
    _ActiveServer,
    _QueuedJob,
    _admission_accept,
    _job_fits_cost_budget,
    _policy_score,
    simulate_policy,
)


ROOT = Path(".").resolve()
OUT = ROOT / "metric/case39/round2_mechanism_20260424_092002"
OUT.mkdir(parents=True, exist_ok=True)


STAGES = [
    {
        "stage": "source_frozen",
        "label": "source-frozen transfer",
        "role": "bridge_transfer_current_main",
        "v1": "metric/case39/phase3_oracle_confirm_v1_native_clean_attack_test/aggregate_summary.json",
        "v2": "metric/case39/phase3_oracle_confirm_v2_native_clean_attack_test/aggregate_summary.json",
    },
    {
        "stage": "winner_replay",
        "label": "source-fixed winner replay",
        "role": "mechanism_isolation",
        "v1": "metric/case39_source_fixed_replay/phase3_oracle_confirm_v1/aggregate_summary.json",
        "v2": "metric/case39_source_fixed_replay/phase3_oracle_confirm_v2/aggregate_summary.json",
    },
    {
        "stage": "anchored_retune",
        "label": "source-anchored retune",
        "role": "repair_attempt",
        "v1": "metric/case39_source_anchor/phase3_oracle_confirm_v1/aggregate_summary.json",
        "v2": "metric/case39_source_anchor/phase3_oracle_confirm_v2/aggregate_summary.json",
    },
    {
        "stage": "native_safeguarded_retune",
        "label": "native safeguarded retune",
        "role": "protocol_internal_negative_control",
        "v1": "metric/case39_localretune_protectedec/phase3_oracle_confirm_v1/aggregate_summary.json",
        "v2": "metric/case39_localretune_protectedec/phase3_oracle_confirm_v2/aggregate_summary.json",
    },
    {
        "stage": "native_unconstrained_retune",
        "label": "native unconstrained retune",
        "role": "stress_test_out_of_protocol",
        "v1": "metric/case39_localretune/phase3_oracle_confirm_v1/aggregate_summary.json",
        "v2": "metric/case39_localretune/phase3_oracle_confirm_v2/aggregate_summary.json",
    },
]

METHODS = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]
METRICS = ["recall", "unnecessary", "cost", "served_ratio", "delay_p95", "backend_fail"]
HIGHER_IS_BETTER = {"recall", "served_ratio"}
RNG_SEED = 20260424


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(values: Sequence[float], prob: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), prob))


def summarize_raw_bank(path: Path) -> Dict[str, Any]:
    raw = np.load(path, allow_pickle=True).item()
    is_attack = np.asarray(raw["is_attack_step"], dtype=int)
    ddd_alarm = np.asarray(raw["ddd_alarm"], dtype=int)
    ang_no = np.maximum(np.asarray(raw["ang_no_summary"], dtype=float), 0.0)
    ang_str = np.maximum(np.asarray(raw["ang_str_summary"], dtype=float), 0.0)
    sev = ang_no * ang_str
    trigger_after_gate = np.asarray(raw.get("trigger_after_gate", np.zeros_like(ddd_alarm)), dtype=int)
    attack_mask = is_attack == 1
    clean_mask = is_attack == 0
    ddd_mask = ddd_alarm == 1
    return {
        "total_steps": int(len(is_attack)),
        "total_clean_steps": int(np.sum(clean_mask)),
        "total_attack_steps": int(np.sum(attack_mask)),
        "total_attack_severity_all_steps_current_proxy": float(np.sum(sev[attack_mask])),
        "ddd_alarms": int(np.sum(ddd_mask)),
        "verified_alarms_trigger_after_gate": int(np.sum(trigger_after_gate == 1)),
        "clean_alarm_jobs": int(np.sum(clean_mask & ddd_mask)),
        "attack_alarm_jobs": int(np.sum(attack_mask & ddd_mask)),
        "ddd_attack_severity_current_proxy": float(np.sum(sev[attack_mask & ddd_mask])),
        "not_alarm_attack_steps": int(np.sum(attack_mask & ~ddd_mask)),
        "not_alarm_attack_severity_current_proxy": float(np.sum(sev[attack_mask & ~ddd_mask])),
    }


def make_cfg(
    *,
    policy_name: str,
    slot_budget: int,
    max_wait_steps: int,
    train_stats: Dict[str, Any],
    tuned_config: Dict[str, Any] | None = None,
    rng_seed: int = 20260402,
) -> SimulationConfig:
    kwargs = dict(
        policy_name=policy_name,
        slot_budget=int(slot_budget),
        max_wait_steps=int(max_wait_steps),
        rng_seed=int(rng_seed),
        cost_budget_window_steps=0,
        window_cost_budget=None,
        mean_pred_busy_steps=float(train_stats.get("mean_pred_busy_steps", 1.0)),
        mean_pred_service_cost=float(train_stats.get("mean_pred_service_cost", 1.0)),
        mean_pred_expected_consequence=float(train_stats.get("mean_pred_expected_consequence", 1.0)),
    )
    if tuned_config:
        kwargs.update(
            v_weight=float(tuned_config["v_weight"]),
            clean_penalty=float(tuned_config["clean_penalty"]),
            age_bonus=float(tuned_config["age_bonus"]),
            urgency_bonus=float(tuned_config["urgency_bonus"]),
            fail_penalty=float(tuned_config["fail_penalty"]),
            busy_penalty=float(tuned_config["busy_penalty"]),
            cost_penalty=float(tuned_config["cost_penalty"]),
            admission_score_threshold=float(tuned_config["admission_score_threshold"]),
        )
    return SimulationConfig(**kwargs)


def simulate_policy_detailed(jobs: Sequence[AlarmJob], *, total_steps: int, cfg: SimulationConfig) -> Dict[str, Any]:
    arrivals: Dict[int, List[AlarmJob]] = {}
    for job in jobs:
        arrivals.setdefault(int(job.arrival_step), []).append(job)

    queue: List[_QueuedJob] = []
    rng = np.random.default_rng(int(cfg.rng_seed))
    active_servers: List[_ActiveServer] = []
    rolling_cost: List[Tuple[int, float]] = []

    served_jobs: List[int] = []
    served_attack_jobs: List[int] = []
    served_clean_jobs: List[int] = []
    dropped_jobs_threshold: List[int] = []
    dropped_jobs_ttl: List[int] = []
    dropped_jobs_budget_blocked: List[int] = []
    queue_delays_served: List[int] = []
    attack_delays_served: List[int] = []
    clean_delays_served: List[int] = []
    selected_score_by_job: Dict[int, float] = {}

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
            rolling_cost = [x for x in rolling_cost if x[0] > int(step) - int(cfg.cost_budget_window_steps)]
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
                dropped_jobs_threshold.append(job.job_id)

        new_queue: List[_QueuedJob] = []
        for item in queue:
            if int(step - item.enqueue_step) > int(cfg.max_wait_steps):
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
                selected_score_by_job[int(item.job.job_id)] = float(score)
                running_cost_window += float(item.job.actual_service_cost)

            selected_ids = {x.job.job_id for x in selected_items}
            for _score, item in scored:
                if item.job.job_id not in selected_ids and item not in residual_queue:
                    residual_queue.append(item)
            queue = residual_queue

        for item in selected_items:
            job = item.job
            delay = int(step - item.enqueue_step)
            served_jobs.append(job.job_id)
            queue_delays_served.append(delay)
            total_service_time += float(job.actual_service_time)
            total_service_cost += float(job.actual_service_cost)
            total_backend_fail += int(job.actual_backend_fail)
            active_servers.append(_ActiveServer(job_id=job.job_id, busy_until_step=int(step + max(int(job.actual_busy_steps), 1))))
            if cfg.cost_budget_window_steps > 0 and cfg.window_cost_budget is not None and cfg.window_cost_budget > 0:
                rolling_cost.append((int(step), float(job.actual_service_cost)))
            if job.is_attack == 1:
                served_attack_jobs.append(job.job_id)
                attack_delays_served.append(delay)
            else:
                served_clean_jobs.append(job.job_id)
                clean_delays_served.append(delay)

        occupied_server_steps += float(len(active_servers))
        queue_len_trace.append(int(len(queue)))
        active_server_trace.append(int(len(active_servers)))

    dropped_jobs_horizon = [item.job.job_id for item in queue]
    official = simulate_policy(jobs, total_steps=total_steps, cfg=cfg)
    official_summary = official["summary"]
    return {
        "official": official,
        "summary": official_summary,
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
        "computed_totals": {
            "total_service_time": float(total_service_time),
            "total_service_cost": float(total_service_cost),
            "total_backend_fail": int(total_backend_fail),
            "budget_blocked_starts": int(budget_blocked_starts),
            "server_utilization_detail": float(occupied_server_steps / max(int(total_steps) * max(int(cfg.slot_budget), 1), 1)),
        },
    }


def variant_by_name(name: str):
    return next(v for v in DEFAULT_VARIANTS if v.name == name)


def prepare_aggregate_context(agg: Dict[str, Any]) -> Dict[str, Any]:
    manifest = agg["confirm_manifest"]
    workdir = Path(manifest["workdir"])
    args = {
        "clean_bank": str(workdir / manifest["clean_bank"]),
        "attack_bank": str(workdir / manifest["attack_bank"]),
        "train_bank": str(workdir / manifest["train_bank"]),
        "n_bins": 20,
        "max_wait_steps": int(manifest["frozen_regime"]["max_wait_steps"]),
        "decision_step_group": int(manifest["frozen_regime"]["decision_step_group"]),
        "busy_time_quantile": float(manifest["frozen_regime"]["busy_time_quantile"]),
        "consequence_blend_verify": 0.70,
        "consequence_mode": "conditional",
        "rng_seed": 20260402,
    }
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args["train_bank"]), args["decision_step_group"])
    posterior_verify = fit_attack_posterior_from_banks(args["clean_bank"], args["attack_bank"], signal_key="score_phys_l2", n_bins=args["n_bins"])
    posterior_ddd = fit_attack_posterior_from_banks(args["clean_bank"], args["attack_bank"], signal_key="ddd_loss_alarm", n_bins=args["n_bins"])
    service_models = fit_service_models_from_mixed_bank(args["train_bank"], signal_key="verify_score", n_bins=args["n_bins"])
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args["n_bins"])
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args["n_bins"])
    severity_models = severity_models_cond if args["consequence_mode"] == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, args["busy_time_quantile"])

    baseline_train_jobs, baseline_total_steps, _baseline_diag = _build_jobs_for_variant(
        arrays_bank=arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=None,
        gain_bundle_by_variant={},
        severity_blend_verify=args["consequence_blend_verify"],
        busy_time_unit=busy_time_unit,
    )
    baseline_train_stats = _job_stats(baseline_train_jobs)

    winner_name = str(agg["winner_dev_selection"]["winner_variant"])
    winner_variant = variant_by_name(winner_name)
    gain_bundle_by_variant: Dict[str, Any] = {}
    if winner_variant.mode == "help_gain":
        gain_bundle_by_variant[winner_variant.name] = _fit_net_gain_models(
            arrays_train,
            clean_scale=float(winner_variant.clean_scale),
            n_bins=args["n_bins"],
        )
    winner_train_jobs, _winner_total_steps, _winner_diag = _build_jobs_for_variant(
        arrays_bank=arrays_train,
        arrays_train=arrays_train,
        posterior_verify=posterior_verify,
        posterior_ddd=posterior_ddd,
        service_models=service_models,
        severity_models=severity_models,
        variant=winner_variant,
        gain_bundle_by_variant=gain_bundle_by_variant,
        severity_blend_verify=args["consequence_blend_verify"],
        busy_time_unit=busy_time_unit,
    )
    winner_train_stats = _job_stats(winner_train_jobs)

    return {
        "manifest": manifest,
        "workdir": workdir,
        "args": args,
        "arrays_train": arrays_train,
        "posterior_verify": posterior_verify,
        "posterior_ddd": posterior_ddd,
        "service_models": service_models,
        "severity_models": severity_models,
        "busy_time_unit": busy_time_unit,
        "baseline_train_stats": baseline_train_stats,
        "winner_variant": winner_variant,
        "winner_train_stats": winner_train_stats,
        "gain_bundle_by_variant": gain_bundle_by_variant,
    }


def build_jobs_for_context(ctx: Dict[str, Any], arrays_test: Dict[str, np.ndarray], *, variant_kind: str):
    variant = None if variant_kind == "baseline" else ctx["winner_variant"]
    return _build_jobs_for_variant(
        arrays_bank=arrays_test,
        arrays_train=ctx["arrays_train"],
        posterior_verify=ctx["posterior_verify"],
        posterior_ddd=ctx["posterior_ddd"],
        service_models=ctx["service_models"],
        severity_models=ctx["severity_models"],
        variant=variant,
        gain_bundle_by_variant=ctx["gain_bundle_by_variant"],
        severity_blend_verify=ctx["args"]["consequence_blend_verify"],
        busy_time_unit=ctx["busy_time_unit"],
    )


def proxy_values(job: AlarmJob, *, cost_scale: float, time_scale: float) -> Dict[str, float]:
    product = max(float(job.meta.get("ang_no", 0.0)), 0.0) * max(float(job.meta.get("ang_str", 0.0)), 0.0)
    additive = max(float(job.meta.get("ang_no", 0.0)), 0.0) + max(float(job.meta.get("ang_str", 0.0)), 0.0)
    backend_ok = 1.0 if int(job.actual_backend_fail) == 0 else 0.0
    recover_ok = 1.0 if int(job.meta.get("recover_fail", 0.0)) == 0 else 0.0
    burden = (
        max(float(job.actual_service_cost), 0.0) / max(cost_scale, 1e-12)
        + max(float(job.actual_service_time), 0.0) / max(time_scale, 1e-12)
        + float(int(job.actual_backend_fail) != 0)
        + float(int(job.meta.get("recover_fail", 0.0)) != 0)
    )
    if int(job.is_attack) != 1:
        return {
            "current_product_proxy": 0.0,
            "additive_proxy": 0.0,
            "backend_success_weighted_proxy": 0.0,
            "recovery_aware_proxy": 0.0,
            "burden_proxy": 0.0,
        }
    return {
        "current_product_proxy": product,
        "additive_proxy": additive,
        "backend_success_weighted_proxy": product * backend_ok,
        "recovery_aware_proxy": product * recover_ok,
        "burden_proxy": burden,
    }


def proxy_recall(jobs: Sequence[AlarmJob], served_attack_ids: Iterable[int]) -> Dict[str, float]:
    served = set(int(x) for x in served_attack_ids)
    attack_jobs = [j for j in jobs if int(j.is_attack) == 1]
    cost_scale = q([float(j.actual_service_cost) for j in attack_jobs], 0.95) or 1.0
    time_scale = q([float(j.actual_service_time) for j in attack_jobs], 0.95) or 1.0
    out: Dict[str, float] = {}
    for name in [
        "current_product_proxy",
        "additive_proxy",
        "backend_success_weighted_proxy",
        "recovery_aware_proxy",
        "burden_proxy",
    ]:
        denom = 0.0
        numer = 0.0
        for job in attack_jobs:
            val = proxy_values(job, cost_scale=cost_scale, time_scale=time_scale)[name]
            denom += val
            if int(job.job_id) in served:
                numer += val
        out[name] = float(numer / denom) if denom > 1e-12 else float("nan")
    return out


def summarize_funnel(
    *,
    stage: Dict[str, str],
    agg_name: str,
    holdout: Dict[str, Any],
    slot: str,
    method: str,
    jobs: Sequence[AlarmJob],
    total_steps: int,
    detailed: Dict[str, Any],
    raw_summary: Dict[str, Any],
) -> Dict[str, Any]:
    official = detailed["summary"]
    served = detailed["served_jobs"]
    served_attacks = detailed["served_attack_jobs"]
    served_clean = detailed["served_clean_jobs"]
    ttl = detailed["dropped_jobs_ttl"]
    horizon = detailed["dropped_jobs_horizon"]
    threshold_drop = detailed["dropped_jobs_threshold"]
    budget_drop = detailed["dropped_jobs_budget_blocked"]
    served_job_objs = [jobs[i] for i in served]
    backend_fail_count = int(sum(int(j.actual_backend_fail) for j in served_job_objs))
    backend_success_count = int(len(served_job_objs) - backend_fail_count)
    delays = [int(x) for x in detailed["queue_delays_served"]]
    total_jobs = int(len(jobs))
    admitted = total_jobs - len(threshold_drop)
    expired = len(ttl) + len(horizon)
    return {
        "stage": stage["stage"],
        "stage_label": stage["label"],
        "stage_role": stage["role"],
        "aggregate": agg_name,
        "holdout_tag": holdout["tag"],
        "family_tag": holdout.get("family_tag"),
        "slot_budget": int(slot),
        "variant": method,
        "total_steps": raw_summary["total_steps"],
        "total_clean_steps": raw_summary["total_clean_steps"],
        "total_attack_steps": raw_summary["total_attack_steps"],
        "total_clean_jobs": raw_summary["clean_alarm_jobs"],
        "total_attack_jobs": raw_summary["attack_alarm_jobs"],
        "total_attack_severity_under_current_proxy": raw_summary["total_attack_severity_all_steps_current_proxy"],
        "ddd_attack_severity_under_current_proxy": raw_summary["ddd_attack_severity_current_proxy"],
        "not_alarm_attack_steps": raw_summary["not_alarm_attack_steps"],
        "not_alarm_attack_severity_under_current_proxy": raw_summary["not_alarm_attack_severity_current_proxy"],
        "ddd_verified_alarms": raw_summary["ddd_alarms"],
        "verified_alarms_trigger_after_gate": raw_summary["verified_alarms_trigger_after_gate"],
        "admitted_jobs": admitted,
        "queued_jobs": admitted,
        "threshold_dropped_jobs": len(threshold_drop),
        "budget_blocked_jobs": len(budget_drop),
        "expired_jobs": expired,
        "ttl_expired_jobs": len(ttl),
        "horizon_unserved_jobs": len(horizon),
        "served_jobs": len(served),
        "served_attack_jobs": len(served_attacks),
        "served_clean_jobs": len(served_clean),
        "unnecessary_interventions": int(official["unnecessary_mtd_count"]),
        "backend_fail_count": backend_fail_count,
        "backend_success_count": backend_success_count,
        "weighted_attack_recall_no_backend_fail": float(official["weighted_attack_recall_no_backend_fail"]),
        "served_ratio": float(official["pred_expected_consequence_served_ratio"]),
        "average_service_time": float(np.mean([j.actual_service_time for j in served_job_objs])) if served_job_objs else 0.0,
        "average_service_cost": float(np.mean([j.actual_service_cost for j in served_job_objs])) if served_job_objs else 0.0,
        "average_service_time_per_step": float(official["average_service_time_per_step"]),
        "average_service_cost_per_step": float(official["average_service_cost_per_step"]),
        "queue_delay_p50": q(delays, 0.50),
        "queue_delay_p95": float(official["queue_delay_p95"]),
        "queue_delay_max": float(official["queue_delay_max"]),
    }


def build_all_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str, str, int, str], Dict[str, Any]]]:
    funnel_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    detail_index: Dict[Tuple[str, str, str, int, str], Dict[str, Any]] = {}

    for stage in STAGES:
        for agg_name in ["v1", "v2"]:
            agg_path = ROOT / stage[agg_name]
            agg = load_json(agg_path)
            ctx = prepare_aggregate_context(agg)
            workdir = ctx["workdir"]
            per_holdout_by_tag = {h["tag"]: h for h in agg["per_holdout_results"]}

            for holdout in ctx["manifest"]["holdouts"]:
                tag = holdout["tag"]
                agg_holdout = per_holdout_by_tag[tag]
                test_bank = workdir / holdout["test_bank"]
                raw_summary = summarize_raw_bank(test_bank)
                arrays_test = _aggregate_arrival_steps(
                    mixed_bank_to_alarm_arrays(str(test_bank)),
                    ctx["args"]["decision_step_group"],
                )
                baseline_jobs, total_steps, _baseline_diag = build_jobs_for_context(ctx, arrays_test, variant_kind="baseline")
                oracle_jobs, total_steps_oracle, _oracle_diag = build_jobs_for_context(ctx, arrays_test, variant_kind="winner")
                assert total_steps == total_steps_oracle

                result_summary = load_json(workdir / holdout["result_summary"])
                for slot in ["1", "2"]:
                    proposed_tuned = result_summary["slot_budget_results"][slot]["tuning"]["proposed_ca_vq_hard"]
                    oracle_tuned = agg_holdout["slot_budget_results"][slot]["winner_tuned_config"]
                    sims = {
                        "phase3_oracle_upgrade": (
                            oracle_jobs,
                            _simulate_with_tuned_phase3(
                                oracle_jobs,
                                total_steps=total_steps,
                                slot_budget=int(slot),
                                tuned_config=oracle_tuned,
                                train_stats=ctx["winner_train_stats"],
                                max_wait_steps=ctx["args"]["max_wait_steps"],
                                rng_seed=ctx["args"]["rng_seed"],
                            ),
                            make_cfg(
                                policy_name="proposed_ca_vq_hard",
                                slot_budget=int(slot),
                                max_wait_steps=ctx["args"]["max_wait_steps"],
                                train_stats=ctx["winner_train_stats"],
                                tuned_config=oracle_tuned,
                                rng_seed=ctx["args"]["rng_seed"],
                            ),
                        ),
                        "phase3_proposed": (
                            baseline_jobs,
                            None,
                            make_cfg(
                                policy_name="proposed_ca_vq_hard",
                                slot_budget=int(slot),
                                max_wait_steps=ctx["args"]["max_wait_steps"],
                                train_stats=ctx["baseline_train_stats"],
                                tuned_config=proposed_tuned,
                                rng_seed=ctx["args"]["rng_seed"],
                            ),
                        ),
                        "topk_expected_consequence": (
                            baseline_jobs,
                            None,
                            make_cfg(
                                policy_name="topk_expected_consequence",
                                slot_budget=int(slot),
                                max_wait_steps=ctx["args"]["max_wait_steps"],
                                train_stats=ctx["baseline_train_stats"],
                                tuned_config=None,
                                rng_seed=ctx["args"]["rng_seed"],
                            ),
                        ),
                    }

                    for method, (jobs, _official_res, cfg) in sims.items():
                        detailed = simulate_policy_detailed(jobs, total_steps=total_steps, cfg=cfg)
                        row = summarize_funnel(
                            stage=stage,
                            agg_name=agg_name,
                            holdout=holdout,
                            slot=slot,
                            method=method,
                            jobs=jobs,
                            total_steps=total_steps,
                            detailed=detailed,
                            raw_summary=raw_summary,
                        )
                        funnel_rows.append(row)
                        recalls = proxy_recall(jobs, detailed["served_attack_jobs"])
                        for proxy_name, proxy_recall_value in recalls.items():
                            proxy_rows.append(
                                {
                                    "stage": stage["stage"],
                                    "stage_label": stage["label"],
                                    "aggregate": agg_name,
                                    "holdout_tag": tag,
                                    "family_tag": holdout.get("family_tag"),
                                    "slot_budget": int(slot),
                                    "variant": method,
                                    "proxy": proxy_name,
                                    "proxy_recall": proxy_recall_value,
                                    "served_attack_jobs": len(detailed["served_attack_jobs"]),
                                    "served_clean_jobs": len(detailed["served_clean_jobs"]),
                                    "backend_fail_count": row["backend_fail_count"],
                                    "average_service_cost_per_step": row["average_service_cost_per_step"],
                                }
                            )
                        detail_index[(stage["stage"], agg_name, tag, int(slot), method)] = {
                            "jobs": jobs,
                            "detailed": detailed,
                            "row": row,
                        }
    return funnel_rows, proxy_rows, detail_index


def mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def metric_value(row: Dict[str, Any], metric: str) -> float:
    mapping = {
        "recall": "weighted_attack_recall_no_backend_fail",
        "unnecessary": "unnecessary_interventions",
        "cost": "average_service_cost_per_step",
        "served_ratio": "served_ratio",
        "delay_p95": "queue_delay_p95",
        "backend_fail": "backend_fail_count",
    }
    return float(row[mapping[metric]])


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def sign_flip_p(deltas: Sequence[float]) -> float:
    arr = np.asarray([x for x in deltas if abs(x) > 1e-12], dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(arr)))
    if n <= 20:
        vals = []
        for mask in range(1 << n):
            signs = np.asarray([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            vals.append(abs(float(np.mean(arr * signs))))
        return float(np.mean(np.asarray(vals) >= observed - 1e-15))
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(20000):
        signs = rng.choice([-1.0, 1.0], size=n)
        vals.append(abs(float(np.mean(arr * signs))))
    return float(np.mean(np.asarray(vals) >= observed - 1e-15))


def bootstrap_ci(deltas: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(deltas, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(RNG_SEED)
    means = []
    for _ in range(20000):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_paired_stats(funnel_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (r["stage"], r["slot_budget"], r["variant"], r["holdout_tag"]): r
        for r in funnel_rows
    }
    comparisons = [
        ("source_frozen_vs_winner_replay", ("source_frozen", "phase3_oracle_upgrade"), ("winner_replay", "phase3_oracle_upgrade")),
        ("source_frozen_vs_anchored_retune", ("source_frozen", "phase3_oracle_upgrade"), ("anchored_retune", "phase3_oracle_upgrade")),
        ("source_frozen_vs_native_safeguarded_retune", ("source_frozen", "phase3_oracle_upgrade"), ("native_safeguarded_retune", "phase3_oracle_upgrade")),
        ("source_frozen_vs_native_unconstrained_retune", ("source_frozen", "phase3_oracle_upgrade"), ("native_unconstrained_retune", "phase3_oracle_upgrade")),
        ("oracle_vs_phase3_proposed", ("source_frozen", "phase3_oracle_upgrade"), ("source_frozen", "phase3_proposed")),
        ("oracle_vs_topk_expected_consequence", ("source_frozen", "phase3_oracle_upgrade"), ("source_frozen", "topk_expected_consequence")),
    ]
    holdouts = sorted({r["holdout_tag"] for r in funnel_rows})
    out = []
    for slot in [1, 2]:
        for comp_name, left, right in comparisons:
            for metric in METRICS:
                deltas = []
                wins = losses = ties = 0
                for tag in holdouts:
                    lrow = by_key.get((left[0], slot, left[1], tag))
                    rrow = by_key.get((right[0], slot, right[1], tag))
                    if not lrow or not rrow:
                        continue
                    delta = metric_value(lrow, metric) - metric_value(rrow, metric)
                    deltas.append(delta)
                    if abs(delta) <= 1e-12:
                        ties += 1
                    else:
                        left_better = delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                        if left_better:
                            wins += 1
                        else:
                            losses += 1
                ci_low, ci_high = bootstrap_ci(deltas)
                out.append(
                    {
                        "comparison": comp_name,
                        "left": f"{left[0]}::{left[1]}",
                        "right": f"{right[0]}::{right[1]}",
                        "slot_budget": slot,
                        "metric": metric,
                        "mean_delta_left_minus_right": float(np.mean(deltas)) if deltas else float("nan"),
                        "median_delta_left_minus_right": float(np.median(deltas)) if deltas else float("nan"),
                        "min_delta": float(np.min(deltas)) if deltas else float("nan"),
                        "max_delta": float(np.max(deltas)) if deltas else float("nan"),
                        "bootstrap95_ci_low": ci_low,
                        "bootstrap95_ci_high": ci_high,
                        "exact_sign_test_p_two_sided": sign_test_p(wins, losses),
                        "sign_flip_permutation_p_two_sided": sign_flip_p(deltas),
                        "n_holdouts": len(deltas),
                        "left_wins": wins,
                        "left_losses": losses,
                        "ties": ties,
                    }
                )
    return out


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v):
        return "NA"
    return f"{v:.{digits}f}"


def write_claim_map() -> None:
    red = '<span style="color:red">RED FLAG</span>'
    rows = [
        (
            "title",
            "Recovery-aware / consequence-aware larger-system success framing, if any current title implies native case39 scale-up.",
            "weak",
            f"{red} Avoid 'native case39' or 'scale-up success'. Use: 'False-alarm-aware MTD orchestration with case39 bridge transfer and stress-test evidence'.",
        ),
        (
            "abstract",
            "Case39 validates the method on a larger native system.",
            "invalid",
            f"{red} Use: 'On case39, we provide a bridge-transfer stress test with native clean/attack/test artifacts but case14-backed train/val calibration; results expose transfer limitations rather than a native success claim.'",
        ),
        (
            "contribution 3",
            "Substantially reduces unnecessary MTD, backend failure burden, defense latency, and cost under the defined operating point.",
            "moderate",
            f"Safe for case14. For case39 add: '{red} case39 evidence is bridge-transfer/stress-test only, not native larger-system proof.'",
        ),
        (
            "Table II role column",
            "If Table II lists case39 as native larger-system evidence or scale-up success.",
            "invalid",
            "Use role labels: 'bridge transfer', 'mechanism isolation', 'source-anchored repair attempt', 'native safeguarded negative control', and 'out-of-protocol stress test'.",
        ),
        (
            "Fig. 4 caption",
            "If Fig. 4 describes case39 as native larger-system evidence.",
            "invalid",
            f"{red} Use: 'Case39 bridge-transfer stress test under frozen case14 dev selection; lower recall/cost profile indicates transfer limitation.'",
        ),
        (
            "Fig. 5 caption",
            "If Fig. 5 says mechanism evidence proves scale-up success.",
            "weak",
            "Use: 'Mechanism decomposition of where recall/cost changes arise across replay, anchored retune, safeguarded retune, and unconstrained local retune.'",
        ),
        (
            "conclusion",
            "The method is validated on a larger native case39 system.",
            "invalid",
            f"{red} Use: 'The larger-system study is currently evidence of bridge transfer and stress behavior; full native case39 train/val remains feasible but not yet canonical.'",
        ),
        (
            "limitation paragraph",
            "Case39 limitations are minor or only computational.",
            "weak",
            f"{red} State explicitly: train/val calibration and winner selection are not fully native, anti-write evidence lacks a current-run STAMP, and severity truth is a proxy.",
        ),
        (
            "consequence-aware wording",
            "Physical consequence / recovery consequence is fully modeled.",
            "weak",
            f"{red} Use: 'learned expected-consequence proxy'. The current truth target is still max(ang_no,0)*max(ang_str,0); recover_fail is available but not yet in the scheduler objective.",
        ),
    ]
    lines = ["# Claim Map", "", "| paper location | original claim | support level | safe replacement wording |", "|---|---|---|---|"]
    for loc, original, level, replacement in rows:
        lines.append(f"| {loc} | {original} | {level} | {replacement} |")
    lines.extend(
        [
            "",
            "## Terms Requiring Replacement",
            "",
            f"- {red} `native case39`: replace with `case39 bridge-transfer stress test` unless train/val and winner selection are made canonical-native.",
            f"- {red} `larger-system evidence`: replace with `larger-system bridge evidence`.",
            f"- {red} `scale-up success`: replace with `scale-up stress behavior / transfer limitation`.",
            f"- {red} `physical consequence` or `recovery consequence`: replace with `expected-consequence proxy` unless a recovery-aware truth label is explicitly used.",
        ]
    )
    (OUT / "claim_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_funnel_summary(funnel_rows: List[Dict[str, Any]]) -> None:
    groups = group_rows(funnel_rows, ["stage", "slot_budget", "variant"])
    lines = ["# Funnel Summary", ""]
    lines.append("This summary is computed by replaying the existing holdout decisions with current artifacts; no new family or retune was run.")
    lines.append("")
    lines.append("| stage | slot | variant | recall | unnecessary | cost/step | served_ratio | expired | backend_fail | delay_p95 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        rows = groups[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key[0]),
                    str(key[1]),
                    str(key[2]),
                    fmt(mean(rows, "weighted_attack_recall_no_backend_fail")),
                    fmt(mean(rows, "unnecessary_interventions")),
                    fmt(mean(rows, "average_service_cost_per_step")),
                    fmt(mean(rows, "served_ratio")),
                    fmt(mean(rows, "expired_jobs")),
                    fmt(mean(rows, "backend_fail_count")),
                    fmt(mean(rows, "queue_delay_p95")),
                ]
            )
            + " |"
        )
    source_rows = groups[("source_frozen", 1, "phase3_oracle_upgrade")] + groups[("source_frozen", 2, "phase3_oracle_upgrade")]
    lines.extend(
        [
            "",
            "## Current Main Stage",
            "",
            f"- Source-frozen oracle average recall across B=1/B=2 rows: `{fmt(mean(source_rows, 'weighted_attack_recall_no_backend_fail'))}`.",
            f"- Source-frozen oracle average expired jobs per holdout-budget: `{fmt(mean(source_rows, 'expired_jobs'))}`.",
            f"- Source-frozen oracle average backend fail count per holdout-budget: `{fmt(mean(source_rows, 'backend_fail_count'))}`.",
        ]
    )
    (OUT / "funnel_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loss_decomposition(funnel_rows: List[Dict[str, Any]]) -> None:
    oracle_rows = [r for r in funnel_rows if r["variant"] == "phase3_oracle_upgrade"]
    groups = group_rows(oracle_rows, ["stage", "slot_budget"])
    lines = ["# Loss Decomposition", ""]
    lines.append("The scheduler denominator is DDD-alarm attack severity. The raw mixed banks also expose attack steps that never became DDD alarm jobs; those are reported separately as not-alarm loss.")
    lines.append("")
    lines.append("| stage | slot | not-alarm severity | expired jobs | horizon unserved | backend_fail | served_attack | recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        rows = groups[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {fmt(mean(rows, 'not_alarm_attack_severity_under_current_proxy'))} | "
            f"{fmt(mean(rows, 'ttl_expired_jobs'))} | {fmt(mean(rows, 'horizon_unserved_jobs'))} | "
            f"{fmt(mean(rows, 'backend_fail_count'))} | {fmt(mean(rows, 'served_attack_jobs'))} | "
            f"{fmt(mean(rows, 'weighted_attack_recall_no_backend_fail'))} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- The largest absolute loss before scheduling is the attack mass that never becomes a DDD-alarm job. Within the scheduler-visible jobs, the dominant B=1 loss is queue pressure: admitted jobs remain queued until TTL/horizon expiration rather than being immediately served.",
            "- Source-frozen keeps materially higher recall than anchored/protected local retunes because it admits and serves more attack jobs under the frozen case14 operating point, accepting higher clean-service and cost burden.",
            "- Winner replay uses native train/val with the source winner and falls between source-frozen and local retunes, which indicates that calibration-bank shift itself explains part of the recall drop.",
            "- Anchored and safeguarded local retunes are conservative: fewer served clean jobs and lower cost, but also sharply lower served attack severity.",
            "- Native unconstrained retune partially recovers B=2 recall but remains unstable at B=1, so B=1 and B=2 bottlenecks are not identical. B=1 is capacity/queue constrained; B=2 is more sensitive to tuned operating point and proxy ranking.",
        ]
    )
    (OUT / "loss_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_statistics(rows: List[Dict[str, Any]]) -> None:
    write_csv(OUT / "paired_statistics.csv", rows)
    lines = ["# Paired Statistics", ""]
    lines.append("Deltas are left minus right. `left_wins` follows metric direction: higher is better for recall/served_ratio; lower is better for unnecessary/cost/delay/backend_fail.")
    lines.append("")
    lines.append("| comparison | slot | metric | mean delta | 95% CI | sign p | sign-flip p | W/L/T |")
    lines.append("|---|---:|---|---:|---|---:|---:|---:|")
    for row in rows:
        if row["metric"] in {"recall", "cost", "backend_fail"}:
            lines.append(
                f"| {row['comparison']} | {row['slot_budget']} | {row['metric']} | "
                f"{fmt(row['mean_delta_left_minus_right'])} | "
                f"[{fmt(row['bootstrap95_ci_low'])}, {fmt(row['bootstrap95_ci_high'])}] | "
                f"{fmt(row['exact_sign_test_p_two_sided'])} | {fmt(row['sign_flip_permutation_p_two_sided'])} | "
                f"{row['left_wins']}/{row['left_losses']}/{row['ties']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Source-frozen is consistently higher-recall than anchored and safeguarded native retunes, but it pays higher intervention/cost burden.",
            "- Source-frozen versus winner replay separates two effects: source-frozen uses case14-backed train/val, while winner replay uses native train/val with the source winner. A positive recall delta here indicates calibration-bank shift is a real mechanism.",
            "- Oracle versus phase3/topk at source-frozen should be treated as an operating-point comparison, not proof of a new native case39 family.",
        ]
    )
    (OUT / "paired_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_proxy_robustness(proxy_rows: List[Dict[str, Any]]) -> None:
    write_csv(OUT / "proxy_robustness.csv", proxy_rows)
    groups = group_rows([r for r in proxy_rows if r["variant"] == "phase3_oracle_upgrade"], ["proxy", "stage", "slot_budget"])
    stage_means: Dict[str, List[Tuple[str, int, float]]] = {}
    for key, rows in groups.items():
        proxy, stage, slot = key
        stage_means.setdefault(proxy, []).append((stage, int(slot), mean(rows, "proxy_recall")))
    lines = ["# Proxy Robustness", ""]
    lines.append("This is post-hoc re-scoring under fixed service decisions. It does not change scheduler decisions.")
    lines.append("")
    lines.append("| proxy | slot | ordering by oracle proxy recall |")
    lines.append("|---|---:|---|")
    for proxy in sorted(stage_means):
        for slot in [1, 2]:
            vals = [(s, v) for s, sl, v in stage_means[proxy] if sl == slot]
            vals.sort(key=lambda x: x[1], reverse=True)
            ordering = " > ".join([f"{s} ({fmt(v)})" for s, v in vals])
            lines.append(f"| {proxy} | {slot} | {ordering} |")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- Stage ordering is broadly stable for source-frozen versus safeguarded/anchored local retunes under product, additive, backend-success, and recovery-aware proxies.",
            "- Source-frozen remains ahead of local protected retune under the tested proxy recalls, but the exact gap is proxy dependent.",
            "- `phase3_oracle_upgrade` is a reasonable operating point when the goal is lower clean-service/cost than phase3/topk at similar source-frozen recall; it is not a fully native case39 success proof.",
            "- Conclusions about `physical consequence` remain weak because the current scheduler target is still proxy-based. The burden proxy is the most direct warning that cost/time/fail penalties can change interpretation.",
            "- `recover_fail` is present in mixed banks and can be used in the next version, but it is not currently in the scheduler objective or primary recall denominator.",
        ]
    )
    (OUT / "proxy_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_feasibility() -> None:
    fit = ROOT / "metric/case39_localretune/mixed_bank_fit_native.npy"
    val = ROOT / "metric/case39_localretune/mixed_bank_eval_native.npy"
    canon_fit = ROOT / "metric/case39/mixed_bank_fit.npy"
    canon_val = ROOT / "metric/case39/mixed_bank_eval.npy"
    case14_fit = ROOT / "metric/case14/mixed_bank_fit.npy"
    case14_val = ROOT / "metric/case14/mixed_bank_eval.npy"
    local_agg = load_json(ROOT / "metric/case39_localretune/phase3_oracle_confirm_v1/aggregate_summary.json")
    uses_native = local_agg["confirm_manifest"]["train_bank"] == "metric/case39_localretune/mixed_bank_fit_native.npy"
    lines = [
        "# Full Native Case39 Feasibility",
        "",
        "| item | value |",
        "|---|---|",
        f"| `mixed_bank_fit_native.npy` exists | {fit.exists()} |",
        f"| `mixed_bank_eval_native.npy` exists | {val.exists()} |",
        f"| fit native is symlink | {fit.is_symlink()} |",
        f"| eval native is symlink | {val.is_symlink()} |",
        f"| canonical fit resolves to | `{canon_fit.resolve()}` |",
        f"| canonical eval resolves to | `{canon_val.resolve()}` |",
        f"| canonical fit equals case14 sha | {sha256_file(canon_fit.resolve()) == sha256_file(case14_fit)} |",
        f"| canonical eval equals case14 sha | {sha256_file(canon_val.resolve()) == sha256_file(case14_val)} |",
        f"| existing localretune uses native banks | {uses_native} |",
        "",
        "## Execution Cost",
        "",
        "- Full native train/val is already executable: the native fit/eval banks exist and current localretune manifests use them.",
        "- Replacing canonical `metric/case39/mixed_bank_fit.npy` and `mixed_bank_eval.npy` requires changing the canonical links/files to the native fit/eval banks, then rerunning confirm with an explicit STAMP.",
        "- If raw measurement regeneration is required, round1 `case39_reality_check.json` estimated about 4.72 CPU hours for full case39 measurement generation in the audit environment. If using existing native banks, confirm-only reruns should be much cheaper and dominated by scheduler simulation.",
        "",
        "## Shortest Safe Route",
        "",
        "1. Create a STAMP before touching canonical files.",
        "2. Hash canonical case14 and case39 clean/attack/train/val banks.",
        "3. Point canonical case39 fit/eval to native `metric/case39_localretune/*_native.npy` or generate a new manifest that references them explicitly.",
        "4. Rerun only the existing confirm pipeline with fixed `oracle_protected_ec` and the frozen regime.",
        "5. Run postrun audit and anti-write checks against both current repo `metric/case14` and old repo `metric/case14`.",
    ]
    (OUT / "native_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance_plan() -> None:
    lines = [
        "# Anti-write / Provenance Plan",
        "",
        "## Why Current Anti-write Evidence Is Insufficient",
        "",
        "- Round1 found no usable current-run STAMP under `/tmp/case39_*`.",
        "- Without a STAMP created before rerun, file mtime checks cannot prove that case14 assets were not modified during the case39 run.",
        "- Existing hash manifests are useful, but they do not replace a before/after write barrier for a new rerun.",
        "",
        "## Next Rerun STAMP Protocol",
        "",
        "```bash",
        "STAMP=/tmp/case39_round3_$(date +%Y%m%d_%H%M%S).stamp",
        "touch \"$STAMP\"",
        "# run the selected case39 pipeline",
        "find metric/case14 -type f -newer \"$STAMP\" -print > anti_write_q1_case14.txt",
        "find /home/pang/projects/DDET-MTD/metric/case14 -type f -newer \"$STAMP\" -print > anti_write_oldrepo_case14.txt",
        "```",
        "",
        "## Files To Hash Before And After",
        "",
        "- `metric/case14/metric_clean_alarm_scores_full.npy`",
        "- `metric/case14/metric_attack_alarm_scores_400.npy`",
        "- `metric/case14/mixed_bank_fit.npy`",
        "- `metric/case14/mixed_bank_eval.npy`",
        "- `metric/case39/metric_clean_alarm_scores_full.npy`",
        "- `metric/case39/metric_attack_alarm_scores_400.npy`",
        "- `metric/case39/mixed_bank_fit.npy`",
        "- `metric/case39/mixed_bank_eval.npy`",
        "- `metric/case39_localretune/mixed_bank_fit_native.npy`",
        "- `metric/case39_localretune/mixed_bank_eval_native.npy`",
        "- all manifest JSON files used by confirm",
        "",
        "## Proof Standard",
        "",
        "- `anti_write_q1_case14.txt` and `anti_write_oldrepo_case14.txt` must be empty.",
        "- Before/after SHA256 for case14 files must match.",
        "- Case39 output manifests must reference case39/native paths for the intended run.",
        "- The final audit bundle must include STAMP path, pre-hash JSON, post-hash JSON, anti-write txt files, and an outputs tree.",
    ]
    (OUT / "provenance_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs_tree() -> None:
    files = sorted({p.name for p in OUT.iterdir() if p.is_file()} | {"outputs_tree.txt"})
    (OUT / "outputs_tree.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    write_claim_map()
    funnel_rows, proxy_rows, _detail_index = build_all_rows()
    write_csv(OUT / "funnel_by_holdout_budget_variant.csv", funnel_rows)
    write_funnel_summary(funnel_rows)
    write_loss_decomposition(funnel_rows)
    paired_rows = build_paired_stats(funnel_rows)
    write_paired_statistics(paired_rows)
    write_proxy_robustness(proxy_rows)
    write_native_feasibility()
    write_provenance_plan()
    write_outputs_tree()
    print(json.dumps({"output_dir": str(OUT), "funnel_rows": len(funnel_rows), "paired_rows": len(paired_rows), "proxy_rows": len(proxy_rows)}, indent=2))


if __name__ == "__main__":
    main()

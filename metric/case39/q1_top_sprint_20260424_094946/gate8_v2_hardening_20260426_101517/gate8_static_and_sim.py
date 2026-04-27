from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPRINT = ROOT / "metric/case39/q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate8_v2_hardening_20260426_101517"
GATE5_SCRIPT = SPRINT / "gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py"
ALT_CASE14_ROOT = Path("/home/pang/projects/DDET-MTD")

from evaluation_budget_scheduler_phase3 import (  # noqa: E402
    _aggregate_arrival_steps,
    _arrival_diagnostics,
    _busy_time_unit_from_fit,
    _derive_cost_budget_from_fit,
    _job_stats,
    _predict_jobs,
    _run_one_policy,
    _threshold_candidates,
    _tune_adaptive_threshold_policy,
    _tune_proposed_ca_policy,
    _tune_threshold_policy,
)
from phase3_holdout_core import _build_policy_cfgs  # noqa: E402
from scheduler.calibration import (  # noqa: E402
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import SimulationConfig  # noqa: E402


EPS = 1e-12
BUDGETS = [1, 2]
ALPHAS = [0.25, 0.5, 1.0]
RNG_SEED = 20260408


def load_gate5():
    spec = importlib.util.spec_from_file_location("gate5_transfer_burden_guard", GATE5_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GATE5_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G5 = load_gate5()
G2 = G5.G2
R2 = G5.R2


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
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def resolve_repo_or_case14_legacy(rel: str | Path) -> Tuple[Path, str]:
    """Resolve a repo-relative path, falling back to the legacy case14 repo for missing raw banks.

    The q1-case39 worktree contains case14 manifests and summaries but is missing the raw
    phase3_confirm_blind_v*/banks files. Using the sibling DDET-MTD path preserves the
    original case14 holdout identities without copying or mutating the current repo.
    """
    path = Path(rel)
    if path.is_absolute():
        return path, "absolute"
    primary = ROOT / path
    if primary.exists():
        return primary, "current_repo"
    legacy = ALT_CASE14_ROOT / path
    if legacy.exists() and str(path).startswith("metric/case14/"):
        return legacy, "legacy_case14_repo_fallback"
    return primary, "missing"


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


def phase3_args(manifest: Dict[str, Any], test_bank: str) -> SimpleNamespace:
    regime = manifest["frozen_regime"]
    clean_bank, _clean_src = resolve_repo_or_case14_legacy(manifest["clean_bank"])
    attack_bank, _attack_src = resolve_repo_or_case14_legacy(manifest["attack_bank"])
    train_bank, _train_src = resolve_repo_or_case14_legacy(manifest["train_bank"])
    tune_bank, _tune_src = resolve_repo_or_case14_legacy(manifest["val_bank"])
    eval_bank, _eval_src = resolve_repo_or_case14_legacy(test_bank)
    return SimpleNamespace(
        clean_bank=str(clean_bank),
        attack_bank=str(attack_bank),
        train_bank=str(train_bank),
        tune_bank=str(tune_bank),
        eval_bank=str(eval_bank),
        path_sources={
            "clean_bank": _clean_src,
            "attack_bank": _attack_src,
            "train_bank": _train_src,
            "val_bank": _tune_src,
            "test_bank": _eval_src,
        },
        output="",
        n_bins=20,
        slot_budget_list=list(regime["slot_budget_list"]),
        max_wait_steps=int(regime["max_wait_steps"]),
        decision_step_group=int(regime["decision_step_group"]),
        busy_time_quantile=float(regime["busy_time_quantile"]),
        use_cost_budget=bool(regime["use_cost_budget"]),
        cost_budget_window_steps=int(regime.get("cost_budget_window_steps", 20)),
        cost_budget_quantile=float(regime.get("cost_budget_quantile", 0.60)),
        threshold_quantiles=[0.50, 0.60, 0.70, 0.80, 0.90],
        adaptive_gain_scale_list=[0.0, 0.10, 0.20, 0.40],
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


def prepare_phase3_context(args: SimpleNamespace) -> Dict[str, Any]:
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.train_bank), int(args.decision_step_group))
    arrays_tune = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.tune_bank), int(args.decision_step_group))
    arrays_eval = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args.eval_bank), int(args.decision_step_group))
    posterior_verify = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="score_phys_l2", n_bins=args.n_bins)
    posterior_ddd = fit_attack_posterior_from_banks(args.clean_bank, args.attack_bank, signal_key="ddd_loss_alarm", n_bins=args.n_bins)
    service_models = fit_service_models_from_mixed_bank(args.train_bank, signal_key="verify_score", n_bins=args.n_bins)
    severity_models_cond = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models_exp = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args.n_bins)
    severity_models = severity_models_cond if args.consequence_mode == "conditional" else severity_models_exp
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args.busy_time_quantile))

    common_predict = dict(
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
    jobs_train, total_steps_train = _predict_jobs(arrays_train, **common_predict)
    jobs_tune, total_steps_tune = _predict_jobs(arrays_tune, **common_predict)
    jobs_eval, total_steps_eval = _predict_jobs(arrays_eval, **common_predict)
    jobs_tune_ddd, _ = _predict_jobs(
        arrays_tune,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    jobs_eval_ddd, _ = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        severity_models=severity_models,
        severity_blend_verify=float(args.consequence_blend_verify),
        consequence_mode=str(args.consequence_mode),
        fit_verify_score=np.asarray(arrays_train["verify_score"], dtype=float),
        busy_time_unit=busy_time_unit,
    )
    train_stats = _job_stats(jobs_train)
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
        "args": args,
        "arrays_train": arrays_train,
        "arrays_tune": arrays_tune,
        "arrays_eval": arrays_eval,
        "jobs_train": jobs_train,
        "jobs_tune": jobs_tune,
        "jobs_tune_ddd": jobs_tune_ddd,
        "jobs_eval": jobs_eval,
        "jobs_eval_ddd": jobs_eval_ddd,
        "total_steps_train": total_steps_train,
        "total_steps_tune": total_steps_tune,
        "total_steps_eval": total_steps_eval,
        "train_stats": train_stats,
        "verify_threshold_candidates": verify_threshold_candidates,
        "ddd_threshold_candidates": ddd_threshold_candidates,
        "ec_threshold_candidates": ec_threshold_candidates,
        "adaptive_gain_candidates": adaptive_gain_candidates,
        "busy_time_unit": busy_time_unit,
        "env": {
            "train_arrival": _arrival_diagnostics(jobs_train, total_steps_train),
            "tune_arrival": _arrival_diagnostics(jobs_tune, total_steps_tune),
            "eval_arrival": _arrival_diagnostics(jobs_eval, total_steps_eval),
        },
    }


def tune_cfgs(ctx: Dict[str, Any], slot_budget: int) -> Tuple[Dict[str, SimulationConfig], Dict[str, Any], Dict[str, List]]:
    args = ctx["args"]
    train_stats = ctx["train_stats"]
    cost_budget_window_steps = 0
    window_cost_budget = None
    if args.use_cost_budget:
        cost_budget_window_steps = int(args.cost_budget_window_steps)
        window_cost_budget = _derive_cost_budget_from_fit(
            ctx["jobs_train"],
            ctx["total_steps_train"],
            window_steps=cost_budget_window_steps,
            q=float(args.cost_budget_quantile),
        )
    cost_budget_per_step = None
    if window_cost_budget is not None and cost_budget_window_steps > 0:
        cost_budget_per_step = float(window_cost_budget) / max(int(cost_budget_window_steps), 1)
    score_kwargs = {
        "max_wait_steps": int(args.max_wait_steps),
        "clean_penalty": float(args.objective_clean_penalty),
        "delay_penalty": float(args.objective_delay_penalty),
        "queue_penalty": float(args.objective_queue_penalty),
        "cost_penalty": float(args.objective_cost_penalty),
        "cost_budget_per_step": cost_budget_per_step,
    }
    thr_verify, tune_thr_verify = _tune_threshold_policy(
        ctx["jobs_tune"],
        ctx["total_steps_tune"],
        threshold_candidates=ctx["verify_threshold_candidates"],
        policy_name="threshold_verify_fifo",
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
        mean_pred_service_cost=train_stats["mean_pred_service_cost"],
        mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
        score_kwargs=score_kwargs,
    )
    thr_ddd, tune_thr_ddd = _tune_threshold_policy(
        ctx["jobs_tune_ddd"],
        ctx["total_steps_tune"],
        threshold_candidates=ctx["ddd_threshold_candidates"],
        policy_name="threshold_ddd_fifo",
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
        mean_pred_service_cost=train_stats["mean_pred_service_cost"],
        mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
        score_kwargs=score_kwargs,
    )
    thr_ec, tune_thr_ec = _tune_threshold_policy(
        ctx["jobs_tune"],
        ctx["total_steps_tune"],
        threshold_candidates=ctx["ec_threshold_candidates"],
        policy_name="threshold_expected_consequence_fifo",
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
        mean_pred_service_cost=train_stats["mean_pred_service_cost"],
        mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
        score_kwargs=score_kwargs,
    )
    adaptive_best, tune_adaptive = _tune_adaptive_threshold_policy(
        ctx["jobs_tune"],
        ctx["total_steps_tune"],
        threshold_candidates=ctx["verify_threshold_candidates"],
        gain_candidates=ctx["adaptive_gain_candidates"],
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
        mean_pred_service_cost=train_stats["mean_pred_service_cost"],
        mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
        score_kwargs=score_kwargs,
    )
    proposed_best, tune_proposed = _tune_proposed_ca_policy(
        ctx["jobs_tune"],
        ctx["total_steps_tune"],
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
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
    cfgs = _build_policy_cfgs(
        slot_budget=slot_budget,
        max_wait_steps=int(args.max_wait_steps),
        rng_seed=int(args.rng_seed),
        cost_budget_window_steps=cost_budget_window_steps,
        window_cost_budget=window_cost_budget,
        mean_pred_busy_steps=train_stats["mean_pred_busy_steps"],
        mean_pred_service_cost=train_stats["mean_pred_service_cost"],
        mean_pred_expected_consequence=train_stats["mean_pred_expected_consequence"],
        thr_verify=thr_verify,
        thr_ddd=thr_ddd,
        thr_ec=thr_ec,
        adaptive_best=adaptive_best,
        proposed_best=proposed_best,
    )
    tune = {
        "threshold_verify_fifo": tune_thr_verify,
        "threshold_ddd_fifo": tune_thr_ddd,
        "threshold_expected_consequence_fifo": tune_thr_ec,
        "adaptive_threshold_verify_fifo": tune_adaptive,
        "proposed_ca_vq_hard": tune_proposed,
    }
    jobs = {
        "default": ctx["jobs_eval"],
        "ddd": ctx["jobs_eval_ddd"],
    }
    return cfgs, tune, jobs


def metric_row(detail: Dict[str, Any], jobs: Sequence[Any], *, method: str, budget: int, holdout_id: str) -> Dict[str, Any]:
    row = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=holdout_id)
    row["recall"] = row["scheduler_recall"]
    row["delay_p95"] = row["delay_p95"]
    row["cost"] = row["cost"]
    row["unnecessary"] = row["unnecessary"]
    return row


def apply_component_guard(jobs: Sequence[Any], *, cfg: SimulationConfig, stats: Dict[str, float], alpha: float, component: str) -> List[Any]:
    if component == "none" or abs(alpha) <= 1e-12:
        return list(jobs)
    mean_ec = max(float(cfg.mean_pred_expected_consequence), EPS)
    v_weight = max(abs(float(cfg.v_weight)), EPS)
    scale = mean_ec / v_weight
    out = []
    for job in jobs:
        fail_z = (float(job.pred_fail_prob) - stats["fail_mean"]) / stats["fail_std"]
        cost_z = (float(job.pred_service_cost) - stats["cost_mean"]) / stats["cost_std"]
        time_z = (float(job.pred_service_time) - stats["time_mean"]) / stats["time_std"]
        if component == "fail_only":
            burden = fail_z
        elif component == "cost_time_only":
            burden = cost_z + time_z
        elif component == "full_trbg":
            burden = fail_z + cost_z + time_z
        else:
            raise ValueError(component)
        guarded_ec = float(job.pred_expected_consequence) - float(alpha) * float(burden) * scale
        meta = dict(job.meta)
        meta["gate8_guard_component"] = component
        meta["gate8_guard_alpha"] = float(alpha)
        out.append(replace(job, pred_expected_consequence=float(guarded_ec), meta=meta))
    return out


def case14_holdouts() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifests = [
        read_json(ROOT / "metric/case14/phase3_confirm_blind_v1/manifest.json"),
        read_json(ROOT / "metric/case14/phase3_confirm_blind_v2/manifest.json"),
    ]
    merged = dict(manifests[0])
    holdouts: List[Dict[str, Any]] = []
    for m in manifests:
        holdouts.extend(m["holdouts"])
    merged["holdouts"] = holdouts
    return merged, holdouts


def run_case14_compat() -> Dict[str, Any]:
    manifest, holdouts = case14_holdouts()
    rows: List[Dict[str, Any]] = []
    path_resolution_rows: List[Dict[str, Any]] = []
    for hold in holdouts:
        args = phase3_args(manifest, hold["test_bank"])
        path_resolution_rows.append(
            {
                "holdout_id": hold["tag"],
                "test_bank": hold["test_bank"],
                "resolved_test_bank": args.eval_bank,
                "test_bank_source": args.path_sources["test_bank"],
                "all_sources": json.dumps(args.path_sources, sort_keys=True),
            }
        )
        ctx = prepare_phase3_context(args)
        stats = G5.guard_stats(ctx["jobs_train"])
        for budget in BUDGETS:
            cfgs, tune, jobs_map = tune_cfgs(ctx, budget)
            static_candidates = ["threshold_verify_fifo", "threshold_ddd_fifo", "threshold_expected_consequence_fifo"]
            def _static_score(name: str) -> float:
                payload = tune.get(name, {})
                if isinstance(payload, dict):
                    for key in ("objective", "best_objective"):
                        if key in payload:
                            return float(payload[key])
                    summary = payload.get("summary", {})
                    if isinstance(summary, dict):
                        return float(summary.get("unnecessary_mtd_count", 0.0))
                return 0.0
            static_best = min(
                static_candidates,
                key=_static_score,
            )
            methods = [
                ("incumbent_queue_aware", "adaptive_threshold_verify_fifo", jobs_map["default"]),
                ("original_proposed_safeguarded", "proposed_ca_vq_hard", jobs_map["default"]),
                ("topk_expected_consequence", "topk_expected_consequence", jobs_map["default"]),
                ("best_static_threshold", static_best, jobs_map["ddd"] if static_best == "threshold_ddd_fifo" else jobs_map["default"]),
            ]
            for out_name, cfg_name, jobs in methods:
                detail = R2.simulate_policy_detailed(jobs, total_steps=ctx["total_steps_eval"], cfg=cfgs[cfg_name])
                row = metric_row(detail, jobs, method=out_name, budget=budget, holdout_id=hold["tag"])
                row["source_policy_name"] = cfg_name
                row["test_bank_source"] = args.path_sources["test_bank"]
                rows.append(row)
            cfg = cfgs["proposed_ca_vq_hard"]
            guarded = apply_component_guard(jobs_map["default"], cfg=cfg, stats=stats, alpha=1.0, component="full_trbg")
            detail = R2.simulate_policy_detailed(guarded, total_steps=ctx["total_steps_eval"], cfg=cfg)
            row = metric_row(detail, guarded, method="proposed_plus_TRBG_source", budget=budget, holdout_id=hold["tag"])
            row["source_policy_name"] = "proposed_ca_vq_hard"
            row["test_bank_source"] = args.path_sources["test_bank"]
            rows.append(row)
    write_csv(OUT / "gate8_case14_input_path_resolution.csv", path_resolution_rows)
    write_csv(OUT / "gate8_case14_trbg_compat_by_holdout.csv", rows)
    summary = aggregate(rows, ["method", "B"])
    write_csv(OUT / "gate8_case14_trbg_compat_summary.csv", summary)
    decision = case14_decision(rows, summary)
    return decision


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    metrics = [
        "recall",
        "scheduler_recall",
        "unnecessary",
        "cost",
        "backend_fail",
        "recover_fail",
        "delay_p95",
        "served_ratio",
        "served_attack_mass",
        "backend_success_attack_mass",
        "clean_service",
    ]
    out = []
    for key, vals in sorted(groups.items()):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(vals)
        for m in metrics:
            if m in vals[0]:
                arr = np.asarray([float(v[m]) for v in vals], dtype=float)
                row[m] = float(np.mean(arr))
                row[f"{m}_median"] = float(np.median(arr))
        out.append(row)
    return out


def case14_decision(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["method"], int(r["B"])): r for r in summary}
    per_budget = []
    for b in BUDGETS:
        trbg = by[("proposed_plus_TRBG_source", b)]
        prop = by[("original_proposed_safeguarded", b)]
        per_budget.append(
            {
                "B": b,
                "recall_retention": float(trbg["recall"]) / max(float(prop["recall"]), EPS),
                "unnecessary_delta": float(trbg["unnecessary"]) - float(prop["unnecessary"]),
                "cost_delta": float(trbg["cost"]) - float(prop["cost"]),
                "backend_fail_delta": float(trbg["backend_fail"]) - float(prop["backend_fail"]),
            }
        )
    avg_ret = float(np.mean([x["recall_retention"] for x in per_budget]))
    improves = any(
        np.mean([x[k] for x in per_budget]) < -1e-12 for k in ["unnecessary_delta", "cost_delta", "backend_fail_delta"]
    )
    major_burden_increase = any(
        np.mean([x[k] for x in per_budget]) > 0.10 * max(abs(float(by[("original_proposed_safeguarded", 1)].get(k.replace("_delta", ""), 1.0))), 1.0)
        for k in ["unnecessary_delta", "backend_fail_delta"]
    )
    if avg_ret >= 0.98 and improves:
        status = "strong"
    elif avg_ret >= 0.95 and not major_burden_increase:
        status = "acceptable"
    else:
        status = "failure"
    lines = [
        "# Gate8 Case14 TRBG Compatibility Stats",
        "",
        "TRBG-source is applied as a fixed burden guard on top of the original proposed safeguarded scheduler; no case14 test holdout is used for parameter selection.",
        "",
        "Input provenance note: the q1-case39 worktree contains the case14 confirm manifests and summaries but not the raw `phase3_confirm_blind_v*/banks` files. Missing raw case14 holdout banks were read from `/home/pang/projects/DDET-MTD/metric/case14/.../banks/` with identical manifest-relative names; `gate8_case14_input_path_resolution.csv` records this fallback.",
        "",
        "| B | recall_retention | unnecessary_delta | cost_delta | backend_fail_delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for x in per_budget:
        lines.append(f"| {x['B']} | {fmt(x['recall_retention'])} | {fmt(x['unnecessary_delta'])} | {fmt(x['cost_delta'])} | {fmt(x['backend_fail_delta'])} |")
    (OUT / "gate8_case14_trbg_compat_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# Gate8 Case14 TRBG Compatibility Decision",
        "",
        f"- Status: `{status}`.",
        f"- Average recall retention: `{fmt(avg_ret)}`.",
        f"- At least one burden metric improves: `{improves}`.",
        "- If TRBG has limited 14-bus gain, it should be framed as a case39 scale-up burden guard rather than a paper-wide replacement.",
        f"- Recommended framing: `{'paper-wide method upgrade' if status == 'strong' else 'case39 scale-up extension'}`.",
    ]
    (OUT / "gate8_case14_trbg_compat_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    return {"status": status, "avg_recall_retention": avg_ret, "per_budget": per_budget}


def case39_source_defs() -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    by_source, _by_topk, _by_winner = G5.source_method_defs()
    sets = G5.calibration_sets(by_source)
    stats = G5.guard_stats(sets["TRBG-source"]["train_jobs"])
    return by_source, stats


def run_ablation() -> Dict[str, Any]:
    by_source, stats = case39_source_defs()
    holdouts = read_json(SPRINT / "gate2_full_native_20260424_100642/gate2_source_frozen_transfer_manifest_used.json")["holdouts"]
    rows: List[Dict[str, Any]] = []
    components = [("none", 0.0), *[(c, a) for c in ["fail_only", "cost_time_only", "full_trbg"] for a in ALPHAS]]
    for hold in holdouts:
        arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
        for budget in BUDGETS:
            src_def = by_source[budget]
            jobs, total_steps, _ = G5.build_jobs_from_arrays(src_def["ctx"], arrays, src_def["variant"])
            cfg = src_def["cfg"]
            for component, alpha in components:
                guarded = apply_component_guard(jobs, cfg=cfg, stats=stats, alpha=alpha, component=component)
                detail = R2.simulate_policy_detailed(guarded, total_steps=total_steps, cfg=cfg)
                method = "source_frozen_transfer" if component == "none" else f"{component}_alpha_{alpha}"
                row = metric_row(detail, guarded, method=method, budget=budget, holdout_id=hold["tag"])
                row["component"] = component
                row["alpha"] = alpha
                rows.append(row)
    write_csv(OUT / "gate8_trbg_component_ablation_grid.csv", rows)
    summary = aggregate(rows, ["component", "alpha", "B"])
    # Add deltas vs source for readability.
    base = {(int(r["B"])): r for r in summary if r["component"] == "none"}
    for r in summary:
        b = int(r["B"])
        r["recall_retention_vs_source"] = float(r["recall"]) / max(float(base[b]["recall"]), EPS)
        r["backend_fail_reduction_vs_source"] = (float(base[b]["backend_fail"]) - float(r["backend_fail"])) / max(float(base[b]["backend_fail"]), EPS)
        r["cost_delta_vs_source"] = float(r["cost"]) - float(base[b]["cost"])
    full = [r for r in summary if r["component"] == "full_trbg" and abs(float(r["alpha"]) - 1.0) < 1e-12]
    fail = [r for r in summary if r["component"] == "fail_only" and abs(float(r["alpha"]) - 1.0) < 1e-12]
    ct = [r for r in summary if r["component"] == "cost_time_only" and abs(float(r["alpha"]) - 1.0) < 1e-12]
    avg = lambda xs, k: float(np.mean([float(x[k]) for x in xs]))
    full_better_fail = avg(full, "backend_fail_reduction_vs_source") >= avg(fail, "backend_fail_reduction_vs_source") - 1e-12 and avg(full, "recall_retention_vs_source") >= avg(fail, "recall_retention_vs_source") - 0.03
    full_better_ct = avg(full, "backend_fail_reduction_vs_source") >= avg(ct, "backend_fail_reduction_vs_source") - 1e-12 and avg(full, "recall_retention_vs_source") >= avg(ct, "recall_retention_vs_source") - 0.03
    driver = "fail_component" if avg(fail, "backend_fail_reduction_vs_source") >= avg(ct, "backend_fail_reduction_vs_source") else "cost_time_component"
    lines = [
        "# Gate8 TRBG Component Ablation Summary",
        "",
        "This ablation is diagnostic only and does not replace locked TRBG-source.",
        "",
        f"- Full TRBG alpha=1 backend_fail reduction: `{fmt(avg(full, 'backend_fail_reduction_vs_source'))}`.",
        f"- Fail-only alpha=1 backend_fail reduction: `{fmt(avg(fail, 'backend_fail_reduction_vs_source'))}`.",
        f"- Cost-time-only alpha=1 backend_fail reduction: `{fmt(avg(ct, 'backend_fail_reduction_vs_source'))}`.",
        f"- Full TRBG better/equivalent vs fail-only under the diagnostic rule: `{full_better_fail}`.",
        f"- Full TRBG better/equivalent vs cost-time-only under the diagnostic rule: `{full_better_ct}`.",
        f"- Backend_fail reduction driver: `{driver}`.",
        "- If a simpler component looks better on confirm, it remains appendix observation because Gate8 is not authorized to replace the locked main method.",
    ]
    (OUT / "gate8_trbg_component_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_lines = [
        "# Gate8 TRBG Component Ablation Stats",
        "",
        md_table(
            [
                {
                    "component": r["component"],
                    "alpha": r["alpha"],
                    "B": r["B"],
                    "recall_retention": fmt(r["recall_retention_vs_source"]),
                    "backend_reduction": fmt(r["backend_fail_reduction_vs_source"]),
                    "cost_delta": fmt(r["cost_delta_vs_source"]),
                }
                for r in summary
            ],
            ["component", "alpha", "B", "recall_retention", "backend_reduction", "cost_delta"],
        ),
    ]
    (OUT / "gate8_trbg_component_ablation_stats.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    write_csv(OUT / "gate8_trbg_component_ablation_summary.csv", summary)
    plot_ablation(summary)
    return {"full_better_fail": full_better_fail, "full_better_ct": full_better_ct, "driver": driver}


def plot_ablation(summary: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"none": "o", "fail_only": "s", "cost_time_only": "^", "full_trbg": "D"}
    colors = {"none": "black", "fail_only": "#c44e52", "cost_time_only": "#4c72b0", "full_trbg": "#55a868"}
    for r in summary:
        ax.scatter(float(r["backend_fail"]), float(r["recall"]), marker=markers.get(r["component"], "o"), color=colors.get(r["component"], "gray"), s=55)
        if r["component"] != "none":
            ax.annotate(f"{r['component'].replace('_only','')}\na={r['alpha']},B={r['B']}", (float(r["backend_fail"]), float(r["recall"])), fontsize=7)
    ax.set_xlabel("backend_fail (lower better)")
    ax.set_ylabel("recall (higher better)")
    ax.set_title("Gate8 TRBG component ablation frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gate8_trbg_ablation_frontier.png", dpi=180)
    fig.savefig(OUT / "figure_gate8_trbg_ablation_frontier.pdf")
    plt.close(fig)


def release_cleanup() -> None:
    fit = ROOT / "metric/case39/mixed_bank_fit.npy"
    ev = ROOT / "metric/case39/mixed_bank_eval.npy"
    scripts = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"}:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "metric/case39/mixed_bank_fit.npy" in txt or "metric/case39/mixed_bank_eval.npy" in txt:
                scripts.append(str(path.relative_to(ROOT)))
    lines = [
        "# Gate8 Path Ambiguity Scan",
        "",
        f"- canonical metric/case39/mixed_bank_fit.npy resolves to: `{fit.resolve(strict=False)}`.",
        f"- canonical metric/case39/mixed_bank_eval.npy resolves to: `{ev.resolve(strict=False)}`.",
        f"- fit resolves to case14: `{'case14' in str(fit.resolve(strict=False))}`.",
        f"- eval resolves to case14: `{'case14' in str(ev.resolve(strict=False))}`.",
        "",
        "## Scripts / docs mentioning canonical case39 fit/eval",
        "",
    ]
    lines += [f"- `{s}`" for s in scripts] or ["- none found"]
    (OUT / "gate8_path_ambiguity_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "gate8_release_tree_plan.txt").write_text(
        "\n".join(
            [
                "metric/",
                "  case39_transfer/",
                "    banks/target_clean_attack_holdouts/",
                "    manifests/source_case14_to_case39.json",
                "    results/gate5_gate6_trbg/",
                "  case39_native/",
                "    banks/mixed_bank_fit_native.npy",
                "    banks/mixed_bank_eval_native.npy",
                "    results/full_native_local_retune/",
                "  case39_q1_sprint/",
                "    gate0_to_gate8_audit_and_rewrite_pack/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "gate8_release_cleanup_dryrun.md").write_text(
        "\n".join(
            [
                "# Gate8 Release Cleanup Dry Run",
                "",
                "No repo structure was changed. Release should split transfer, native, and sprint artifacts into explicit trees.",
                "",
                "- `metric/case39_transfer/`: case14 train/val to case39 target evidence.",
                "- `metric/case39_native/`: native case39 fit/eval and local-retune artifacts.",
                "- `metric/case39_q1_sprint/`: Gate0-Gate8 evidence packs.",
                "- Canonical case39 fit/eval symlinks must be removed or made unambiguous before release.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "gate8_readme_patch_plan.md").write_text(
        "\n".join(
            [
                "# Gate8 README Patch Plan",
                "",
                "- State that source-frozen transfer means case14 train/val -> case39 target.",
                "- State that full-native means native case39 train/val.",
                "- Mark old pre-fix attack-side artifacts invalid/caution-only.",
                "- State Gate6 is recombined stress replication.",
                "- State Gate7/Gate8 fresh solver outputs are sanity evidence unless enough banks accumulate for full validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"fit_case14": "case14" in str(fit.resolve(strict=False)), "eval_case14": "case14" in str(ev.resolve(strict=False)), "mentions": len(scripts)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    case14 = run_case14_compat()
    ablation = run_ablation()
    cleanup = release_cleanup()
    write_json(OUT / "gate8_static_results.json", {"case14": case14, "ablation": ablation, "cleanup": cleanup})


if __name__ == "__main__":
    main()

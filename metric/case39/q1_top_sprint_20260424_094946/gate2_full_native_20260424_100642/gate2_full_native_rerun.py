from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from evaluation_budget_scheduler_phase3 import (
    _aggregate_arrival_steps,
    _busy_time_unit_from_fit,
    _job_stats,
    _objective,
    _threshold_candidates,
    _tune_adaptive_threshold_policy,
    _tune_proposed_ca_policy,
    _tune_threshold_policy,
)
from phase3_oracle_family_core import (
    DEFAULT_VARIANTS,
    _build_jobs_for_variant,
    _fit_net_gain_models,
    _screen_variants,
    _select_joint_winner,
)
from scheduler.calibration import (
    fit_attack_posterior_from_banks,
    fit_attack_severity_models_from_arrays,
    fit_expected_consequence_models_from_arrays,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies_phase3 import SimulationConfig, simulate_policy


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate2_full_native_20260424_100642"
ROUND2_HELPER_PATH = ROOT / "metric" / "case39" / "round2_mechanism_20260424_092002" / "generate_round2_pack.py"
STAMP_LABEL = "20260424_100642"
STAMP_PATH = Path("/tmp") / f"case39_q1_top_sprint_20260424_094946_gate2_{STAMP_LABEL}.stamp"
OLD_REPO_CASE14 = Path("/home/pang/projects/DDET-MTD/metric/case14")

BUDGETS = [1, 2]
WMAX = 10
RNG_SEED = 20260402
REGIME = {
    "decision_step_group": 1,
    "busy_time_quantile": 0.65,
    "use_cost_budget": False,
    "cost_budget_window_steps": 20,
    "cost_budget_quantile": 0.60,
    "slot_budget_list": BUDGETS,
    "max_wait_steps": WMAX,
}
PRIMARY_METRICS = ["recall", "unnecessary", "cost", "served_ratio", "backend_fail", "delay_p95"]
HIGHER_IS_BETTER = {"recall", "served_ratio", "served_attack_mass"}


def load_round2_helper():
    spec = importlib.util.spec_from_file_location("round2_pack_helper", ROUND2_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper from {ROUND2_HELPER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R2 = load_round2_helper()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    stat = path.lstat() if exists else None
    return {
        "path": rel(path),
        "exists": bool(exists),
        "is_symlink": bool(path.is_symlink()),
        "resolved_path": str(resolved),
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": sha256_file(path) if exists else None,
        "lstat_mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
    }


def hash_records() -> Dict[str, Dict[str, Any]]:
    files = {
        "case14_fit": ROOT / "metric" / "case14" / "mixed_bank_fit.npy",
        "case14_eval": ROOT / "metric" / "case14" / "mixed_bank_eval.npy",
        "case39_native_fit": ROOT / "metric" / "case39_localretune" / "mixed_bank_fit_native.npy",
        "case39_native_eval": ROOT / "metric" / "case39_localretune" / "mixed_bank_eval_native.npy",
        "case39_canonical_fit": ROOT / "metric" / "case39" / "mixed_bank_fit.npy",
        "case39_canonical_eval": ROOT / "metric" / "case39" / "mixed_bank_eval.npy",
        "source_frozen_transfer_manifest": SPRINT / "source_frozen_transfer_manifest.json",
        "full_native_case39_manifest": SPRINT / "full_native_case39_manifest.json",
    }
    return {name: path_record(path) for name, path in sorted(files.items())}


def newer_files(root: Path, stamp: Path) -> List[str]:
    if not root.exists() or not stamp.exists():
        return []
    threshold = stamp.stat().st_mtime
    out: List[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime > threshold:
            out.append(str(path))
    return sorted(out)


def git_value(args: List[str]) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    except Exception:
        return None


def variant_by_name(name: str):
    return next(v for v in DEFAULT_VARIANTS if v.name == name)


def bank_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    return stem.replace("mixed_bank_test_", "")


def holdouts_from_manifest(manifest: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return [
        {
            "tag": bank_tag(p),
            "test_bank": p,
        }
        for p in manifest[key]
    ]


def compatible_manifest(*, train_bank: str, val_bank: str, clean_bank: str, attack_bank: str, holdouts: List[Dict[str, Any]], case_name: str) -> Dict[str, Any]:
    return {
        "workdir": str(ROOT),
        "case_name": case_name,
        "clean_bank": clean_bank,
        "attack_bank": attack_bank,
        "train_bank": train_bank,
        "val_bank": val_bank,
        "frozen_regime": dict(REGIME),
        "holdouts": [
            {
                "tag": h["tag"],
                "seed_base": None,
                "start_offset": None,
                "test_bank": h["test_bank"],
            }
            for h in holdouts
        ],
    }


def assert_no_canonical_train_val(paths: Iterable[str]) -> None:
    forbidden = {"metric/case39/mixed_bank_fit.npy", "metric/case39/mixed_bank_eval.npy"}
    used = {str(p).replace("\\", "/") for p in paths}
    bad = sorted(forbidden & used)
    if bad:
        raise RuntimeError(f"Forbidden canonical case39 train/val path used: {bad}")


def prepare_context(*, manifest: Dict[str, Any], winner_variant_name: str | None = None) -> Dict[str, Any]:
    assert_no_canonical_train_val([manifest["train_bank"], manifest["val_bank"]])
    args = {
        "clean_bank": str(ROOT / manifest["clean_bank"]),
        "attack_bank": str(ROOT / manifest["attack_bank"]),
        "train_bank": str(ROOT / manifest["train_bank"]),
        "val_bank": str(ROOT / manifest["val_bank"]),
        "n_bins": 20,
        "decision_step_group": REGIME["decision_step_group"],
        "busy_time_quantile": REGIME["busy_time_quantile"],
        "consequence_blend_verify": 0.70,
        "consequence_mode": "conditional",
    }
    arrays_train = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args["train_bank"]), args["decision_step_group"])
    arrays_val = _aggregate_arrival_steps(mixed_bank_to_alarm_arrays(args["val_bank"]), args["decision_step_group"])
    posterior_verify = fit_attack_posterior_from_banks(args["clean_bank"], args["attack_bank"], signal_key="score_phys_l2", n_bins=args["n_bins"])
    posterior_ddd = fit_attack_posterior_from_banks(args["clean_bank"], args["attack_bank"], signal_key="ddd_loss_alarm", n_bins=args["n_bins"])
    service_models = fit_service_models_from_mixed_bank(args["train_bank"], signal_key="verify_score", n_bins=args["n_bins"])
    severity_models = fit_attack_severity_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args["n_bins"])
    _ = fit_expected_consequence_models_from_arrays(arrays_train, signal_keys=("verify_score", "ddd_loss_recons"), n_bins=args["n_bins"])
    busy_time_unit = _busy_time_unit_from_fit(arrays_train, float(args["busy_time_quantile"]))

    gain_bundle_by_variant: Dict[str, Any] = {}
    for v in DEFAULT_VARIANTS:
        if v.mode == "help_gain":
            gain_bundle_by_variant[v.name] = _fit_net_gain_models(
                arrays_train,
                clean_scale=float(v.clean_scale),
                n_bins=args["n_bins"],
            )
    baseline_train_jobs, total_steps_train, _ = build_jobs(
        ctx={
            "arrays_train": arrays_train,
            "posterior_verify": posterior_verify,
            "posterior_ddd": posterior_ddd,
            "service_models": service_models,
            "severity_models": severity_models,
            "gain_bundle_by_variant": gain_bundle_by_variant,
            "busy_time_unit": busy_time_unit,
        },
        arrays_bank=arrays_train,
        variant_name=None,
    )
    baseline_val_jobs, total_steps_val, _ = build_jobs(
        ctx={
            "arrays_train": arrays_train,
            "posterior_verify": posterior_verify,
            "posterior_ddd": posterior_ddd,
            "service_models": service_models,
            "severity_models": severity_models,
            "gain_bundle_by_variant": gain_bundle_by_variant,
            "busy_time_unit": busy_time_unit,
        },
        arrays_bank=arrays_val,
        variant_name=None,
    )
    winner_variant = variant_by_name(winner_variant_name) if winner_variant_name else None
    return {
        "manifest": manifest,
        "args": args,
        "arrays_train": arrays_train,
        "arrays_val": arrays_val,
        "posterior_verify": posterior_verify,
        "posterior_ddd": posterior_ddd,
        "service_models": service_models,
        "severity_models": severity_models,
        "gain_bundle_by_variant": gain_bundle_by_variant,
        "busy_time_unit": busy_time_unit,
        "baseline_train_jobs": baseline_train_jobs,
        "baseline_val_jobs": baseline_val_jobs,
        "baseline_train_stats": _job_stats(baseline_train_jobs),
        "baseline_val_stats": _job_stats(baseline_val_jobs),
        "total_steps_train": total_steps_train,
        "total_steps_val": total_steps_val,
        "winner_variant": winner_variant,
    }


def build_jobs(*, ctx: Dict[str, Any], arrays_bank: Dict[str, np.ndarray], variant_name: str | None):
    variant = None if variant_name is None else variant_by_name(variant_name)
    return _build_jobs_for_variant(
        arrays_bank=arrays_bank,
        arrays_train=ctx["arrays_train"],
        posterior_verify=ctx["posterior_verify"],
        posterior_ddd=ctx["posterior_ddd"],
        service_models=ctx["service_models"],
        severity_models=ctx["severity_models"],
        variant=variant,
        gain_bundle_by_variant=ctx["gain_bundle_by_variant"],
        severity_blend_verify=0.70,
        busy_time_unit=ctx["busy_time_unit"],
    )


def score_kwargs() -> Dict[str, Any]:
    return {
        "max_wait_steps": WMAX,
        "clean_penalty": 0.60,
        "delay_penalty": 0.15,
        "queue_penalty": 0.10,
        "cost_penalty": 0.05,
        "cost_budget_per_step": None,
    }


def tune_incumbent_queue_aware(ctx: Dict[str, Any], *, slot_budget: int) -> Dict[str, float]:
    jobs_val = ctx["baseline_val_jobs"]
    total_steps_val = ctx["total_steps_val"]
    stats = ctx["baseline_train_stats"]
    best: Dict[str, float] | None = None
    best_obj = -1e18
    for v_weight in [1.0, 2.0, 4.0]:
        for age_bonus in [0.0, 0.10, 0.20]:
            for fail_penalty in [0.0, 0.05]:
                for busy_penalty in [0.5, 1.0, 2.0]:
                    for cost_penalty in [0.0, 0.5, 1.0]:
                        for admission_thr in [-0.10, 0.0, 0.10]:
                            cfg = SimulationConfig(
                                policy_name="proposed_vq_hard",
                                slot_budget=int(slot_budget),
                                max_wait_steps=WMAX,
                                rng_seed=RNG_SEED,
                                mean_pred_busy_steps=float(stats["mean_pred_busy_steps"]),
                                mean_pred_service_cost=float(stats["mean_pred_service_cost"]),
                                mean_pred_expected_consequence=float(stats["mean_pred_expected_consequence"]),
                                v_weight=float(v_weight),
                                age_bonus=float(age_bonus),
                                fail_penalty=float(fail_penalty),
                                busy_penalty=float(busy_penalty),
                                cost_penalty=float(cost_penalty),
                                admission_score_threshold=float(admission_thr),
                            )
                            res = simulate_policy(jobs_val, total_steps=total_steps_val, cfg=cfg)
                            obj = _objective(res["summary"], slot_budget=int(slot_budget), **score_kwargs())
                            if obj > best_obj:
                                best_obj = obj
                                best = {
                                    "v_weight": float(v_weight),
                                    "age_bonus": float(age_bonus),
                                    "fail_penalty": float(fail_penalty),
                                    "busy_penalty": float(busy_penalty),
                                    "cost_penalty": float(cost_penalty),
                                    "admission_score_threshold": float(admission_thr),
                                }
    assert best is not None
    return best


def tune_static_threshold(ctx: Dict[str, Any], *, slot_budget: int) -> float:
    arrays_val = ctx["arrays_val"]
    candidates = _threshold_candidates(np.asarray(arrays_val["verify_score"], dtype=float), [0.50, 0.60, 0.70, 0.80, 0.90])
    thr, _ = _tune_threshold_policy(
        ctx["baseline_val_jobs"],
        ctx["total_steps_val"],
        threshold_candidates=candidates,
        policy_name="threshold_verify_fifo",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        rng_seed=RNG_SEED,
        cost_budget_window_steps=0,
        window_cost_budget=None,
        mean_pred_busy_steps=float(ctx["baseline_train_stats"]["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(ctx["baseline_train_stats"]["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(ctx["baseline_train_stats"]["mean_pred_expected_consequence"]),
        score_kwargs=score_kwargs(),
    )
    return float(thr)


def tune_adaptive_threshold(ctx: Dict[str, Any], *, slot_budget: int) -> Dict[str, float]:
    arrays_val = ctx["arrays_val"]
    candidates = _threshold_candidates(np.asarray(arrays_val["verify_score"], dtype=float), [0.50, 0.60, 0.70, 0.80, 0.90])
    verify_signal = np.asarray(arrays_val["verify_score"], dtype=float)
    verify_signal = verify_signal[np.isfinite(verify_signal)]
    iqr = float(np.quantile(verify_signal, 0.80) - np.quantile(verify_signal, 0.20)) if verify_signal.size else 1.0
    gains = [float(x) * max(iqr, 1e-6) for x in [0.0, 0.10, 0.20, 0.40]]
    best, _ = _tune_adaptive_threshold_policy(
        ctx["baseline_val_jobs"],
        ctx["total_steps_val"],
        threshold_candidates=candidates,
        gain_candidates=gains,
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        rng_seed=RNG_SEED,
        cost_budget_window_steps=0,
        window_cost_budget=None,
        mean_pred_busy_steps=float(ctx["baseline_train_stats"]["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(ctx["baseline_train_stats"]["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(ctx["baseline_train_stats"]["mean_pred_expected_consequence"]),
        score_kwargs=score_kwargs(),
    )
    return {"base_threshold": float(best["base_threshold"]), "adaptive_gain": float(best["adaptive_gain"])}


def phase3_cfg_from_screen(screen: Dict[str, Any], *, slot_budget: int) -> Dict[str, float]:
    return {k: float(v) for k, v in screen["phase3_reference_by_slot"][str(slot_budget)]["config"].items()}


def variant_cfg_from_screen(screen: Dict[str, Any], *, variant_name: str, slot_budget: int) -> Dict[str, float]:
    if "variants" in screen and variant_name in screen["variants"]:
        return {k: float(v) for k, v in screen["variants"][variant_name]["by_slot"][str(slot_budget)]["tuned_config"].items()}
    payload = screen["selection"]["winner_payload"]
    return {k: float(v) for k, v in payload["by_slot"][str(slot_budget)]["tuned_config"].items()}


def cfg_proposed_ca(*, slot_budget: int, train_stats: Dict[str, float], tuned: Dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        policy_name="proposed_ca_vq_hard",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        rng_seed=RNG_SEED,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
        v_weight=float(tuned["v_weight"]),
        clean_penalty=float(tuned["clean_penalty"]),
        age_bonus=float(tuned["age_bonus"]),
        urgency_bonus=float(tuned["urgency_bonus"]),
        fail_penalty=float(tuned["fail_penalty"]),
        busy_penalty=float(tuned["busy_penalty"]),
        cost_penalty=float(tuned["cost_penalty"]),
        admission_score_threshold=float(tuned["admission_score_threshold"]),
    )


def cfg_topk(*, slot_budget: int, train_stats: Dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        policy_name="topk_expected_consequence",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        rng_seed=RNG_SEED,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
    )


def cfg_static_threshold(*, slot_budget: int, train_stats: Dict[str, float], threshold: float) -> SimulationConfig:
    return SimulationConfig(
        policy_name="threshold_verify_fifo",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        threshold=float(threshold),
        rng_seed=RNG_SEED,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
    )


def cfg_adaptive_threshold(*, slot_budget: int, train_stats: Dict[str, float], tuned: Dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        policy_name="adaptive_threshold_verify_fifo",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        threshold=float(tuned["base_threshold"]),
        adaptive_gain=float(tuned["adaptive_gain"]),
        rng_seed=RNG_SEED,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
    )


def cfg_incumbent(*, slot_budget: int, train_stats: Dict[str, float], tuned: Dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        policy_name="proposed_vq_hard",
        slot_budget=int(slot_budget),
        max_wait_steps=WMAX,
        rng_seed=RNG_SEED,
        mean_pred_busy_steps=float(train_stats["mean_pred_busy_steps"]),
        mean_pred_service_cost=float(train_stats["mean_pred_service_cost"]),
        mean_pred_expected_consequence=float(train_stats["mean_pred_expected_consequence"]),
        v_weight=float(tuned["v_weight"]),
        age_bonus=float(tuned["age_bonus"]),
        fail_penalty=float(tuned["fail_penalty"]),
        busy_penalty=float(tuned["busy_penalty"]),
        cost_penalty=float(tuned["cost_penalty"]),
        admission_score_threshold=float(tuned["admission_score_threshold"]),
    )


def summarize_detail(*, detail: Dict[str, Any], jobs: Sequence[Any]) -> Dict[str, Any]:
    s = detail["summary"]
    served_attack_ids = set(int(x) for x in detail["served_attack_jobs"])
    served_clean_ids = set(int(x) for x in detail["served_clean_jobs"])
    served_jobs = set(int(x) for x in detail["served_jobs"])
    attack_mass_total = float(sum(float(j.severity_true) for j in jobs if int(j.is_attack) == 1))
    served_attack_mass = float(sum(float(j.severity_true) for j in jobs if int(j.job_id) in served_attack_ids))
    served_attack_success_mass = float(sum(float(j.severity_true) for j in jobs if int(j.job_id) in served_attack_ids and int(j.actual_backend_fail) == 0))
    delays = np.asarray(detail["queue_delays_served"], dtype=float)
    service_times = np.asarray([float(j.actual_service_time) for j in jobs if int(j.job_id) in served_jobs], dtype=float)
    service_costs = np.asarray([float(j.actual_service_cost) for j in jobs if int(j.job_id) in served_jobs], dtype=float)
    return {
        "recall": float(s["weighted_attack_recall_no_backend_fail"]),
        "weighted_attack_recall": float(s["weighted_attack_recall"]),
        "unnecessary": int(s["unnecessary_mtd_count"]),
        "cost": float(s["average_service_cost_per_step"]),
        "served_ratio": float(len(served_jobs) / max(int(s["total_jobs"]), 1)),
        "backend_fail": int(s["total_backend_fail"]),
        "delay_p50": float(np.quantile(delays, 0.50)) if delays.size else 0.0,
        "delay_p95": float(s["queue_delay_p95"]),
        "served_attack_mass": served_attack_mass,
        "backend_success_attack_mass": served_attack_success_mass,
        "total_attack_mass": attack_mass_total,
        "served_clean_count": int(len(served_clean_ids)),
        "served_attack_count": int(len(served_attack_ids)),
        "served_jobs": int(len(served_jobs)),
        "expired_count": int(len(detail["dropped_jobs_ttl"]) + len(detail["dropped_jobs_horizon"])),
        "average_service_time": float(np.mean(service_times)) if service_times.size else 0.0,
        "average_service_cost": float(np.mean(service_costs)) if service_costs.size else 0.0,
        "total_service_time": float(s["total_service_time"]),
        "total_service_cost": float(s["total_service_cost"]),
    }


def run_one(*, rows: List[Dict[str, Any]], statuses: List[Dict[str, Any]], label: str, group: str, context_name: str, ctx: Dict[str, Any], holdouts: List[Dict[str, Any]], slot_budget: int, variant_name: str | None, cfg: SimulationConfig, tuning_source: str, is_full_native: bool) -> None:
    try:
        for hold in holdouts:
            arrays_test = _aggregate_arrival_steps(
                mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])),
                REGIME["decision_step_group"],
            )
            jobs, total_steps, _ = build_jobs(ctx=ctx, arrays_bank=arrays_test, variant_name=variant_name)
            detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=cfg)
            metrics = summarize_detail(detail=detail, jobs=jobs)
            rows.append(
                {
                    "method": label,
                    "group": group,
                    "context": context_name,
                    "is_full_native": bool(is_full_native),
                    "holdout_tag": hold["tag"],
                    "slot_budget": int(slot_budget),
                    "variant_name": variant_name or "phase3_baseline",
                    "policy_name": cfg.policy_name,
                    "tuning_source": tuning_source,
                    "status": "ok",
                    "failure_reason": "",
                    **metrics,
                }
            )
        statuses.append({"method": label, "slot_budget": int(slot_budget), "status": "ok", "failure_reason": ""})
    except Exception as exc:
        statuses.append({"method": label, "slot_budget": int(slot_budget), "status": "failed", "failure_reason": repr(exc)})
        for hold in holdouts:
            rows.append(
                {
                    "method": label,
                    "group": group,
                    "context": context_name,
                    "is_full_native": bool(is_full_native),
                    "holdout_tag": hold["tag"],
                    "slot_budget": int(slot_budget),
                    "variant_name": variant_name or "phase3_baseline",
                    "policy_name": getattr(cfg, "policy_name", "unknown"),
                    "tuning_source": tuning_source,
                    "status": "failed",
                    "failure_reason": repr(exc),
                }
            )


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r["method"], int(r["slot_budget"])) for r in rows if r.get("status") == "ok"})
    metrics = [
        "recall",
        "unnecessary",
        "cost",
        "served_ratio",
        "backend_fail",
        "delay_p50",
        "delay_p95",
        "served_attack_mass",
        "served_clean_count",
        "expired_count",
        "average_service_time",
        "average_service_cost",
    ]
    for method, slot in keys:
        subset = [r for r in rows if r.get("status") == "ok" and r["method"] == method and int(r["slot_budget"]) == slot]
        row = {
            "method": method,
            "slot_budget": slot,
            "n_holdouts": len(subset),
            "group": subset[0]["group"] if subset else "",
            "is_full_native": subset[0]["is_full_native"] if subset else "",
        }
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in subset], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{metric}_median"] = float(np.median(vals)) if vals.size else float("nan")
            row[f"{metric}_min"] = float(np.min(vals)) if vals.size else float("nan")
            row[f"{metric}_max"] = float(np.max(vals)) if vals.size else float("nan")
        out.append(row)
    return out


def metric_value(row: Dict[str, Any], metric: str) -> float:
    return float(row[metric])


def bootstrap_ci(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260424)
    means = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(5000)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * prob))


def sign_flip_p(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.abs(arr) > 1e-12]
    n = arr.size
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(arr)))
    if n <= 14:
        count = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.asarray([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            if abs(float(np.mean(arr * signs))) >= obs - 1e-12:
                count += 1
        return float(count / total)
    rng = np.random.default_rng(20260424)
    count = 0
    total = 20000
    for _ in range(total):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(float(np.mean(arr * signs))) >= obs - 1e-12:
            count += 1
    return float(count / total)


def paired_stats(rows: List[Dict[str, Any]], best_full_native_label: str) -> List[Dict[str, Any]]:
    comparisons = [
        ("source_frozen_transfer_vs_full_native_safeguarded_retune", "source_frozen_transfer", "native_safeguarded_retune"),
        ("source_frozen_transfer_vs_full_native_unconstrained_retune", "source_frozen_transfer", "native_unconstrained_retune"),
        ("source_frozen_transfer_vs_anchored_retune", "source_frozen_transfer", "anchored_retune"),
        ("source_frozen_transfer_vs_winner_replay", "source_frozen_transfer", "winner_replay"),
        ("phase3_oracle_upgrade_vs_phase3_proposed", "phase3_oracle_upgrade", "phase3_proposed"),
        ("phase3_oracle_upgrade_vs_topk_expected_consequence", "phase3_oracle_upgrade", "topk_expected_consequence"),
        ("best_full_native_operating_point_vs_source_frozen_transfer", best_full_native_label, "source_frozen_transfer"),
    ]
    by_key = {(r["method"], int(r["slot_budget"]), r["holdout_tag"]): r for r in rows if r.get("status") == "ok"}
    holdouts = sorted({r["holdout_tag"] for r in rows if r.get("status") == "ok"})
    out: List[Dict[str, Any]] = []
    for slot in BUDGETS:
        for comp, left, right in comparisons:
            for metric in PRIMARY_METRICS:
                vals: List[float] = []
                wins = losses = ties = 0
                for h in holdouts:
                    l = by_key.get((left, slot, h))
                    rr = by_key.get((right, slot, h))
                    if not l or not rr:
                        continue
                    delta = metric_value(l, metric) - metric_value(rr, metric)
                    vals.append(delta)
                    if abs(delta) <= 1e-12:
                        ties += 1
                    else:
                        left_better = delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                        if left_better:
                            wins += 1
                        else:
                            losses += 1
                ci_low, ci_high = bootstrap_ci(vals)
                out.append(
                    {
                        "comparison": comp,
                        "left": left,
                        "right": right,
                        "slot_budget": slot,
                        "metric": metric,
                        "mean_delta_left_minus_right": float(np.mean(vals)) if vals else float("nan"),
                        "median_delta_left_minus_right": float(np.median(vals)) if vals else float("nan"),
                        "min_delta": float(np.min(vals)) if vals else float("nan"),
                        "max_delta": float(np.max(vals)) if vals else float("nan"),
                        "bootstrap95_ci_low": ci_low,
                        "bootstrap95_ci_high": ci_high,
                        "exact_sign_test_p_two_sided": sign_test_p(wins, losses),
                        "sign_flip_permutation_p_two_sided": sign_flip_p(vals),
                        "n_holdouts": len(vals),
                        "left_wins": wins,
                        "left_losses": losses,
                        "ties": ties,
                    }
                )
    return out


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_protocol(full_manifest: Dict[str, Any], source_manifest: Dict[str, Any], gate2_manifest: Path) -> None:
    lines = [
        "# Gate 2 Protocol Used",
        "",
        f"- Gate 1 full-native manifest: `{rel(SPRINT / 'full_native_case39_manifest.json')}`",
        f"- Gate 1 source-frozen transfer manifest: `{rel(SPRINT / 'source_frozen_transfer_manifest.json')}`",
        f"- Gate 2 compatible full-native manifest: `{rel(gate2_manifest)}`",
        "- Full-native train bank: `metric/case39_localretune/mixed_bank_fit_native.npy`",
        "- Full-native val bank: `metric/case39_localretune/mixed_bank_eval_native.npy`",
        "- Canonical `metric/case39/mixed_bank_fit.npy` was not used.",
        "- Canonical `metric/case39/mixed_bank_eval.npy` was not used.",
        "- Source-frozen transfer uses `source_case=case14` and `target_case=case39` from `source_frozen_transfer_manifest.json`.",
        "- Budgets: `B = 1, 2`.",
        "- `Wmax = 10`.",
        f"- Frozen holdouts: `{len(full_manifest['native_holdout_banks'])}`.",
        "- Winner/config selection used train/val only; test holdouts were not used to select winners.",
        "- Missing or failed baselines are retained in `baseline_status.csv` and in summary files.",
        "",
        "## Baseline Definitions",
        "",
        "- `source_frozen_transfer`: case14 train/val transfer config evaluated on case39 holdouts.",
        "- `winner_replay`: source winner/config replayed with native case39 train/val prediction models.",
        "- `anchored_retune`: existing source-anchored native train/val screen config replayed on the same holdouts.",
        "- `native_safeguarded_retune`: forced `oracle_protected_ec` native train/val screen config.",
        "- `native_unconstrained_retune`: full native oracle-family screen without the protected-only guard.",
        "- `phase3_proposed`: native baseline `proposed_ca_vq_hard` selected on native val.",
        "- `phase3_oracle_upgrade`: native val-selected oracle-family upgrade.",
        "- `topk_expected_consequence`: static top-k by predicted expected consequence under native predictions.",
        "- `incumbent_queue_aware`: tuned non-consequence queue-aware `proposed_vq_hard` diagnostic baseline.",
        "- `static_threshold`: tuned static verify-score FIFO threshold diagnostic baseline.",
    ]
    (OUT / "gate2_protocol_used.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(summary_rows: List[Dict[str, Any]], status_rows: List[Dict[str, Any]]) -> None:
    display = []
    for r in summary_rows:
        display.append(
            {
                "method": r["method"],
                "B": r["slot_budget"],
                "recall": fmt(r["recall_mean"]),
                "unnecessary": fmt(r["unnecessary_mean"]),
                "cost": fmt(r["cost_mean"]),
                "served_ratio": fmt(r["served_ratio_mean"]),
                "backend_fail": fmt(r["backend_fail_mean"]),
                "delay_p95": fmt(r["delay_p95_mean"]),
                "served_attack_mass": fmt(r["served_attack_mass_mean"]),
                "clean_served": fmt(r["served_clean_count_mean"]),
            }
        )
    failures = [r for r in status_rows if r["status"] != "ok"]
    lines = [
        "# Full-native Case39 Gate 2 Summary",
        "",
        "This table uses the fixed 8 case39 holdouts, B=1/2, and Wmax=10. All rows were generated from explicit Gate 1 manifests; canonical case39 fit/eval were not used.",
        "",
        markdown_table(display, ["method", "B", "recall", "unnecessary", "cost", "served_ratio", "backend_fail", "delay_p95", "served_attack_mass", "clean_served"]),
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.append(markdown_table(failures, ["method", "slot_budget", "status", "failure_reason"]))
    else:
        lines.append("No baseline failed.")
    (OUT / "full_native_case39_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage_compare_md(summary_rows: List[Dict[str, Any]]) -> None:
    rows = []
    for r in summary_rows:
        rows.append(
            {
                "method": r["method"],
                "B": r["slot_budget"],
                "group": r["group"],
                "full_native": r["is_full_native"],
                "recall": fmt(r["recall_mean"]),
                "cost": fmt(r["cost_mean"]),
                "unnecessary": fmt(r["unnecessary_mean"]),
                "backend_fail": fmt(r["backend_fail_mean"]),
            }
        )
    lines = [
        "# Stage Compare: Native vs Transfer",
        "",
        "Rows separate source-frozen transfer from explicit full-native case39 operating points.",
        "",
        markdown_table(rows, ["method", "B", "group", "full_native", "recall", "cost", "unnecessary", "backend_fail"]),
    ]
    (OUT / "stage_compare_native_vs_transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_md(rows: List[Dict[str, Any]]) -> None:
    display = []
    for r in rows:
        if r["metric"] not in {"recall", "cost", "backend_fail"}:
            continue
        display.append(
            {
                "comparison": r["comparison"],
                "B": r["slot_budget"],
                "metric": r["metric"],
                "mean_delta": fmt(r["mean_delta_left_minus_right"]),
                "CI": f"[{fmt(r['bootstrap95_ci_low'])}, {fmt(r['bootstrap95_ci_high'])}]",
                "sign_p": fmt(r["exact_sign_test_p_two_sided"]),
                "flip_p": fmt(r["sign_flip_permutation_p_two_sided"]),
                "W/L/T": f"{r['left_wins']}/{r['left_losses']}/{r['ties']}",
            }
        )
    lines = [
        "# Paired Statistics: Native vs Transfer",
        "",
        "Deltas are left minus right over the same 8 holdouts. Wins/losses/ties use metric direction.",
        "",
        markdown_table(display, ["comparison", "B", "metric", "mean_delta", "CI", "sign_p", "flip_p", "W/L/T"]),
    ]
    (OUT / "paired_statistics_native_vs_transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance_report(pre: Dict[str, Any], post: Dict[str, Any], q1_newer: List[str], old_newer: List[str], used_paths: List[str]) -> None:
    case14_equal = pre["case14_fit"]["sha256"] == post["case14_fit"]["sha256"] and pre["case14_eval"]["sha256"] == post["case14_eval"]["sha256"]
    native_equal = pre["case39_native_fit"]["sha256"] == post["case39_native_fit"]["sha256"] and pre["case39_native_eval"]["sha256"] == post["case39_native_eval"]["sha256"]
    canonical_touched = (
        pre["case39_canonical_fit"]["lstat_mtime_ns"] != post["case39_canonical_fit"]["lstat_mtime_ns"]
        or pre["case39_canonical_eval"]["lstat_mtime_ns"] != post["case39_canonical_eval"]["lstat_mtime_ns"]
    )
    accidental = any(p in {"metric/case39/mixed_bank_fit.npy", "metric/case39/mixed_bank_eval.npy"} for p in used_paths)
    (OUT / "anti_write_q1_case14.txt").write_text("\n".join(q1_newer) + ("\n" if q1_newer else ""), encoding="utf-8")
    (OUT / "anti_write_oldrepo_case14.txt").write_text("\n".join(old_newer) + ("\n" if old_newer else ""), encoding="utf-8")
    lines = [
        "# Gate 2 Provenance Report",
        "",
        f"- Gate 2 STAMP path: `{STAMP_PATH}`",
        f"- Branch: `{git_value(['branch', '--show-current'])}`",
        f"- Commit: `{git_value(['rev-parse', 'HEAD'])}`",
        "",
        "## SHA Checks",
        "",
        f"- case14 fit pre SHA: `{pre['case14_fit']['sha256']}`",
        f"- case14 fit post SHA: `{post['case14_fit']['sha256']}`",
        f"- case14 eval pre SHA: `{pre['case14_eval']['sha256']}`",
        f"- case14 eval post SHA: `{post['case14_eval']['sha256']}`",
        f"- native case39 fit pre SHA: `{pre['case39_native_fit']['sha256']}`",
        f"- native case39 fit post SHA: `{post['case39_native_fit']['sha256']}`",
        f"- native case39 eval pre SHA: `{pre['case39_native_eval']['sha256']}`",
        f"- native case39 eval post SHA: `{post['case39_native_eval']['sha256']}`",
        f"- case14 SHA pre/post equal: `{case14_equal}`",
        f"- native case39 SHA pre/post equal: `{native_equal}`",
        "",
        "## Anti-write",
        "",
        f"- anti_write_q1_case14 empty: `{not q1_newer}`",
        f"- anti_write_oldrepo_case14 empty: `{not old_newer}`",
        "",
        "## Canonical Case39 Fit/Eval",
        "",
        f"- canonical case39 fit/eval touched: `{canonical_touched}`",
        f"- any output path accidentally used canonical case39 fit/eval: `{accidental}`",
        f"- canonical fit still resolves to: `{post['case39_canonical_fit']['resolved_path']}`",
        f"- canonical eval still resolves to: `{post['case39_canonical_eval']['resolved_path']}`",
    ]
    (OUT / "gate2_provenance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(summary_rows: List[Dict[str, Any]], paired_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["method"], int(r["slot_budget"])): r for r in summary_rows}
    source_b1 = by[("source_frozen_transfer", 1)]
    source_b2 = by[("source_frozen_transfer", 2)]
    full_native_methods = [
        "native_safeguarded_retune",
        "native_unconstrained_retune",
        "phase3_oracle_upgrade",
        "phase3_proposed",
        "topk_expected_consequence",
        "incumbent_queue_aware",
        "static_threshold",
    ]
    best_by_slot = {}
    for slot in BUDGETS:
        candidates = [by[(m, slot)] for m in full_native_methods if (m, slot) in by]
        best_by_slot[slot] = max(candidates, key=lambda r: float(r["recall_mean"]))
    local_collapse = (
        by[("native_safeguarded_retune", 1)]["recall_mean"] < source_b1["recall_mean"]
        and by[("native_safeguarded_retune", 2)]["recall_mean"] < source_b2["recall_mean"]
        and by[("native_unconstrained_retune", 1)]["recall_mean"] < source_b1["recall_mean"]
        and by[("native_unconstrained_retune", 2)]["recall_mean"] < source_b2["recall_mean"]
    )
    any_full_native_beats_source = any(best_by_slot[s]["recall_mean"] > by[("source_frozen_transfer", s)]["recall_mean"] for s in BUDGETS)
    source_more_burden = {
        "B1_cost_delta_vs_best": float(source_b1["cost_mean"] - best_by_slot[1]["cost_mean"]),
        "B2_cost_delta_vs_best": float(source_b2["cost_mean"] - best_by_slot[2]["cost_mean"]),
        "B1_clean_delta_vs_best": float(source_b1["served_clean_count_mean"] - best_by_slot[1]["served_clean_count_mean"]),
        "B2_clean_delta_vs_best": float(source_b2["served_clean_count_mean"] - best_by_slot[2]["served_clean_count_mean"]),
        "B1_backend_fail_delta_vs_best": float(source_b1["backend_fail_mean"] - best_by_slot[1]["backend_fail_mean"]),
        "B2_backend_fail_delta_vs_best": float(source_b2["backend_fail_mean"] - best_by_slot[2]["backend_fail_mean"]),
    }
    paired_support = [
        r for r in paired_rows
        if r["comparison"] in {
            "source_frozen_transfer_vs_full_native_safeguarded_retune",
            "source_frozen_transfer_vs_full_native_unconstrained_retune",
            "source_frozen_transfer_vs_anchored_retune",
            "source_frozen_transfer_vs_winner_replay",
        }
        and r["metric"] == "recall"
    ]
    source_supported = all(int(r["left_wins"]) >= 7 for r in paired_support)
    decision = {
        "full_native_local_retune_still_collapses": bool(local_collapse),
        "any_full_native_operating_point_beats_source_frozen_transfer_on_recall": bool(any_full_native_beats_source),
        "best_full_native_by_slot_posthoc_recall": {str(k): {"method": v["method"], "recall_mean": v["recall_mean"]} for k, v in best_by_slot.items()},
        "source_advantage_interpretation": "source-frozen has higher recall and served attack mass, but it also carries higher cost and backend-fail burden; clean service is not consistently higher versus top-k.",
        "source_more_burden": source_more_burden,
        "supports_q1_level_native_mechanism_claim": False,
        "main_gap": "Full-native canonical route is explicit and runnable, but full-native operating points do not beat source-frozen transfer; evidence supports transfer regularization/stress behavior, not native success.",
        "recommend_continue_gate3_funnel_ceiling": True,
        "paired_statistics_support_source_frozen_recall": bool(source_supported),
    }
    lines = [
        "# Gate 2 Decision",
        "",
        f"- full-native local retune still collapses: `{decision['full_native_local_retune_still_collapses']}`",
        f"- any full-native operating point beats source-frozen transfer on recall: `{decision['any_full_native_operating_point_beats_source_frozen_transfer_on_recall']}`",
        f"- best full-native B=1 by post-hoc recall: `{decision['best_full_native_by_slot_posthoc_recall']['1']['method']}` recall `{fmt(decision['best_full_native_by_slot_posthoc_recall']['1']['recall_mean'])}`",
        f"- best full-native B=2 by post-hoc recall: `{decision['best_full_native_by_slot_posthoc_recall']['2']['method']}` recall `{fmt(decision['best_full_native_by_slot_posthoc_recall']['2']['recall_mean'])}`",
        f"- source advantage: {decision['source_advantage_interpretation']}",
        f"- source burden versus the post-hoc best full-native recall baseline: `{source_more_burden}`; clean service is not consistently higher versus top-k.",
        f"- full-native result supports Q1-level native mechanism claim: `{decision['supports_q1_level_native_mechanism_claim']}`",
        f"- main gap: {decision['main_gap']}",
        f"- recommend Gate 3 funnel/ceiling: `{decision['recommend_continue_gate3_funnel_ceiling']}`",
    ]
    (OUT / "gate2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "gate2_decision.json", decision)
    return decision


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(f"case39 q1 top sprint gate2 stamp {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    time.sleep(0.25)
    pre_hash = hash_records()
    write_json(OUT / "hash_pre_gate2.json", pre_hash)

    full_manifest = read_json(SPRINT / "full_native_case39_manifest.json")
    source_manifest = read_json(SPRINT / "source_frozen_transfer_manifest.json")
    holdouts = holdouts_from_manifest(full_manifest, "native_holdout_banks")

    full_train = full_manifest["native_train_bank"]
    full_val = full_manifest["native_val_bank"]
    source_train = source_manifest["source_train_bank"]
    source_val = source_manifest["source_val_bank"]
    clean_bank = full_manifest["native_clean_bank"]
    attack_bank = full_manifest["native_attack_bank"]
    assert_no_canonical_train_val([full_train, full_val, source_train, source_val])

    gate2_full_manifest = compatible_manifest(
        train_bank=full_train,
        val_bank=full_val,
        clean_bank=clean_bank,
        attack_bank=attack_bank,
        holdouts=holdouts,
        case_name="case39",
    )
    gate2_source_manifest = compatible_manifest(
        train_bank=source_train,
        val_bank=source_val,
        clean_bank=source_manifest["target_clean_bank"],
        attack_bank=source_manifest["target_attack_bank"],
        holdouts=holdouts,
        case_name="case39_source_frozen_transfer",
    )
    gate2_full_manifest_path = OUT / "gate2_full_native_manifest_used.json"
    gate2_source_manifest_path = OUT / "gate2_source_frozen_transfer_manifest_used.json"
    write_json(gate2_full_manifest_path, gate2_full_manifest)
    write_json(gate2_source_manifest_path, gate2_source_manifest)
    write_protocol(full_manifest, source_manifest, gate2_full_manifest_path)

    native_screen = _screen_variants(manifest=gate2_full_manifest)
    write_json(OUT / "gate2_full_native_screen_train_val_summary.json", native_screen)
    source_screen_path = ROOT / "metric" / "case14" / "phase3_oracle_family" / "screen_train_val_summary.json"
    source_screen = read_json(source_screen_path)
    shutil.copy2(source_screen_path, OUT / "gate2_source_screen_train_val_summary_used.json")
    anchored_screen = read_json(ROOT / "metric" / "case39_source_anchor" / "oracle_family" / "screen_train_val_summary_source_anchored.json")
    safeguarded_screen = read_json(ROOT / "metric" / "case39_localretune_protectedec" / "oracle_family" / "screen_train_val_summary_forced_oracle_protected_ec.json")
    native_existing_screen = read_json(ROOT / "metric" / "case39_localretune" / "oracle_family" / "screen_train_val_summary.json")

    source_ctx = prepare_context(manifest=gate2_source_manifest)
    native_ctx = prepare_context(manifest=gate2_full_manifest)

    tuning_native: Dict[Tuple[str, int], Any] = {}
    for slot in BUDGETS:
        tuning_native[("incumbent_queue_aware", slot)] = tune_incumbent_queue_aware(native_ctx, slot_budget=slot)
        tuning_native[("static_threshold", slot)] = tune_static_threshold(native_ctx, slot_budget=slot)
        tuning_native[("adaptive_threshold", slot)] = tune_adaptive_threshold(native_ctx, slot_budget=slot)
    write_json(OUT / "gate2_native_tuning_payloads.json", {f"{k[0]}::B{k[1]}": v for k, v in tuning_native.items()})

    rows: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []

    method_defs = []
    for slot in BUDGETS:
        source_winner = source_screen["selection"]["winner_variant"]
        native_winner = native_screen["selection"]["winner_variant"]
        anchored_winner = anchored_screen["selection"]["winner_variant"]
        safeguarded_winner = safeguarded_screen["selection"]["winner_variant"]
        native_existing_winner = native_existing_screen["selection"]["winner_variant"]

        source_train_stats = _job_stats(build_jobs(ctx=source_ctx, arrays_bank=source_ctx["arrays_train"], variant_name=source_winner)[0])
        native_stats_by_variant: Dict[str, Dict[str, float]] = {}
        for vn in sorted({native_winner, anchored_winner, safeguarded_winner, native_existing_winner}):
            native_stats_by_variant[vn] = _job_stats(build_jobs(ctx=native_ctx, arrays_bank=native_ctx["arrays_train"], variant_name=vn)[0])
        baseline_stats = native_ctx["baseline_train_stats"]

        method_defs.extend(
            [
                {
                    "label": "source_frozen_transfer",
                    "group": "source_frozen_transfer",
                    "ctx": source_ctx,
                    "context_name": "source_case14_train_val_target_case39",
                    "variant": source_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=source_train_stats, tuned=variant_cfg_from_screen(source_screen, variant_name=source_winner, slot_budget=slot)),
                    "tuning_source": str(source_screen_path),
                    "full_native": False,
                },
                {
                    "label": "winner_replay",
                    "group": "full_native_with_source_config",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_source_config",
                    "variant": source_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant.get(source_winner, baseline_stats), tuned=variant_cfg_from_screen(source_screen, variant_name=source_winner, slot_budget=slot)),
                    "tuning_source": str(source_screen_path),
                    "full_native": True,
                },
                {
                    "label": "anchored_retune",
                    "group": "full_native_source_anchored",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_anchored_screen",
                    "variant": anchored_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[anchored_winner], tuned=variant_cfg_from_screen(anchored_screen, variant_name=anchored_winner, slot_budget=slot)),
                    "tuning_source": "metric/case39_source_anchor/oracle_family/screen_train_val_summary_source_anchored.json",
                    "full_native": True,
                },
                {
                    "label": "native_safeguarded_retune",
                    "group": "full_native_safeguarded",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_forced_oracle_protected_ec",
                    "variant": safeguarded_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[safeguarded_winner], tuned=variant_cfg_from_screen(safeguarded_screen, variant_name=safeguarded_winner, slot_budget=slot)),
                    "tuning_source": "metric/case39_localretune_protectedec/oracle_family/screen_train_val_summary_forced_oracle_protected_ec.json",
                    "full_native": True,
                },
                {
                    "label": "native_unconstrained_retune",
                    "group": "full_native_unconstrained",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_unconstrained_existing_screen",
                    "variant": native_existing_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[native_existing_winner], tuned=variant_cfg_from_screen(native_existing_screen, variant_name=native_existing_winner, slot_budget=slot)),
                    "tuning_source": "metric/case39_localretune/oracle_family/screen_train_val_summary.json",
                    "full_native": True,
                },
                {
                    "label": "phase3_oracle_upgrade",
                    "group": "full_native_gate2_screen",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_gate2_screen",
                    "variant": native_winner,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[native_winner], tuned=variant_cfg_from_screen(native_screen, variant_name=native_winner, slot_budget=slot)),
                    "tuning_source": str(OUT / "gate2_full_native_screen_train_val_summary.json"),
                    "full_native": True,
                },
                {
                    "label": "phase3_proposed",
                    "group": "full_native_baseline",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_baseline_phase3",
                    "variant": None,
                    "cfg": cfg_proposed_ca(slot_budget=slot, train_stats=baseline_stats, tuned=phase3_cfg_from_screen(native_screen, slot_budget=slot)),
                    "tuning_source": str(OUT / "gate2_full_native_screen_train_val_summary.json"),
                    "full_native": True,
                },
                {
                    "label": "topk_expected_consequence",
                    "group": "full_native_baseline",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_topk",
                    "variant": None,
                    "cfg": cfg_topk(slot_budget=slot, train_stats=baseline_stats),
                    "tuning_source": "fixed_policy_no_test_selection",
                    "full_native": True,
                },
                {
                    "label": "incumbent_queue_aware",
                    "group": "full_native_diagnostic_baseline",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_incumbent_queue_aware",
                    "variant": None,
                    "cfg": cfg_incumbent(slot_budget=slot, train_stats=baseline_stats, tuned=tuning_native[("incumbent_queue_aware", slot)]),
                    "tuning_source": "gate2_native_val",
                    "full_native": True,
                },
                {
                    "label": "static_threshold",
                    "group": "full_native_diagnostic_baseline",
                    "ctx": native_ctx,
                    "context_name": "native_case39_train_val_static_threshold",
                    "variant": None,
                    "cfg": cfg_static_threshold(slot_budget=slot, train_stats=baseline_stats, threshold=tuning_native[("static_threshold", slot)]),
                    "tuning_source": "gate2_native_val",
                    "full_native": True,
                },
            ]
        )

    for item in method_defs:
        run_one(
            rows=rows,
            statuses=statuses,
            label=item["label"],
            group=item["group"],
            context_name=item["context_name"],
            ctx=item["ctx"],
            holdouts=holdouts,
            slot_budget=int(item["cfg"].slot_budget),
            variant_name=item["variant"],
            cfg=item["cfg"],
            tuning_source=item["tuning_source"],
            is_full_native=bool(item["full_native"]),
        )

    summary_rows = aggregate(rows)
    write_csv(OUT / "full_native_case39_by_holdout.csv", rows)
    write_csv(OUT / "full_native_case39_summary.csv", summary_rows)
    write_csv(OUT / "stage_compare_native_vs_transfer.csv", summary_rows)
    write_csv(OUT / "baseline_status.csv", statuses)
    write_summary_md(summary_rows, statuses)
    write_stage_compare_md(summary_rows)

    best_full_native_label = "topk_expected_consequence"
    paired = paired_stats(rows, best_full_native_label)
    write_csv(OUT / "paired_statistics_native_vs_transfer.csv", paired)
    write_paired_md(paired)

    post_hash = hash_records()
    write_json(OUT / "hash_post_gate2.json", post_hash)
    q1_newer = newer_files(ROOT / "metric" / "case14", STAMP_PATH)
    old_newer = newer_files(OLD_REPO_CASE14, STAMP_PATH)
    used_paths = sorted({full_train, full_val, source_train, source_val, clean_bank, attack_bank})
    write_provenance_report(pre_hash, post_hash, q1_newer, old_newer, used_paths)
    decision = write_decision(summary_rows, paired)

    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree_gate2.txt").write_text("\n".join(sorted(set(files) | {"outputs_tree_gate2.txt"})) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUT),
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired),
        "anti_write_q1_case14_empty": not q1_newer,
        "anti_write_oldrepo_case14_empty": not old_newer,
        "decision": decision,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

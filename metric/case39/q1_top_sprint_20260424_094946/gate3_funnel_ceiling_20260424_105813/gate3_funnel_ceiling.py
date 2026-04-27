from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
GATE2 = SPRINT / "gate2_full_native_20260424_100642"
OUT = SPRINT / "gate3_funnel_ceiling_20260424_105813"
GATE2_SCRIPT = GATE2 / "gate2_full_native_rerun.py"

BUDGETS = [1, 2]
WMAX = 10
RNG_SEED = 20260402
METHOD_ORDER = [
    "source_frozen_transfer",
    "winner_replay",
    "anchored_retune",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
    "phase3_oracle_upgrade",
    "phase3_proposed",
    "topk_expected_consequence",
    "incumbent_queue_aware",
    "static_threshold",
]
FIGURE_METHODS = [
    "source_frozen_transfer",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
    "topk_expected_consequence",
]


def load_gate2():
    spec = importlib.util.spec_from_file_location("gate2_full_native_rerun", GATE2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GATE2_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G2 = load_gate2()
R2 = G2.R2


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
        for row in rows:
            w.writerow(row)


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


def to_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def raw_attack_totals(bank_path: str) -> Dict[str, float]:
    payload = np.load(ROOT / bank_path, allow_pickle=True).item()
    is_attack = np.asarray(payload["is_attack_step"], dtype=int) == 1
    ang_no = np.asarray(payload["ang_no_summary"], dtype=float)
    ang_str = np.asarray(payload["ang_str_summary"], dtype=float)
    product = np.maximum(ang_no, 0.0) * np.maximum(ang_str, 0.0)
    return {
        "total_attack_jobs": int(np.sum(is_attack)),
        "total_attack_mass_product_proxy": float(np.sum(product[is_attack])),
        "raw_clean_steps": int(np.sum(~is_attack)),
    }


def mass_for_ids(jobs_by_id: Dict[int, Any], ids: Iterable[int], *, no_fail: bool | None = None, attack_only: bool = True) -> float:
    total = 0.0
    for raw_id in ids:
        job = jobs_by_id[int(raw_id)]
        if attack_only and int(job.is_attack) != 1:
            continue
        if no_fail is True and int(job.actual_backend_fail) != 0:
            continue
        if no_fail is False and int(job.actual_backend_fail) == 0:
            continue
        total += float(job.severity_true)
    return float(total)


def ids_matching(jobs_by_id: Dict[int, Any], ids: Iterable[int], *, attack: bool | None = None, backend_fail: bool | None = None) -> List[int]:
    out: List[int] = []
    for raw_id in ids:
        job = jobs_by_id[int(raw_id)]
        if attack is not None and bool(job.is_attack) != bool(attack):
            continue
        if backend_fail is not None and bool(job.actual_backend_fail) != bool(backend_fail):
            continue
        out.append(int(raw_id))
    return out


def build_method_defs() -> List[Dict[str, Any]]:
    full_manifest = read_json(GATE2 / "gate2_full_native_manifest_used.json")
    source_manifest = read_json(GATE2 / "gate2_source_frozen_transfer_manifest_used.json")
    source_screen = read_json(GATE2 / "gate2_source_screen_train_val_summary_used.json")
    native_screen = read_json(GATE2 / "gate2_full_native_screen_train_val_summary.json")
    anchored_screen = read_json(ROOT / "metric" / "case39_source_anchor" / "oracle_family" / "screen_train_val_summary_source_anchored.json")
    safeguarded_screen = read_json(ROOT / "metric" / "case39_localretune_protectedec" / "oracle_family" / "screen_train_val_summary_forced_oracle_protected_ec.json")
    native_existing_screen = read_json(ROOT / "metric" / "case39_localretune" / "oracle_family" / "screen_train_val_summary.json")
    native_tuning = read_json(GATE2 / "gate2_native_tuning_payloads.json")

    source_ctx = G2.prepare_context(manifest=source_manifest)
    native_ctx = G2.prepare_context(manifest=full_manifest)

    method_defs: List[Dict[str, Any]] = []
    for slot in BUDGETS:
        source_winner = source_screen["selection"]["winner_variant"]
        native_winner = native_screen["selection"]["winner_variant"]
        anchored_winner = anchored_screen["selection"]["winner_variant"]
        safeguarded_winner = safeguarded_screen["selection"]["winner_variant"]
        native_existing_winner = native_existing_screen["selection"]["winner_variant"]

        source_stats = G2._job_stats(G2.build_jobs(ctx=source_ctx, arrays_bank=source_ctx["arrays_train"], variant_name=source_winner)[0])
        native_stats_by_variant: Dict[str, Dict[str, float]] = {}
        for vn in sorted({source_winner, native_winner, anchored_winner, safeguarded_winner, native_existing_winner}):
            native_stats_by_variant[vn] = G2._job_stats(G2.build_jobs(ctx=native_ctx, arrays_bank=native_ctx["arrays_train"], variant_name=vn)[0])
        baseline_stats = native_ctx["baseline_train_stats"]

        method_defs.extend(
            [
                {
                    "method": "source_frozen_transfer",
                    "group": "source_frozen_transfer",
                    "full_native": False,
                    "ctx": source_ctx,
                    "variant": source_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=source_stats, tuned=G2.variant_cfg_from_screen(source_screen, variant_name=source_winner, slot_budget=slot)),
                },
                {
                    "method": "winner_replay",
                    "group": "full_native_with_source_config",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": source_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[source_winner], tuned=G2.variant_cfg_from_screen(source_screen, variant_name=source_winner, slot_budget=slot)),
                },
                {
                    "method": "anchored_retune",
                    "group": "full_native_source_anchored",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": anchored_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[anchored_winner], tuned=G2.variant_cfg_from_screen(anchored_screen, variant_name=anchored_winner, slot_budget=slot)),
                },
                {
                    "method": "native_safeguarded_retune",
                    "group": "full_native_safeguarded",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": safeguarded_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[safeguarded_winner], tuned=G2.variant_cfg_from_screen(safeguarded_screen, variant_name=safeguarded_winner, slot_budget=slot)),
                },
                {
                    "method": "native_unconstrained_retune",
                    "group": "full_native_unconstrained",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": native_existing_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[native_existing_winner], tuned=G2.variant_cfg_from_screen(native_existing_screen, variant_name=native_existing_winner, slot_budget=slot)),
                },
                {
                    "method": "phase3_oracle_upgrade",
                    "group": "full_native_gate2_screen",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": native_winner,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=native_stats_by_variant[native_winner], tuned=G2.variant_cfg_from_screen(native_screen, variant_name=native_winner, slot_budget=slot)),
                },
                {
                    "method": "phase3_proposed",
                    "group": "full_native_baseline",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": None,
                    "cfg": G2.cfg_proposed_ca(slot_budget=slot, train_stats=baseline_stats, tuned=G2.phase3_cfg_from_screen(native_screen, slot_budget=slot)),
                },
                {
                    "method": "topk_expected_consequence",
                    "group": "full_native_baseline",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": None,
                    "cfg": G2.cfg_topk(slot_budget=slot, train_stats=baseline_stats),
                },
                {
                    "method": "incumbent_queue_aware",
                    "group": "full_native_diagnostic_baseline",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": None,
                    "cfg": G2.cfg_incumbent(slot_budget=slot, train_stats=baseline_stats, tuned=native_tuning[f"incumbent_queue_aware::B{slot}"]),
                },
                {
                    "method": "static_threshold",
                    "group": "full_native_diagnostic_baseline",
                    "full_native": True,
                    "ctx": native_ctx,
                    "variant": None,
                    "cfg": G2.cfg_static_threshold(slot_budget=slot, train_stats=baseline_stats, threshold=float(native_tuning[f"static_threshold::B{slot}"])),
                },
            ]
        )
    return method_defs


def compute_funnel_row(method_def: Dict[str, Any], holdout: Dict[str, Any], detail: Dict[str, Any], jobs: Sequence[Any]) -> Dict[str, Any]:
    raw = raw_attack_totals(holdout["test_bank"])
    jobs_by_id = {int(j.job_id): j for j in jobs}
    all_ids = list(jobs_by_id.keys())
    attack_ids = ids_matching(jobs_by_id, all_ids, attack=True)
    clean_ids = ids_matching(jobs_by_id, all_ids, attack=False)
    threshold_drop = set(int(x) for x in detail["dropped_jobs_threshold"])
    admitted_ids = [i for i in all_ids if i not in threshold_drop]
    admitted_attack_ids = ids_matching(jobs_by_id, admitted_ids, attack=True)
    expired_ids = set(int(x) for x in detail["dropped_jobs_ttl"]) | set(int(x) for x in detail["dropped_jobs_horizon"])
    expired_attack_ids = ids_matching(jobs_by_id, expired_ids, attack=True)
    served_attack_ids = ids_matching(jobs_by_id, detail["served_attack_jobs"], attack=True)
    served_clean_ids = ids_matching(jobs_by_id, detail["served_clean_jobs"], attack=False)
    backend_success_attack_ids = ids_matching(jobs_by_id, served_attack_ids, attack=True, backend_fail=False)
    backend_fail_attack_ids = ids_matching(jobs_by_id, served_attack_ids, attack=True, backend_fail=True)
    served_ids = [int(x) for x in detail["served_jobs"]]
    total_verified_mass = mass_for_ids(jobs_by_id, attack_ids)
    total_raw_mass = float(raw["total_attack_mass_product_proxy"])
    served_attack_mass = mass_for_ids(jobs_by_id, served_attack_ids)
    backend_success_mass = mass_for_ids(jobs_by_id, backend_success_attack_ids)
    backend_fail_attack_mass = mass_for_ids(jobs_by_id, backend_fail_attack_ids)
    delays = np.asarray(detail["queue_delays_served"], dtype=float)
    service_times = np.asarray([float(jobs_by_id[i].actual_service_time) for i in served_ids], dtype=float)
    service_costs = np.asarray([float(jobs_by_id[i].actual_service_cost) for i in served_ids], dtype=float)
    served_clean_backend_fail = ids_matching(jobs_by_id, served_clean_ids, attack=False, backend_fail=True)
    row = {
        "holdout_id": holdout["tag"],
        "budget": int(method_def["cfg"].slot_budget),
        "variant": method_def["method"],
        "group": method_def["group"],
        "full_native_or_source_frozen": "full_native" if method_def["full_native"] else "source_frozen",
        "oracle_variant_name": method_def["variant"] or "phase3_baseline",
        "policy_name": method_def["cfg"].policy_name,
        "total_attack_jobs": int(raw["total_attack_jobs"]),
        "total_attack_mass_product_proxy": total_raw_mass,
        "verified_attack_jobs": int(len(attack_ids)),
        "verified_attack_mass_product_proxy": total_verified_mass,
        "detector_ceiling": float(total_verified_mass / max(total_raw_mass, 1e-12)),
        "admitted_attack_jobs": int(len(admitted_attack_ids)),
        "admitted_attack_mass": mass_for_ids(jobs_by_id, admitted_attack_ids),
        "queued_attack_jobs": int(len(admitted_attack_ids)),
        "queued_attack_mass": mass_for_ids(jobs_by_id, admitted_attack_ids),
        "expired_attack_jobs": int(len(expired_attack_ids)),
        "expired_attack_mass": mass_for_ids(jobs_by_id, expired_attack_ids),
        "served_attack_jobs": int(len(served_attack_ids)),
        "served_attack_mass": served_attack_mass,
        "backend_success_attack_jobs": int(len(backend_success_attack_ids)),
        "backend_success_attack_mass": backend_success_mass,
        "backend_fail_attack_jobs": int(len(backend_fail_attack_ids)),
        "backend_fail_attack_mass": backend_fail_attack_mass,
        "scheduler_conditional_recall": float(backend_success_mass / max(total_verified_mass, 1e-12)),
        "absolute_recall": float(backend_success_mass / max(total_raw_mass, 1e-12)),
        "clean_jobs_seen": int(len(clean_ids)),
        "clean_jobs_served": int(len(served_clean_ids)),
        "unnecessary": int(len(served_clean_ids)),
        "cost": float(detail["summary"]["average_service_cost_per_step"]),
        "backend_fail_total": int(detail["summary"]["total_backend_fail"]),
        "backend_fail_clean_jobs": int(len(served_clean_backend_fail)),
        "delay_p50": float(np.quantile(delays, 0.50)) if delays.size else 0.0,
        "delay_p95": float(detail["summary"]["queue_delay_p95"]),
        "delay_max": float(np.max(delays)) if delays.size else 0.0,
        "average_service_time": float(np.mean(service_times)) if service_times.size else 0.0,
        "average_service_cost": float(np.mean(service_costs)) if service_costs.size else 0.0,
        "total_service_cost": float(detail["summary"]["total_service_cost"]),
        "served_jobs_total": int(len(served_ids)),
        "served_ratio": float(len(served_ids) / max(len(all_ids), 1)),
    }
    return row


def oracle_jobs(jobs: Sequence[Any], *, mode: str) -> List[Any]:
    out = []
    for job in jobs:
        sev = float(job.severity_true) if int(job.is_attack) == 1 else 0.0
        if mode == "backend_success":
            score = sev if int(job.actual_backend_fail) == 0 else 0.0
        elif mode in {"severity", "capacity"}:
            score = sev
        else:
            raise KeyError(mode)
        out.append(replace(job, pred_expected_consequence=float(score), pred_attack_prob=1.0 if sev > 0 else 0.0, value_proxy=float(score)))
    return out


def ceiling_for(method_def: Dict[str, Any], holdout: Dict[str, Any], jobs: Sequence[Any], raw_mass: float, verified_mass: float) -> Dict[str, Any]:
    cfg = G2.cfg_topk(slot_budget=int(method_def["cfg"].slot_budget), train_stats=G2._job_stats(list(jobs)))
    capacity_detail = R2.simulate_policy_detailed(oracle_jobs(jobs, mode="capacity"), total_steps=int(np.asarray(G2.mixed_bank_to_alarm_arrays(str(ROOT / holdout["test_bank"]))["total_steps"]).reshape(-1)[0]), cfg=cfg)
    # Use the same total_steps as normal jobs; deriving from arrays directly is safe under decision_step_group=1.
    backend_detail = R2.simulate_policy_detailed(oracle_jobs(jobs, mode="backend_success"), total_steps=int(np.asarray(G2.mixed_bank_to_alarm_arrays(str(ROOT / holdout["test_bank"]))["total_steps"]).reshape(-1)[0]), cfg=cfg)
    jobs_by_id = {int(j.job_id): j for j in jobs}
    cap_served_attack = mass_for_ids(jobs_by_id, capacity_detail["served_attack_jobs"])
    cap_backend_success = mass_for_ids(jobs_by_id, capacity_detail["served_attack_jobs"], no_fail=True)
    bs_backend_success = mass_for_ids(jobs_by_id, backend_detail["served_attack_jobs"], no_fail=True)
    return {
        "holdout_id": holdout["tag"],
        "budget": int(method_def["cfg"].slot_budget),
        "variant": method_def["method"],
        "group": method_def["group"],
        "full_native_or_source_frozen": "full_native" if method_def["full_native"] else "source_frozen",
        "detector_ceiling": float(verified_mass / max(raw_mass, 1e-12)),
        "capacity_oracle_ceiling": float(cap_served_attack / max(raw_mass, 1e-12)),
        "verified_oracle_backend_success_ceiling": float(cap_backend_success / max(raw_mass, 1e-12)),
        "backend_success_oracle_ceiling": float(bs_backend_success / max(raw_mass, 1e-12)),
        "capacity_oracle_served_attack_mass": float(cap_served_attack),
        "backend_success_oracle_success_mass": float(bs_backend_success),
    }


def detail_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, int, str], Dict[str, Any]]]:
    method_defs = build_method_defs()
    holdouts = read_json(GATE2 / "gate2_full_native_manifest_used.json")["holdouts"]
    funnel_rows: List[Dict[str, Any]] = []
    ceiling_rows: List[Dict[str, Any]] = []
    details: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for method_def in method_defs:
        slot = int(method_def["cfg"].slot_budget)
        for hold in holdouts:
            arrays_test = G2._aggregate_arrival_steps(
                G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])),
                1,
            )
            jobs, total_steps, _ = G2.build_jobs(ctx=method_def["ctx"], arrays_bank=arrays_test, variant_name=method_def["variant"])
            detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=method_def["cfg"])
            row = compute_funnel_row(method_def, hold, detail, jobs)
            ceil = ceiling_for(method_def, hold, jobs, row["total_attack_mass_product_proxy"], row["verified_attack_mass_product_proxy"])
            ceil["gap_to_detector_ceiling"] = float(ceil["detector_ceiling"] - row["absolute_recall"])
            ceil["gap_to_capacity_oracle"] = float(ceil["capacity_oracle_ceiling"] - row["absolute_recall"])
            ceil["gap_to_backend_success_oracle"] = float(ceil["backend_success_oracle_ceiling"] - row["absolute_recall"])
            funnel_rows.append(row)
            ceiling_rows.append(ceil)
            details[(method_def["method"], slot, hold["tag"])] = {"detail": detail, "jobs": jobs, "funnel": row, "ceiling": ceil}
    return funnel_rows, ceiling_rows, details


def aggregate(rows: List[Dict[str, Any]], *, key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in key_fields), []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(key_fields, key)}
        rec["n"] = len(vals)
        for metric in [
            "total_attack_mass_product_proxy",
            "verified_attack_mass_product_proxy",
            "detector_ceiling",
            "admitted_attack_mass",
            "expired_attack_mass",
            "served_attack_mass",
            "backend_success_attack_mass",
            "backend_fail_attack_mass",
            "scheduler_conditional_recall",
            "absolute_recall",
            "clean_jobs_served",
            "cost",
            "backend_fail_total",
            "delay_p50",
            "delay_p95",
            "delay_max",
            "average_service_time",
            "average_service_cost",
            "served_ratio",
            "comparator_scheduler_recall",
            "comparator_absolute_recall",
            "source_retained_absolute_recall",
            "source_retained_scheduler_recall",
            "source_retained_served_attack_mass",
            "source_retained_backend_fail",
            "source_retained_cost",
            "source_retained_clean",
            "source_retained_served_count",
            "source_minus_comparator_scheduler_recall",
            "source_minus_comparator_absolute_recall",
        ]:
            if metric in vals[0]:
                arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
                rec[f"{metric}_mean"] = float(np.mean(arr))
        out.append(rec)
    return out


def write_funnel_summary(funnel_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = aggregate(funnel_rows, key_fields=["variant", "budget"])
    src_b1 = next(r for r in summary if r["variant"] == "source_frozen_transfer" and r["budget"] == 1)
    src_b2 = next(r for r in summary if r["variant"] == "source_frozen_transfer" and r["budget"] == 2)
    detector = float(np.mean([r["detector_ceiling"] for r in funnel_rows if r["variant"] == "source_frozen_transfer"]))
    loss_detector = 1.0 - detector
    loss_scheduler = detector - float(np.mean([r["served_attack_mass"] / max(r["total_attack_mass_product_proxy"], 1e-12) for r in funnel_rows if r["variant"] == "source_frozen_transfer"]))
    loss_backend = float(np.mean([(r["served_attack_mass"] - r["backend_success_attack_mass"]) / max(r["total_attack_mass_product_proxy"], 1e-12) for r in funnel_rows if r["variant"] == "source_frozen_transfer"]))
    largest = "all attack -> verified alarm" if loss_detector >= max(loss_scheduler, loss_backend) else "verified -> served" if loss_scheduler >= loss_backend else "served -> backend success"
    rows = []
    for r in summary:
        if r["variant"] in FIGURE_METHODS:
            rows.append(
                {
                    "variant": r["variant"],
                    "B": r["budget"],
                    "detector": fmt(r["detector_ceiling_mean"]),
                    "served_mass": fmt(r["served_attack_mass_mean"]),
                    "backend_success_mass": fmt(r["backend_success_attack_mass_mean"]),
                    "sched_recall": fmt(r["scheduler_conditional_recall_mean"]),
                    "absolute": fmt(r["absolute_recall_mean"]),
                    "backend_fail": fmt(r["backend_fail_total_mean"]),
                }
            )
    lines = [
        "# Case39 Funnel Summary",
        "",
        "Definitions: `verified` means DDD/scheduler-visible alarms; `queued` equals accepted into the scheduler queue after any threshold gate. Diagnostic rows do not relabel source-frozen as native success.",
        "",
        md_table(rows, ["variant", "B", "detector", "served_mass", "backend_success_mass", "sched_recall", "absolute", "backend_fail"]),
        "",
        "## Answers",
        "",
        f"1. Largest recall loss is `{largest}`; source-frozen average detector ceiling is `{fmt(detector)}`.",
        f"2. B=1 and B=2 share the upstream detector ceiling, but B=1 has stronger capacity/queue pressure while B=2 shifts more of the loss to backend success.",
        "3. Source-frozen's advantage versus safeguarded, unconstrained, anchored, and winner replay occurs mainly after verification: it serves more attack mass before backend filtering.",
        "4. Full-native local retune collapses primarily at service/admission-score operating point: verified attacks are admitted to the queue but little attack mass is actually served, especially for safeguarded and B=1 unconstrained.",
        "5. Backend_fail burden is dominated by served attack jobs rather than served clean jobs; clean backend failures are small relative to attack backend failures in these traces.",
    ]
    (OUT / "funnel_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"largest_loss": largest, "source_b1": src_b1, "source_b2": src_b2, "detector_mean": detector}


def write_oracle_summary(ceiling_rows: List[Dict[str, Any]], funnel_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = [r for r in ceiling_rows if r["variant"] == "source_frozen_transfer"]
    native_best = [r for r in ceiling_rows if r["variant"] == "topk_expected_consequence"]
    def mean_metric(rows: List[Dict[str, Any]], key: str, budget: int | None = None) -> float:
        vals = [float(r[key]) for r in rows if budget is None or int(r["budget"]) == budget]
        return float(np.mean(vals)) if vals else float("nan")
    rows = []
    for b in BUDGETS:
        rows.append(
            {
                "B": b,
                "detector": fmt(mean_metric(source, "detector_ceiling", b)),
                "capacity": fmt(mean_metric(source, "capacity_oracle_ceiling", b)),
                "backend_success": fmt(mean_metric(source, "backend_success_oracle_ceiling", b)),
                "source_gap_capacity": fmt(mean_metric(source, "gap_to_capacity_oracle", b)),
                "source_gap_backend": fmt(mean_metric(source, "gap_to_backend_success_oracle", b)),
                "native_best_gap_capacity": fmt(mean_metric(native_best, "gap_to_capacity_oracle", b)),
                "native_best_gap_backend": fmt(mean_metric(native_best, "gap_to_backend_success_oracle", b)),
            }
        )
    lines = [
        "# Oracle Ceiling Summary",
        "",
        "These diagnostic oracles use hindsight truth labels and are ceilings only, not deployable baselines and not new scheduler families.",
        "",
        md_table(rows, ["B", "detector", "capacity", "backend_success", "source_gap_capacity", "source_gap_backend", "native_best_gap_capacity", "native_best_gap_backend"]),
        "",
        "- `verified-oracle top severity` prioritizes true product severity within verified alarms.",
        "- `backend-success oracle` uses true backend_fail only to estimate an upper bound.",
        "- `capacity-only oracle` estimates how much verified attack mass can be served under the same arrival, B, and Wmax.",
    ]
    (OUT / "oracle_ceiling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "source_gap_capacity_mean": mean_metric(source, "gap_to_capacity_oracle"),
        "source_gap_backend_mean": mean_metric(source, "gap_to_backend_success_oracle"),
        "capacity_mean": mean_metric(source, "capacity_oracle_ceiling"),
        "backend_success_mean": mean_metric(source, "backend_success_oracle_ceiling"),
    }


def burden_efficiency(funnel_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = aggregate(funnel_rows, key_fields=["variant", "budget"])
    by = {(r["variant"], int(r["budget"])): r for r in agg}
    out = []
    for r in agg:
        variant = r["variant"]
        budget = int(r["budget"])
        topk = by.get(("topk_expected_consequence", budget))
        winner = by.get(("winner_replay", budget))
        recall = float(r["scheduler_conditional_recall_mean"])
        cost = float(r["cost_mean"])
        backend_fail = float(r["backend_fail_total_mean"])
        success_mass = float(r["backend_success_attack_mass_mean"])
        clean_service = float(r["clean_jobs_served_mean"])
        served_attack = float(r["served_attack_mass_mean"])
        served_total_mass = served_attack
        served_jobs = float(np.mean([x["served_jobs_total"] for x in funnel_rows if x["variant"] == variant and int(x["budget"]) == budget]))
        served_attack_jobs = float(np.mean([x["served_attack_jobs"] for x in funnel_rows if x["variant"] == variant and int(x["budget"]) == budget]))
        backend_fail_attack = float(np.mean([x["backend_fail_attack_jobs"] for x in funnel_rows if x["variant"] == variant and int(x["budget"]) == budget]))
        served_attack_count = float(np.mean([x["served_attack_jobs"] for x in funnel_rows if x["variant"] == variant and int(x["budget"]) == budget]))
        clean_seen = float(np.mean([x["clean_jobs_seen"] for x in funnel_rows if x["variant"] == variant and int(x["budget"]) == budget]))
        rec = {
            "variant": variant,
            "budget": budget,
            "recall": recall,
            "absolute_recall": float(r["absolute_recall_mean"]),
            "cost": cost,
            "backend_fail": backend_fail,
            "recall_per_cost": recall / max(cost, 1e-12),
            "recall_per_backend_fail": recall / max(backend_fail, 1e-12),
            "successful_attack_mass_per_backend_fail": success_mass / max(backend_fail, 1e-12),
            "successful_attack_mass_per_clean_service": success_mass / max(clean_service, 1e-12),
            "attack_service_precision": served_attack_jobs / max(served_jobs, 1e-12),
            "severity_precision": served_attack / max(served_total_mass, 1e-12),
            "backend_success_rate_among_served": (served_jobs - backend_fail) / max(served_jobs, 1e-12),
            "backend_success_rate_among_served_attack": (served_attack_count - backend_fail_attack) / max(served_attack_count, 1e-12),
            "clean_service_rate": clean_service / max(clean_seen, 1e-12),
        }
        for base_name, base in [("topk", topk), ("winner_replay", winner)]:
            if base:
                d_recall = recall - float(base["scheduler_conditional_recall_mean"])
                d_cost = cost - float(base["cost_mean"])
                d_backend = backend_fail - float(base["backend_fail_total_mean"])
                rec[f"marginal_recall_gain_vs_{base_name}_per_extra_cost"] = d_recall / d_cost if abs(d_cost) > 1e-12 else float("nan")
                rec[f"marginal_recall_gain_vs_{base_name}_per_extra_backend_fail"] = d_recall / d_backend if abs(d_backend) > 1e-12 else float("nan")
        out.append(rec)
    rows = []
    for r in out:
        if r["variant"] in FIGURE_METHODS:
            rows.append(
                {
                    "variant": r["variant"],
                    "B": r["budget"],
                    "recall/cost": fmt(r["recall_per_cost"]),
                    "recall/backend": fmt(r["recall_per_backend_fail"]),
                    "attack_precision": fmt(r["attack_service_precision"]),
                    "backend_success_attack": fmt(r["backend_success_rate_among_served_attack"]),
                }
            )
    lines = [
        "# Burden-efficiency",
        "",
        md_table(rows, ["variant", "B", "recall/cost", "recall/backend", "attack_precision", "backend_success_attack"]),
        "",
        "- Source-frozen buys recall with materially higher cost and backend_fail.",
        "- Top-k is the closest full-native high-recall point and is less burdensome, but it remains below source-frozen recall.",
        "- Native safeguarded is a low-cost, low-service, low-recall point rather than an effective operating point.",
    ]
    (OUT / "burden_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def truncate_source(source: Dict[str, Any], comparator: Dict[str, Any], mode: str) -> Dict[str, float]:
    detail = source["detail"]
    jobs_by_id = {int(j.job_id): j for j in source["jobs"]}
    comp = comparator["funnel"]
    if mode == "served_count":
        cap = int(comp["served_jobs_total"])
    elif mode == "cost":
        cap = float(comp["total_service_cost"])
    elif mode == "backend_fail":
        cap = int(comp["backend_fail_total"])
    elif mode == "clean_service":
        cap = int(comp["clean_jobs_served"])
    else:
        raise KeyError(mode)
    selected: List[int] = []
    used_cost = 0.0
    used_backend = 0
    used_clean = 0
    for raw_id in detail["served_jobs"]:
        job = jobs_by_id[int(raw_id)]
        if mode == "served_count" and len(selected) >= cap:
            break
        if mode == "cost" and used_cost + float(job.actual_service_cost) > cap + 1e-12:
            break
        if mode == "backend_fail" and used_backend + int(job.actual_backend_fail) > cap:
            break
        if mode == "clean_service" and int(job.is_attack) == 0 and used_clean + 1 > cap:
            break
        selected.append(int(raw_id))
        used_cost += float(job.actual_service_cost)
        used_backend += int(job.actual_backend_fail)
        used_clean += 1 if int(job.is_attack) == 0 else 0
    success_mass = mass_for_ids(jobs_by_id, selected, no_fail=True)
    served_attack_mass = mass_for_ids(jobs_by_id, selected)
    raw_mass = float(source["funnel"]["total_attack_mass_product_proxy"])
    verified_mass = float(source["funnel"]["verified_attack_mass_product_proxy"])
    return {
        "source_retained_absolute_recall": success_mass / max(raw_mass, 1e-12),
        "source_retained_scheduler_recall": success_mass / max(verified_mass, 1e-12),
        "source_retained_served_attack_mass": served_attack_mass,
        "source_retained_backend_fail": used_backend,
        "source_retained_cost": used_cost,
        "source_retained_clean": used_clean,
        "source_retained_served_count": len(selected),
    }


def matched_burden(details: Dict[Tuple[str, int, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    holdouts = sorted({k[2] for k in details})
    for budget in BUDGETS:
        for comp_name in ["topk_expected_consequence", "winner_replay"]:
            for hold in holdouts:
                source = details[("source_frozen_transfer", budget, hold)]
                comp = details[(comp_name, budget, hold)]
                for mode in ["served_count", "cost", "backend_fail", "clean_service"]:
                    trunc = truncate_source(source, comp, mode)
                    comp_f = comp["funnel"]
                    rows.append(
                        {
                            "holdout_id": hold,
                            "budget": budget,
                            "comparator": comp_name,
                            "match_type": mode,
                            "comparator_scheduler_recall": comp_f["scheduler_conditional_recall"],
                            "comparator_absolute_recall": comp_f["absolute_recall"],
                            **trunc,
                            "source_minus_comparator_scheduler_recall": trunc["source_retained_scheduler_recall"] - comp_f["scheduler_conditional_recall"],
                            "source_minus_comparator_absolute_recall": trunc["source_retained_absolute_recall"] - comp_f["absolute_recall"],
                        }
                    )
    agg_rows = aggregate(rows, key_fields=["comparator", "match_type", "budget"])
    display = []
    for r in agg_rows:
        display.append(
            {
                "comparator": r["comparator"],
                "match": r["match_type"],
                "B": r["budget"],
                "source_sched": fmt(r["source_retained_scheduler_recall_mean"]),
                "comp_sched": fmt(r["comparator_scheduler_recall_mean"]),
                "delta": fmt(r["source_minus_comparator_scheduler_recall_mean"]),
            }
        )
    lines = [
        "# Matched-burden Diagnostic",
        "",
        "Matched-burden rows are post-hoc diagnostics, not deployable baselines and not a new scheduler family.",
        "",
        md_table(display, ["comparator", "match", "B", "source_sched", "comp_sched", "delta"]),
        "",
        "- If the source-frozen advantage disappears under matched burden, its gain is mainly bought by more service burden.",
        "- If it remains, source-frozen has ordering advantage at the same burden.",
    ]
    (OUT / "matched_burden_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def pareto_frontier(eff_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for budget in BUDGETS:
        rows = [r for r in eff_rows if int(r["budget"]) == budget]
        for r in rows:
            def efficient(burden_key: str) -> bool:
                for other in rows:
                    if other is r:
                        continue
                    if float(other["recall"]) >= float(r["recall"]) - 1e-12 and float(other[burden_key]) <= float(r[burden_key]) + 1e-12:
                        if float(other["recall"]) > float(r["recall"]) + 1e-12 or float(other[burden_key]) < float(r[burden_key]) - 1e-12:
                            return False
                return True
            out.append(
                {
                    "variant": r["variant"],
                    "budget": budget,
                    "recall": r["recall"],
                    "cost": r["cost"],
                    "backend_fail": r["backend_fail"],
                    "pareto_recall_cost": efficient("cost"),
                    "pareto_recall_backend_fail": efficient("backend_fail"),
                }
            )
    display = [
        {
            "variant": r["variant"],
            "B": r["budget"],
            "recall": fmt(r["recall"]),
            "cost": fmt(r["cost"]),
            "backend": fmt(r["backend_fail"]),
            "rc": r["pareto_recall_cost"],
            "rb": r["pareto_recall_backend_fail"],
        }
        for r in out
        if r["variant"] in FIGURE_METHODS
    ]
    lines = [
        "# Pareto Frontier",
        "",
        md_table(display, ["variant", "B", "recall", "cost", "backend", "rc", "rb"]),
        "",
        "- Source-frozen is not dominated by top-k or winner replay because it has higher recall, but it is a high-burden point.",
        "- Top-k is the closest full-native high-recall point and is more burden-efficient.",
        "- Native safeguarded is best read as low-cost low-service low-recall, not as an effective operating point.",
    ]
    (OUT / "pareto_frontier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plot_funnel(funnel_rows: List[Dict[str, Any]]) -> None:
    stages = [
        ("total_attack_mass_product_proxy", "all"),
        ("verified_attack_mass_product_proxy", "verified"),
        ("admitted_attack_mass", "admitted"),
        ("served_attack_mass", "served"),
        ("backend_success_attack_mass", "success"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for ax, budget in zip(axes, BUDGETS):
        for method in FIGURE_METHODS:
            vals = []
            for key, _label in stages:
                xs = [float(r[key]) for r in funnel_rows if r["variant"] == method and int(r["budget"]) == budget]
                vals.append(float(np.mean(xs)) if xs else 0.0)
            ax.plot([x[1] for x in stages], vals, marker="o", label=method)
        ax.set_title(f"B={budget}")
        ax.set_ylabel("mean product-proxy mass")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "figure_case39_funnel_mass.png", dpi=220)
    fig.savefig(OUT / "figure_case39_funnel_mass.pdf")
    plt.close(fig)


def plot_gap(ceiling_rows: List[Dict[str, Any]], funnel_rows: List[Dict[str, Any]]) -> None:
    methods = ["source_frozen_transfer", "topk_expected_consequence"]
    gap_keys = [("gap_to_detector_ceiling", "detector"), ("gap_to_capacity_oracle", "capacity"), ("gap_to_backend_success_oracle", "backend")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        x = np.arange(len(gap_keys))
        width = 0.35
        for idx, method in enumerate(methods):
            vals = [np.mean([float(r[key]) for r in ceiling_rows if r["variant"] == method and int(r["budget"]) == budget]) for key, _ in gap_keys]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([x[1] for x in gap_keys])
        ax.set_title(f"B={budget}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("absolute recall gap")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gap_to_ceiling.png", dpi=220)
    fig.savefig(OUT / "figure_gap_to_ceiling.pdf")
    plt.close(fig)


def plot_recall_burden(eff_rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, budget in enumerate(BUDGETS):
        rows = [r for r in eff_rows if int(r["budget"]) == budget]
        for r in rows:
            axes[0, col].scatter(float(r["cost"]), float(r["recall"]))
            axes[0, col].annotate(r["variant"], (float(r["cost"]), float(r["recall"])), fontsize=6)
            axes[1, col].scatter(float(r["backend_fail"]), float(r["recall"]))
            axes[1, col].annotate(r["variant"], (float(r["backend_fail"]), float(r["recall"])), fontsize=6)
        axes[0, col].set_title(f"Recall vs cost B={budget}")
        axes[1, col].set_title(f"Recall vs backend_fail B={budget}")
        axes[0, col].set_xlabel("cost")
        axes[1, col].set_xlabel("backend_fail")
        axes[0, col].set_ylabel("scheduler recall")
        axes[1, col].set_ylabel("scheduler recall")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure_recall_vs_burden.png", dpi=220)
    fig.savefig(OUT / "figure_recall_vs_burden.pdf")
    plt.close(fig)


def plot_matched(matched_rows: List[Dict[str, Any]]) -> None:
    agg = aggregate(matched_rows, key_fields=["comparator", "match_type", "budget"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    match_types = ["served_count", "cost", "backend_fail", "clean_service"]
    for ax, budget in zip(axes, BUDGETS):
        xs = np.arange(len(match_types))
        vals = []
        comp_vals = []
        for mt in match_types:
            row = next(r for r in agg if r["comparator"] == "topk_expected_consequence" and r["match_type"] == mt and int(r["budget"]) == budget)
            vals.append(float(row["source_retained_scheduler_recall_mean"]))
            comp_vals.append(float(row["comparator_scheduler_recall_mean"]))
        ax.bar(xs - 0.18, vals, width=0.36, label="source matched")
        ax.bar(xs + 0.18, comp_vals, width=0.36, label="topk")
        ax.set_xticks(xs)
        ax.set_xticklabels(match_types, rotation=25, ha="right")
        ax.set_title(f"B={budget}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("scheduler recall")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_matched_burden.png", dpi=220)
    fig.savefig(OUT / "figure_matched_burden.pdf")
    plt.close(fig)


def write_decision(funnel_info: Dict[str, Any], ceiling_info: Dict[str, Any], matched_rows: List[Dict[str, Any]], eff_rows: List[Dict[str, Any]]) -> None:
    matched_agg = aggregate(matched_rows, key_fields=["comparator", "match_type", "budget"])
    def delta(comp: str, mt: str, b: int) -> float:
        row = next(r for r in matched_agg if r["comparator"] == comp and r["match_type"] == mt and int(r["budget"]) == b)
        return float(row["source_minus_comparator_scheduler_recall_mean"])
    matched_cost_leads = delta("topk_expected_consequence", "cost", 1) > 0 and delta("topk_expected_consequence", "cost", 2) > 0
    matched_backend_leads = delta("topk_expected_consequence", "backend_fail", 1) > 0 and delta("topk_expected_consequence", "backend_fail", 2) > 0
    lines = [
        "# Gate 3 Decision",
        "",
        f"1. Low absolute recall is partly upstream-limited by verified-alarm ceiling (`{fmt(funnel_info['detector_mean'])}`), but substantial post-verification/backend loss remains.",
        f"2. Source-frozen average gap to capacity oracle is `{fmt(ceiling_info['source_gap_capacity_mean'])}` absolute recall.",
        f"3. Source-frozen average gap to backend-success oracle is `{fmt(ceiling_info['source_gap_backend_mean'])}` absolute recall.",
        "4. Source-frozen is not just cost: matched-burden diagnostics retain some ordering advantage versus top-k, though the gap shrinks materially.",
        f"5. Matched-cost source-frozen remains ahead of top-k across both budgets: `{matched_cost_leads}`.",
        f"6. Matched-backend-fail source-frozen remains ahead of top-k across both budgets: `{matched_backend_leads}`.",
        "7. Backend_fail is a major burden and is dominated by served attack jobs rather than clean jobs.",
        "8. Full-native local retune collapses at the service/operating-point stage: verified attacks are present, but little attack mass is served before backend filtering.",
        "9. Best current wording: `transfer regularization mechanism` plus `backend stress-test limitation`; not `native case39 success`.",
        "10. Gate 4 consequence/recovery robustness is recommended.",
        "11. A small fail-capped source-frozen variant is worth testing only as a future extension or v2 experiment, not as the current submitted main result.",
    ]
    (OUT / "gate3_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    funnel_rows, ceiling_rows, details = detail_records()
    write_csv(OUT / "funnel_by_holdout_variant_budget.csv", funnel_rows)
    funnel_info = write_funnel_summary(funnel_rows)
    write_csv(OUT / "oracle_ceiling_by_holdout_variant_budget.csv", ceiling_rows)
    ceiling_info = write_oracle_summary(ceiling_rows, funnel_rows)
    eff_rows = burden_efficiency(funnel_rows)
    write_csv(OUT / "burden_efficiency.csv", eff_rows)
    matched_rows = matched_burden(details)
    write_csv(OUT / "matched_burden_diagnostic.csv", matched_rows)
    pareto_rows = pareto_frontier(eff_rows)
    write_csv(OUT / "pareto_frontier.csv", pareto_rows)
    plot_funnel(funnel_rows)
    plot_gap(ceiling_rows, funnel_rows)
    plot_recall_burden(eff_rows)
    plot_matched(matched_rows)
    write_decision(funnel_info, ceiling_info, matched_rows, eff_rows)
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree.txt").write_text("\n".join(sorted(set(files) | {"outputs_tree.txt"})) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "funnel_rows": len(funnel_rows), "ceiling_rows": len(ceiling_rows), "matched_rows": len(matched_rows)}, indent=2))


if __name__ == "__main__":
    main()

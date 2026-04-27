from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPRINT = ROOT / "metric/case39/q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate9_fail_only_guard_20260426_224426"
G5_SCRIPT = SPRINT / "gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py"
G8_SCRIPT = SPRINT / "gate8_v2_hardening_20260426_101517/gate8_static_and_sim.py"
RELEASE = SPRINT / "release_candidate"
ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
BUDGETS = [1, 2]
EPS = 1e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G5 = load_module("gate5_transfer_burden_guard", G5_SCRIPT)
G8 = load_module("gate8_static_and_sim", G8_SCRIPT)
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
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def fail_stats(jobs: Sequence[Any]) -> Dict[str, float]:
    fail = np.asarray([float(j.pred_fail_prob) for j in jobs], dtype=float)
    return {
        "fail_mean": float(np.mean(fail)) if fail.size else 0.0,
        "fail_std": float(np.std(fail) if fail.size and np.std(fail) > 1e-12 else 1.0),
    }


def apply_fail_guard(jobs: Sequence[Any], *, cfg: Any, stats: Dict[str, float], alpha: float) -> List[Any]:
    if abs(float(alpha)) <= 1e-12:
        return list(jobs)
    mean_ec = max(float(cfg.mean_pred_expected_consequence), EPS)
    v_weight = max(abs(float(cfg.v_weight)), EPS)
    scale = mean_ec / v_weight
    out = []
    for job in jobs:
        fail_z = (float(job.pred_fail_prob) - stats["fail_mean"]) / stats["fail_std"]
        guarded_ec = float(job.pred_expected_consequence) - float(alpha) * fail_z * scale
        meta = dict(job.meta)
        meta["gate9_guard"] = "fail_only"
        meta["gate9_alpha"] = float(alpha)
        meta["gate9_actual_backend_fail_used_in_decision"] = 0.0
        meta["gate9_actual_recover_fail_used_in_decision"] = 0.0
        out.append(replace(job, pred_expected_consequence=float(guarded_ec), meta=meta))
    return out


def run_fail_guard(jobs: Sequence[Any], *, total_steps: int, cfg: Any, stats: Dict[str, float], alpha: float):
    guarded = apply_fail_guard(jobs, cfg=cfg, stats=stats, alpha=alpha)
    detail = R2.simulate_policy_detailed(guarded, total_steps=total_steps, cfg=cfg)
    return detail, guarded


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    metrics = [
        "recall",
        "scheduler_recall",
        "backend_fail",
        "cost",
        "unnecessary",
        "recover_fail",
        "served_ratio",
        "served_attack_mass",
        "backend_success_attack_mass",
        "clean_service",
        "served_clean_count",
        "delay_p50",
        "delay_p95",
        "average_service_time",
        "average_service_cost",
    ]
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(vals)
        for metric in metrics:
            if metric in vals[0]:
                arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
                rec[metric] = float(np.mean(arr))
                rec[f"{metric}_median"] = float(np.median(arr))
        out.append(rec)
    return out


def static_best_name(tune: Dict[str, Any]) -> str:
    candidates = ["threshold_verify_fifo", "threshold_ddd_fifo", "threshold_expected_consequence_fifo"]

    def score(name: str) -> float:
        payload = tune.get(name, {})
        if isinstance(payload, dict):
            for key in ("objective", "best_objective"):
                if key in payload:
                    return float(payload[key])
            summary = payload.get("summary", {})
            if isinstance(summary, dict):
                return float(summary.get("unnecessary_mtd_count", 0.0))
        return 0.0

    return min(candidates, key=score)


def select_alpha_from_case14_dev() -> Dict[str, Any]:
    manifest, _holdouts = G8.case14_holdouts()
    args = G8.phase3_args(manifest, manifest["val_bank"])
    ctx = G8.prepare_phase3_context(args)
    stats = fail_stats(ctx["jobs_train"])
    rows: List[Dict[str, Any]] = []
    base_by_budget: Dict[int, Dict[str, Any]] = {}
    for budget in BUDGETS:
        cfgs, _tune, jobs_map = G8.tune_cfgs(ctx, budget)
        cfg = cfgs["proposed_ca_vq_hard"]
        base_detail = R2.simulate_policy_detailed(jobs_map["default"], total_steps=ctx["total_steps_eval"], cfg=cfg)
        base = G5.summarize_detail(base_detail, jobs_map["default"], method="original_proposed_dev", budget=budget, holdout_id="case14_source_val")
        base["alpha"] = 0.0
        base_by_budget[budget] = base
        for alpha in ALPHAS:
            detail, guarded = run_fail_guard(jobs_map["default"], total_steps=ctx["total_steps_eval"], cfg=cfg, stats=stats, alpha=alpha)
            row = G5.summarize_detail(
                detail,
                guarded,
                method="TRFG-source-dev",
                budget=budget,
                holdout_id="case14_source_val",
                calibration_mode="source",
                alpha=alpha,
            )
            row["baseline_recall"] = base["recall"]
            row["recall_retention_vs_original_proposed"] = float(row["recall"]) / max(float(base["recall"]), EPS)
            row["unnecessary_limit"] = float(base["unnecessary"]) * 1.05
            row["cost_limit"] = float(base["cost"]) * 1.05
            row["selection_rule_passed_budget"] = bool(
                row["recall_retention_vs_original_proposed"] >= 0.98
                and float(row["unnecessary"]) <= float(row["unnecessary_limit"]) + 1e-12
                and float(row["cost"]) <= float(row["cost_limit"]) + 1e-12
            )
            rows.append(row)
    write_csv(OUT / "gate9_case14_calibration_grid.csv", rows)
    by_alpha: List[Dict[str, Any]] = []
    for alpha in ALPHAS:
        vals = [r for r in rows if abs(float(r["alpha"]) - alpha) < 1e-12]
        by_alpha.append(
            {
                "alpha": alpha,
                "passed_all_budgets": all(bool(r["selection_rule_passed_budget"]) for r in vals),
                "avg_recall_retention": float(np.mean([float(r["recall_retention_vs_original_proposed"]) for r in vals])),
                "avg_backend_fail": float(np.mean([float(r["backend_fail"]) for r in vals])),
                "avg_cost": float(np.mean([float(r["cost"]) for r in vals])),
                "avg_unnecessary": float(np.mean([float(r["unnecessary"]) for r in vals])),
            }
        )
    passed = [x for x in by_alpha if x["passed_all_budgets"]]
    failed_source_dev = not bool(passed)
    if passed:
        selected = sorted(passed, key=lambda x: (x["avg_backend_fail"], x["avg_cost"], x["avg_unnecessary"], x["alpha"]))[0]
    else:
        selected = next(x for x in by_alpha if abs(float(x["alpha"])) <= 1e-12)
    payload = {
        "selection_split": "case14 source val",
        "alpha_grid": ALPHAS,
        "selected_alpha": float(selected["alpha"]),
        "fail_guard_failed_source_dev": failed_source_dev,
        "selection_rule": "pass all B budgets on recall>=0.98, unnecessary<=+5%, cost<=+5%, then lowest backend_fail/cost/unnecessary/alpha",
        "alpha0_exact_reproduces_original_proposed": alpha0_exact(rows, base_by_budget),
        "by_alpha": by_alpha,
        "normalization": stats,
        "no_test_selection": True,
    }
    write_json(OUT / "gate9_case14_selected_alpha.json", payload)
    return payload


def alpha0_exact(rows: List[Dict[str, Any]], base_by_budget: Dict[int, Dict[str, Any]]) -> bool:
    for budget in BUDGETS:
        row = next(r for r in rows if int(r["B"]) == budget and abs(float(r["alpha"])) <= 1e-12)
        base = base_by_budget[budget]
        for metric in ["recall", "backend_fail", "cost", "unnecessary", "served_ratio"]:
            if abs(float(row[metric]) - float(base[metric])) > 1e-10:
                return False
    return True


def select_native_fail_alpha() -> Dict[str, Any]:
    by_source, _by_topk, _by_winner = G5.source_method_defs()
    sets = G5.calibration_sets(by_source)
    native = sets["TRBG-native-burden"]
    source_def_by_budget = by_source
    stats = fail_stats(native["train_jobs"])
    rows: List[Dict[str, Any]] = []
    base_by_budget: Dict[int, Dict[str, Any]] = {}
    for budget in BUDGETS:
        cfg = source_def_by_budget[budget]["cfg"]
        detail = R2.simulate_policy_detailed(native["dev_jobs"], total_steps=native["dev_steps"], cfg=cfg)
        base = G5.summarize_detail(detail, native["dev_jobs"], method="native_fail_source_alpha0_dev", budget=budget, holdout_id="native_case39_val")
        base_by_budget[budget] = base
        for alpha in ALPHAS:
            detail, guarded = run_fail_guard(native["dev_jobs"], total_steps=native["dev_steps"], cfg=cfg, stats=stats, alpha=alpha)
            row = G5.summarize_detail(
                detail,
                guarded,
                method="TRFG-native-fail-dev",
                budget=budget,
                holdout_id="native_case39_val",
                calibration_mode="native-fail",
                alpha=alpha,
            )
            row["recall_retention_vs_alpha0"] = float(row["recall"]) / max(float(base["recall"]), EPS)
            row["selection_rule_passed_budget"] = bool(row["recall_retention_vs_alpha0"] >= 0.95)
            rows.append(row)
    by_alpha = []
    for alpha in ALPHAS:
        vals = [r for r in rows if abs(float(r["alpha"]) - alpha) < 1e-12]
        by_alpha.append(
            {
                "alpha": alpha,
                "passed_all_budgets": all(bool(r["selection_rule_passed_budget"]) for r in vals),
                "avg_recall_retention": float(np.mean([float(r["recall_retention_vs_alpha0"]) for r in vals])),
                "avg_backend_fail": float(np.mean([float(r["backend_fail"]) for r in vals])),
                "avg_cost": float(np.mean([float(r["cost"]) for r in vals])),
                "avg_unnecessary": float(np.mean([float(r["unnecessary"]) for r in vals])),
            }
        )
    passed = [x for x in by_alpha if x["passed_all_budgets"]]
    selected = sorted(passed or by_alpha, key=lambda x: (not x["passed_all_budgets"], x["avg_backend_fail"], x["avg_cost"], x["avg_unnecessary"], x["alpha"]))[0]
    return {
        "selection_split": "explicit native case39 val",
        "selected_alpha": float(selected["alpha"]),
        "failed_native_dev": not bool(passed),
        "selection_rule": "diagnostic only: recall retention >=0.95 vs alpha0 for all budgets, then lowest backend_fail/cost/unnecessary/alpha",
        "by_alpha": by_alpha,
        "normalization": stats,
        "no_test_selection": True,
    }


def run_case14_confirm(selected_alpha: float) -> Dict[str, Any]:
    manifest, holdouts = G8.case14_holdouts()
    rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    for hold in holdouts:
        args = G8.phase3_args(manifest, hold["test_bank"])
        path_rows.append(
            {
                "holdout_id": hold["tag"],
                "test_bank": hold["test_bank"],
                "resolved_test_bank": args.eval_bank,
                "test_bank_source": args.path_sources["test_bank"],
            }
        )
        ctx = G8.prepare_phase3_context(args)
        stats = fail_stats(ctx["jobs_train"])
        full_stats = G5.guard_stats(ctx["jobs_train"])
        for budget in BUDGETS:
            cfgs, tune, jobs_map = G8.tune_cfgs(ctx, budget)
            static_best = static_best_name(tune)
            base_methods = [
                ("incumbent_queue_aware", "adaptive_threshold_verify_fifo", jobs_map["default"]),
                ("original_proposed_safeguarded", "proposed_ca_vq_hard", jobs_map["default"]),
                ("source_frozen_transfer_alpha0_equivalent", "proposed_ca_vq_hard", jobs_map["default"]),
                ("topk_expected_consequence", "topk_expected_consequence", jobs_map["default"]),
                ("best_static_threshold", static_best, jobs_map["ddd"] if static_best == "threshold_ddd_fifo" else jobs_map["default"]),
            ]
            for method, cfg_name, jobs in base_methods:
                detail = R2.simulate_policy_detailed(jobs, total_steps=ctx["total_steps_eval"], cfg=cfgs[cfg_name])
                row = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=hold["tag"])
                row["source_policy_name"] = cfg_name
                row["test_bank_source"] = args.path_sources["test_bank"]
                rows.append(row)
            cfg = cfgs["proposed_ca_vq_hard"]
            detail, guarded = run_fail_guard(jobs_map["default"], total_steps=ctx["total_steps_eval"], cfg=cfg, stats=stats, alpha=selected_alpha)
            row = G5.summarize_detail(detail, guarded, method="TRFG-source", budget=budget, holdout_id=hold["tag"], calibration_mode="source", alpha=selected_alpha)
            row["source_policy_name"] = "proposed_ca_vq_hard"
            row["test_bank_source"] = args.path_sources["test_bank"]
            rows.append(row)
            guarded_full = G8.apply_component_guard(jobs_map["default"], cfg=cfg, stats=full_stats, alpha=1.0, component="full_trbg")
            detail = R2.simulate_policy_detailed(guarded_full, total_steps=ctx["total_steps_eval"], cfg=cfg)
            row = G5.summarize_detail(detail, guarded_full, method="TRBG-source", budget=budget, holdout_id=hold["tag"], calibration_mode="source", alpha=1.0)
            row["source_policy_name"] = "proposed_ca_vq_hard"
            row["test_bank_source"] = args.path_sources["test_bank"]
            rows.append(row)
    write_csv(OUT / "gate9_case14_confirm_by_holdout.csv", rows)
    write_csv(OUT / "gate9_case14_input_path_resolution.csv", path_rows)
    summary = aggregate(rows, ["method", "B"])
    write_csv(OUT / "gate9_case14_confirm_summary.csv", summary)
    return write_case14_decision(summary, selected_alpha)


def write_case14_decision(summary: List[Dict[str, Any]], selected_alpha: float) -> Dict[str, Any]:
    by = {(r["method"], int(r["B"])): r for r in summary}
    per_budget = []
    for budget in BUDGETS:
        trfg = by[("TRFG-source", budget)]
        prop = by[("original_proposed_safeguarded", budget)]
        trbg = by[("TRBG-source", budget)]
        per_budget.append(
            {
                "B": budget,
                "recall_retention_vs_original": float(trfg["recall"]) / max(float(prop["recall"]), EPS),
                "unnecessary_delta_vs_original": float(trfg["unnecessary"]) - float(prop["unnecessary"]),
                "cost_delta_vs_original": float(trfg["cost"]) - float(prop["cost"]),
                "backend_fail_delta_vs_original": float(trfg["backend_fail"]) - float(prop["backend_fail"]),
                "unnecessary_delta_vs_TRBG": float(trfg["unnecessary"]) - float(trbg["unnecessary"]),
            }
        )
    avg_ret = float(np.mean([r["recall_retention_vs_original"] for r in per_budget]))
    unnecessary_ok = all(
        float(by[("TRFG-source", b)]["unnecessary"]) <= float(by[("original_proposed_safeguarded", b)]["unnecessary"]) * 1.05 + 1e-12
        for b in BUDGETS
    )
    cost_ok = all(
        float(by[("TRFG-source", b)]["cost"]) <= float(by[("original_proposed_safeguarded", b)]["cost"]) * 1.05 + 1e-12
        for b in BUDGETS
    )
    pass_compat = bool(avg_ret >= 0.98 and unnecessary_ok and cost_ok)
    lines = [
        "# Gate9 Case14 Compatibility Stats",
        "",
        f"- Selected TRFG-source alpha: `{selected_alpha}`.",
        f"- Compatibility passed: `{pass_compat}`.",
        f"- Average recall retention vs original proposed: `{fmt(avg_ret)}`.",
        f"- Unnecessary within +5% for all budgets: `{unnecessary_ok}`.",
        f"- Cost within +5% for all budgets: `{cost_ok}`.",
        "",
        md_table(
            [
                {
                    "B": r["B"],
                    "recall_retention": fmt(r["recall_retention_vs_original"]),
                    "unnecessary_delta": fmt(r["unnecessary_delta_vs_original"]),
                    "cost_delta": fmt(r["cost_delta_vs_original"]),
                    "backend_delta": fmt(r["backend_fail_delta_vs_original"]),
                    "unnecessary_delta_vs_TRBG": fmt(r["unnecessary_delta_vs_TRBG"]),
                }
                for r in per_budget
            ],
            ["B", "recall_retention", "unnecessary_delta", "cost_delta", "backend_delta", "unnecessary_delta_vs_TRBG"],
        ),
    ]
    (OUT / "gate9_case14_compatibility_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = [
        "# Gate9 Case14 Decision",
        "",
        f"- TRFG-source compatibility status: `{'pass' if pass_compat else 'fail'}`.",
        f"- Recall retention >=98%: `{avg_ret >= 0.98}`.",
        f"- Unnecessary no longer explodes relative to original proposed: `{unnecessary_ok}`.",
        f"- Cost/backend_fail direction: cost_ok=`{cost_ok}`; backend deltas are in the stats table.",
        f"- Paper-wide method upgrade allowed by 14-bus compatibility: `{pass_compat}`.",
        f"- Recommended framing: `{'paper-wide method upgrade candidate' if pass_compat else 'case39 scale-up extension or appendix diagnostic'}`.",
    ]
    (OUT / "gate9_case14_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    return {"pass_compat": pass_compat, "avg_recall_retention": avg_ret, "per_budget": per_budget}


def method_defs() -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for d in G5.G3.build_method_defs():
        out[(d["method"], int(d["cfg"].slot_budget))] = d
    return out


def run_case39_original8_secondary(source_sel: Dict[str, Any], native_sel: Dict[str, Any]) -> Dict[str, Any]:
    defs = method_defs()
    by_source, _by_topk, _by_winner = G5.source_method_defs()
    source_sets = G5.calibration_sets(by_source)
    _rows_by_mode, _selected, full_stats_by_mode = G5.calibration_grid(by_source)
    source_stats = fail_stats(source_sets["TRBG-source"]["train_jobs"])
    native_stats = fail_stats(source_sets["TRBG-native-burden"]["train_jobs"])
    holdouts = read_json(SPRINT / "gate2_full_native_20260424_100642/gate2_source_frozen_transfer_manifest_used.json")["holdouts"]
    rows: List[Dict[str, Any]] = []
    for hold in holdouts:
        arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
        for budget in BUDGETS:
            for method in [
                "source_frozen_transfer",
                "topk_expected_consequence",
                "winner_replay",
                "native_safeguarded_retune",
                "native_unconstrained_retune",
            ]:
                d = defs[(method, budget)]
                jobs, total_steps, _ = G5.build_jobs_from_arrays(d["ctx"], arrays, d["variant"])
                detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=d["cfg"])
                row = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=hold["tag"])
                row["diagnostic_scope"] = "secondary_original8_only"
                rows.append(row)
            src = by_source[budget]
            jobs, total_steps, _ = G5.build_jobs_from_arrays(src["ctx"], arrays, src["variant"])
            detail, guarded, _cap = G5.run_guard(jobs, total_steps=total_steps, cfg=src["cfg"], stats=full_stats_by_mode["TRBG-source"], alpha=1.0, fail_cap_quantile=1.0)
            row = G5.summarize_detail(detail, guarded, method="TRBG-source", budget=budget, holdout_id=hold["tag"], calibration_mode="source", alpha=1.0, fail_cap_quantile=1.0)
            row["diagnostic_scope"] = "secondary_original8_only"
            rows.append(row)
            for method, stats, alpha in [
                ("TRFG-source", source_stats, float(source_sel["selected_alpha"])),
                ("TRFG-native-fail", native_stats, float(native_sel["selected_alpha"])),
            ]:
                detail, guarded = run_fail_guard(jobs, total_steps=total_steps, cfg=src["cfg"], stats=stats, alpha=alpha)
                row = G5.summarize_detail(detail, guarded, method=method, budget=budget, holdout_id=hold["tag"], calibration_mode=method, alpha=alpha)
                row["diagnostic_scope"] = "secondary_original8_only"
                rows.append(row)
    write_csv(OUT / "gate9_case39_original8_secondary_by_holdout.csv", rows)
    summary = aggregate(rows, ["method", "B"])
    write_csv(OUT / "gate9_case39_original8_secondary_summary.csv", summary)
    write_original8_reports(summary)
    return {"rows": len(rows), "summary_rows": len(summary)}


def write_original8_reports(summary: List[Dict[str, Any]]) -> None:
    lines = [
        "# Gate9 Case39 Original 8-Holdout Secondary Stats",
        "",
        "This is secondary diagnostic only, not primary confirmatory evidence, because fail-only was motivated by Gate8 diagnostics.",
        "",
        md_table(
            [
                {
                    "method": r["method"],
                    "B": r["B"],
                    "recall": fmt(r["recall"]),
                    "backend_fail": fmt(r["backend_fail"]),
                    "cost": fmt(r["cost"]),
                    "unnecessary": fmt(r["unnecessary"]),
                    "recover_fail": fmt(r["recover_fail"]),
                }
                for r in summary
            ],
            ["method", "B", "recall", "backend_fail", "cost", "unnecessary", "recover_fail"],
        ),
    ]
    (OUT / "gate9_case39_original8_secondary_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pareto = pareto_rows(summary)
    lines = [
        "# Gate9 Case39 Original 8-Holdout Secondary Pareto",
        "",
        "Secondary diagnostic only; not used for method replacement.",
        "",
        md_table(pareto, ["method", "B", "recall_cost_efficient", "recall_backend_efficient", "dominated_by"]),
    ]
    (OUT / "gate9_case39_original8_secondary_pareto.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pareto_rows(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in summary:
        dominated_cost = []
        dominated_backend = []
        for q in summary:
            if int(q["B"]) != int(r["B"]) or q["method"] == r["method"]:
                continue
            if float(q["recall"]) >= float(r["recall"]) - 1e-12 and float(q["cost"]) <= float(r["cost"]) + 1e-12 and (
                float(q["recall"]) > float(r["recall"]) + 1e-12 or float(q["cost"]) < float(r["cost"]) - 1e-12
            ):
                dominated_cost.append(q["method"])
            if float(q["recall"]) >= float(r["recall"]) - 1e-12 and float(q["backend_fail"]) <= float(r["backend_fail"]) + 1e-12 and (
                float(q["recall"]) > float(r["recall"]) + 1e-12 or float(q["backend_fail"]) < float(r["backend_fail"]) - 1e-12
            ):
                dominated_backend.append(q["method"])
        rows.append(
            {
                "method": r["method"],
                "B": r["B"],
                "recall_cost_efficient": not dominated_cost,
                "recall_backend_efficient": not dominated_backend,
                "dominated_by": ";".join(sorted(set(dominated_cost + dominated_backend))),
            }
        )
    return rows


def physical_sanity_audit() -> None:
    sample_paths = [
        SPRINT / "gate8_v2_hardening_20260426_101517/reduced_fresh/mixed_bank_test_gate8_reduced_interleaved_0.npy",
        ROOT / "metric/case39/phase3_confirm_blind_v1/banks/mixed_bank_test_cfA_frontloaded_0_seed20260511_off1020.npy",
    ]
    available: Dict[str, Any] = {}
    for path in sample_paths:
        if not path.exists():
            continue
        obj = np.load(path, allow_pickle=True).item()
        available[str(path)] = sorted(obj.keys())
    desired = [
        "voltage magnitude deviation",
        "phase angle deviation norm",
        "state-estimation error norm",
        "branch-flow deviation or overload count",
        "measurement residual change",
        "backend_success/backend_fail/recover_fail",
        "service_cost/service_time",
    ]
    rows = []
    keys_flat = " ".join(" ".join(v) for v in available.values())
    checks = {
        "voltage_dev_proxy": any(x in keys_flat for x in ["voltage_dev", "v_mag", "voltage_magnitude_deviation"]),
        "angle_dev_proxy": any(x in keys_flat for x in ["angle_dev", "phase_angle", "vang", "ang_no_summary"]),
        "state_error_proxy": any(x in keys_flat for x in ["verify_score", "state_error", "recover"]),
        "branch_flow_dev_proxy": any(x in keys_flat for x in ["branch_flow", "overload", "flow_dev"]),
        "measurement_residual_proxy": "ddd_loss_recons" in keys_flat,
        "backend_recovery_service_fields": all(x in keys_flat for x in ["backend_fail", "recover_fail", "stage_one_time", "delta_cost_one"]),
    }
    for name, ok in checks.items():
        rows.append({"proxy": name, "available": ok, "interpretation": "limited sanity only" if ok else "not available without new OPF/flow logging"})
    write_csv(OUT / "gate9_physical_sanity_by_method.csv", rows)
    write_json(OUT / "gate9_physical_sanity_field_inventory.json", available)
    enough = bool(checks["state_error_proxy"] and checks["measurement_residual_proxy"])
    summary = [
        "# Gate9 Physical Sanity Summary",
        "",
        "- Full OPF/branch-flow physical consequence labels are not available in the existing banks.",
        f"- Limited state/residual sanity fields available: `{enough}`.",
        "- Voltage magnitude and branch-flow deviation proxies should be reported as not available unless new OPF/flow logging is added.",
        "- Therefore this can support at most a physical-deviation sanity appendix, not a physical-consequence-aware scheduler claim.",
    ]
    (OUT / "gate9_physical_sanity_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    corr = [
        "# Gate9 Physical Proxy Rank Correlation",
        "",
        "Composite OPF/flow-based physical-deviation proxy is not available from the current fields, so rank correlation against product_proxy is not computed.",
        "",
        "- product_proxy vs physical-deviation proxy direction consistency: `not_available_without_new_OPF_or_flow_logging`.",
        "- TRFG/TRBG/source-frozen ordering under physical-deviation proxy: `not_available`.",
        "- Main-text physical sanity check: `not_ready`; appendix field audit only.",
    ]
    (OUT / "gate9_physical_proxy_rank_correlation.md").write_text("\n".join(corr) + "\n", encoding="utf-8")


def release_candidate() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    for d in ["case39_transfer", "case39_native", "case39_q1_sprint"]:
        (RELEASE / d).mkdir(parents=True, exist_ok=True)
    transfer_manifest = read_json(SPRINT / "source_frozen_transfer_manifest.json")
    native_manifest = read_json(SPRINT / "full_native_case39_manifest.json")
    write_json(RELEASE / "MANIFEST_TRANSFER.json", transfer_manifest)
    write_json(RELEASE / "MANIFEST_FULL_NATIVE.json", native_manifest)
    # Classify canonical case39 fit/eval references.
    refs = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".json"}:
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "metric/case39/mixed_bank_fit.npy" in txt or "metric/case39/mixed_bank_eval.npy" in txt:
            rel = str(path.relative_to(ROOT))
            if "q1_top_sprint" in rel or "audit" in rel or "round" in rel:
                cat = "safe_historical"
            elif rel.endswith(".md") or rel.endswith(".json"):
                cat = "docs_only"
            else:
                cat = "must_patch"
            refs.append({"path": rel, "category": cat})
    write_csv(RELEASE / "canonical_case39_fit_eval_reference_classification.csv", refs)
    readme = [
        "# Release Candidate README",
        "",
        "This release candidate separates case39 transfer evidence, full-native case39 evidence, and Q1 sprint audit artifacts.",
        "",
        "- `case39_transfer/`: source-frozen transfer means case14 train/val -> case39 target.",
        "- `case39_native/`: full-native means explicit native case39 train/val.",
        "- `case39_q1_sprint/`: Gate0-Gate9 audit, mechanism, fresh sanity, and rewrite evidence.",
        "- Ambiguous canonical `metric/case39/mixed_bank_fit.npy` / `mixed_bank_eval.npy` symlinks are not placed in this release candidate.",
        "- Gate6 is recombined stress replication.",
        "- Gate7/Gate8/Gate9 fresh outputs are sanity evidence, not full 8-bank statistical validation unless enough banks accumulate.",
    ]
    (RELEASE / "README_RELEASE.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    invalid = [
        "# Invalid / Caution Artifacts",
        "",
        "- Old pre-fix attack-side summaries are invalid or caution-only unless explicitly regenerated after the path/provenance fixes.",
        "- Any artifact using ambiguous canonical case39 fit/eval paths must be treated as suspect until classified and patched.",
    ]
    (RELEASE / "INVALID_ARTIFACTS.md").write_text("\n".join(invalid) + "\n", encoding="utf-8")
    repro = [
        "# Reproduce Gate Summary",
        "",
        "- Gate1-Gate4: provenance, consistency, funnel, proxy robustness.",
        "- Gate5: TRBG-source internal moderate success.",
        "- Gate6: locked recombined stress replication.",
        "- Gate7: one-bank fresh physical-solver sanity check.",
        "- Gate8: 14-bus compatibility diagnostic, component ablation, four reduced fresh sanity banks.",
        "- Gate9: pre-registered fail-only guard validation.",
    ]
    (RELEASE / "REPRODUCE_GATE_SUMMARY.md").write_text("\n".join(repro) + "\n", encoding="utf-8")
    # Lightweight placeholders so no ambiguous symlinks are introduced.
    for sub, text in [
        ("case39_transfer/README.md", "Source-frozen case14 train/val -> case39 target transfer evidence.\n"),
        ("case39_native/README.md", "Explicit native case39 train/val evidence; no canonical symlink ambiguity.\n"),
        ("case39_q1_sprint/README.md", "Gate0-Gate9 sprint outputs by timestamp; historical artifacts retain caveats.\n"),
    ]:
        (RELEASE / sub).write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    source_sel = select_alpha_from_case14_dev()
    native_sel = select_native_fail_alpha()
    selected = {
        "TRFG-source": source_sel,
        "TRFG-native-fail": native_sel,
        "strict_no_test_selection": True,
        "gate8_confirm_not_used_as_selector": True,
    }
    write_json(OUT / "gate9_selected_alphas.json", selected)
    case14 = run_case14_confirm(float(source_sel["selected_alpha"]))
    original8 = run_case39_original8_secondary(source_sel, native_sel)
    physical_sanity_audit()
    release_candidate()
    write_json(OUT / "gate9_static_results.json", {"case14": case14, "original8": original8, "selected": selected})


if __name__ == "__main__":
    main()


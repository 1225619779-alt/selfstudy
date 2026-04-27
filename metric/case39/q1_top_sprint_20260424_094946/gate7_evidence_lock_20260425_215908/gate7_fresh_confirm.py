from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPRINT = ROOT / "metric/case39/q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate7_evidence_lock_20260425_215908"
G5_SCRIPT = SPRINT / "gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py"
BANK = OUT / "fresh_banks/mixed_bank_test_fresh_fullsolver_540_seed20260711_off1500.npy"
PARTIAL = OUT / "partials/mixed_bank_test_fresh_fullsolver_540_seed20260711_off1500.resume.partial.npy"
RUNTIME = OUT / "logs/runtime_resume_steps.jsonl"
G6C_RUNTIME = SPRINT / "gate6c_checkpoint_fullsolver_20260425_0820/logs/runtime_steps.jsonl"
G6C_INPUT = OUT / "partials/gate6c_seed20260711_step218_input.partial.npy"

METHODS = {
    "source_frozen_transfer",
    "topk_expected_consequence",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
}
BUDGETS = [1, 2]
EPS = 1e-12


def load_gate5():
    spec = importlib.util.spec_from_file_location("gate5_transfer_burden_guard", G5_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {G5_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G5 = load_gate5()
G2 = G5.G2
R2 = G5.R2


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


def bank_summary(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True).item()
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": obj.get("status"),
        "completed_steps": int(obj.get("completed_steps", obj.get("summary", {}).get("total_steps", 0))),
        "total_steps_requested": int(obj.get("total_steps_requested", obj.get("summary", {}).get("total_steps", 0))),
        "case_name": obj.get("case_name"),
        "model_path": obj.get("model_path"),
        "summary": obj.get("summary", {}),
    }


def runtime_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source, path in [("gate6c_initial", G6C_RUNTIME), ("gate7_resume", RUNTIME)]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["runtime_source"] = source
            rows.append(row)
    return rows


def summarize_runtime() -> Dict[str, Any]:
    rows = runtime_rows()
    if not rows:
        return {}
    elapsed = np.asarray([float(r["elapsed_step_seconds"]) for r in rows], dtype=float)
    return {
        "runtime_rows": len(rows),
        "completed_steps": int(max(int(r["step"]) for r in rows)),
        "elapsed_total_seconds_combined_sum": float(np.sum(elapsed)),
        "step_seconds_mean": float(np.mean(elapsed)),
        "step_seconds_median": float(np.median(elapsed)),
        "step_seconds_p95": float(np.quantile(elapsed, 0.95)),
        "step_seconds_max": float(np.max(elapsed)),
        "ddd_alarm": int(sum(int(r["ddd_alarm"]) for r in rows)),
        "trigger_after_gate": int(sum(int(r["trigger_after_gate"]) for r in rows)),
        "backend_fail": int(sum(int(r["backend_fail"]) for r in rows)),
        "recover_fail": int(sum(int(r["recover_fail"]) for r in rows)),
    }


def method_defs() -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for d in G5.G3.build_method_defs():
        if d["method"] in METHODS:
            out[(d["method"], int(d["cfg"].slot_budget))] = d
    return out


def block_for_step(step: int) -> Tuple[int, str]:
    blocks = [
        (0, 119, "strong3_front"),
        (120, 179, "clean_after_strong3"),
        (180, 269, "strong2_mid"),
        (270, 359, "clean_mid"),
        (360, 419, "weak1_late"),
        (420, 539, "clean_tail"),
    ]
    for i, (lo, hi, name) in enumerate(blocks):
        if lo <= int(step) <= hi:
            return i, name
    return -1, "unknown"


def block_rows(method: str, budget: int, detail: Dict[str, Any], jobs: Sequence[Any]) -> List[Dict[str, Any]]:
    by_id = {int(j.job_id): j for j in jobs}
    served = set(int(x) for x in detail["served_jobs"])
    rows = []
    for block_id, block_name in [(i, n) for i, n in enumerate(["strong3_front", "clean_after_strong3", "strong2_mid", "clean_mid", "weak1_late", "clean_tail"])]:
        ids = [int(j.job_id) for j in jobs if block_for_step(int(j.arrival_step))[0] == block_id]
        attack_ids = [i for i in ids if int(by_id[i].is_attack) == 1]
        clean_ids = [i for i in ids if int(by_id[i].is_attack) == 0]
        served_ids = [i for i in ids if i in served]
        served_attack = [i for i in attack_ids if i in served]
        served_clean = [i for i in clean_ids if i in served]
        total_mass = float(sum(float(by_id[i].severity_true) for i in attack_ids))
        served_mass = float(sum(float(by_id[i].severity_true) for i in served_attack))
        success_mass = float(sum(float(by_id[i].severity_true) for i in served_attack if int(by_id[i].actual_backend_fail) == 0))
        rows.append(
            {
                "method": method,
                "B": budget,
                "block_id": block_id,
                "block_name": block_name,
                "jobs": len(ids),
                "attack_jobs": len(attack_ids),
                "clean_jobs": len(clean_ids),
                "served_jobs": len(served_ids),
                "served_attack_jobs": len(served_attack),
                "served_clean_jobs": len(served_clean),
                "total_attack_mass": total_mass,
                "served_attack_mass": served_mass,
                "backend_success_attack_mass": success_mass,
                "block_recall": success_mass / max(total_mass, EPS),
                "backend_fail": int(sum(int(by_id[i].actual_backend_fail) for i in served_ids)),
                "recover_fail": int(sum(int(float(by_id[i].meta.get("recover_fail", 0.0))) for i in served_ids)),
            }
        )
    return rows


def main() -> None:
    if not BANK.exists():
        raise SystemExit(f"Fresh fullsolver bank not found: {BANK}")
    bank = bank_summary(BANK)
    manifest = {
        "interpretation": "one fresh physical-solver sanity check, not 8-bank statistical validation",
        "bank": bank,
        "resume_input": str(G6C_INPUT),
        "runtime_summary": summarize_runtime(),
        "fixed_trbg_source": {"alpha": 1.0, "fail_cap_quantile": 1.0, "calibration_mode": "source"},
    }
    (OUT / "fresh_fullsolver_bank_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(BANK)), 1)
    defs = method_defs()
    by_source, _by_topk, _by_winner = G5.source_method_defs()
    _rows_by_mode, selected, stats_by_mode = G5.calibration_grid(by_source)
    summary_rows: List[Dict[str, Any]] = []
    block_out: List[Dict[str, Any]] = []
    details: Dict[Tuple[str, int], Dict[str, Any]] = {}
    jobs_by_method: Dict[Tuple[str, int], Sequence[Any]] = {}

    for budget in BUDGETS:
        for method in sorted(METHODS):
            d = defs[(method, budget)]
            jobs, total_steps, _ = G5.build_jobs_from_arrays(d["ctx"], arrays, d["variant"])
            detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=d["cfg"])
            rec = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id="fresh_fullsolver_1bank")
            rec["fresh_bank"] = "fresh_fullsolver_540_seed20260711_off1500"
            rec["validation_scope"] = "one_bank_sanity_check"
            summary_rows.append(rec)
            details[(method, budget)] = detail
            jobs_by_method[(method, budget)] = jobs
            block_out.extend(block_rows(method, budget, detail, jobs))

        source_def = by_source[budget]
        jobs, total_steps, _ = G5.build_jobs_from_arrays(source_def["ctx"], arrays, source_def["variant"])
        detail, guarded, _cap = G5.run_guard(
            jobs,
            total_steps=total_steps,
            cfg=source_def["cfg"],
            stats=stats_by_mode["TRBG-source"],
            alpha=1.0,
            fail_cap_quantile=1.0,
        )
        rec = G5.summarize_detail(
            detail,
            guarded,
            method="TRBG-source",
            budget=budget,
            holdout_id="fresh_fullsolver_1bank",
            calibration_mode="TRBG-source",
            alpha=1.0,
            fail_cap_quantile=1.0,
        )
        rec["fresh_bank"] = "fresh_fullsolver_540_seed20260711_off1500"
        rec["validation_scope"] = "one_bank_sanity_check"
        summary_rows.append(rec)
        details[("TRBG-source", budget)] = detail
        jobs_by_method[("TRBG-source", budget)] = guarded
        block_out.extend(block_rows("TRBG-source", budget, detail, guarded))

    write_csv(OUT / "fresh_fullsolver_confirm_summary.csv", summary_rows)
    write_csv(OUT / "fresh_fullsolver_confirm_by_step_or_block.csv", block_out)

    by = {(r["method"], int(r["B"])): r for r in summary_rows}
    comparison_rows = []
    for budget in BUDGETS:
        src = by[("source_frozen_transfer", budget)]
        trbg = by[("TRBG-source", budget)]
        topk = by[("topk_expected_consequence", budget)]
        comparison_rows.append(
            {
                "B": budget,
                "trbg_recall": trbg["scheduler_recall"],
                "source_recall": src["scheduler_recall"],
                "topk_recall": topk["scheduler_recall"],
                "recall_retention_vs_source": float(trbg["scheduler_recall"]) / max(float(src["scheduler_recall"]), EPS),
                "backend_fail_reduction_vs_source": (float(src["backend_fail"]) - float(trbg["backend_fail"])) / max(float(src["backend_fail"]), EPS),
                "cost_delta_vs_source": float(trbg["cost"]) - float(src["cost"]),
                "recover_delta_vs_source": float(trbg["recover_fail"]) - float(src["recover_fail"]),
                "trbg_above_topk_recall": float(trbg["scheduler_recall"]) > float(topk["scheduler_recall"]),
            }
        )
    write_csv(OUT / "fresh_fullsolver_paired_or_block_stats.csv", comparison_rows)
    avg_retention = float(np.mean([r["recall_retention_vs_source"] for r in comparison_rows]))
    avg_backend_reduction = float(np.mean([r["backend_fail_reduction_vs_source"] for r in comparison_rows]))
    avg_cost_delta = float(np.mean([r["cost_delta_vs_source"] for r in comparison_rows]))
    avg_recover_delta = float(np.mean([r["recover_delta_vs_source"] for r in comparison_rows]))
    lines = [
        "# Fresh Full-Solver Runtime Report",
        "",
        "This is one completed fresh physical-solver sanity bank, not an 8-bank statistical validation.",
        "",
        f"- completed steps: `{bank['completed_steps']}/{bank['total_steps_requested']}`",
        f"- case: `{bank['case_name']}`",
        f"- model: `{bank['model_path']}`",
        f"- runtime rows: `{manifest['runtime_summary'].get('runtime_rows')}`",
        f"- step seconds mean/median/p95/max: `{fmt(manifest['runtime_summary'].get('step_seconds_mean'))}` / `{fmt(manifest['runtime_summary'].get('step_seconds_median'))}` / `{fmt(manifest['runtime_summary'].get('step_seconds_p95'))}` / `{fmt(manifest['runtime_summary'].get('step_seconds_max'))}`",
    ]
    (OUT / "fresh_fullsolver_runtime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_lines = [
        "# Fresh Full-Solver Paired / Block Stats",
        "",
        "Only one fresh bank is available, so these are directional B=1/B=2 sanity comparisons rather than a holdout-level paired statistical test.",
        "",
        "| B | recall_retention_vs_source | backend_fail_reduction_vs_source | cost_delta_vs_source | TRBG above topk recall |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in comparison_rows:
        stats_lines.append(
            f"| {r['B']} | {fmt(r['recall_retention_vs_source'])} | {fmt(r['backend_fail_reduction_vs_source'])} | {fmt(r['cost_delta_vs_source'])} | {r['trbg_above_topk_recall']} |"
        )
    (OUT / "fresh_fullsolver_paired_or_block_stats.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    decision = [
        "# Fresh Full-Solver Decision",
        "",
        "- Scope: `one_bank_fresh_physical_solver_sanity_check`.",
        "- This is not 8-bank statistical validation.",
        f"- Average recall retention vs source-frozen: `{fmt(avg_retention)}`.",
        f"- Average backend_fail reduction vs source-frozen: `{fmt(avg_backend_reduction)}`.",
        f"- Average cost delta vs source-frozen: `{fmt(avg_cost_delta)}`.",
        f"- Average recover_fail delta vs source-frozen: `{fmt(avg_recover_delta)}`.",
        f"- TRBG-source above topk recall for all budgets: `{all(bool(r['trbg_above_topk_recall']) for r in comparison_rows)}`.",
        f"- Severe reverse result: `{avg_retention < 0.9 or avg_backend_reduction < -0.05}`.",
    ]
    (OUT / "fresh_fullsolver_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

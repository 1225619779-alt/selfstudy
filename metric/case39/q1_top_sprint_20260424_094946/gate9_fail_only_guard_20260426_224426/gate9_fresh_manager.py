from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
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
CHECKPOINT_SCRIPT = OUT / "checkpointed_evaluation_mixed_timeline.py"
G5_SCRIPT = SPRINT / "gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py"
PYTHON = Path("/home/pang/projects/DDET-MTD/.venv_rocm/bin/python")
BUDGETS = [1, 2]
EPS = 1e-12
CHECKPOINT_EVERY = 5
MAX_WALL_SECONDS_PER_BANK = 43200.0
TRFG_SOURCE_ALPHA = 0.25
TRFG_NATIVE_ALPHA = 2.0

BANKS = [
    {
        "bank_id": "gate9_sparse_front_0",
        "seed": 20260911,
        "offset": 2400,
        "schedule": "clean:100;att-1-0.12:15;clean:80;att-2-0.22:15;clean:30",
    },
    {
        "bank_id": "gate9_late_mixed_1",
        "seed": 20260912,
        "offset": 2480,
        "schedule": "clean:140;att-2-0.18:20;clean:40;att-3-0.28:20;clean:20",
    },
    {
        "bank_id": "gate9_alternating_short_2",
        "seed": 20260913,
        "offset": 2560,
        "schedule": "clean:60;att-1-0.10:10;clean:50;att-2-0.22:10;clean:50;att-3-0.32:10;clean:50",
    },
    {
        "bank_id": "gate9_dense_middle_3",
        "seed": 20260914,
        "offset": 2640,
        "schedule": "clean:90;att-1-0.12:15;att-2-0.18:15;clean:60;att-3-0.30:20;clean:40",
    },
]

BASE_METHODS = {
    "source_frozen_transfer",
    "topk_expected_consequence",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
}


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def bank_paths(bank: Dict[str, Any]) -> Dict[str, Path]:
    bid = bank["bank_id"]
    return {
        "output": OUT / "reduced_fresh" / f"mixed_bank_test_{bid}.npy",
        "partial": OUT / "partials" / f"mixed_bank_test_{bid}.partial.npy",
        "runtime": OUT / "logs" / f"runtime_{bid}.jsonl",
        "stdout": OUT / "logs" / f"{bid}.stdout.log",
        "stderr": OUT / "logs" / f"{bid}.stderr.log",
    }


def load_npy_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    obj = np.load(path, allow_pickle=True).item()
    return {
        "exists": True,
        "status": obj.get("status"),
        "completed_steps": int(obj.get("completed_steps", obj.get("summary", {}).get("total_steps", 0))),
        "total_steps_requested": int(obj.get("total_steps_requested", obj.get("summary", {}).get("total_steps", 0))),
        "case_name": obj.get("case_name"),
        "model_path": obj.get("model_path"),
        "summary": obj.get("summary", {}),
    }


def runtime_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"runtime_rows": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"runtime_rows": 0}
    elapsed = np.asarray([float(r["elapsed_step_seconds"]) for r in rows], dtype=float)
    return {
        "runtime_rows": len(rows),
        "last_step": int(max(int(r["step"]) for r in rows)),
        "step_seconds_mean": float(np.mean(elapsed)),
        "step_seconds_median": float(np.median(elapsed)),
        "step_seconds_p95": float(np.quantile(elapsed, 0.95)),
        "step_seconds_max": float(np.max(elapsed)),
        "backend_fail": int(sum(int(r.get("backend_fail", 0)) for r in rows)),
        "recover_fail": int(sum(int(r.get("recover_fail", 0)) for r in rows)),
    }


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
        out.append(replace(job, pred_expected_consequence=float(guarded_ec), meta=meta))
    return out


def run_fail_guard(jobs: Sequence[Any], *, total_steps: int, cfg: Any, stats: Dict[str, float], alpha: float):
    guarded = apply_fail_guard(jobs, cfg=cfg, stats=stats, alpha=alpha)
    detail = R2.simulate_policy_detailed(guarded, total_steps=total_steps, cfg=cfg)
    return detail, guarded


def method_defs() -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for d in G5.G3.build_method_defs():
        if d["method"] in BASE_METHODS:
            out[(d["method"], int(d["cfg"].slot_budget))] = d
    return out


def generate_bank(bank: Dict[str, Any]) -> Dict[str, Any]:
    paths = bank_paths(bank)
    out_summary = load_npy_summary(paths["output"])
    if out_summary.get("status") == "complete":
        return {**bank, "exit_code": 0, "skipped_existing_complete": True}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(CHECKPOINT_SCRIPT),
        "--schedule",
        str(bank["schedule"]),
        "--start_offset",
        str(bank["offset"]),
        "--seed_base",
        str(bank["seed"]),
        "--output",
        str(paths["output"]),
        "--partial_output",
        str(paths["partial"]),
        "--runtime_jsonl",
        str(paths["runtime"]),
        "--checkpoint_every",
        str(CHECKPOINT_EVERY),
        "--max_wall_seconds",
        str(MAX_WALL_SECONDS_PER_BANK),
    ]
    env = dict(os.environ)
    env["DDET_CASE_NAME"] = "case39"
    start = time.time()
    with paths["stdout"].open("a", encoding="utf-8") as out, paths["stderr"].open("a", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err)
    return {**bank, "exit_code": int(proc.returncode), "wall_seconds": float(time.time() - start), "cmd": " ".join(cmd)}


def collect_status(run_rows: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    by_id = {r["bank_id"]: r for r in (run_rows or [])}
    rows = []
    for bank in BANKS:
        paths = bank_paths(bank)
        out_summary = load_npy_summary(paths["output"])
        partial_summary = load_npy_summary(paths["partial"])
        rows.append(
            {
                **bank,
                "output": str(paths["output"]),
                "partial": str(paths["partial"]),
                "runtime_jsonl": str(paths["runtime"]),
                "stdout_log": str(paths["stdout"]),
                "stderr_log": str(paths["stderr"]),
                "exit_code": by_id.get(bank["bank_id"], {}).get("exit_code"),
                "complete": bool(out_summary.get("status") == "complete"),
                "output_completed_steps": out_summary.get("completed_steps", 0),
                "partial_status": partial_summary.get("status"),
                "partial_completed_steps": partial_summary.get("completed_steps", 0),
                "runtime": runtime_summary(paths["runtime"]),
            }
        )
    return rows


def write_manifest(status_rows: List[Dict[str, Any]]) -> None:
    write_json(
        OUT / "gate9_fresh_bank_manifest.json",
        {
            "scope": "Gate9 pre-registered reduced fresh physical-solver validation",
            "interpretation": "fresh sanity evidence, not full 8-bank statistical validation",
            "TRFG-source": {"alpha": TRFG_SOURCE_ALPHA, "calibration": "case14 source train/val"},
            "TRFG-native-fail": {"alpha": TRFG_NATIVE_ALPHA, "calibration": "explicit native case39 train/val", "diagnostic_only": True},
            "checkpoint_every": CHECKPOINT_EVERY,
            "serial_execution": True,
            "banks": status_rows,
        },
    )


def generate_banks() -> List[Dict[str, Any]]:
    run_rows = []
    for bank in BANKS:
        result = generate_bank(bank)
        run_rows.append(result)
        write_manifest(collect_status(run_rows))
    return collect_status(run_rows)


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    metrics = [
        "recall",
        "scheduler_recall",
        "backend_fail",
        "cost",
        "recover_fail",
        "unnecessary",
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


def confirm(status_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in status_rows if r["complete"]]
    defs = method_defs()
    by_source, _topk, _winner = G5.source_method_defs()
    source_sets = G5.calibration_sets(by_source)
    _rows_by_mode, _selected, full_stats = G5.calibration_grid(by_source)
    source_stats = fail_stats(source_sets["TRBG-source"]["train_jobs"])
    native_stats = fail_stats(source_sets["TRBG-native-burden"]["train_jobs"])
    rows: List[Dict[str, Any]] = []
    for bank in completed:
        arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(bank["output"]), 1)
        for budget in BUDGETS:
            for method in sorted(BASE_METHODS):
                d = defs[(method, budget)]
                jobs, total_steps, _ = G5.build_jobs_from_arrays(d["ctx"], arrays, d["variant"])
                detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=d["cfg"])
                rec = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=bank["bank_id"])
                rec["fresh_bank"] = bank["bank_id"]
                rec["validation_scope"] = "gate9_reduced_fresh_confirm"
                rows.append(rec)
            src = by_source[budget]
            jobs, total_steps, _ = G5.build_jobs_from_arrays(src["ctx"], arrays, src["variant"])
            detail, guarded, _cap = G5.run_guard(jobs, total_steps=total_steps, cfg=src["cfg"], stats=full_stats["TRBG-source"], alpha=1.0, fail_cap_quantile=1.0)
            rec = G5.summarize_detail(detail, guarded, method="TRBG-source", budget=budget, holdout_id=bank["bank_id"], calibration_mode="source", alpha=1.0, fail_cap_quantile=1.0)
            rec["fresh_bank"] = bank["bank_id"]
            rec["validation_scope"] = "gate9_reduced_fresh_confirm"
            rows.append(rec)
            for method, stats, alpha in [
                ("TRFG-source", source_stats, TRFG_SOURCE_ALPHA),
                ("TRFG-native-fail", native_stats, TRFG_NATIVE_ALPHA),
            ]:
                detail, guarded = run_fail_guard(jobs, total_steps=total_steps, cfg=src["cfg"], stats=stats, alpha=alpha)
                rec = G5.summarize_detail(detail, guarded, method=method, budget=budget, holdout_id=bank["bank_id"], calibration_mode=method, alpha=alpha)
                rec["fresh_bank"] = bank["bank_id"]
                rec["validation_scope"] = "gate9_reduced_fresh_confirm"
                rows.append(rec)
    write_csv(OUT / "gate9_fresh_confirm_by_bank_budget.csv", rows)
    summary = aggregate(rows, ["method", "B"]) if rows else []
    write_csv(OUT / "gate9_fresh_confirm_summary.csv", summary)
    return write_reports(status_rows, rows, summary)


def paired_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["method"], int(r["B"]), r["fresh_bank"]): r for r in rows}
    out = []
    banks = sorted({r["fresh_bank"] for r in rows})
    for bank in banks:
        for budget in BUDGETS:
            src = by.get(("source_frozen_transfer", budget, bank))
            trfg = by.get(("TRFG-source", budget, bank))
            trbg = by.get(("TRBG-source", budget, bank))
            topk = by.get(("topk_expected_consequence", budget, bank))
            win = by.get(("winner_replay", budget, bank))
            if not src or not trfg:
                continue
            out.append(
                {
                    "fresh_bank": bank,
                    "B": budget,
                    "recall_retention_vs_source": float(trfg["recall"]) / max(float(src["recall"]), EPS),
                    "backend_fail_reduction_vs_source": (float(src["backend_fail"]) - float(trfg["backend_fail"])) / max(float(src["backend_fail"]), EPS),
                    "cost_delta_vs_source": float(trfg["cost"]) - float(src["cost"]),
                    "recover_delta_vs_source": float(trfg["recover_fail"]) - float(src["recover_fail"]),
                    "unnecessary_delta_vs_source": float(trfg["unnecessary"]) - float(src["unnecessary"]),
                    "recall_delta_vs_topk": float(trfg["recall"]) - float(topk["recall"]) if topk else float("nan"),
                    "recall_delta_vs_TRBG": float(trfg["recall"]) - float(trbg["recall"]) if trbg else float("nan"),
                    "backend_fail_delta_vs_TRBG": float(trfg["backend_fail"]) - float(trbg["backend_fail"]) if trbg else float("nan"),
                    "cost_delta_vs_TRBG": float(trfg["cost"]) - float(trbg["cost"]) if trbg else float("nan"),
                    "recall_delta_vs_winner": float(trfg["recall"]) - float(win["recall"]) if win else float("nan"),
                }
            )
    return out


def pareto_flags(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def plot_frontier(summary: List[Dict[str, Any]], xkey: str, path_base: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in summary:
        marker = "o" if int(r["B"]) == 1 else "s"
        ax.scatter(float(r[xkey]), float(r["recall"]), marker=marker, s=55)
        ax.annotate(f"{r['method']}\nB={r['B']}", (float(r[xkey]), float(r["recall"])), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recall")
    ax.set_title(f"Gate9 fresh recall vs {xlabel}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def write_reports(status_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    paired = paired_rows(rows)
    write_csv(OUT / "gate9_trfg_vs_trbg_stats.csv", paired)
    completed = sum(1 for r in status_rows if r["complete"])
    avg_ret = float(np.mean([r["recall_retention_vs_source"] for r in paired])) if paired else float("nan")
    avg_backend_red = float(np.mean([r["backend_fail_reduction_vs_source"] for r in paired])) if paired else float("nan")
    avg_cost_delta = float(np.mean([r["cost_delta_vs_source"] for r in paired])) if paired else float("nan")
    avg_recover_delta = float(np.mean([r["recover_delta_vs_source"] for r in paired])) if paired else float("nan")
    avg_unnec_delta = float(np.mean([r["unnecessary_delta_vs_source"] for r in paired])) if paired else float("nan")
    avg_topk_delta = float(np.mean([r["recall_delta_vs_topk"] for r in paired])) if paired else float("nan")
    avg_trbg_recall_delta = float(np.mean([r["recall_delta_vs_TRBG"] for r in paired])) if paired else float("nan")
    avg_trbg_backend_delta = float(np.mean([r["backend_fail_delta_vs_TRBG"] for r in paired])) if paired else float("nan")
    pareto = pareto_flags(summary)
    write_csv(OUT / "gate9_fresh_pareto_frontier.csv", pareto)
    plot_frontier(summary, "backend_fail", OUT / "figure_gate9_fresh_recall_backend_frontier", "backend_fail")
    plot_frontier(summary, "cost", OUT / "figure_gate9_fresh_recall_cost_frontier", "cost")
    runtime = [
        "# Gate9 Fresh Runtime Report",
        "",
        "Reduced fresh physical-solver sanity check; not full 8-bank statistical validation.",
        "",
        f"- attempted banks: `{len(status_rows)}`",
        f"- completed banks: `{completed}`",
        f"- serial execution: `True`",
        "",
        "| bank_id | complete | completed_steps | exit_code | mean_step_sec | p95_step_sec | backend_fail | recover_fail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in status_rows:
        rt = r["runtime"]
        runtime.append(
            f"| {r['bank_id']} | {r['complete']} | {r['partial_completed_steps']} | {r['exit_code']} | {fmt(rt.get('step_seconds_mean'))} | {fmt(rt.get('step_seconds_p95'))} | {rt.get('backend_fail', 0)} | {rt.get('recover_fail', 0)} |"
        )
    (OUT / "gate9_fresh_runtime_report.md").write_text("\n".join(runtime) + "\n", encoding="utf-8")
    stats = [
        "# Gate9 Fresh Paired / Block Stats",
        "",
        "| fresh_bank | B | recall_retention_vs_source | backend_fail_reduction_vs_source | cost_delta_vs_source | recall_delta_vs_topk | recall_delta_vs_TRBG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in paired:
        stats.append(
            f"| {r['fresh_bank']} | {r['B']} | {fmt(r['recall_retention_vs_source'])} | {fmt(r['backend_fail_reduction_vs_source'])} | {fmt(r['cost_delta_vs_source'])} | {fmt(r['recall_delta_vs_topk'])} | {fmt(r['recall_delta_vs_TRBG'])} |"
        )
    (OUT / "gate9_fresh_paired_or_block_stats.md").write_text("\n".join(stats) + "\n", encoding="utf-8")
    pareto_md = [
        "# Gate9 Fresh Pareto Frontier",
        "",
        "| method | B | recall-cost efficient | recall-backend efficient | dominated_by |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for r in pareto:
        pareto_md.append(f"| {r['method']} | {r['B']} | {r['recall_cost_efficient']} | {r['recall_backend_efficient']} | {r['dominated_by']} |")
    (OUT / "gate9_fresh_pareto_frontier.md").write_text("\n".join(pareto_md) + "\n", encoding="utf-8")
    trfg_rows = [r for r in pareto if r["method"] == "TRFG-source"]
    trfg_not_dominated = all(bool(r["recall_cost_efficient"]) or bool(r["recall_backend_efficient"]) for r in trfg_rows)
    decision = {
        "completed_banks": completed,
        "avg_recall_retention_vs_source": avg_ret,
        "avg_backend_fail_reduction_vs_source": avg_backend_red,
        "avg_cost_delta_vs_source": avg_cost_delta,
        "avg_recover_delta_vs_source": avg_recover_delta,
        "avg_unnecessary_delta_vs_source": avg_unnec_delta,
        "avg_recall_delta_vs_topk": avg_topk_delta,
        "avg_recall_delta_vs_TRBG": avg_trbg_recall_delta,
        "avg_backend_fail_delta_vs_TRBG": avg_trbg_backend_delta,
        "trfg_not_pareto_dominated_in_at_least_one_plane_each_budget": trfg_not_dominated,
        "primary_success": bool(avg_ret >= 0.95 and avg_backend_red >= 0.08 and avg_cost_delta <= 1e-12 and avg_topk_delta >= -0.01 and trfg_not_dominated),
        "strong_success": bool(avg_ret >= 0.97 and avg_backend_red >= 0.10 and avg_cost_delta < -1e-12),
    }
    write_json(OUT / "gate9_fresh_results.json", decision)
    fresh_decision = [
        "# Gate9 Fresh Decision",
        "",
        f"- Completed reduced fresh banks: `{completed}/{len(status_rows)}`.",
        f"- Recall retention vs source: `{fmt(avg_ret)}`.",
        f"- Backend_fail reduction vs source: `{fmt(avg_backend_red)}`.",
        f"- Cost delta vs source: `{fmt(avg_cost_delta)}`.",
        f"- Recover_fail delta vs source: `{fmt(avg_recover_delta)}`.",
        f"- Unnecessary delta vs source: `{fmt(avg_unnec_delta)}`.",
        f"- Recall delta vs topk: `{fmt(avg_topk_delta)}`.",
        f"- Recall delta vs TRBG-source: `{fmt(avg_trbg_recall_delta)}`.",
        f"- Backend_fail delta vs TRBG-source: `{fmt(avg_trbg_backend_delta)}`.",
        f"- Primary success: `{decision['primary_success']}`.",
        f"- Strong success before 14-bus compatibility check: `{decision['strong_success']}`.",
    ]
    (OUT / "gate9_fresh_decision.md").write_text("\n".join(fresh_decision) + "\n", encoding="utf-8")
    vs = [
        "# Gate9 TRFG vs TRBG Summary",
        "",
        f"- Average recall delta TRFG-source minus TRBG-source: `{fmt(avg_trbg_recall_delta)}`.",
        f"- Average backend_fail delta TRFG-source minus TRBG-source: `{fmt(avg_trbg_backend_delta)}`.",
        "- Fail-only is compared under a pre-registered Gate9 alpha, so this is eligible for method replacement consideration.",
    ]
    (OUT / "gate9_trfg_vs_trbg_summary.md").write_text("\n".join(vs) + "\n", encoding="utf-8")
    (OUT / "gate9_trfg_vs_trbg_pareto.md").write_text("\n".join(pareto_md) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    for d in ["logs", "partials", "reduced_fresh", "figures"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    status = generate_banks()
    write_manifest(status)
    decision = confirm(status)
    write_json(OUT / "gate9_fresh_manager_done.json", decision)


if __name__ == "__main__":
    main()


from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPRINT = ROOT / "metric/case39/q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate8_v2_hardening_20260426_101517"
CHECKPOINT_SCRIPT = OUT / "checkpointed_evaluation_mixed_timeline.py"
G5_SCRIPT = SPRINT / "gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py"
PYTHON = Path("/home/pang/projects/DDET-MTD/.venv_rocm/bin/python")
MAX_WORKERS = 2
MAX_WALL_SECONDS_PER_BANK = 43200.0
CHECKPOINT_EVERY = 5
BUDGETS = [1, 2]
EPS = 1e-12

BANKS = [
    {
        "bank_id": "reduced_interleaved_0",
        "seed": 20260811,
        "offset": 2100,
        "schedule": "clean:80;att-1-0.15:20;clean:40;att-2-0.20:20;clean:40;att-3-0.30:20;clean:20",
    },
    {
        "bank_id": "reduced_tailheavy_1",
        "seed": 20260812,
        "offset": 2160,
        "schedule": "clean:150;att-1-0.10:15;clean:30;att-2-0.20:15;att-3-0.35:30",
    },
    {
        "bank_id": "reduced_cleanheavy_2",
        "seed": 20260813,
        "offset": 2220,
        "schedule": "clean:120;att-1-0.15:20;clean:60;att-2-0.25:20;clean:20",
    },
    {
        "bank_id": "reduced_attackpulse_3",
        "seed": 20260814,
        "offset": 2280,
        "schedule": "clean:90;att-3-0.35:30;clean:60;att-1-0.15:30;clean:30",
    },
]

METHODS = {
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
        "output": OUT / "reduced_fresh" / f"mixed_bank_test_gate8_{bid}.npy",
        "partial": OUT / "partials" / f"mixed_bank_test_gate8_{bid}.partial.npy",
        "runtime": OUT / "logs" / f"runtime_gate8_{bid}.jsonl",
        "stdout": OUT / "logs" / f"gate8_{bid}.stdout.log",
        "stderr": OUT / "logs" / f"gate8_{bid}.stderr.log",
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


def write_manifest(status_rows: List[Dict[str, Any]]) -> None:
    payload = {
        "scope": "Gate8 reduced fresh physical-solver sanity extension",
        "interpretation": "small reduced fresh sanity set, not full statistical validation",
        "fixed_trbg_source": {"alpha": 1.0, "fail_cap_quantile": 1.0, "calibration_mode": "source"},
        "concurrency_cap": MAX_WORKERS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "max_wall_seconds_per_bank": MAX_WALL_SECONDS_PER_BANK,
        "banks": status_rows,
    }
    (OUT / "gate8_fresh_bank_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_bank(bank: Dict[str, Any]) -> Dict[str, Any]:
    paths = bank_paths(bank)
    if load_npy_summary(paths["output"]).get("status") == "complete":
        return {**bank, "exit_code": 0, "skipped_existing_complete": True}
    for key in ("output", "partial", "runtime", "stdout", "stderr"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
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


def generate_banks() -> List[Dict[str, Any]]:
    status_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(run_bank, bank) for bank in BANKS]
        for fut in as_completed(futs):
            status_rows.append(fut.result())
            write_manifest(collect_status(status_rows))
    return collect_status(status_rows)


def collect_status(run_rows: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    by_id = {r["bank_id"]: r for r in (run_rows or [])}
    rows = []
    for bank in BANKS:
        paths = bank_paths(bank)
        out_summary = load_npy_summary(paths["output"])
        partial_summary = load_npy_summary(paths["partial"])
        row = {
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
        rows.append(row)
    return rows


def method_defs() -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for d in G5.G3.build_method_defs():
        if d["method"] in METHODS:
            out[(d["method"], int(d["cfg"].slot_budget))] = d
    return out


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    metrics = [
        "scheduler_recall",
        "backend_fail",
        "cost",
        "recover_fail",
        "unnecessary",
        "served_ratio",
        "served_attack_mass",
        "backend_success_attack_mass",
        "clean_service",
        "delay_p50",
        "delay_p95",
        "mean_service_time",
        "mean_service_cost",
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


def confirm_completed_banks(status_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in status_rows if r["complete"]]
    defs = method_defs()
    by_source, _by_topk, _by_winner = G5.source_method_defs()
    _rows_by_mode, _selected, stats_by_mode = G5.calibration_grid(by_source)
    rows: List[Dict[str, Any]] = []
    for bank in completed:
        arrays = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(bank["output"]), 1)
        for budget in BUDGETS:
            for method in sorted(METHODS):
                d = defs[(method, budget)]
                jobs, total_steps, _ = G5.build_jobs_from_arrays(d["ctx"], arrays, d["variant"])
                detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=d["cfg"])
                rec = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=bank["bank_id"])
                rec["fresh_bank"] = bank["bank_id"]
                rec["validation_scope"] = "gate8_reduced_fresh_sanity"
                rows.append(rec)

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
                holdout_id=bank["bank_id"],
                calibration_mode="TRBG-source",
                alpha=1.0,
                fail_cap_quantile=1.0,
            )
            rec["fresh_bank"] = bank["bank_id"]
            rec["validation_scope"] = "gate8_reduced_fresh_sanity"
            rows.append(rec)

    write_csv(OUT / "gate8_fresh_confirm_by_bank.csv", rows)
    summary = aggregate(rows, ["method", "B"]) if rows else []
    write_csv(OUT / "gate8_fresh_confirm_summary.csv", summary)
    return write_fresh_decision(status_rows, rows, summary)


def write_fresh_decision(status_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["method"], int(r["B"]), r["fresh_bank"]): r for r in rows}
    paired = []
    for bank in sorted({r["fresh_bank"] for r in rows}):
        for budget in BUDGETS:
            src = by.get(("source_frozen_transfer", budget, bank))
            trbg = by.get(("TRBG-source", budget, bank))
            topk = by.get(("topk_expected_consequence", budget, bank))
            if not src or not trbg or not topk:
                continue
            paired.append(
                {
                    "fresh_bank": bank,
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
    write_csv(OUT / "gate8_fresh_paired_or_block_stats.csv", paired)
    completed_count = sum(1 for r in status_rows if r["complete"])
    attempted_count = len(status_rows)
    avg_retention = float(np.mean([r["recall_retention_vs_source"] for r in paired])) if paired else float("nan")
    avg_backend_reduction = float(np.mean([r["backend_fail_reduction_vs_source"] for r in paired])) if paired else float("nan")
    avg_cost_delta = float(np.mean([r["cost_delta_vs_source"] for r in paired])) if paired else float("nan")
    avg_recover_delta = float(np.mean([r["recover_delta_vs_source"] for r in paired])) if paired else float("nan")
    serious_reverse = bool(avg_retention < 0.90 or avg_backend_reduction < -0.05) if paired else True
    report = [
        "# Gate8 Fresh Runtime Report",
        "",
        "This is a reduced fresh physical-solver sanity check, not full statistical validation.",
        "",
        f"- attempted banks: `{attempted_count}`",
        f"- completed banks: `{completed_count}`",
        f"- steps per bank: `240`",
        f"- concurrency cap: `{MAX_WORKERS}`",
        "",
        "| bank_id | complete | completed_steps | exit_code | mean_step_sec | p95_step_sec | backend_fail | recover_fail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in status_rows:
        rt = r["runtime"]
        report.append(
            f"| {r['bank_id']} | {r['complete']} | {r['partial_completed_steps']} | {r['exit_code']} | {fmt(rt.get('step_seconds_mean'))} | {fmt(rt.get('step_seconds_p95'))} | {rt.get('backend_fail', 0)} | {rt.get('recover_fail', 0)} |"
        )
    (OUT / "gate8_fresh_runtime_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    stats_md = [
        "# Gate8 Fresh Paired / Block Stats",
        "",
        "Directional reduced-bank comparisons. Banks are not used for parameter selection.",
        "",
        "| fresh_bank | B | recall_retention_vs_source | backend_fail_reduction_vs_source | cost_delta_vs_source | TRBG above topk recall |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in paired:
        stats_md.append(
            f"| {r['fresh_bank']} | {r['B']} | {fmt(r['recall_retention_vs_source'])} | {fmt(r['backend_fail_reduction_vs_source'])} | {fmt(r['cost_delta_vs_source'])} | {r['trbg_above_topk_recall']} |"
        )
    (OUT / "gate8_fresh_paired_or_block_stats.md").write_text("\n".join(stats_md) + "\n", encoding="utf-8")
    decision = [
        "# Gate8 Fresh Decision",
        "",
        "- Scope: `reduced_fresh_physical_solver_sanity_check`.",
        "- This is not full statistical validation.",
        f"- Completed banks: `{completed_count}/{attempted_count}`.",
        f"- Average recall retention vs source-frozen: `{fmt(avg_retention)}`.",
        f"- Average backend_fail reduction vs source-frozen: `{fmt(avg_backend_reduction)}`.",
        f"- Average cost delta vs source-frozen: `{fmt(avg_cost_delta)}`.",
        f"- Average recover_fail delta vs source-frozen: `{fmt(avg_recover_delta)}`.",
        f"- TRBG-source above topk recall for all completed bank/budget pairs: `{all(bool(r['trbg_above_topk_recall']) for r in paired) if paired else False}`.",
        f"- Serious reverse result: `{serious_reverse}`.",
        "- If completed bank count remains small, this should be written as small fresh sanity evidence only.",
    ]
    (OUT / "gate8_fresh_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    return {
        "completed_banks": completed_count,
        "attempted_banks": attempted_count,
        "avg_recall_retention": avg_retention,
        "avg_backend_reduction": avg_backend_reduction,
        "avg_cost_delta": avg_cost_delta,
        "avg_recover_delta": avg_recover_delta,
        "serious_reverse": serious_reverse,
    }


def main() -> None:
    for d in ["reduced_fresh", "partials", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    status = generate_banks()
    write_manifest(status)
    decision = confirm_completed_banks(status)
    (OUT / "gate8_fresh_results.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


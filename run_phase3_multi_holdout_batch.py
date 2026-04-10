#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate multiple fresh mixed holdout banks, run the frozen phase-3 holdout "
            "evaluation on each bank, and aggregate mean/std + paired deltas."
        )
    )
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--python_exe", type=str, default=sys.executable)

    p.add_argument("--clean_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_clean_alarm_scores_full.npy")
    p.add_argument("--attack_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_attack_alarm_scores_400.npy")
    p.add_argument("--train_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_fit.npy")
    p.add_argument("--val_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_eval.npy")

    p.add_argument("--schedule", type=str, default="clean:120;att-1-0.2:60;clean:60;att-2-0.2:60;clean:60;att-3-0.3:60;clean:120")
    p.add_argument("--num_holdouts", type=int, default=5)
    p.add_argument("--seed_base_start", type=int, default=20260421)
    p.add_argument("--start_offset_start", type=int, default=480)
    p.add_argument("--start_offset_step", type=int, default=60)
    p.add_argument("--tau_verify", type=float, default=-1.0)

    # Frozen regime selected from validation ranking
    p.add_argument("--decision_step_group", type=int, default=1)
    p.add_argument("--busy_time_quantile", type=float, default=0.65)
    p.add_argument("--use_cost_budget", action="store_true")
    p.add_argument("--cost_budget_window_steps", type=int, default=20)
    p.add_argument("--cost_budget_quantile", type=float, default=0.60)
    p.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    p.add_argument("--max_wait_steps", type=int, default=10)

    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--threshold_quantiles", type=float, nargs="*", default=[0.50, 0.60, 0.70, 0.80, 0.90])
    p.add_argument("--adaptive_gain_scale_list", type=float, nargs="*", default=[0.0, 0.10, 0.20, 0.40])
    p.add_argument("--consequence_blend_verify", type=float, default=0.70)
    p.add_argument("--consequence_mode", type=str, default="conditional", choices=["conditional", "expected"])
    p.add_argument("--objective_clean_penalty", type=float, default=0.60)
    p.add_argument("--objective_delay_penalty", type=float, default=0.15)
    p.add_argument("--objective_queue_penalty", type=float, default=0.10)
    p.add_argument("--objective_cost_penalty", type=float, default=0.05)
    p.add_argument("--vq_v_grid", type=float, nargs="*", default=[1.0, 2.0, 4.0])
    p.add_argument("--vq_age_grid", type=float, nargs="*", default=[0.0, 0.10, 0.20])
    p.add_argument("--vq_urgency_grid", type=float, nargs="*", default=[0.0, 0.10, 0.20])
    p.add_argument("--vq_fail_grid", type=float, nargs="*", default=[0.0, 0.05])
    p.add_argument("--vq_busy_grid", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    p.add_argument("--vq_cost_grid", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    p.add_argument("--vq_clean_grid", type=float, nargs="*", default=[0.0, 0.20, 0.50])
    p.add_argument("--vq_admission_threshold_grid", type=float, nargs="*", default=[-0.10, 0.0, 0.10])
    p.add_argument("--rng_seed", type=int, default=20260402)

    p.add_argument("--out_dir", type=str, default=f"metric/{DEFAULT_CASE_NAME}/phase3_multi_holdout")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_and_log(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("[CMD] " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    mean = sum(xs) / len(xs)
    std = stats.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(xs),
        "max": max(xs),
    }


def select_best_threshold(eval_compact: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    threshold_names = [
        "threshold_verify_fifo",
        "threshold_ddd_fifo",
        "threshold_expected_consequence_fifo",
        "adaptive_threshold_verify_fifo",
    ]
    candidates = []
    for name in threshold_names:
        if name not in eval_compact:
            continue
        payload = eval_compact[name]
        candidates.append(
            (
                float(payload["weighted_attack_recall_no_backend_fail"]),
                -float(payload["unnecessary_mtd_count"]),
                -float(payload["queue_delay_p95"]),
                name,
                payload,
            )
        )
    if not candidates:
        raise KeyError("No threshold-family policies found in eval_compact")
    _, _, _, name, payload = max(candidates)
    return name, payload


def aggregate_holdouts(summary_paths: List[Path], *, manifest: Dict[str, Any], out_json: Path) -> None:
    holdouts: List[Dict[str, Any]] = [_load_json(p) for p in summary_paths]
    slot_keys = sorted(holdouts[0]["slot_budget_results"].keys(), key=lambda x: int(x))
    aggregate: Dict[str, Any] = {
        "frozen_regime": manifest["frozen_regime"],
        "manifest": manifest,
        "n_holdouts": len(holdouts),
        "slot_budget_aggregates": {},
    }

    for slot_key in slot_keys:
        # Collect all policies present in first holdout
        policies = sorted(holdouts[0]["slot_budget_results"][slot_key]["eval_compact"].keys())
        policy_metric_table: Dict[str, Dict[str, List[float]]] = {}
        for policy in policies:
            policy_metric_table[policy] = {
                "weighted_attack_recall_no_backend_fail": [],
                "unnecessary_mtd_count": [],
                "queue_delay_p95": [],
                "average_service_cost_per_step": [],
                "pred_expected_consequence_served_ratio": [],
            }
        paired = {
            "proposed_vs_best_threshold": {
                "delta_recall": [],
                "delta_unnecessary": [],
                "delta_delay_p95": [],
                "delta_cost_per_step": [],
                "proposed_wins_on_recall": 0,
                "proposed_no_worse_unnecessary": 0,
            },
            "proposed_vs_topk_expected_consequence": {
                "delta_recall": [],
                "delta_unnecessary": [],
                "delta_delay_p95": [],
                "delta_cost_per_step": [],
                "proposed_lower_unnecessary": 0,
                "proposed_lower_cost": 0,
            },
        }
        threshold_winner_names: List[str] = []

        for result in holdouts:
            eval_compact = result["slot_budget_results"][slot_key]["eval_compact"]
            for policy in policies:
                p = eval_compact[policy]
                for metric in policy_metric_table[policy].keys():
                    policy_metric_table[policy][metric].append(float(p[metric]))

            proposed = eval_compact["proposed_ca_vq_hard"]
            thr_name, thr = select_best_threshold(eval_compact)
            threshold_winner_names.append(thr_name)
            topk_ec = eval_compact["topk_expected_consequence"]

            drec = float(proposed["weighted_attack_recall_no_backend_fail"]) - float(thr["weighted_attack_recall_no_backend_fail"])
            dun = float(proposed["unnecessary_mtd_count"]) - float(thr["unnecessary_mtd_count"])
            ddel = float(proposed["queue_delay_p95"]) - float(thr["queue_delay_p95"])
            dcost = float(proposed["average_service_cost_per_step"]) - float(thr["average_service_cost_per_step"])
            paired["proposed_vs_best_threshold"]["delta_recall"].append(drec)
            paired["proposed_vs_best_threshold"]["delta_unnecessary"].append(dun)
            paired["proposed_vs_best_threshold"]["delta_delay_p95"].append(ddel)
            paired["proposed_vs_best_threshold"]["delta_cost_per_step"].append(dcost)
            if drec > 0:
                paired["proposed_vs_best_threshold"]["proposed_wins_on_recall"] += 1
            if dun <= 0:
                paired["proposed_vs_best_threshold"]["proposed_no_worse_unnecessary"] += 1

            drec2 = float(proposed["weighted_attack_recall_no_backend_fail"]) - float(topk_ec["weighted_attack_recall_no_backend_fail"])
            dun2 = float(proposed["unnecessary_mtd_count"]) - float(topk_ec["unnecessary_mtd_count"])
            ddel2 = float(proposed["queue_delay_p95"]) - float(topk_ec["queue_delay_p95"])
            dcost2 = float(proposed["average_service_cost_per_step"]) - float(topk_ec["average_service_cost_per_step"])
            paired["proposed_vs_topk_expected_consequence"]["delta_recall"].append(drec2)
            paired["proposed_vs_topk_expected_consequence"]["delta_unnecessary"].append(dun2)
            paired["proposed_vs_topk_expected_consequence"]["delta_delay_p95"].append(ddel2)
            paired["proposed_vs_topk_expected_consequence"]["delta_cost_per_step"].append(dcost2)
            if dun2 < 0:
                paired["proposed_vs_topk_expected_consequence"]["proposed_lower_unnecessary"] += 1
            if dcost2 < 0:
                paired["proposed_vs_topk_expected_consequence"]["proposed_lower_cost"] += 1

        policy_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for policy, metric_table in policy_metric_table.items():
            policy_stats[policy] = {metric: mean_std(values) for metric, values in metric_table.items()}

        paired_stats = {}
        for name, payload in paired.items():
            paired_stats[name] = {
                "delta_recall": mean_std(payload["delta_recall"]),
                "delta_unnecessary": mean_std(payload["delta_unnecessary"]),
                "delta_delay_p95": mean_std(payload["delta_delay_p95"]),
                "delta_cost_per_step": mean_std(payload["delta_cost_per_step"]),
            }
            for k, v in payload.items():
                if not isinstance(v, list):
                    paired_stats[name][k] = int(v)

        threshold_winner_hist: Dict[str, int] = {}
        for name in threshold_winner_names:
            threshold_winner_hist[name] = threshold_winner_hist.get(name, 0) + 1

        aggregate["slot_budget_aggregates"][slot_key] = {
            "policy_stats": policy_stats,
            "paired_stats": paired_stats,
            "best_threshold_frequency": threshold_winner_hist,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    workdir = Path(os.path.expanduser(args.workdir)).resolve()
    python_exe = str(args.python_exe)
    out_dir = (workdir / args.out_dir).resolve()
    banks_dir = out_dir / "banks"
    results_dir = out_dir / "results"
    logs_dir = out_dir / "logs"
    ensure_dir(banks_dir)
    ensure_dir(results_dir)
    ensure_dir(logs_dir)

    print(f"[INFO] Workdir: {workdir}")
    print(f"[INFO] Python : {python_exe}")
    print(f"[INFO] Out dir: {out_dir}")
    print(
        f"[INFO] Frozen regime: decision_step_group={args.decision_step_group}, "
        f"busy_time_quantile={args.busy_time_quantile}, use_cost_budget={args.use_cost_budget}"
    )

    if not (workdir / "evaluation_mixed_timeline.py").exists():
        raise FileNotFoundError(f"Missing evaluation_mixed_timeline.py in {workdir}")
    if not (workdir / "evaluation_budget_scheduler_phase3_holdout.py").exists():
        raise FileNotFoundError(f"Missing evaluation_budget_scheduler_phase3_holdout.py in {workdir}")

    manifest: Dict[str, Any] = {
        "workdir": str(workdir),
        "clean_bank": args.clean_bank,
        "attack_bank": args.attack_bank,
        "train_bank": args.train_bank,
        "val_bank": args.val_bank,
        "schedule": args.schedule,
        "frozen_regime": {
            "decision_step_group": args.decision_step_group,
            "busy_time_quantile": args.busy_time_quantile,
            "use_cost_budget": bool(args.use_cost_budget),
            "cost_budget_window_steps": args.cost_budget_window_steps,
            "cost_budget_quantile": args.cost_budget_quantile,
            "slot_budget_list": list(args.slot_budget_list),
            "max_wait_steps": args.max_wait_steps,
        },
        "holdouts": [],
    }

    summary_paths: List[Path] = []

    for i in range(args.num_holdouts):
        seed_base = int(args.seed_base_start + i)
        start_offset = int(args.start_offset_start + i * args.start_offset_step)
        tag = f"h{i:02d}_seed{seed_base}_off{start_offset}"
        bank_path = banks_dir / f"mixed_bank_test_{tag}.npy"
        result_path = results_dir / f"budget_scheduler_phase3_holdout_{tag}.npy"
        summary_path = result_path.with_suffix(".summary.json")

        manifest["holdouts"].append({
            "tag": tag,
            "seed_base": seed_base,
            "start_offset": start_offset,
            "test_bank": str(bank_path.relative_to(workdir)),
            "result_npy": str(result_path.relative_to(workdir)),
            "result_summary": str(summary_path.relative_to(workdir)),
        })

        if args.force or not bank_path.exists():
            cmd = [
                python_exe,
                "evaluation_mixed_timeline.py",
                "--tau_verify", str(args.tau_verify),
                "--schedule", args.schedule,
                "--seed_base", str(seed_base),
                "--start_offset", str(start_offset),
                "--output", str(bank_path.relative_to(workdir)),
            ]
            print("\n" + "=" * 90)
            print(f"[RUN] generate_bank {tag}")
            print(" ".join(cmd))
            print(f"[LOG] {logs_dir / (tag + '_generate.log')}")
            print("=" * 90)
            run_and_log(cmd, cwd=workdir, log_path=logs_dir / f"{tag}_generate.log")
        else:
            print(f"[SKIP] Existing bank: {bank_path}")

        if args.force or not summary_path.exists():
            cmd = [
                python_exe,
                "evaluation_budget_scheduler_phase3_holdout.py",
                "--clean_bank", args.clean_bank,
                "--attack_bank", args.attack_bank,
                "--train_bank", args.train_bank,
                "--val_bank", args.val_bank,
                "--test_bank", str(bank_path.relative_to(workdir)),
                "--output", str(result_path.relative_to(workdir)),
                "--slot_budget_list", *[str(x) for x in args.slot_budget_list],
                "--max_wait_steps", str(args.max_wait_steps),
                "--decision_step_group", str(args.decision_step_group),
                "--busy_time_quantile", str(args.busy_time_quantile),
                "--n_bins", str(args.n_bins),
                "--threshold_quantiles", *[str(x) for x in args.threshold_quantiles],
                "--adaptive_gain_scale_list", *[str(x) for x in args.adaptive_gain_scale_list],
                "--consequence_blend_verify", str(args.consequence_blend_verify),
                "--consequence_mode", str(args.consequence_mode),
                "--objective_clean_penalty", str(args.objective_clean_penalty),
                "--objective_delay_penalty", str(args.objective_delay_penalty),
                "--objective_queue_penalty", str(args.objective_queue_penalty),
                "--objective_cost_penalty", str(args.objective_cost_penalty),
                "--vq_v_grid", *[str(x) for x in args.vq_v_grid],
                "--vq_age_grid", *[str(x) for x in args.vq_age_grid],
                "--vq_urgency_grid", *[str(x) for x in args.vq_urgency_grid],
                "--vq_fail_grid", *[str(x) for x in args.vq_fail_grid],
                "--vq_busy_grid", *[str(x) for x in args.vq_busy_grid],
                "--vq_cost_grid", *[str(x) for x in args.vq_cost_grid],
                "--vq_clean_grid", *[str(x) for x in args.vq_clean_grid],
                "--vq_admission_threshold_grid", *[str(x) for x in args.vq_admission_threshold_grid],
                "--rng_seed", str(args.rng_seed),
            ]
            if args.use_cost_budget:
                cmd.extend([
                    "--use_cost_budget",
                    "--cost_budget_window_steps", str(args.cost_budget_window_steps),
                    "--cost_budget_quantile", str(args.cost_budget_quantile),
                ])
            print("\n" + "=" * 90)
            print(f"[RUN] holdout_eval {tag}")
            print(" ".join(cmd))
            print(f"[LOG] {logs_dir / (tag + '_holdout.log')}")
            print("=" * 90)
            run_and_log(cmd, cwd=workdir, log_path=logs_dir / f"{tag}_holdout.log")
        else:
            print(f"[SKIP] Existing holdout summary: {summary_path}")

        summary_paths.append(summary_path)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    aggregate_path = out_dir / "aggregate_summary.json"
    aggregate_holdouts(summary_paths, manifest=manifest, out_json=aggregate_path)

    print("\n" + "=" * 90)
    print("[ALL DONE] Multi-holdout batch finished.")
    print(f"[MANIFEST]  {manifest_path}")
    print(f"[AGGREGATE] {aggregate_path}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def load_metric(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def fmt(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def clean_summary(path: str, regime: str) -> Dict[str, Any]:
    d = load_metric(path)
    g = d["group_key"]
    return {
        "regime": regime,
        "path": path,
        "tau": float(d["tau_verify"]),
        "total_clean_sample": int(d["total_clean_sample"][g]),
        "total_DDD_alarm": int(d["total_DDD_alarm"][g]),
        "total_trigger_after_verification": int(d["total_trigger_after_verification"][g]),
        "false_alarm_rate": float(d["false_alarm_rate"][g]),
        "trigger_rate": float(d["trigger_rate"][g]),
        "useless_mtd_rate": float(d["useless_mtd_rate"][g]),
        "fail_per_alarm": float(d["fail_per_alarm"][g]),
        "stage_one_time_per_alarm": float(d["stage_one_time_per_alarm"][g]),
        "stage_two_time_per_alarm": float(d["stage_two_time_per_alarm"][g]),
        "delta_cost_one_per_alarm": float(d["delta_cost_one_per_alarm"][g]),
        "delta_cost_two_per_alarm": float(d["delta_cost_two_per_alarm"][g]),
        "recovery_error_count": int(np.sum(np.asarray(d.get("recovery_error_alarm", {}).get(g, []), dtype=bool))),
        "backend_metric_fail_count": int(np.sum(np.asarray(d.get("backend_metric_fail_alarm", {}).get(g, []), dtype=bool))),
        "backend_mtd_fail_count": int(np.nansum(np.asarray(d.get("backend_mtd_fail_alarm", {}).get(g, []), dtype=float))),
    }


def attack_summary(path: str, regime: str) -> Dict[str, Any]:
    d = load_metric(path)
    total_alarm = 0
    total_trigger = 0
    backend_metric_fail = 0
    backend_mtd_fail = 0
    recovery_error = 0
    group_arr = {}
    for g in ATTACK_GROUPS:
        alarm = int(np.asarray(d["TP_DDD"].get(g, []), dtype=bool).sum())
        trig = int(np.asarray(d.get("trigger_after_verification", {}).get(g, []), dtype=bool).sum())
        total_alarm += alarm
        total_trigger += trig
        backend_metric_fail += int(np.asarray(d.get("backend_metric_fail", {}).get(g, []), dtype=bool).sum())
        backend_mtd_fail += int(np.nansum(np.asarray(d.get("fail", {}).get(g, []), dtype=float)))
        recovery_error += int(np.asarray(d.get("recovery_error", {}).get(g, []), dtype=bool).sum())
        group_arr[g] = float(trig / alarm) if alarm > 0 else float("nan")
    return {
        "regime": regime,
        "path": path,
        "tau": float(d["metadata"]["tau_verify"]),
        "overall_arr": float(total_trigger / total_alarm) if total_alarm > 0 else float("nan"),
        "total_alarm": total_alarm,
        "total_trigger": total_trigger,
        "recovery_error_count": recovery_error,
        "backend_metric_fail_count": backend_metric_fail,
        "backend_mtd_fail_count": backend_mtd_fail,
        "group_arr": group_arr,
    }


def write_clean_report(path: str, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Case39 Exact Tau Clean Summary", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['regime']}",
                "",
                f"- path: `{row['path']}`",
                f"- tau_verify: `{fmt(row['tau'])}`",
                f"- total_clean_sample: `{row['total_clean_sample']}`",
                f"- total_DDD_alarm: `{row['total_DDD_alarm']}`",
                f"- total_trigger_after_verification: `{row['total_trigger_after_verification']}`",
                f"- false_alarm_rate: `{fmt(row['false_alarm_rate'])}`",
                f"- trigger_rate: `{fmt(row['trigger_rate'])}`",
                f"- useless_mtd_rate: `{fmt(row['useless_mtd_rate'])}`",
                f"- fail_per_alarm: `{fmt(row['fail_per_alarm'])}`",
                f"- stage_one_time_per_alarm: `{fmt(row['stage_one_time_per_alarm'])}`",
                f"- stage_two_time_per_alarm: `{fmt(row['stage_two_time_per_alarm'])}`",
                f"- delta_cost_one_per_alarm: `{fmt(row['delta_cost_one_per_alarm'])}`",
                f"- delta_cost_two_per_alarm: `{fmt(row['delta_cost_two_per_alarm'])}`",
                f"- recovery_error_count: `{row['recovery_error_count']}`",
                f"- backend_metric_fail_count: `{row['backend_metric_fail_count']}`",
                f"- backend_mtd_fail_count: `{row['backend_mtd_fail_count']}`",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_attack_report(path: str, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Case39 Exact Tau Attack Summary", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['regime']}",
                "",
                f"- path: `{row['path']}`",
                f"- tau_verify: `{fmt(row['tau'])}`",
                f"- total_alarm: `{row['total_alarm']}`",
                f"- total_trigger: `{row['total_trigger']}`",
                f"- overall_arr: `{fmt(row['overall_arr'])}`",
                f"- recovery_error_count: `{row['recovery_error_count']}`",
                f"- backend_metric_fail_count: `{row['backend_metric_fail_count']}`",
                f"- backend_mtd_fail_count: `{row['backend_mtd_fail_count']}`",
                "- groupwise_arr:",
            ]
        )
        for g, arr in row["group_arr"].items():
            lines.append(f"  - `{g}`: `{fmt(arr)}`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Write exact-tau clean/attack markdown summaries for case39.")
    p.add_argument("--clean-baseline", required=True)
    p.add_argument("--clean-main", required=True)
    p.add_argument("--clean-strict", required=True)
    p.add_argument("--attack-baseline", required=True)
    p.add_argument("--attack-main", required=True)
    p.add_argument("--attack-strict", required=True)
    p.add_argument("--output-clean-md", required=True)
    p.add_argument("--output-attack-md", required=True)
    args = p.parse_args()

    clean_rows = [
        clean_summary(args.clean_baseline, "baseline"),
        clean_summary(args.clean_main, "main"),
        clean_summary(args.clean_strict, "strict"),
    ]
    attack_rows = [
        attack_summary(args.attack_baseline, "baseline"),
        attack_summary(args.attack_main, "main"),
        attack_summary(args.attack_strict, "strict"),
    ]

    Path(args.output_clean_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_attack_md).parent.mkdir(parents=True, exist_ok=True)
    write_clean_report(args.output_clean_md, clean_rows)
    write_attack_report(args.output_attack_md, attack_rows)
    print(f"Saved clean report: {args.output_clean_md}")
    print(f"Saved attack report: {args.output_attack_md}")


if __name__ == "__main__":
    main()

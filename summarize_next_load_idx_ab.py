from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def load_metric(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def arr_bool(values) -> np.ndarray:
    return np.asarray(list(values), dtype=bool)


def arr_float(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def finite_mean(values: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(values)
    denom = int(finite.sum())
    if denom == 0:
        return float("nan"), 0
    return float(np.mean(values[finite])), denom


def summarize(path: str, label: str, regime: str) -> Dict[str, Any]:
    d = load_metric(path)
    total_alarm = 0
    total_trigger = 0
    backend_metric_fail = 0
    backend_mtd_fail = 0
    cost_no_all: List[float] = []
    cost_two_all: List[float] = []

    for g in ATTACK_GROUPS:
        total_alarm += int(arr_bool(d["TP_DDD"].get(g, [])).sum())
        total_trigger += int(arr_bool(d.get("trigger_after_verification", {}).get(g, [])).sum())
        backend_metric_fail += int(arr_bool(d.get("backend_metric_fail", {}).get(g, [])).sum())
        backend_mtd_fail += int(np.nansum(arr_float(d.get("fail", {}).get(g, []))))
        cost_no_all.extend(arr_float(d.get("cost_no_mtd", {}).get(g, [])).tolist())
        cost_two_all.extend(arr_float(d.get("cost_with_mtd_two", {}).get(g, [])).tolist())

    stage1_mean, stage1_denom = finite_mean(arr_float(d.get("mtd_stage_one_time", [])))
    stage2_mean, stage2_denom = finite_mean(arr_float(d.get("mtd_stage_two_time", [])))
    cost_no = arr_float(cost_no_all)
    cost_two = arr_float(cost_two_all)
    delta = cost_two - cost_no if cost_no.size and cost_two.size else np.asarray([], dtype=float)
    delta_mean, delta_denom = finite_mean(delta)

    return {
        "variant": label,
        "regime": regime,
        "metric_path": path,
        "tau_verify": float(d["metadata"]["tau_verify"]),
        "overall_arr": safe_rate(total_trigger, total_alarm),
        "total_alarms": total_alarm,
        "total_triggers": total_trigger,
        "backend_metric_fail_count": backend_metric_fail,
        "backend_metric_fail_rate_among_triggers": safe_rate(backend_metric_fail, total_trigger),
        "backend_mtd_fail_count": backend_mtd_fail,
        "backend_mtd_fail_rate_among_triggers": safe_rate(backend_mtd_fail, total_trigger),
        "stage_one_time_mean_triggered_finite": stage1_mean,
        "stage_one_time_denominator": stage1_denom,
        "stage_two_time_mean_triggered_finite": stage2_mean,
        "stage_two_time_denominator": stage2_denom,
        "delta_cost_two_mean_finite": delta_mean,
        "delta_cost_two_denominator": delta_denom,
    }


def fmt(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize next_load_idx A/B attack comparison.")
    p.add_argument("--old-baseline", required=True)
    p.add_argument("--old-main", required=True)
    p.add_argument("--old-strict", required=True)
    p.add_argument("--new-baseline", required=True)
    p.add_argument("--new-main", required=True)
    p.add_argument("--new-strict", required=True)
    p.add_argument("--output-csv", default="metric/case39/next_load_idx_ab_check.csv")
    p.add_argument("--output-md", default="reports/next_load_idx_ab_check.md")
    p.add_argument("--output-json", default="metric/case39/next_load_idx_ab_check.json")
    args = p.parse_args()

    rows = [
        summarize(args.old_baseline, "old_offset7", "baseline"),
        summarize(args.old_main, "old_offset7", "main"),
        summarize(args.old_strict, "old_offset7", "strict"),
        summarize(args.new_baseline, "new_sample_length", "baseline"),
        summarize(args.new_main, "new_sample_length", "main"),
        summarize(args.new_strict, "new_sample_length", "strict"),
    ]

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = ["# Next Load Index A/B Check", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['variant']} / {row['regime']}",
                "",
                f"- metric_path: `{row['metric_path']}`",
                f"- tau_verify: `{fmt(row['tau_verify'])}`",
                f"- overall_arr: `{fmt(row['overall_arr'])}`",
                f"- backend_metric_fail_rate_among_triggers: `{fmt(row['backend_metric_fail_rate_among_triggers'])}`",
                f"- backend_mtd_fail_rate_among_triggers: `{fmt(row['backend_mtd_fail_rate_among_triggers'])}`",
                f"- stage_one_time_mean_triggered_finite: `{fmt(row['stage_one_time_mean_triggered_finite'])}` over `{row['stage_one_time_denominator']}`",
                f"- stage_two_time_mean_triggered_finite: `{fmt(row['stage_two_time_mean_triggered_finite'])}` over `{row['stage_two_time_denominator']}`",
                f"- delta_cost_two_mean_finite: `{fmt(row['delta_cost_two_mean_finite'])}` over `{row['delta_cost_two_denominator']}`",
                "",
            ]
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD: {out_md}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

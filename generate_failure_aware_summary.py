from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def load_metric(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def arr_bool(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=bool)


def arr_float(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def safe_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom)


def finite_mean(arr: np.ndarray) -> tuple[float, int, int]:
    finite = np.isfinite(arr)
    denom = int(finite.sum())
    nan_count = int((~finite).sum())
    if denom == 0:
        return float("nan"), 0, nan_count
    return float(np.mean(arr[finite])), denom, nan_count


def cost_delta_stats(cost_no: np.ndarray, cost_with: np.ndarray) -> tuple[float, int, int]:
    if cost_no.size == 0 or cost_with.size == 0:
        return float("nan"), 0, 0
    pair_n = int(min(cost_no.size, cost_with.size))
    if pair_n == 0:
        return float("nan"), 0, 0
    delta = cost_with[:pair_n] - cost_no[:pair_n]
    return finite_mean(delta)


def summarize_clean(path: str, regime: str) -> Dict[str, Any]:
    d = load_metric(path)
    g = d["group_key"]

    total_alarms = int(d["total_DDD_alarm"][g])
    total_triggers = int(d["total_trigger_after_verification"][g])

    recovery_error_alarm = arr_bool(d.get("recovery_error_alarm", {}).get(g, []))
    backend_mtd_fail_alarm = arr_float(d.get("backend_mtd_fail_alarm", {}).get(g, d.get("fail_alarm", {}).get(g, [])))
    backend_metric_fail_alarm = arr_float(d.get("backend_metric_fail_alarm", {}).get(g, []))

    mtd_stage_one_time = arr_float(d.get("mtd_stage_one_time", []))
    mtd_stage_two_time = arr_float(d.get("mtd_stage_two_time", []))
    cost_no = arr_float(d.get("cost_no_mtd", {}).get(g, []))
    cost_one = arr_float(d.get("cost_with_mtd_one", {}).get(g, []))
    cost_two = arr_float(d.get("cost_with_mtd_two", {}).get(g, []))

    stage_one_mean, stage_one_denom, stage_one_nan = finite_mean(mtd_stage_one_time)
    stage_two_mean, stage_two_denom, stage_two_nan = finite_mean(mtd_stage_two_time)
    delta_one_mean, delta_one_denom, delta_one_nan = cost_delta_stats(cost_no, cost_one)
    delta_two_mean, delta_two_denom, delta_two_nan = cost_delta_stats(cost_no, cost_two)

    recovery_error_count = int(recovery_error_alarm.sum())
    backend_metric_fail_count = int(np.nansum(backend_metric_fail_alarm)) if backend_metric_fail_alarm.size else 0
    backend_mtd_fail_count = int(np.nansum(backend_mtd_fail_alarm)) if backend_mtd_fail_alarm.size else 0

    return {
        "scenario": "clean",
        "regime": regime,
        "metric_path": path,
        "tau_verify": float(d["tau_verify"]),
        "total_alarms": total_alarms,
        "total_triggers": total_triggers,
        "recovery_error_count": recovery_error_count,
        "recovery_error_rate_among_alarms": safe_rate(recovery_error_count, total_alarms),
        "recovery_error_policy": d.get("recovery_error_policy", "unknown"),
        "backend_metric_success_count": int(total_triggers - backend_metric_fail_count),
        "backend_metric_fail_count": backend_metric_fail_count,
        "backend_metric_fail_rate_among_triggers": safe_rate(backend_metric_fail_count, total_triggers),
        "backend_metric_fail_rate_among_alarms": safe_rate(backend_metric_fail_count, total_alarms),
        "backend_mtd_fail_count": backend_mtd_fail_count,
        "backend_mtd_fail_rate_among_triggers": safe_rate(backend_mtd_fail_count, total_triggers),
        "backend_mtd_fail_rate_among_alarms": safe_rate(backend_mtd_fail_count, total_alarms),
        "stage_one_time_success_only_mean": stage_one_mean,
        "stage_one_time_denominator": stage_one_denom,
        "stage_one_time_nan_count": stage_one_nan,
        "stage_two_time_success_only_mean": stage_two_mean,
        "stage_two_time_denominator": stage_two_denom,
        "stage_two_time_nan_count": stage_two_nan,
        "delta_cost_one_success_only_mean": delta_one_mean,
        "delta_cost_one_denominator": delta_one_denom,
        "delta_cost_one_nan_count": delta_one_nan,
        "delta_cost_two_success_only_mean": delta_two_mean,
        "delta_cost_two_denominator": delta_two_denom,
        "delta_cost_two_nan_count": delta_two_nan,
        "legacy_false_alarm_rate": float(d["false_alarm_rate"][g]),
        "legacy_trigger_rate": float(d["trigger_rate"][g]),
        "legacy_useless_mtd_rate": float(d["useless_mtd_rate"][g]),
        "legacy_fail_per_alarm": float(d["fail_per_alarm"][g]),
        "legacy_stage_one_time_per_alarm": float(d["stage_one_time_per_alarm"][g]),
        "legacy_stage_two_time_per_alarm": float(d["stage_two_time_per_alarm"][g]),
        "legacy_delta_cost_one_per_alarm": float(d["delta_cost_one_per_alarm"][g]),
        "legacy_delta_cost_two_per_alarm": float(d["delta_cost_two_per_alarm"][g]),
    }


def summarize_attack(path: str, regime: str) -> Dict[str, Any]:
    d = load_metric(path)

    total_alarms = 0
    total_triggers = 0
    recovery_error_count = 0
    backend_metric_fail_count = 0
    backend_mtd_fail_count = 0
    cost_no_all: List[float] = []
    cost_one_all: List[float] = []
    cost_two_all: List[float] = []

    for g in ATTACK_GROUPS:
        total_alarms += int(arr_bool(d["TP_DDD"].get(g, [])).sum())
        total_triggers += int(arr_bool(d.get("trigger_after_verification", {}).get(g, [])).sum())
        recovery_error_count += int(arr_bool(d.get("recovery_error", {}).get(g, [])).sum())
        backend_metric_fail_count += int(arr_bool(d.get("backend_metric_fail", {}).get(g, [])).sum())
        backend_mtd_fail_count += int(np.nansum(arr_float(d.get("fail", {}).get(g, []))))
        cost_no_all.extend(arr_float(d.get("cost_no_mtd", {}).get(g, [])).tolist())
        cost_one_all.extend(arr_float(d.get("cost_with_mtd_one", {}).get(g, [])).tolist())
        cost_two_all.extend(arr_float(d.get("cost_with_mtd_two", {}).get(g, [])).tolist())

    mtd_stage_one_time = arr_float(d.get("mtd_stage_one_time", []))
    mtd_stage_two_time = arr_float(d.get("mtd_stage_two_time", []))
    cost_no = arr_float(cost_no_all)
    cost_one = arr_float(cost_one_all)
    cost_two = arr_float(cost_two_all)

    stage_one_mean, stage_one_denom, stage_one_nan = finite_mean(mtd_stage_one_time)
    stage_two_mean, stage_two_denom, stage_two_nan = finite_mean(mtd_stage_two_time)
    delta_one_mean, delta_one_denom, delta_one_nan = cost_delta_stats(cost_no, cost_one)
    delta_two_mean, delta_two_denom, delta_two_nan = cost_delta_stats(cost_no, cost_two)

    return {
        "scenario": "attack",
        "regime": regime,
        "metric_path": path,
        "tau_verify": float(d["metadata"]["tau_verify"]),
        "total_alarms": total_alarms,
        "total_triggers": total_triggers,
        "recovery_error_count": recovery_error_count,
        "recovery_error_rate_among_alarms": safe_rate(recovery_error_count, total_alarms),
        "recovery_error_policy": d["metadata"].get("recovery_error_policy", "unknown"),
        "backend_metric_success_count": int(total_triggers - backend_metric_fail_count),
        "backend_metric_fail_count": backend_metric_fail_count,
        "backend_metric_fail_rate_among_triggers": safe_rate(backend_metric_fail_count, total_triggers),
        "backend_metric_fail_rate_among_alarms": safe_rate(backend_metric_fail_count, total_alarms),
        "backend_mtd_fail_count": backend_mtd_fail_count,
        "backend_mtd_fail_rate_among_triggers": safe_rate(backend_mtd_fail_count, total_triggers),
        "backend_mtd_fail_rate_among_alarms": safe_rate(backend_mtd_fail_count, total_alarms),
        "stage_one_time_success_only_mean": stage_one_mean,
        "stage_one_time_denominator": stage_one_denom,
        "stage_one_time_nan_count": stage_one_nan,
        "stage_two_time_success_only_mean": stage_two_mean,
        "stage_two_time_denominator": stage_two_denom,
        "stage_two_time_nan_count": stage_two_nan,
        "delta_cost_one_success_only_mean": delta_one_mean,
        "delta_cost_one_denominator": delta_one_denom,
        "delta_cost_one_nan_count": delta_one_nan,
        "delta_cost_two_success_only_mean": delta_two_mean,
        "delta_cost_two_denominator": delta_two_denom,
        "delta_cost_two_nan_count": delta_two_nan,
        "overall_arr": safe_rate(total_triggers, total_alarms),
    }


def fmt(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def write_markdown(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Case39 Failure-Aware Statistics")
    lines.append("")
    lines.append("This report keeps failure denominators explicit.")
    lines.append("")
    for scenario in ["clean", "attack"]:
        lines.append(f"## {scenario.capitalize()}")
        lines.append("")
        for row in payload[scenario]:
            lines.append(f"### {row['regime']}")
            lines.append("")
            lines.append(f"- metric_path: `{row['metric_path']}`")
            lines.append(f"- tau_verify: `{fmt(row['tau_verify'])}`")
            lines.append(f"- total_alarms: `{row['total_alarms']}`")
            lines.append(f"- total_triggers: `{row['total_triggers']}`")
            lines.append(f"- recovery_error_count: `{row['recovery_error_count']}`")
            lines.append(f"- recovery_error_rate_among_alarms: `{fmt(row['recovery_error_rate_among_alarms'])}`")
            lines.append(f"- recovery_error_policy: `{row['recovery_error_policy']}`")
            lines.append(f"- backend_metric_success_count: `{row['backend_metric_success_count']}`")
            lines.append(f"- backend_metric_fail_count: `{row['backend_metric_fail_count']}`")
            lines.append(f"- backend_metric_fail_rate_among_triggers: `{fmt(row['backend_metric_fail_rate_among_triggers'])}`")
            lines.append(f"- backend_metric_fail_rate_among_alarms: `{fmt(row['backend_metric_fail_rate_among_alarms'])}`")
            lines.append(f"- backend_mtd_fail_count: `{row['backend_mtd_fail_count']}`")
            lines.append(f"- backend_mtd_fail_rate_among_triggers: `{fmt(row['backend_mtd_fail_rate_among_triggers'])}`")
            lines.append(f"- backend_mtd_fail_rate_among_alarms: `{fmt(row['backend_mtd_fail_rate_among_alarms'])}`")
            lines.append(f"- stage_one_time_success_only_mean: `{fmt(row['stage_one_time_success_only_mean'])}` over `{row['stage_one_time_denominator']}` finite samples, nan_count=`{row['stage_one_time_nan_count']}`")
            lines.append(f"- stage_two_time_success_only_mean: `{fmt(row['stage_two_time_success_only_mean'])}` over `{row['stage_two_time_denominator']}` finite samples, nan_count=`{row['stage_two_time_nan_count']}`")
            lines.append(f"- delta_cost_one_success_only_mean: `{fmt(row['delta_cost_one_success_only_mean'])}` over `{row['delta_cost_one_denominator']}` finite samples, nan_count=`{row['delta_cost_one_nan_count']}`")
            lines.append(f"- delta_cost_two_success_only_mean: `{fmt(row['delta_cost_two_success_only_mean'])}` over `{row['delta_cost_two_denominator']}` finite samples, nan_count=`{row['delta_cost_two_nan_count']}`")
            if scenario == "clean":
                lines.append(f"- legacy_fail_per_alarm: `{fmt(row['legacy_fail_per_alarm'])}`")
                lines.append(f"- legacy_stage_two_time_per_alarm: `{fmt(row['legacy_stage_two_time_per_alarm'])}`")
            else:
                lines.append(f"- overall_arr: `{fmt(row['overall_arr'])}`")
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate failure-aware summaries for case39 clean/attack metrics.")
    p.add_argument("--clean-baseline", required=True)
    p.add_argument("--clean-main", required=True)
    p.add_argument("--clean-strict", required=True)
    p.add_argument("--attack-baseline", required=True)
    p.add_argument("--attack-main", required=True)
    p.add_argument("--attack-strict", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    payload = {
        "clean": [
            summarize_clean(args.clean_baseline, "baseline"),
            summarize_clean(args.clean_main, "main"),
            summarize_clean(args.clean_strict, "strict"),
        ],
        "attack": [
            summarize_attack(args.attack_baseline, "baseline"),
            summarize_attack(args.attack_main, "main"),
            summarize_attack(args.attack_strict, "strict"),
        ],
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    rows = payload["clean"] + payload["attack"]
    headers = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, float) and np.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    write_markdown(str(out_md), payload)
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD: {out_md}")


if __name__ == "__main__":
    main()

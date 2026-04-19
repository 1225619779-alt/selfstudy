from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path("metric/case39")
OUT_JSON = ROOT / "backend_failure_audit.json"
OUT_TXT = ROOT / "backend_failure_audit.txt"

ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def load_metric(path: Path) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def mean_or_nan(values: List[Any]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def quantiles_or_empty(values: List[Any]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    return [float(x) for x in np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])]


def summarize_clean(path: Path) -> Dict[str, Any]:
    d = load_metric(path)
    g = d["group_key"]
    trigger_n = int(d["total_trigger_after_verification"][g])
    verify_n = int(d["total_DDD_alarm"][g])
    fail_vals = list(d["fail"][g])
    obj_one = list(d["obj_one"][g])
    obj_two = list(d["obj_two"][g])

    return {
        "path": str(path),
        "tau_verify": float(d["tau_verify"]),
        "total_clean_sample": int(d["total_clean_sample"][g]),
        "total_DDD_alarm": verify_n,
        "total_trigger_after_verification": trigger_n,
        "trigger_rate": float(d["trigger_rate"][g]),
        "uMTD_rate": float(d["useless_mtd_rate"][g]),
        "fail_per_alarm": float(d["fail_per_alarm"][g]),
        "fail_mean_triggered_only": mean_or_nan(fail_vals),
        "post_mtd_opf_converge_mean": mean_or_nan(list(d.get("post_mtd_opf_converge", []))),
        "obj_one_quantiles": quantiles_or_empty(obj_one),
        "obj_two_quantiles": quantiles_or_empty(obj_two),
        "stage_one_time_per_alarm": float(d["stage_one_time_per_alarm"][g]),
        "stage_two_time_per_alarm": float(d["stage_two_time_per_alarm"][g]),
        "delta_cost_one_per_alarm": float(d["delta_cost_one_per_alarm"][g]),
        "delta_cost_two_per_alarm": float(d["delta_cost_two_per_alarm"][g]),
    }


def summarize_attack(path: Path) -> Dict[str, Any]:
    d = load_metric(path)
    total_alarm = 0
    total_trigger = 0
    all_fail: List[Any] = []
    all_obj_one: List[Any] = []
    all_obj_two: List[Any] = []
    all_backend_metric_fail: List[Any] = []
    group_rows = []

    for g in ATTACK_GROUPS:
        front_end = int(np.asarray(d["TP_DDD"].get(g, []), dtype=bool).sum())
        backend = int(np.asarray(d["trigger_after_verification"].get(g, []), dtype=bool).sum())
        total_alarm += front_end
        total_trigger += backend

        fail_vals = list(d["fail"].get(g, []))
        obj_one = list(d["obj_one"].get(g, []))
        obj_two = list(d["obj_two"].get(g, []))
        backend_metric_fail = list(d.get("backend_metric_fail", {}).get(g, []))

        all_fail.extend(fail_vals)
        all_obj_one.extend(obj_one)
        all_obj_two.extend(obj_two)
        all_backend_metric_fail.extend(backend_metric_fail)

        group_rows.append(
            {
                "group": g,
                "front_end_alarms": front_end,
                "backend_triggers": backend,
                "arr": float(backend / front_end) if front_end else float("nan"),
                "backend_metric_fail_rate": mean_or_nan(backend_metric_fail),
                "fail_rate_triggered_only": mean_or_nan(fail_vals),
            }
        )

    return {
        "path": str(path),
        "tau_verify": float(d["metadata"]["tau_verify"]),
        "total_DDD_alarm": total_alarm,
        "total_trigger_after_gate": total_trigger,
        "overall_arr": float(total_trigger / total_alarm) if total_alarm else float("nan"),
        "fail_rate_triggered_only": mean_or_nan(all_fail),
        "backend_metric_fail_rate": mean_or_nan(all_backend_metric_fail),
        "post_mtd_opf_converge_mean": mean_or_nan(list(d.get("post_mtd_opf_converge", []))),
        "mean_stage_one_time_triggered": mean_or_nan(list(d.get("mtd_stage_one_time", []))),
        "mean_stage_two_time_triggered": mean_or_nan(list(d.get("mtd_stage_two_time", []))),
        "obj_one_quantiles": quantiles_or_empty(all_obj_one),
        "obj_two_quantiles": quantiles_or_empty(all_obj_two),
        "groups": group_rows,
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
    clean_paths = [
        ROOT / "metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy",
        ROOT / "metric_event_trigger_clean_tau_0.013_mode_0_0.03_1.1.npy",
        ROOT / "metric_event_trigger_clean_tau_0.016_mode_0_0.03_1.1.npy",
    ]
    attack_paths = [
        ROOT / "metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy",
        ROOT / "metric_event_trigger_tau_0.013_mode_0_0.03_1.1.npy",
        ROOT / "metric_event_trigger_tau_0.016_mode_0_0.03_1.1.npy",
    ]

    payload = {
        "clean": [summarize_clean(p) for p in clean_paths],
        "attack": [summarize_attack(p) for p in attack_paths],
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("Case39 backend failure audit")
    lines.append("")
    lines.append("[clean]")
    for row in payload["clean"]:
        lines.append(
            "tau={tau} trigger_rate={trigger_rate} uMTD_rate={uMTD_rate} "
            "fail_per_alarm={fail_per_alarm} obj1_q50={obj1_q50} obj2_q50={obj2_q50}".format(
                tau=fmt(row["tau_verify"]),
                trigger_rate=fmt(row["trigger_rate"]),
                uMTD_rate=fmt(row["uMTD_rate"]),
                fail_per_alarm=fmt(row["fail_per_alarm"]),
                obj1_q50=fmt(row["obj_one_quantiles"][3] if row["obj_one_quantiles"] else np.nan),
                obj2_q50=fmt(row["obj_two_quantiles"][3] if row["obj_two_quantiles"] else np.nan),
            )
        )
    lines.append("")
    lines.append("[attack]")
    for row in payload["attack"]:
        lines.append(
            "tau={tau} overall_arr={overall_arr} fail_rate_triggered_only={fail_rate} "
            "backend_metric_fail_rate={backend_fail} post_mtd_opf_converge_mean={opf}".format(
                tau=fmt(row["tau_verify"]),
                overall_arr=fmt(row["overall_arr"]),
                fail_rate=fmt(row["fail_rate_triggered_only"]),
                backend_fail=fmt(row["backend_metric_fail_rate"]),
                opf=fmt(row["post_mtd_opf_converge_mean"]),
            )
        )
        for g in row["groups"]:
            lines.append(
                "  {group}: arr={arr} backend_metric_fail_rate={bmf} fail_rate_triggered_only={fail}".format(
                    group=g["group"],
                    arr=fmt(g["arr"]),
                    bmf=fmt(g["backend_metric_fail_rate"]),
                    fail=fmt(g["fail_rate_triggered_only"]),
                )
            )

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_BASELINE = "metric/case14/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
DEFAULT_MAIN = "metric/case14/metric_event_trigger_clean_tau_0.021_mode_0_0.03_1.1.npy"
STRICT_CANDIDATES = [
    "metric/case14/metric_event_trigger_clean_tau_0.03_mode_0_0.03_1.1.npy",
    "metric/case14/metric_event_trigger_clean_tau_0.030_mode_0_0.03_1.1.npy",
]


PAPER_COLUMNS = [
    ("row_label", "Row"),
    ("tau_label", "Tau"),
    ("total_clean_sample", "Total clean samples"),
    ("total_ddd_alarm", "DDD false alarms"),
    ("front_end_far", "Front-end FAR"),
    ("trigger_count", "Triggered backend MTD"),
    ("skip_count", "Skipped alarms"),
    ("backend_deployment_rate_among_alarms", "Backend MTD deployment rate among alarms"),
    ("alarm_rejection_rate", "Alarm rejection rate"),
    ("unnecessary_mtd_deployment_rate", "Unnecessary MTD deployment rate"),
    ("backend_failure_rate_per_false_alarm", "Backend failure rate per false alarm"),
    ("mean_stage_i_defense_time_per_false_alarm", "Mean stage-I defense time per false alarm"),
    ("mean_stage_ii_defense_time_per_false_alarm", "Mean stage-II defense time per false alarm"),
    ("mean_stage_i_incremental_operating_cost_per_false_alarm", "Mean stage-I incremental operating cost per false alarm"),
    ("mean_stage_ii_incremental_operating_cost_per_false_alarm", "Mean stage-II incremental operating cost per false alarm"),
    ("red_unnecessary_mtd_pct", "Reduction of unnecessary MTD deployment rate (%)"),
    ("red_failure_pct", "Reduction of backend failure rate per false alarm (%)"),
    ("red_stage_i_time_pct", "Reduction of mean stage-I defense time per false alarm (%)"),
    ("red_stage_ii_time_pct", "Reduction of mean stage-II defense time per false alarm (%)"),
    ("red_stage_i_cost_pct", "Reduction of mean stage-I incremental operating cost per false alarm (%)"),
    ("red_stage_ii_cost_pct", "Reduction of mean stage-II incremental operating cost per false alarm (%)"),
]

SCALAR_FIELDS = [
    "total_clean_sample",
    "total_DDD_alarm",
    "false_alarm_rate",
]

LIST_FIELDS = [
    "TP_DDD",
    "clean_alarm_idx",
    "verify_score",
]


# ---------- helpers ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a paper-ready clean main table from baseline/main/strict clean metric files."
    )
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--main", type=str, default=DEFAULT_MAIN)
    parser.add_argument(
        "--strict",
        type=str,
        default="",
        help="Optional strict tau metric file. If omitted, the script will try common 0.03 / 0.030 filenames.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14",
        help="Directory to save CSV / markdown / txt summaries.",
    )
    parser.add_argument("--baseline_label", type=str, default="Baseline")
    parser.add_argument("--main_label", type=str, default="Main OP")
    parser.add_argument("--strict_label", type=str, default="Strict OP")
    parser.add_argument("--baseline_tau_label", type=str, default="-1.0")
    parser.add_argument("--main_tau_label", type=str, default="0.021")
    parser.add_argument("--strict_tau_label", type=str, default="0.030")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")



def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return -1



def resolve_strict_path(user_path: str) -> Optional[str]:
    if user_path:
        return user_path if os.path.exists(user_path) else None
    for cand in STRICT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None



def load_result(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    obj = np.load(path, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise TypeError(f"Loaded object is not a dict: {type(obj)}")
    return obj



def detect_group_key(data: Dict[str, Any]) -> str:
    candidate_fields = [
        "total_clean_sample",
        "total_DDD_alarm",
        "TP_DDD",
        "verify_score",
        "trigger_after_verification",
    ]
    for field in candidate_fields:
        value = data.get(field, None)
        if isinstance(value, dict) and len(value) > 0:
            keys = list(value.keys())
            if len(keys) == 1:
                return keys[0]
            if "(0,0.0)" in keys:
                return "(0,0.0)"
            return keys[0]
    raise KeyError("Cannot detect group key from metric file")



def get_group_value(data: Dict[str, Any], field: str, group_key: str, default: Any = None) -> Any:
    value = data.get(field, None)
    if isinstance(value, dict):
        return value.get(group_key, default)
    return default



def compare_lists(a: List[Any], b: List[Any], rtol: float, atol: float) -> Tuple[bool, str]:
    if len(a) != len(b):
        return False, f"length mismatch: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return True, "both empty"

    numeric = True
    aa: List[float] = []
    bb: List[float] = []
    for x, y in zip(a, b):
        try:
            aa.append(float(x))
            bb.append(float(y))
        except Exception:
            numeric = False
            break

    if numeric:
        aa_arr = np.asarray(aa, dtype=float)
        bb_arr = np.asarray(bb, dtype=float)
        same = np.allclose(aa_arr, bb_arr, rtol=rtol, atol=atol, equal_nan=True)
        if same:
            return True, "allclose"
        diff_pos = np.where(~np.isclose(aa_arr, bb_arr, rtol=rtol, atol=atol, equal_nan=True))[0]
        first = int(diff_pos[0])
        return False, f"first numeric mismatch at pos={first}: {aa_arr[first]} vs {bb_arr[first]}"

    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"first mismatch at pos={i}: {x} vs {y}"
    return True, "exact match"



def pct_reduction(baseline: float, current: float) -> float:
    baseline = safe_float(baseline)
    current = safe_float(current)
    if math.isnan(baseline) or math.isnan(current) or abs(baseline) < 1e-15:
        return float("nan")
    return 100.0 * (baseline - current) / baseline



def fmt_num(x: Any, digits: int = 6) -> str:
    x = safe_float(x)
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"



def infer_tau_label(path: str, fallback: str) -> str:
    m = re.search(r"tau_([-+]?\d*\.?\d+)", os.path.basename(path))
    if m:
        return m.group(1)
    return fallback



def build_row(data: Dict[str, Any], group_key: str, row_label: str, tau_label: str) -> Dict[str, Any]:
    total_clean_sample = safe_int(get_group_value(data, "total_clean_sample", group_key, -1))
    total_ddd_alarm = safe_int(get_group_value(data, "total_DDD_alarm", group_key, -1))
    front_end_far = safe_float(get_group_value(data, "false_alarm_rate", group_key, float("nan")))
    trigger_count = safe_int(get_group_value(data, "total_trigger_after_verification", group_key, -1))
    skip_count = safe_int(get_group_value(data, "total_skip_by_verification", group_key, -1))

    backend_deployment_rate_among_alarms = safe_float(
        get_group_value(data, "trigger_rate", group_key, float("nan"))
    )
    if math.isnan(backend_deployment_rate_among_alarms) and total_ddd_alarm > 0 and trigger_count >= 0:
        backend_deployment_rate_among_alarms = trigger_count / total_ddd_alarm

    alarm_rejection_rate = safe_float(get_group_value(data, "skip_rate", group_key, float("nan")))
    if math.isnan(alarm_rejection_rate) and total_ddd_alarm > 0 and skip_count >= 0:
        alarm_rejection_rate = skip_count / total_ddd_alarm

    unnecessary_mtd_deployment_rate = safe_float(
        get_group_value(data, "useless_mtd_rate", group_key, float("nan"))
    )
    if math.isnan(unnecessary_mtd_deployment_rate) and total_clean_sample > 0 and trigger_count >= 0:
        unnecessary_mtd_deployment_rate = trigger_count / total_clean_sample

    return {
        "row_label": row_label,
        "tau_label": tau_label,
        "total_clean_sample": total_clean_sample,
        "total_ddd_alarm": total_ddd_alarm,
        "front_end_far": front_end_far,
        "trigger_count": trigger_count,
        "skip_count": skip_count,
        "backend_deployment_rate_among_alarms": backend_deployment_rate_among_alarms,
        "alarm_rejection_rate": alarm_rejection_rate,
        "unnecessary_mtd_deployment_rate": unnecessary_mtd_deployment_rate,
        "backend_failure_rate_per_false_alarm": safe_float(
            get_group_value(data, "fail_per_alarm", group_key, float("nan"))
        ),
        "mean_stage_i_defense_time_per_false_alarm": safe_float(
            get_group_value(data, "stage_one_time_per_alarm", group_key, float("nan"))
        ),
        "mean_stage_ii_defense_time_per_false_alarm": safe_float(
            get_group_value(data, "stage_two_time_per_alarm", group_key, float("nan"))
        ),
        "mean_stage_i_incremental_operating_cost_per_false_alarm": safe_float(
            get_group_value(data, "delta_cost_one_per_alarm", group_key, float("nan"))
        ),
        "mean_stage_ii_incremental_operating_cost_per_false_alarm": safe_float(
            get_group_value(data, "delta_cost_two_per_alarm", group_key, float("nan"))
        ),
    }



def add_reduction_columns(rows: List[Dict[str, Any]]) -> None:
    baseline = rows[0]
    for row in rows:
        row["red_unnecessary_mtd_pct"] = pct_reduction(
            baseline["unnecessary_mtd_deployment_rate"], row["unnecessary_mtd_deployment_rate"]
        )
        row["red_failure_pct"] = pct_reduction(
            baseline["backend_failure_rate_per_false_alarm"], row["backend_failure_rate_per_false_alarm"]
        )
        row["red_stage_i_time_pct"] = pct_reduction(
            baseline["mean_stage_i_defense_time_per_false_alarm"], row["mean_stage_i_defense_time_per_false_alarm"]
        )
        row["red_stage_ii_time_pct"] = pct_reduction(
            baseline["mean_stage_ii_defense_time_per_false_alarm"], row["mean_stage_ii_defense_time_per_false_alarm"]
        )
        row["red_stage_i_cost_pct"] = pct_reduction(
            baseline["mean_stage_i_incremental_operating_cost_per_false_alarm"],
            row["mean_stage_i_incremental_operating_cost_per_false_alarm"],
        )
        row["red_stage_ii_cost_pct"] = pct_reduction(
            baseline["mean_stage_ii_incremental_operating_cost_per_false_alarm"],
            row["mean_stage_ii_incremental_operating_cost_per_false_alarm"],
        )



def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k, _ in PAPER_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)



def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [title for _, title in PAPER_COLUMNS]
    keys = [key for key, _ in PAPER_COLUMNS]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        vals: List[str] = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, (float, np.floating)):
                vals.append(fmt_num(value, digits=6))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_checks(path: Path, check_lines: List[str]) -> None:
    path.write_text("\n".join(check_lines) + "\n", encoding="utf-8")



def main() -> None:
    args = parse_args()

    strict_path = resolve_strict_path(args.strict)

    files: List[Tuple[str, str, str]] = [
        (args.baseline_label, args.baseline, args.baseline_tau_label),
        (args.main_label, args.main, args.main_tau_label),
    ]
    if strict_path is not None:
        files.append((args.strict_label, strict_path, args.strict_tau_label))

    loaded: List[Tuple[str, str, Dict[str, Any], str]] = []
    for row_label, path, tau_label in files:
        data = load_result(path)
        group_key = detect_group_key(data)
        tau_label = infer_tau_label(path, tau_label)
        loaded.append((row_label, path, data, group_key))
        print(f"[LOAD] {row_label}: {path}")
        print(f"       group_key = {group_key}, tau_label = {tau_label}")

    # ---- paired-comparison consistency checks ----
    check_lines: List[str] = []
    baseline_label, baseline_path, baseline_data, baseline_group = loaded[0]
    for row_label, path, data, group_key in loaded[1:]:
        check_lines.append(f"=== Compare {baseline_label} vs {row_label} ===")
        for field in SCALAR_FIELDS:
            a = get_group_value(baseline_data, field, baseline_group, float("nan"))
            b = get_group_value(data, field, group_key, float("nan"))
            same = False
            try:
                same = math.isclose(float(a), float(b), rel_tol=args.rtol, abs_tol=args.atol)
            except Exception:
                same = (a == b)
            check_lines.append(f"{field}: {same} | baseline={a} | {row_label}={b}")
        for field in LIST_FIELDS:
            a = get_group_value(baseline_data, field, baseline_group, []) or []
            b = get_group_value(data, field, group_key, []) or []
            same, detail = compare_lists(list(a), list(b), args.rtol, args.atol)
            check_lines.append(f"{field}: {same} | {detail}")
        check_lines.append("")

    # ---- build rows ----
    rows: List[Dict[str, Any]] = []
    for (row_label, path, data, group_key), (_, _, tau_label) in zip(loaded, files):
        rows.append(build_row(data, group_key, row_label, tau_label))

    # If strict path was auto-resolved, we want tau label inferred from filename.
    for row, (row_label, path, data, group_key) in zip(rows, loaded):
        row["tau_label"] = infer_tau_label(path, row["tau_label"])

    add_reduction_columns(rows)

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "clean_main_table.csv"
    md_path = out_dir / "clean_main_table.md"
    txt_path = out_dir / "clean_main_table_checks.txt"
    npy_path = out_dir / "clean_main_table.npy"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    write_checks(txt_path, check_lines)
    np.save(npy_path, {"rows": rows, "checks": check_lines}, allow_pickle=True)

    print("\n==== Saved ====")
    print(csv_path)
    print(md_path)
    print(txt_path)
    print(npy_path)

    print("\n==== Paper-ready main table preview ====")
    for row in rows:
        print(
            f"{row['row_label']:>10s} | tau={row['tau_label']:<6s} | "
            f"FAR={fmt_num(row['front_end_far'])} | trig={row['trigger_count']:>4d} | "
            f"uMTD={fmt_num(row['unnecessary_mtd_deployment_rate'])} | "
            f"fail/alarm={fmt_num(row['backend_failure_rate_per_false_alarm'])} | "
            f"t1/alarm={fmt_num(row['mean_stage_i_defense_time_per_false_alarm'])} | "
            f"t2/alarm={fmt_num(row['mean_stage_ii_defense_time_per_false_alarm'])} | "
            f"c1/alarm={fmt_num(row['mean_stage_i_incremental_operating_cost_per_false_alarm'])} | "
            f"c2/alarm={fmt_num(row['mean_stage_ii_incremental_operating_cost_per_false_alarm'])}"
        )

    print("\n==== Consistency checks ====")
    for line in check_lines:
        print(line)


if __name__ == "__main__":
    main()

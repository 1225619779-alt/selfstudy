from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

"""
Compare two clean false-alarm metric files produced by evaluation_event_trigger_clean.py.

This version keeps the original research logic unchanged:
- same file loading strategy (.npy -> dict)
- same consistency checks
- same core clean-case metrics

High-value additions for paper analysis:
1) neutral naming (no hard-coded baseline/gated wording required)
2) more paper-oriented metric labels
3) automatic incremental trade-off wording for tau-to-tau comparison
4) optional Markdown / CSV export
5) backward-compatible CLI aliases for --baseline / --gated
"""

# =========================
# Default file paths (kept for backward compatibility)
# =========================
DEFAULT_LEFT = "metric/case14/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
DEFAULT_RIGHT = "metric/case14/metric_event_trigger_clean_tau_0.021_mode_0_0.03_1.1.npy"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    paper_name: str
    unit: str
    decimals: int = 6
    is_rate: bool = False
    lower_is_better: Optional[bool] = None


CORE_METRICS: List[MetricSpec] = [
    MetricSpec("tau_verify", "Verification threshold $\\tau$", "raw", decimals=6, is_rate=False, lower_is_better=None),
    MetricSpec("total_clean_sample", "# Clean test samples", "count", decimals=0, is_rate=False, lower_is_better=None),
    MetricSpec("total_DDD_alarm", "# DDD alarms on clean data", "count", decimals=0, is_rate=False, lower_is_better=None),
    MetricSpec("total_trigger_after_verification", "# Post-verification MTD activations", "count", decimals=0, is_rate=False, lower_is_better=True),
    MetricSpec("total_skip_by_verification", "# Post-verification MTD suppressions", "count", decimals=0, is_rate=False, lower_is_better=False),
    MetricSpec("false_alarm_rate", "False-alarm rate (FAR)", "rate", decimals=4, is_rate=True, lower_is_better=True),
    MetricSpec("trigger_rate", "Post-verification activation rate", "rate", decimals=4, is_rate=True, lower_is_better=True),
    MetricSpec("skip_rate", "Post-verification suppression rate", "rate", decimals=4, is_rate=True, lower_is_better=False),
    MetricSpec("useless_mtd_rate", "Unnecessary MTD activation rate", "rate", decimals=4, is_rate=True, lower_is_better=True),
    MetricSpec("fail_per_alarm", "MTD failure rate per DDD alarm", "rate", decimals=4, is_rate=True, lower_is_better=True),
    MetricSpec("stage_one_time_per_alarm", "Stage-I execution time per DDD alarm", "time", decimals=4, is_rate=False, lower_is_better=True),
    MetricSpec("stage_two_time_per_alarm", "Stage-II execution time per DDD alarm", "time", decimals=4, is_rate=False, lower_is_better=True),
    MetricSpec("delta_cost_one_per_alarm", "Stage-I incremental cost per DDD alarm", "cost", decimals=4, is_rate=False, lower_is_better=True),
    MetricSpec("delta_cost_two_per_alarm", "Stage-II incremental cost per DDD alarm", "cost", decimals=4, is_rate=False, lower_is_better=True),
]

CONSISTENCY_SEQUENCE_FIELDS: Tuple[str, ...] = (
    "TP_DDD",
    "clean_alarm_idx",
    "verify_score",
)

CONSISTENCY_SCALAR_FIELDS: Tuple[str, ...] = (
    "total_clean_sample",
    "total_DDD_alarm",
    "false_alarm_rate",
)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two clean false-alarm metric files with consistency checks and paper-ready tables."
    )
    parser.add_argument(
        "--left",
        "--baseline",
        dest="left",
        type=str,
        default=DEFAULT_LEFT,
        help="Path to the first metric file (.npy). Alias: --baseline",
    )
    parser.add_argument(
        "--right",
        "--gated",
        dest="right",
        type=str,
        default=DEFAULT_RIGHT,
        help="Path to the second metric file (.npy). Alias: --gated",
    )
    parser.add_argument(
        "--left-label",
        dest="left_label",
        type=str,
        default=None,
        help="Neutral display label for the first file, e.g. 'tau=0.021' or 'Method A'.",
    )
    parser.add_argument(
        "--right-label",
        dest="right_label",
        type=str,
        default=None,
        help="Neutral display label for the second file, e.g. 'tau=0.030' or 'Method B'.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for numeric comparisons.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for numeric comparisons.",
    )
    parser.add_argument(
        "--export-md",
        type=str,
        default=None,
        help="Optional path to export a Markdown report.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional path to export the main summary table as CSV.",
    )
    parser.add_argument(
        "--no-percent-rates",
        action="store_true",
        help="Display rate metrics as raw fractions instead of percentages.",
    )
    return parser.parse_args()


# =========================
# Helpers
# =========================
def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def fmt_mtime(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def load_result(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError(f"Loaded object is not a dict: {type(data)}")
    return data


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def approx_equal_scalar(a: Any, b: Any, rtol: float, atol: float) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except Exception:
        return a == b


def pct_change(old: Any, new: Any) -> float:
    try:
        old_f = float(old)
        new_f = float(new)
        if math.isclose(old_f, 0.0, abs_tol=1e-15):
            return float("nan")
        return (new_f - old_f) / abs(old_f) * 100.0
    except Exception:
        return float("nan")


def pct_reduction(old: Any, new: Any) -> float:
    try:
        old_f = float(old)
        new_f = float(new)
        if math.isclose(old_f, 0.0, abs_tol=1e-15):
            return float("nan")
        return (old_f - new_f) / abs(old_f) * 100.0
    except Exception:
        return float("nan")


def fmt_plain_num(x: Any, digits: int = 6) -> str:
    try:
        xf = float(x)
        if math.isnan(xf):
            return "N/A"
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def fmt_metric_value(spec: MetricSpec, value: Any, percent_rates: bool = True) -> str:
    try:
        vf = float(value)
        if math.isnan(vf):
            return "N/A"
        if spec.is_rate and percent_rates:
            return f"{vf * 100.0:.{spec.decimals}f}%"
        if spec.unit == "count":
            if abs(vf - round(vf)) < 1e-12:
                return str(int(round(vf)))
            return f"{vf:.{spec.decimals}f}"
        return f"{vf:.{spec.decimals}f}"
    except Exception:
        return str(value)


def fmt_delta(spec: MetricSpec, left: Any, right: Any, percent_rates: bool = True) -> str:
    try:
        left_f = float(left)
        right_f = float(right)
        delta = right_f - left_f
        if spec.is_rate and percent_rates:
            return f"{delta * 100.0:+.{spec.decimals}f} pp"
        return f"{delta:+.{spec.decimals}f}"
    except Exception:
        return "N/A"


def normalize_sequence(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def compare_sequences(a: Sequence[Any], b: Sequence[Any], rtol: float, atol: float) -> Tuple[bool, str]:
    aa = normalize_sequence(a)
    bb = normalize_sequence(b)

    if len(aa) != len(bb):
        return False, f"length mismatch: {len(aa)} vs {len(bb)}"
    if len(aa) == 0:
        return True, "both empty"

    try:
        arr_a = np.asarray(aa, dtype=float)
        arr_b = np.asarray(bb, dtype=float)
        ok = np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        return bool(ok), "allclose" if ok else "not allclose"
    except Exception:
        ok = (aa == bb)
        return ok, "exact match" if ok else "not equal"


def detect_group_key(data: Dict[str, Any]) -> str:
    candidate_fields = [
        "TP_DDD",
        "verify_score",
        "trigger_after_verification",
        "skip_by_verification",
        "clean_alarm_idx",
    ]
    for field in candidate_fields:
        value = data.get(field)
        if isinstance(value, dict) and len(value) > 0:
            keys = list(value.keys())
            if len(keys) == 1:
                return keys[0]
            if "(0,0.0)" in keys:
                return "(0,0.0)"
            return keys[0]
    group_key = data.get("group_key")
    if isinstance(group_key, str):
        return group_key
    raise KeyError("Cannot detect group_key from metric file.")


def get_field_value(data: Dict[str, Any], field: str, group_key: str, default: Any = float("nan")) -> Any:
    value = data.get(field, default)
    if isinstance(value, dict):
        return value.get(group_key, default)
    return value


def infer_label(data: Dict[str, Any], fallback: str) -> str:
    tau = data.get("tau_verify", None)
    if tau is not None:
        try:
            return f"tau={float(tau):.3f}"
        except Exception:
            return f"tau={tau}"
    return fallback


def get_tau(data: Dict[str, Any], group_key: str) -> float:
    return safe_float(get_field_value(data, "tau_verify", group_key, float("nan")))


def is_reference_tau(tau: float) -> bool:
    return (not math.isnan(tau)) and tau < 0.0


def same_special_tradeoff_pair(t1: float, t2: float, tol: float = 1e-9) -> bool:
    if math.isnan(t1) or math.isnan(t2):
        return False
    return (
        (math.isclose(t1, 0.021, abs_tol=tol) and math.isclose(t2, 0.030, abs_tol=tol))
        or (math.isclose(t1, 0.030, abs_tol=tol) and math.isclose(t2, 0.021, abs_tol=tol))
    )


def infer_comparison_mode(left_summary: Dict[str, Any], right_summary: Dict[str, Any]) -> str:
    tau_l = safe_float(left_summary.get("tau_verify"))
    tau_r = safe_float(right_summary.get("tau_verify"))

    if is_reference_tau(tau_l) ^ is_reference_tau(tau_r):
        return "reference_vs_threshold"
    if (not math.isnan(tau_l)) and (not math.isnan(tau_r)) and not math.isclose(tau_l, tau_r, abs_tol=1e-12):
        return "threshold_tradeoff"
    return "generic"


def print_file_info(title: str, path: str, data: Dict[str, Any], group_key: str, label: str) -> None:
    print_header(title)
    print(f"label          : {label}")
    print(f"path           : {os.path.abspath(path)}")
    print(f"modified time  : {fmt_mtime(path)}")
    print(f"tau_verify     : {data.get('tau_verify', 'N/A')}")
    print(f"group_key      : {group_key}")
    print(f"top-level keys : {list(data.keys())}")


def run_consistency_checks(
    left: Dict[str, Any],
    right: Dict[str, Any],
    left_label: str,
    right_label: str,
    group_key_left: str,
    group_key_right: str,
    rtol: float,
    atol: float,
) -> List[Dict[str, Any]]:
    print_header("CONSISTENCY CHECKS")

    results: List[Dict[str, Any]] = []

    same_group_key = (group_key_left == group_key_right)
    print(f"[group_key] same? {same_group_key} | {left_label}={group_key_left} | {right_label}={group_key_right}")
    results.append(
        {
            "check": "group_key",
            "passed": same_group_key,
            "detail": f"{left_label}={group_key_left}, {right_label}={group_key_right}",
        }
    )

    for field in CONSISTENCY_SCALAR_FIELDS:
        lv = get_field_value(left, field, group_key_left)
        rv = get_field_value(right, field, group_key_right)
        ok = approx_equal_scalar(lv, rv, rtol=rtol, atol=atol)
        print(f"[{field}] same? {ok} | {left_label}={lv} | {right_label}={rv}")
        results.append(
            {
                "check": field,
                "passed": ok,
                "detail": f"{left_label}={lv}, {right_label}={rv}",
            }
        )

    for field in CONSISTENCY_SEQUENCE_FIELDS:
        lv = get_field_value(left, field, group_key_left, [])
        rv = get_field_value(right, field, group_key_right, [])
        ok, detail = compare_sequences(lv, rv, rtol=rtol, atol=atol)
        print(f"[{field}] same? {ok} | {detail}")
        results.append(
            {
                "check": field,
                "passed": ok,
                "detail": detail,
            }
        )

    trigger_left = safe_float(get_field_value(left, "total_trigger_after_verification", group_key_left, 0.0))
    skip_left = safe_float(get_field_value(left, "total_skip_by_verification", group_key_left, 0.0))
    alarm_left = safe_float(get_field_value(left, "total_DDD_alarm", group_key_left, 0.0))
    trigger_right = safe_float(get_field_value(right, "total_trigger_after_verification", group_key_right, 0.0))
    skip_right = safe_float(get_field_value(right, "total_skip_by_verification", group_key_right, 0.0))
    alarm_right = safe_float(get_field_value(right, "total_DDD_alarm", group_key_right, 0.0))

    ok_left = approx_equal_scalar(trigger_left + skip_left, alarm_left, rtol=rtol, atol=atol)
    ok_right = approx_equal_scalar(trigger_right + skip_right, alarm_right, rtol=rtol, atol=atol)

    print("\nGate sanity:")
    print(
        f"{left_label}: total_trigger_after_verification + total_skip_by_verification = "
        f"{fmt_plain_num(trigger_left, 0)} + {fmt_plain_num(skip_left, 0)} = {fmt_plain_num(trigger_left + skip_left, 0)} | "
        f"total_DDD_alarm = {fmt_plain_num(alarm_left, 0)} | passed={ok_left}"
    )
    print(
        f"{right_label}: total_trigger_after_verification + total_skip_by_verification = "
        f"{fmt_plain_num(trigger_right, 0)} + {fmt_plain_num(skip_right, 0)} = {fmt_plain_num(trigger_right + skip_right, 0)} | "
        f"total_DDD_alarm = {fmt_plain_num(alarm_right, 0)} | passed={ok_right}"
    )
    results.extend(
        [
            {"check": f"gate_sanity::{left_label}", "passed": ok_left, "detail": f"trigger+skip={trigger_left + skip_left}, total_DDD_alarm={alarm_left}"},
            {"check": f"gate_sanity::{right_label}", "passed": ok_right, "detail": f"trigger+skip={trigger_right + skip_right}, total_DDD_alarm={alarm_right}"},
        ]
    )

    return results


def summarize(data: Dict[str, Any], group_key: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for spec in CORE_METRICS:
        summary[spec.key] = get_field_value(data, spec.key, group_key)
    return summary


def build_summary_rows(
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    left_label: str,
    right_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in CORE_METRICS:
        lv = left_summary.get(spec.key, float("nan"))
        rv = right_summary.get(spec.key, float("nan"))
        rows.append(
            {
                "metric_key": spec.key,
                "paper_metric": spec.paper_name,
                "unit": spec.unit,
                f"{left_label}_raw": lv,
                f"{right_label}_raw": rv,
                "delta_abs": safe_float(rv) - safe_float(lv) if not (is_nan(lv) or is_nan(rv)) else float("nan"),
                "change_pct_of_left": pct_change(lv, rv),
                "reduction_pct_from_left": pct_reduction(lv, rv),
            }
        )
    return rows


def print_summary_table(
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    left_label: str,
    right_label: str,
    percent_rates: bool,
) -> None:
    print_header("SUMMARY TABLE (PAPER-READY, CLEAN FALSE-ALARM)")
    header = (
        f"{'paper metric [raw key]':<52} "
        f"{left_label:>18} "
        f"{right_label:>18} "
        f"{'abs. delta':>18}"
    )
    print(header)
    print("-" * len(header))

    for spec in CORE_METRICS:
        lv = left_summary.get(spec.key, float("nan"))
        rv = right_summary.get(spec.key, float("nan"))
        metric_name = f"{spec.paper_name} [{spec.key}]"
        print(
            f"{metric_name:<52} "
            f"{fmt_metric_value(spec, lv, percent_rates):>18} "
            f"{fmt_metric_value(spec, rv, percent_rates):>18} "
            f"{fmt_delta(spec, lv, rv, percent_rates):>18}"
        )


def print_effect_table(
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    left_label: str,
    right_label: str,
    percent_rates: bool,
) -> None:
    print_header(f"RELATIVE CHANGE ({right_label} RELATIVE TO {left_label})")
    header = f"{'paper metric [raw key]':<52} {'% change vs left':>18} {'note':>18}"
    print(header)
    print("-" * len(header))

    for spec in CORE_METRICS:
        if spec.key in {"tau_verify", "total_clean_sample", "total_DDD_alarm"}:
            continue
        lv = left_summary.get(spec.key, float("nan"))
        rv = right_summary.get(spec.key, float("nan"))
        change = pct_change(lv, rv)
        if spec.lower_is_better is True:
            note = "lower is better"
        elif spec.lower_is_better is False:
            note = "higher is better"
        else:
            note = "descriptive"
        metric_name = f"{spec.paper_name} [{spec.key}]"
        print(f"{metric_name:<52} {fmt_plain_num(change, 2) + '%':>18} {note:>18}")


def pick_left_right_for_tradeoff(
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    left_label: str,
    right_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    tau_l = safe_float(left_summary.get("tau_verify"))
    tau_r = safe_float(right_summary.get("tau_verify"))
    if (not math.isnan(tau_l)) and (not math.isnan(tau_r)) and tau_l <= tau_r:
        return left_summary, right_summary, left_label, right_label
    return right_summary, left_summary, right_label, left_label


def print_takeaways(
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    left_label: str,
    right_label: str,
) -> List[str]:
    print_header("AUTO TAKEAWAYS")
    takeaways: List[str] = []

    mode = infer_comparison_mode(left_summary, right_summary)

    if mode == "reference_vs_threshold":
        trigger_red = pct_reduction(left_summary["trigger_rate"], right_summary["trigger_rate"])
        useless_red = pct_reduction(left_summary["useless_mtd_rate"], right_summary["useless_mtd_rate"])
        fail_red = pct_reduction(left_summary["fail_per_alarm"], right_summary["fail_per_alarm"])
        t1_red = pct_reduction(left_summary["stage_one_time_per_alarm"], right_summary["stage_one_time_per_alarm"])
        t2_red = pct_reduction(left_summary["stage_two_time_per_alarm"], right_summary["stage_two_time_per_alarm"])
        c1_red = pct_reduction(left_summary["delta_cost_one_per_alarm"], right_summary["delta_cost_one_per_alarm"])
        c2_red = pct_reduction(left_summary["delta_cost_two_per_alarm"], right_summary["delta_cost_two_per_alarm"])

        takeaways = [
            f"Compared with {left_label}, {right_label} reduces the post-verification activation rate by {fmt_plain_num(trigger_red, 2)}%.",
            f"Compared with {left_label}, {right_label} reduces the unnecessary MTD activation rate by {fmt_plain_num(useless_red, 2)}%.",
            f"The per-alarm MTD failure rate changes by {fmt_plain_num(fail_red, 2)}% (reported as reduction when positive).",
            f"The Stage-I and Stage-II time per DDD alarm change by {fmt_plain_num(t1_red, 2)}% and {fmt_plain_num(t2_red, 2)}%, respectively.",
            f"The Stage-I and Stage-II incremental cost per DDD alarm change by {fmt_plain_num(c1_red, 2)}% and {fmt_plain_num(c2_red, 2)}%, respectively.",
        ]
    elif mode == "threshold_tradeoff":
        low, high, low_label, high_label = pick_left_right_for_tradeoff(left_summary, right_summary, left_label, right_label)

        trig_change = pct_change(low["trigger_rate"], high["trigger_rate"])
        skip_change = pct_change(low["skip_rate"], high["skip_rate"])
        fail_change = pct_change(low["fail_per_alarm"], high["fail_per_alarm"])
        t1_change = pct_change(low["stage_one_time_per_alarm"], high["stage_one_time_per_alarm"])
        t2_change = pct_change(low["stage_two_time_per_alarm"], high["stage_two_time_per_alarm"])
        c1_change = pct_change(low["delta_cost_one_per_alarm"], high["delta_cost_one_per_alarm"])
        c2_change = pct_change(low["delta_cost_two_per_alarm"], high["delta_cost_two_per_alarm"])

        if same_special_tradeoff_pair(safe_float(low["tau_verify"]), safe_float(high["tau_verify"])):
            takeaways.append(
                f"Incremental threshold trade-off: increasing the verification threshold from {low_label} to {high_label} should be interpreted as a threshold tuning step, not as the total effect of adding verification gating."
            )
        else:
            takeaways.append(
                f"Incremental threshold trade-off: moving from {low_label} to {high_label} is a threshold tuning comparison rather than a reference-vs-gated total comparison."
            )

        takeaways.extend(
            [
                f"When moving from {low_label} to {high_label}, the post-verification activation rate changes from {fmt_plain_num(low['trigger_rate'], 6)} to {fmt_plain_num(high['trigger_rate'], 6)} ({fmt_plain_num(trig_change, 2)}% relative change).",
                f"Over the same threshold step, the post-verification suppression rate changes from {fmt_plain_num(low['skip_rate'], 6)} to {fmt_plain_num(high['skip_rate'], 6)} ({fmt_plain_num(skip_change, 2)}% relative change).",
                f"The per-alarm burden changes are incremental: fail rate {fmt_plain_num(fail_change, 2)}%, Stage-I time {fmt_plain_num(t1_change, 2)}%, Stage-II time {fmt_plain_num(t2_change, 2)}%, Stage-I cost {fmt_plain_num(c1_change, 2)}%, Stage-II cost {fmt_plain_num(c2_change, 2)}%.",
                "This wording avoids overstating threshold-to-threshold changes as if they were total gains over an always-trigger reference.",
            ]
        )
    else:
        takeaways = [
            f"This is a neutral pairwise comparison between {left_label} and {right_label}.",
            f"Post-verification activation rate: {fmt_plain_num(left_summary['trigger_rate'], 6)} -> {fmt_plain_num(right_summary['trigger_rate'], 6)}.",
            f"Unnecessary MTD activation rate: {fmt_plain_num(left_summary['useless_mtd_rate'], 6)} -> {fmt_plain_num(right_summary['useless_mtd_rate'], 6)}.",
            f"Stage-I/II time per DDD alarm: {fmt_plain_num(left_summary['stage_one_time_per_alarm'], 6)} / {fmt_plain_num(left_summary['stage_two_time_per_alarm'], 6)} -> {fmt_plain_num(right_summary['stage_one_time_per_alarm'], 6)} / {fmt_plain_num(right_summary['stage_two_time_per_alarm'], 6)}.",
        ]

    for line in takeaways:
        print(f"- {line}")
    return takeaways


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def export_csv(
    path: str,
    rows: Iterable[Dict[str, Any]],
    left_label: str,
    right_label: str,
) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "metric_key",
        "paper_metric",
        "unit",
        f"{left_label}_raw",
        f"{right_label}_raw",
        "delta_abs",
        "change_pct_of_left",
        "reduction_pct_from_left",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown_report(
    left_path: str,
    right_path: str,
    left_label: str,
    right_label: str,
    left_summary: Dict[str, Any],
    right_summary: Dict[str, Any],
    consistency: List[Dict[str, Any]],
    takeaways: List[str],
    percent_rates: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Clean False-Alarm Comparison Report")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- **{left_label}**: `{left_path}`")
    lines.append(f"- **{right_label}**: `{right_path}`")
    lines.append("")
    lines.append("## Consistency checks")
    lines.append("")
    lines.append("| Check | Passed | Detail |")
    lines.append("|---|---:|---|")
    for item in consistency:
        status = "True" if item["passed"] else "False"
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {item['check']} | {status} | {detail} |")
    lines.append("")
    lines.append("## Main summary table")
    lines.append("")
    lines.append(f"| Paper metric | Raw key | {left_label} | {right_label} | Abs. delta (right-left) |")
    lines.append("|---|---|---:|---:|---:|")
    for spec in CORE_METRICS:
        lv = left_summary.get(spec.key, float("nan"))
        rv = right_summary.get(spec.key, float("nan"))
        lines.append(
            f"| {spec.paper_name} | `{spec.key}` | {fmt_metric_value(spec, lv, percent_rates)} | {fmt_metric_value(spec, rv, percent_rates)} | {fmt_delta(spec, lv, rv, percent_rates)} |"
        )
    lines.append("")
    lines.append("## Auto takeaways")
    lines.append("")
    for line in takeaways:
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def export_markdown(path: str, content: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()

    left = load_result(args.left)
    right = load_result(args.right)

    group_key_left = detect_group_key(left)
    group_key_right = detect_group_key(right)

    left_label = args.left_label or infer_label(left, "Method A")
    right_label = args.right_label or infer_label(right, "Method B")

    print_file_info("LEFT FILE", args.left, left, group_key_left, left_label)
    print_file_info("RIGHT FILE", args.right, right, group_key_right, right_label)

    consistency = run_consistency_checks(
        left=left,
        right=right,
        left_label=left_label,
        right_label=right_label,
        group_key_left=group_key_left,
        group_key_right=group_key_right,
        rtol=args.rtol,
        atol=args.atol,
    )

    left_summary = summarize(left, group_key_left)
    right_summary = summarize(right, group_key_right)

    percent_rates = not args.no_percent_rates
    print_summary_table(
        left_summary=left_summary,
        right_summary=right_summary,
        left_label=left_label,
        right_label=right_label,
        percent_rates=percent_rates,
    )
    print_effect_table(
        left_summary=left_summary,
        right_summary=right_summary,
        left_label=left_label,
        right_label=right_label,
        percent_rates=percent_rates,
    )
    takeaways = print_takeaways(
        left_summary=left_summary,
        right_summary=right_summary,
        left_label=left_label,
        right_label=right_label,
    )

    rows = build_summary_rows(
        left_summary=left_summary,
        right_summary=right_summary,
        left_label=left_label,
        right_label=right_label,
    )

    if args.export_csv:
        export_csv(args.export_csv, rows, left_label, right_label)
        print(f"\nSaved CSV summary to: {os.path.abspath(args.export_csv)}")

    if args.export_md:
        md_content = build_markdown_report(
            left_path=args.left,
            right_path=args.right,
            left_label=left_label,
            right_label=right_label,
            left_summary=left_summary,
            right_summary=right_summary,
            consistency=consistency,
            takeaways=takeaways,
            percent_rates=percent_rates,
        )
        export_markdown(args.export_md, md_content)
        print(f"Saved Markdown report to: {os.path.abspath(args.export_md)}")


if __name__ == "__main__":
    main()

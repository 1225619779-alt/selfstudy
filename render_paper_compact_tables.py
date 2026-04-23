from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROTECTED_GROUPS = ["(2,0.3)", "(3,0.2)", "(3,0.3)"]
ALL_ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


CASE14_BASELINE_CLEAN = "metric/case14/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
CASE14_MAIN_CLEAN = "metric/case14/metric_event_trigger_clean_tau_0.033183758162_mode_0_0.03_1.1.npy"
CASE14_STRICT_CLEAN = "metric/case14/metric_event_trigger_clean_tau_0.036751313717_mode_0_0.03_1.1.npy"
CASE14_BASELINE_MIXED = "metric/case14/metric_mixed_timeline_tau_-1.0.npy"
CASE14_MAIN_MIXED = "metric/case14/metric_mixed_timeline_tau_0.03318.npy"
CASE14_STRICT_MIXED = "metric/case14/metric_mixed_timeline_tau_0.03675.npy"
CASE14_GATE_ABLATION = "metric/case14/metric_gate_ablation_summary_0.03318_0.03675.csv"

CASE39_BASELINE_CLEAN = "metric/case39/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
CASE39_MAIN_CLEAN = "metric/case39/metric_event_trigger_clean_tau_0.013319196253_mode_0_0.03_1.1.npy"
CASE39_STRICT_CLEAN = "metric/case39/metric_event_trigger_clean_tau_0.016153267226_mode_0_0.03_1.1.npy"
CASE39_MAIN_ATTACK = "metric/case39/metric_event_trigger_tau_0.013319196253_mode_0_0.03_1.1.npy"
CASE39_STRICT_ATTACK = "metric/case39/metric_event_trigger_tau_0.016153267226_mode_0_0.03_1.1.npy"
CASE39_GATE_ABLATION = "metric/case39/gate_ablation_case39.csv"
CASE39_MINIMAL_ABLATION = "metric/case39/minimal_score_ablation.csv"

OUT_MD = "reports/paper_compact_tables.md"
OUT_JSON = "reports/paper_compact_tables.json"


def load_dict(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def detect_clean_group_key(metric: Dict[str, Any]) -> str:
    for field in ["total_clean_sample", "total_DDD_alarm", "total_trigger_after_verification"]:
        if isinstance(metric.get(field), dict) and metric[field]:
            keys = list(metric[field].keys())
            if "(0,0.0)" in keys:
                return "(0,0.0)"
            return keys[0]
    raise KeyError(f"Cannot detect clean group key for metric.")


def clean_row(metric_path: str, label: str) -> Dict[str, Any]:
    d = load_dict(metric_path)
    g = detect_clean_group_key(d)
    return {
        "label": label,
        "total_clean_sample": int(d["total_clean_sample"][g]),
        "total_DDD_alarm": int(d["total_DDD_alarm"][g]),
        "total_trigger_after_verification": int(d["total_trigger_after_verification"][g]),
        "trigger_rate": float(d["trigger_rate"][g]),
        "useless_mtd_rate": float(d["useless_mtd_rate"][g]),
        "fail_per_alarm": float(d["fail_per_alarm"][g]),
        "stage_one_time_per_alarm": float(d["stage_one_time_per_alarm"][g]),
        "stage_two_time_per_alarm": float(d["stage_two_time_per_alarm"][g]),
        "delta_cost_one_per_alarm": float(d["delta_cost_one_per_alarm"][g]),
        "delta_cost_two_per_alarm": float(d["delta_cost_two_per_alarm"][g]),
    }


def attack_summary(metric_path: str, label: str) -> Dict[str, Any]:
    d = load_dict(metric_path)
    total_alarms = 0
    total_triggers = 0
    group_arrs: Dict[str, float] = {}
    for g in ALL_ATTACK_GROUPS:
        alarms = int(np.asarray(d["TP_DDD"][g], dtype=bool).sum())
        triggers = int(np.asarray(d["trigger_after_verification"][g], dtype=bool).sum())
        total_alarms += alarms
        total_triggers += triggers
        group_arrs[g] = float(triggers / alarms) if alarms else float("nan")
    protected_min = float(np.nanmin(np.asarray([group_arrs[g] for g in PROTECTED_GROUPS], dtype=float)))
    return {
        "label": label,
        "overall_arr": float(total_triggers / total_alarms) if total_alarms else float("nan"),
        "protected_min_arr": protected_min,
        "group_arrs": group_arrs,
    }


def mixed_row(metric_path: str, label: str) -> Dict[str, Any]:
    d = load_dict(metric_path)
    s = d["summary"]
    return {
        "label": label,
        "total_DDD_alarm": int(s["total_DDD_alarm"]),
        "total_trigger_after_gate": int(s["total_trigger_after_gate"]),
        "total_skip_by_gate": int(s["total_skip_by_gate"]),
        "total_backend_fail": int(s["total_backend_fail"]),
        "final_cumulative_stage_time": float(s["final_cumulative_stage_time"]),
        "final_cumulative_delta_cost": float(s["final_cumulative_delta_cost"]),
    }


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_num_row(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def select_comparator_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        rr = to_num_row(row)
        if rr["budget_label"] in {"Main OP", "Strict OP"} and rr["score_name"] in {"detector_loss", "proposed_phys_score"}:
            out.append(rr)
    return out


def fmt(x: Any) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return "nan"
        return f"{x:.6f}"
    return str(x)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    case14_clean_rows = [
        clean_row(CASE14_BASELINE_CLEAN, "baseline"),
        clean_row(CASE14_MAIN_CLEAN, "main"),
        clean_row(CASE14_STRICT_CLEAN, "strict"),
    ]
    case14_mixed_rows = [
        mixed_row(CASE14_BASELINE_MIXED, "baseline"),
        mixed_row(CASE14_MAIN_MIXED, "main"),
        mixed_row(CASE14_STRICT_MIXED, "strict"),
    ]
    case14_gate_rows = select_comparator_rows(read_csv_rows(CASE14_GATE_ABLATION))

    case39_clean_rows = [
        clean_row(CASE39_BASELINE_CLEAN, "baseline"),
        clean_row(CASE39_MAIN_CLEAN, "main"),
        clean_row(CASE39_STRICT_CLEAN, "strict"),
    ]
    case39_attack_rows = [
        attack_summary(CASE39_MAIN_ATTACK, "main"),
        attack_summary(CASE39_STRICT_ATTACK, "strict"),
    ]
    case39_gate_rows = select_comparator_rows(read_csv_rows(CASE39_GATE_ABLATION))
    case39_ablation_rows = [to_num_row(r) for r in read_csv_rows(CASE39_MINIMAL_ABLATION)]

    payload = {
        "case14_clean_exact": case14_clean_rows,
        "case14_mixed": case14_mixed_rows,
        "case14_gate_comparator": case14_gate_rows,
        "case39_clean_exact": case39_clean_rows,
        "case39_attack_exact": case39_attack_rows,
        "case39_gate_comparator": case39_gate_rows,
        "case39_minimal_score_ablation": case39_ablation_rows,
    }

    md_parts = [
        "# Paper Compact Tables",
        "",
        "## Case14 Clean Exact",
        markdown_table(
            case14_clean_rows,
            [
                "label",
                "total_clean_sample",
                "total_DDD_alarm",
                "total_trigger_after_verification",
                "trigger_rate",
                "useless_mtd_rate",
                "fail_per_alarm",
                "stage_one_time_per_alarm",
                "stage_two_time_per_alarm",
                "delta_cost_one_per_alarm",
                "delta_cost_two_per_alarm",
            ],
        ),
        "",
        "## Case14 Mixed Timeline",
        markdown_table(
            case14_mixed_rows,
            [
                "label",
                "total_DDD_alarm",
                "total_trigger_after_gate",
                "total_skip_by_gate",
                "total_backend_fail",
                "final_cumulative_stage_time",
                "final_cumulative_delta_cost",
            ],
        ),
        "",
        "## Case14 Gate Comparator",
        markdown_table(
            case14_gate_rows,
            [
                "budget_label",
                "score_name",
                "trigger_rate",
                "fail_per_alarm",
                "stage_two_time_per_alarm",
                "delta_cost_two_per_alarm",
                "attack_retention_overall",
                "strong_retention",
            ],
        ),
        "",
        "## Case39 Stress Benchmark Clean Exact",
        markdown_table(
            case39_clean_rows,
            [
                "label",
                "total_clean_sample",
                "total_DDD_alarm",
                "total_trigger_after_verification",
                "trigger_rate",
                "useless_mtd_rate",
                "fail_per_alarm",
                "stage_two_time_per_alarm",
                "delta_cost_two_per_alarm",
            ],
        ),
        "",
        "## Case39 Stress Benchmark Attack Exact",
        markdown_table(
            case39_attack_rows,
            [
                "label",
                "overall_arr",
                "protected_min_arr",
            ],
        ),
        "",
        "## Case39 Gate Comparator",
        markdown_table(
            case39_gate_rows,
            [
                "budget_label",
                "score_name",
                "trigger_rate",
                "fail_per_alarm",
                "stage_two_time_per_alarm",
                "delta_cost_two_per_alarm",
                "attack_retention_overall",
                "strong_retention",
            ],
        ),
        "",
        "## Case39 Minimal Score Ablation",
        markdown_table(
            case39_ablation_rows,
            [
                "regime",
                "score_family",
                "tau_valid",
                "valid_clean_trigger_count",
                "test_clean_trigger_count",
                "test_attack_trigger_count",
                "test_attack_overall_arr",
                "test_attack_protected_min_arr",
                "test_clean_backend_mtd_fail_count",
                "test_clean_stage_two_time_per_alarm",
                "test_clean_delta_cost_two_per_alarm",
                "test_clean_fail_per_alarm",
                "test_clean_fail_per_trigger",
            ],
        ),
        "",
        "## Positioning",
        "- case14 remains the main detailed evidence.",
        "- case39 is an additional stress-test / limitation benchmark.",
        "- Detector-loss gate retains slightly more attack alarms, while the recovery-aware physical score is more backend-aware in stage-two burden/cost.",
        "- Do not claim robust end-to-end case39 backend success.",
        "",
    ]

    Path(OUT_MD).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_MD).write_text("\n".join(md_parts), encoding="utf-8")
    Path(OUT_JSON).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved md: {OUT_MD}")
    print(f"Saved json: {OUT_JSON}")


if __name__ == "__main__":
    main()

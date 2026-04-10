#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

METHOD_KEYS = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]
SLOTS = ["1", "2"]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_stage(summary: Dict[str, Any]) -> str:
    for k in ["stage", "label", "native_case39_stage"]:
        if k in summary and isinstance(summary[k], str):
            return summary[k]
    # derive from path-like fields when possible
    if "used_summary_path" in summary:
        return Path(str(summary["used_summary_path"]).strip()).stem
    return "unknown_stage"


def merged_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if "merged_8_holdouts" in summary:
        return summary["merged_8_holdouts"]
    raise KeyError("Summary missing 'merged_8_holdouts'")


def collect_holdout_rows(agg_v1: Dict[str, Any], agg_v2: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    rows: Dict[str, Dict[str, List[Dict[str, float]]]] = {s: {m: [] for m in METHOD_KEYS} for s in SLOTS}
    for agg in [agg_v1, agg_v2]:
        for holdout in agg.get("per_holdout_results", []):
            sbr = holdout.get("slot_budget_results", {})
            for slot in SLOTS:
                slot_res = sbr.get(slot, {})
                for method in METHOD_KEYS:
                    if method in slot_res:
                        m = slot_res[method]
                        rows[slot][method].append({
                            "recall": float(m["weighted_attack_recall_no_backend_fail"]),
                            "unnecessary": float(m["unnecessary_mtd_count"]),
                            "cost": float(m["average_service_cost_per_step"]),
                            "served_ratio": float(m["pred_expected_consequence_served_ratio"]),
                            "delay_p95": float(m["queue_delay_p95"]),
                        })
    return rows


def paired_deltas(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> Dict[str, Dict[str, float | List[float]]]:
    if len(a) != len(b):
        raise ValueError(f"Paired length mismatch: {len(a)} vs {len(b)}")
    metrics = ["recall", "unnecessary", "cost", "served_ratio", "delay_p95"]
    out: Dict[str, Dict[str, float | List[float]]] = {}
    for m in metrics:
        vals = [x[m] - y[m] for x, y in zip(a, b)]
        out[m] = {
            "mean": mean(vals) if vals else 0.0,
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "raw_deltas": vals,
        }
    return out


def compare_stage(stage_name: str, summary: Dict[str, Any], agg_v1: Dict[str, Any], agg_v2: Dict[str, Any]) -> Dict[str, Any]:
    merged = merged_from_summary(summary)
    rows = collect_holdout_rows(agg_v1, agg_v2)
    out = {
        "stage": get_stage(summary),
        "label": stage_name,
        "merged_8_holdouts": merged,
        "n_holdouts": sum(len(rows[s][METHOD_KEYS[0]]) for s in SLOTS) // 2,
        "paired": {},
    }
    for slot in SLOTS:
        out["paired"][slot] = {
            "oracle_vs_phase3": paired_deltas(rows[slot]["phase3_oracle_upgrade"], rows[slot]["phase3_proposed"]),
            "oracle_vs_topk_expected": paired_deltas(rows[slot]["phase3_oracle_upgrade"], rows[slot]["topk_expected_consequence"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust stage comparison for case39 across transfer/local-retune outputs")
    ap.add_argument("--transfer_summary", required=True)
    ap.add_argument("--transfer_v1", required=True)
    ap.add_argument("--transfer_v2", required=True)
    ap.add_argument("--local_protected_summary", required=True)
    ap.add_argument("--local_protected_v1", required=True)
    ap.add_argument("--local_protected_v2", required=True)
    ap.add_argument("--local_unconstrained_summary", required=True)
    ap.add_argument("--local_unconstrained_v1", required=True)
    ap.add_argument("--local_unconstrained_v2", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    transfer_summary = load_json(args.transfer_summary)
    transfer_v1 = load_json(args.transfer_v1)
    transfer_v2 = load_json(args.transfer_v2)
    lp_summary = load_json(args.local_protected_summary)
    lp_v1 = load_json(args.local_protected_v1)
    lp_v2 = load_json(args.local_protected_v2)
    lu_summary = load_json(args.local_unconstrained_summary)
    lu_v1 = load_json(args.local_unconstrained_v1)
    lu_v2 = load_json(args.local_unconstrained_v2)

    stages = {
        "transfer_frozen_dev": compare_stage("transfer_frozen_dev", transfer_summary, transfer_v1, transfer_v2),
        "local_retune_protectedec": compare_stage("local_retune_protectedec", lp_summary, lp_v1, lp_v2),
        "local_retune_unconstrained": compare_stage("local_retune_unconstrained", lu_summary, lu_v1, lu_v2),
    }

    # high-level comparisons on merged means
    cross: Dict[str, Any] = {}
    for slot in SLOTS:
        cross[slot] = {}
        tf = stages["transfer_frozen_dev"]["merged_8_holdouts"][slot]["phase3_oracle_upgrade"]
        lp = stages["local_retune_protectedec"]["merged_8_holdouts"][slot]["phase3_oracle_upgrade"]
        lu = stages["local_retune_unconstrained"]["merged_8_holdouts"][slot]["phase3_oracle_upgrade"]
        cross[slot]["transfer_vs_local_protectedec"] = {
            "delta_recall": tf["mean_recall"] - lp["mean_recall"],
            "delta_unnecessary": tf["mean_unnecessary"] - lp["mean_unnecessary"],
            "delta_cost": tf["mean_cost"] - lp["mean_cost"],
        }
        cross[slot]["transfer_vs_local_unconstrained"] = {
            "delta_recall": tf["mean_recall"] - lu["mean_recall"],
            "delta_unnecessary": tf["mean_unnecessary"] - lu["mean_unnecessary"],
            "delta_cost": tf["mean_cost"] - lu["mean_cost"],
        }

    out = {
        "method": "case39_stage_compare_significance_v2",
        "stages": stages,
        "cross_stage_merged_deltas": cross,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({
        "output": str(out_path),
        "transfer_stage": stages["transfer_frozen_dev"]["stage"],
        "local_protected_stage": stages["local_retune_protectedec"]["stage"],
        "local_unconstrained_stage": stages["local_retune_unconstrained"]["stage"],
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

METRICS = [
    "weighted_attack_recall_no_backend_fail",
    "unnecessary_mtd_count",
    "queue_delay_p95",
    "average_service_cost_per_step",
    "pred_expected_consequence_served_ratio",
]


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stats(vals: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _paired(rows: List[Dict[str, Any]], a_key: str, b_key: str) -> Dict[str, Any]:
    def _metric_delta(metric: str) -> List[float]:
        return [float(r[a_key][metric]) - float(r[b_key][metric]) for r in rows]

    delta_rec = _metric_delta("weighted_attack_recall_no_backend_fail")
    delta_un = _metric_delta("unnecessary_mtd_count")
    delta_del = _metric_delta("queue_delay_p95")
    delta_cost = _metric_delta("average_service_cost_per_step")
    delta_ec = _metric_delta("pred_expected_consequence_served_ratio")

    arr_rec = np.asarray(delta_rec, dtype=float)
    arr_un = np.asarray(delta_un, dtype=float)
    arr_del = np.asarray(delta_del, dtype=float)
    arr_cost = np.asarray(delta_cost, dtype=float)
    arr_ec = np.asarray(delta_ec, dtype=float)

    return {
        "delta_recall": _stats(arr_rec),
        "delta_unnecessary": _stats(arr_un),
        "delta_delay_p95": _stats(arr_del),
        "delta_cost_per_step": _stats(arr_cost),
        "delta_pred_expected_consequence_served_ratio": _stats(arr_ec),
        "wins_on_recall": int(np.sum(arr_rec > 0)),
        "ties_on_recall": int(np.sum(arr_rec == 0)),
        "lower_unnecessary": int(np.sum(arr_un < 0)),
        "lower_delay": int(np.sum(arr_del < 0)),
        "lower_cost": int(np.sum(arr_cost < 0)),
        "higher_served_ratio": int(np.sum(arr_ec > 0)),
    }


def _family_grouped(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family_tag", "unknown"))].append(row)
    return dict(grouped)


def _policy_stats(rows: List[Dict[str, Any]], policy_key: str) -> Dict[str, Dict[str, float]]:
    return {metric: _stats(float(r[policy_key][metric]) for r in rows) for metric in METRICS}


def _scaled_formal_decision(slot_budget_aggregates: Dict[str, Any], n_holdouts: int) -> Dict[str, Any]:
    threshold_required = int(math.ceil(0.875 * float(n_holdouts)))
    per_slot: Dict[str, Any] = {}
    threshold_ok = True
    efficiency_ok = True
    recall_ok = True

    for slot, payload in slot_budget_aggregates.items():
        wins_vs_best_threshold = int(payload["paired_stats"]["oracle_vs_best_threshold"]["wins_on_recall"])
        oracle_stats = payload["policy_stats"]["phase3_oracle_upgrade"]
        phase3_stats = payload["policy_stats"]["phase3_proposed"]

        mean_recall_oracle = float(oracle_stats["weighted_attack_recall_no_backend_fail"]["mean"])
        mean_recall_phase3 = float(phase3_stats["weighted_attack_recall_no_backend_fail"]["mean"])
        mean_unnecessary_oracle = float(oracle_stats["unnecessary_mtd_count"]["mean"])
        mean_unnecessary_phase3 = float(phase3_stats["unnecessary_mtd_count"]["mean"])
        mean_cost_oracle = float(oracle_stats["average_service_cost_per_step"]["mean"])
        mean_cost_phase3 = float(phase3_stats["average_service_cost_per_step"]["mean"])

        slot_threshold_ok = wins_vs_best_threshold >= threshold_required
        slot_efficiency_ok = (
            mean_recall_oracle >= mean_recall_phase3 - 0.01
            and mean_unnecessary_oracle < mean_unnecessary_phase3
            and mean_cost_oracle <= mean_cost_phase3
        )
        slot_recall_ok = (
            mean_recall_oracle > mean_recall_phase3
            and mean_unnecessary_oracle <= mean_unnecessary_phase3 + 1.0
        )

        per_slot[str(slot)] = {
            "wins_vs_best_threshold": wins_vs_best_threshold,
            "required_wins_vs_best_threshold": threshold_required,
            "passes_threshold_baseline": bool(slot_threshold_ok),
            "passes_efficiency_type": bool(slot_efficiency_ok),
            "passes_recall_type": bool(slot_recall_ok),
            "mean_recall_oracle": mean_recall_oracle,
            "mean_recall_phase3": mean_recall_phase3,
            "mean_unnecessary_oracle": mean_unnecessary_oracle,
            "mean_unnecessary_phase3": mean_unnecessary_phase3,
            "mean_cost_oracle": mean_cost_oracle,
            "mean_cost_phase3": mean_cost_phase3,
        }

        threshold_ok = threshold_ok and slot_threshold_ok
        efficiency_ok = efficiency_ok and slot_efficiency_ok
        recall_ok = recall_ok and slot_recall_ok

    overall = threshold_ok and (efficiency_ok or recall_ok)
    overall_mode = None
    if overall:
        if recall_ok:
            overall_mode = "recall_type"
        elif efficiency_ok:
            overall_mode = "efficiency_type"

    return {
        "n_holdouts": int(n_holdouts),
        "required_threshold_wins_per_slot": threshold_required,
        "per_slot": per_slot,
        "passes_threshold_baseline_all_slots": bool(threshold_ok),
        "passes_efficiency_type_all_slots": bool(efficiency_ok),
        "passes_recall_type_all_slots": bool(recall_ok),
        "passes_overall_rule": bool(overall),
        "overall_mode": overall_mode,
    }


def merge_confirm_summaries(paths: List[str]) -> Dict[str, Any]:
    payloads = [_load_json(p) for p in paths]
    if not payloads:
        raise ValueError("No inputs provided.")

    winner_variants = {json.dumps(p["winner_variant"], sort_keys=True) for p in payloads}
    if len(winner_variants) != 1:
        raise ValueError("All inputs must share the same winner_variant.")

    per_holdout: List[Dict[str, Any]] = []
    seen_tags = set()
    for payload in payloads:
        for row in payload["per_holdout_results"]:
            tag = str(row["tag"])
            if tag in seen_tags:
                raise ValueError(f"Duplicate holdout tag across inputs: {tag}")
            seen_tags.add(tag)
            per_holdout.append(row)

    slot_keys = sorted({str(k) for p in payloads for k in p["slot_budget_aggregates"].keys()}, key=int)
    slot_budget_aggregates: Dict[str, Any] = {}
    family_breakdown: Dict[str, Any] = {}

    for slot in slot_keys:
        rows = [r["slot_budget_results"][slot] for r in per_holdout]
        policy_stats = {
            "phase3_oracle_upgrade": _policy_stats(rows, "phase3_oracle_upgrade"),
            "phase3_proposed": _policy_stats(rows, "phase3_proposed"),
            "topk_expected_consequence": _policy_stats(rows, "topk_expected_consequence"),
            "best_threshold": _policy_stats(rows, "best_threshold"),
        }
        best_threshold_frequency = dict(Counter(str(r["best_threshold_name"]) for r in rows))
        paired_stats = {
            "oracle_vs_phase3": _paired(rows, "phase3_oracle_upgrade", "phase3_proposed"),
            "oracle_vs_best_threshold": _paired(rows, "phase3_oracle_upgrade", "best_threshold"),
            "oracle_vs_topk_expected": _paired(rows, "phase3_oracle_upgrade", "topk_expected_consequence"),
        }
        slot_budget_aggregates[slot] = {
            "policy_stats": policy_stats,
            "paired_stats": paired_stats,
            "best_threshold_frequency": best_threshold_frequency,
        }

        fam_group_rows = _family_grouped(per_holdout)
        family_breakdown[slot] = {}
        for family_tag, fam_rows in fam_group_rows.items():
            local_rows = [r["slot_budget_results"][slot] for r in fam_rows]
            family_breakdown[slot][family_tag] = {
                "n_holdouts": len(local_rows),
                "policy_stats": {
                    "phase3_oracle_upgrade": _policy_stats(local_rows, "phase3_oracle_upgrade"),
                    "phase3_proposed": _policy_stats(local_rows, "phase3_proposed"),
                    "best_threshold": _policy_stats(local_rows, "best_threshold"),
                },
                "paired_stats": {
                    "oracle_vs_phase3": _paired(local_rows, "phase3_oracle_upgrade", "phase3_proposed"),
                    "oracle_vs_best_threshold": _paired(local_rows, "phase3_oracle_upgrade", "best_threshold"),
                },
            }

    out = {
        "method": "phase3_oracle_upgrade_confirm_merged",
        "source_paths": [str(Path(p).resolve()) for p in paths],
        "winner_variant": payloads[0]["winner_variant"],
        "winner_dev_selection": payloads[0].get("winner_dev_selection", {}),
        "n_sources": len(payloads),
        "n_holdouts": len(per_holdout),
        "holdout_tags": [str(r["tag"]) for r in per_holdout],
        "slot_budget_aggregates": slot_budget_aggregates,
        "family_breakdown": family_breakdown,
        "formal_decision_summary_scaled": _scaled_formal_decision(slot_budget_aggregates, len(per_holdout)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge one or more phase3 oracle blind-confirm aggregate summaries and recompute a scaled formal decision summary.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input aggregate_summary.json files from blind confirm runs.")
    parser.add_argument("--output", required=True, help="Output merged summary json path.")
    args = parser.parse_args()

    merged = merge_confirm_summaries(args.inputs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output": str(out_path.resolve()),
        "n_inputs": len(args.inputs),
        "n_holdouts": int(merged["n_holdouts"]),
        "winner_variant": merged["winner_variant"]["name"],
        "passes_overall_rule": bool(merged["formal_decision_summary_scaled"]["passes_overall_rule"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

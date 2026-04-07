
#!/usr/bin/env python3
"""
Merge phase3 oracle fixed-ablation aggregate summaries across multiple confirm manifests.

Usage:
  python merge_phase3_oracle_ablation.py \
    --inputs path/to/v1_ablation.json path/to/v2_ablation.json \
    --output metric/case14/phase3_oracle_ablation_merged/aggregate_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple


POLICIES = ["phase3_reference", "oracle_fused_ec", "oracle_protected_ec"]
METRICS = [
    "weighted_attack_recall_no_backend_fail",
    "unnecessary_mtd_count",
    "queue_delay_p95",
    "average_service_cost_per_step",
    "pred_expected_consequence_served_ratio",
]


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_holdouts(summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return summary.get("per_holdout_results", [])


def _collect_records(summaries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    """
    Returns:
      slot_budget_records[slot_budget][policy] = list of per-holdout metric dicts
      flat_records = list of enriched per-holdout dicts
    """
    slot_budget_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    flat_records: List[Dict[str, Any]] = []

    for src_idx, summary in enumerate(summaries):
        for hold in _iter_holdouts(summary):
            tag = hold["tag"]
            family_tag = hold.get("family_tag", "unknown_family")
            for slot_budget, res in hold["slot_budget_results"].items():
                slot_budget_records.setdefault(slot_budget, {})
                enriched = {
                    "source_index": src_idx,
                    "tag": tag,
                    "family_tag": family_tag,
                    "slot_budget": slot_budget,
                    "schedule": hold.get("schedule"),
                    "seed_base": hold.get("seed_base"),
                    "start_offset": hold.get("start_offset"),
                    "test_bank": hold.get("test_bank"),
                    "slot_budget_results": {},
                }
                for policy in POLICIES:
                    if policy not in res:
                        raise KeyError(f"Policy {policy} missing in holdout {tag}, slot {slot_budget}")
                    slot_budget_records[slot_budget].setdefault(policy, []).append(res[policy])
                    enriched["slot_budget_results"][policy] = res[policy]
                # keep useful contextual baselines if present
                for maybe in ["best_threshold", "phase3_proposed", "topk_expected_consequence", "best_threshold_name"]:
                    if maybe in res:
                        enriched["slot_budget_results"][maybe] = res[maybe]
                flat_records.append(enriched)
    return slot_budget_records, flat_records


def _build_policy_stats(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for policy, items in records.items():
        out[policy] = {}
        for metric in METRICS:
            out[policy][metric] = _stats([float(x[metric]) for x in items])
    return out


def _build_paired_stats(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    def pairwise(a_name: str, b_name: str) -> Dict[str, Any]:
        a_items = records[a_name]
        b_items = records[b_name]
        deltas = {
            "delta_recall": [],
            "delta_unnecessary": [],
            "delta_delay_p95": [],
            "delta_cost_per_step": [],
            "delta_pred_expected_consequence_served_ratio": [],
        }
        wins_on_recall = 0
        ties_on_recall = 0
        lower_unnecessary = 0
        lower_delay = 0
        lower_cost = 0
        higher_served_ratio = 0

        for a, b in zip(a_items, b_items):
            drec = float(b["weighted_attack_recall_no_backend_fail"] - a["weighted_attack_recall_no_backend_fail"])
            dun = float(b["unnecessary_mtd_count"] - a["unnecessary_mtd_count"])
            ddelay = float(b["queue_delay_p95"] - a["queue_delay_p95"])
            dcost = float(b["average_service_cost_per_step"] - a["average_service_cost_per_step"])
            dratio = float(
                b["pred_expected_consequence_served_ratio"] - a["pred_expected_consequence_served_ratio"]
            )

            deltas["delta_recall"].append(drec)
            deltas["delta_unnecessary"].append(dun)
            deltas["delta_delay_p95"].append(ddelay)
            deltas["delta_cost_per_step"].append(dcost)
            deltas["delta_pred_expected_consequence_served_ratio"].append(dratio)

            if drec > 0:
                wins_on_recall += 1
            elif abs(drec) < 1e-12:
                ties_on_recall += 1
            if dun < 0:
                lower_unnecessary += 1
            if ddelay < 0:
                lower_delay += 1
            if dcost < 0:
                lower_cost += 1
            if dratio > 0:
                higher_served_ratio += 1

        out = {}
        for k, vals in deltas.items():
            out[k] = _stats(vals)
        out.update(
            {
                "wins_on_recall": wins_on_recall,
                "ties_on_recall": ties_on_recall,
                "lower_unnecessary": lower_unnecessary,
                "lower_delay": lower_delay,
                "lower_cost": lower_cost,
                "higher_served_ratio": higher_served_ratio,
            }
        )
        return out

    return {
        "fused_vs_phase3": pairwise("phase3_reference", "oracle_fused_ec"),
        "protected_vs_phase3": pairwise("phase3_reference", "oracle_protected_ec"),
        "protected_vs_fused": pairwise("oracle_fused_ec", "oracle_protected_ec"),
    }


def _build_family_breakdown(flat_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in flat_records:
        grouped.setdefault((rec["slot_budget"], rec["family_tag"]), []).append(rec)

    for (slot_budget, family_tag), items in grouped.items():
        slot_out = out.setdefault(slot_budget, {})
        per_policy: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}
        for item in items:
            for policy in POLICIES:
                per_policy[policy].append(item["slot_budget_results"][policy])

        slot_out[family_tag] = {
            "n_holdouts": len(items),
            "policy_stats": _build_policy_stats(per_policy),
            "paired_stats": _build_paired_stats(per_policy),
        }
    return out


def merge_summaries(input_paths: List[Path]) -> Dict[str, Any]:
    summaries = [_load_json(p) for p in input_paths]
    slot_budget_records, flat_records = _collect_records(summaries)

    merged = {
        "method": "phase3_oracle_fixed_ablation_merged",
        "source_paths": [str(p.resolve()) for p in input_paths],
        "n_sources": len(input_paths),
        "n_holdouts": len({rec["tag"] for rec in flat_records}),
        "holdout_tags": sorted({rec["tag"] for rec in flat_records}),
        "slot_budget_aggregates": {},
        "family_breakdown": _build_family_breakdown(flat_records),
    }

    if summaries:
        # carry one representative manifest/order block for convenience
        s0 = summaries[0]
        merged["variant_order"] = s0.get("variant_order", POLICIES)
        merged["variant_payloads"] = s0.get("variant_payloads", {})
        merged["dev_screen_summary_path"] = s0.get("dev_screen_summary_path")

    for slot_budget, records in sorted(slot_budget_records.items(), key=lambda kv: int(kv[0])):
        merged["slot_budget_aggregates"][slot_budget] = {
            "policy_stats": _build_policy_stats(records),
            "paired_stats": _build_paired_stats(records),
        }

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input ablation aggregate_summary.json files")
    parser.add_argument("--output", required=True, help="Merged output JSON path")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_summaries(input_paths)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "merged_output": str(output_path.resolve()),
        "n_sources": merged["n_sources"],
        "n_holdouts": merged["n_holdouts"],
    }, indent=2))


if __name__ == "__main__":
    main()

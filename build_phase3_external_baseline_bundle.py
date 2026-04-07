#!/usr/bin/env python3
"""Build manuscript-ready baseline tables from merged confirm summary.

This script expands the current paper bundle by explicitly surfacing the
historical threshold-style baseline and the aggressive consequence-only baseline
that already exist inside the merged confirm summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mean(policy_stats: Dict[str, Any], metric: str) -> float:
    return float(policy_stats[metric]["mean"])


def build_main_rows(confirm: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aggs = confirm["slot_budget_aggregates"]
    for slot_str in sorted(aggs.keys(), key=int):
        stats = aggs[slot_str]["policy_stats"]
        oracle_name = "phase3_oracle_upgrade"
        phase3_name = "phase3_proposed"
        threshold_name = "best_threshold"
        topk_name = "topk_expected_consequence"

        row = {
            "slot_budget": int(slot_str),
            "oracle_method": "oracle_protected_ec",
            "oracle_recall_mean": _mean(stats[oracle_name], "weighted_attack_recall_no_backend_fail"),
            "oracle_unnecessary_mean": _mean(stats[oracle_name], "unnecessary_mtd_count"),
            "oracle_delay_mean": _mean(stats[oracle_name], "queue_delay_p95"),
            "oracle_cost_mean": _mean(stats[oracle_name], "average_service_cost_per_step"),
            "phase3_recall_mean": _mean(stats[phase3_name], "weighted_attack_recall_no_backend_fail"),
            "phase3_unnecessary_mean": _mean(stats[phase3_name], "unnecessary_mtd_count"),
            "phase3_delay_mean": _mean(stats[phase3_name], "queue_delay_p95"),
            "phase3_cost_mean": _mean(stats[phase3_name], "average_service_cost_per_step"),
            "historical_baseline_name": "best_threshold_family (DDET-style static trigger family)",
            "historical_recall_mean": _mean(stats[threshold_name], "weighted_attack_recall_no_backend_fail"),
            "historical_unnecessary_mean": _mean(stats[threshold_name], "unnecessary_mtd_count"),
            "historical_delay_mean": _mean(stats[threshold_name], "queue_delay_p95"),
            "historical_cost_mean": _mean(stats[threshold_name], "average_service_cost_per_step"),
            "aggressive_baseline_name": "topk_expected_consequence",
            "aggressive_recall_mean": _mean(stats[topk_name], "weighted_attack_recall_no_backend_fail"),
            "aggressive_unnecessary_mean": _mean(stats[topk_name], "unnecessary_mtd_count"),
            "aggressive_delay_mean": _mean(stats[topk_name], "queue_delay_p95"),
            "aggressive_cost_mean": _mean(stats[topk_name], "average_service_cost_per_step"),
        }
        rows.append(row)
    return rows


def build_family_rows(confirm: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    family_breakdown = confirm.get("family_breakdown", {})
    for slot_str in sorted(family_breakdown.keys(), key=int):
        slot_block = family_breakdown[slot_str]
        for family_tag, family_data in sorted(slot_block.items()):
            stats = family_data["policy_stats"]
            rows.append(
                {
                    "slot_budget": int(slot_str),
                    "family_tag": family_tag,
                    "n_holdouts": int(family_data.get("n_holdouts", 0)),
                    "oracle_recall_mean": _mean(stats["phase3_oracle_upgrade"], "weighted_attack_recall_no_backend_fail"),
                    "phase3_recall_mean": _mean(stats["phase3_proposed"], "weighted_attack_recall_no_backend_fail"),
                    "historical_recall_mean": _mean(stats["best_threshold"], "weighted_attack_recall_no_backend_fail"),
                    "oracle_unnecessary_mean": _mean(stats["phase3_oracle_upgrade"], "unnecessary_mtd_count"),
                    "phase3_unnecessary_mean": _mean(stats["phase3_proposed"], "unnecessary_mtd_count"),
                    "historical_unnecessary_mean": _mean(stats["best_threshold"], "unnecessary_mtd_count"),
                }
            )
    return rows


def build_summary(confirm: Dict[str, Any], rows: List[Dict[str, Any]], family_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    threshold_note = (
        "The historical baseline reported here is the best member of the static threshold family "
        "already evaluated on the same confirm holdouts. In most holdouts it resolves to a "
        "verify-score or DDD-score FIFO trigger, which is the closest historical counterpart to "
        "the original DDET-MTD event-trigger style under the current resource-constrained backend setting."
    )
    return {
        "method": "phase3_external_baseline_bundle",
        "source_confirm": confirm.get("source_paths", confirm.get("confirm_source", [])),
        "winner_variant": confirm.get("winner_variant", {}),
        "n_confirm_holdouts": int(confirm.get("n_holdouts", 0)),
        "main_table_rows_full": rows,
        "family_breakdown_rows": family_rows,
        "historical_baseline_note": threshold_note,
        "best_threshold_frequency": {
            slot: confirm.get("slot_budget_aggregates", {}).get(str(slot), {}).get("best_threshold_frequency", {})
            for slot in [1, 2]
        },
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True, help="Merged confirm aggregate summary JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    confirm = load_json(Path(args.confirm))
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    main_rows = build_main_rows(confirm)
    family_rows = build_family_rows(confirm)
    summary = build_summary(confirm, main_rows, family_rows)

    with (out_dir / "external_baseline_bundle_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_csv(out_dir / "external_baseline_main_table.csv", main_rows)
    write_csv(out_dir / "external_baseline_family_breakdown.csv", family_rows)

    print(
        json.dumps(
            {
                "summary_path": str(out_dir / "external_baseline_bundle_summary.json"),
                "main_table_csv": str(out_dir / "external_baseline_main_table.csv"),
                "family_breakdown_csv": str(out_dir / "external_baseline_family_breakdown.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

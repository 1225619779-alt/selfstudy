
#!/usr/bin/env python3
"""
Build paper-ready tables from merged confirm summary + merged ablation summary.

Usage:
  python build_phase3_paper_bundle.py \
    --confirm metric/case14/phase3_confirm_combined_v1_v2/aggregate_summary_merged.json \
    --ablation metric/case14/phase3_oracle_ablation_merged/aggregate_summary.json \
    --output_dir metric/case14/phase3_paper_bundle
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(confirm: Dict[str, Any], ablation: Dict[str, Any]) -> Dict[str, Any]:
    main_rows = []
    for slot_budget in ["1", "2"]:
        agg = confirm["slot_budget_aggregates"][slot_budget]
        oracle = agg["policy_stats"]["phase3_oracle_upgrade"]
        phase3 = agg["policy_stats"]["phase3_proposed"]
        paired = agg["paired_stats"]["oracle_vs_phase3"]

        main_rows.append({
            "slot_budget": int(slot_budget),
            "oracle_recall_mean": oracle["weighted_attack_recall_no_backend_fail"]["mean"],
            "phase3_recall_mean": phase3["weighted_attack_recall_no_backend_fail"]["mean"],
            "delta_recall_mean": paired["delta_recall"]["mean"],
            "oracle_unnecessary_mean": oracle["unnecessary_mtd_count"]["mean"],
            "phase3_unnecessary_mean": phase3["unnecessary_mtd_count"]["mean"],
            "delta_unnecessary_mean": paired["delta_unnecessary"]["mean"],
            "oracle_cost_mean": oracle["average_service_cost_per_step"]["mean"],
            "phase3_cost_mean": phase3["average_service_cost_per_step"]["mean"],
            "delta_cost_mean": paired["delta_cost_per_step"]["mean"],
            "oracle_delay_mean": oracle["queue_delay_p95"]["mean"],
            "phase3_delay_mean": phase3["queue_delay_p95"]["mean"],
            "delta_delay_mean": paired["delta_delay_p95"]["mean"],
            "wins_on_recall_vs_phase3": paired["wins_on_recall"],
            "ties_on_recall_vs_phase3": paired["ties_on_recall"],
            "lower_unnecessary_vs_phase3": paired["lower_unnecessary"],
            "wins_vs_best_threshold": agg["paired_stats"]["oracle_vs_best_threshold"]["wins_on_recall"],
        })

    ablation_rows = []
    for slot_budget in ["1", "2"]:
        agg = ablation["slot_budget_aggregates"][slot_budget]
        for policy in ["phase3_reference", "oracle_fused_ec", "oracle_protected_ec"]:
            ps = agg["policy_stats"][policy]
            ablation_rows.append({
                "slot_budget": int(slot_budget),
                "policy": policy,
                "recall_mean": ps["weighted_attack_recall_no_backend_fail"]["mean"],
                "unnecessary_mean": ps["unnecessary_mtd_count"]["mean"],
                "delay_mean": ps["queue_delay_p95"]["mean"],
                "cost_mean": ps["average_service_cost_per_step"]["mean"],
                "served_ratio_mean": ps["pred_expected_consequence_served_ratio"]["mean"],
            })

    bundle = {
        "confirm_source": confirm.get("source_paths", []),
        "ablation_source": ablation.get("source_paths", []),
        "winner_variant": confirm.get("winner_variant"),
        "winner_dev_selection": confirm.get("winner_dev_selection"),
        "n_confirm_holdouts": confirm.get("n_holdouts"),
        "main_table_rows": main_rows,
        "ablation_table_rows": ablation_rows,
    }
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True, help="Merged confirm summary JSON")
    parser.add_argument("--ablation", required=True, help="Merged ablation summary JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    confirm = _load_json(Path(args.confirm))
    ablation = _load_json(Path(args.ablation))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(confirm, ablation)

    with (out_dir / "paper_bundle_summary.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    _write_csv(out_dir / "table_confirm_main.csv", bundle["main_table_rows"])
    _write_csv(out_dir / "table_ablation.csv", bundle["ablation_table_rows"])

    print(json.dumps({
        "output_dir": str(out_dir.resolve()),
        "summary_json": str((out_dir / "paper_bundle_summary.json").resolve()),
        "confirm_table_csv": str((out_dir / "table_confirm_main.csv").resolve()),
        "ablation_table_csv": str((out_dir / "table_ablation.csv").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

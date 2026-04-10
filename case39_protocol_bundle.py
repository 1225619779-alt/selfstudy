#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

METHODS = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_agg(agg1: Dict[str, Any], agg2: Dict[str, Any]) -> Dict[str, Any]:
    per_holdout = list(agg1.get("per_holdout_results", [])) + list(agg2.get("per_holdout_results", []))
    merged: Dict[str, Any] = {"n_holdouts": len(per_holdout), "merged_8_holdouts": {}, "paired": {}}
    for slot in ["1", "2"]:
        merged["merged_8_holdouts"][slot] = {}
        for method in METHODS:
            vals = []
            for h in per_holdout:
                r = h["slot_budget_results"][slot][method]
                vals.append(r)
            merged["merged_8_holdouts"][slot][method] = {
                "mean_recall": mean(v["weighted_attack_recall_no_backend_fail"] for v in vals),
                "mean_unnecessary": mean(v["unnecessary_mtd_count"] for v in vals),
                "mean_cost": mean(v["average_service_cost_per_step"] for v in vals),
                "mean_served_ratio": mean(v["pred_expected_consequence_served_ratio"] for v in vals),
            }
        # paired deltas
        oracle = "phase3_oracle_upgrade"
        phase3 = "phase3_proposed"
        topk = "topk_expected_consequence"
        for other_name, other in [("oracle_vs_phase3", phase3), ("oracle_vs_topk_expected", topk)]:
            deltas: Dict[str, List[float]] = {k: [] for k in ["recall", "unnecessary", "cost", "served_ratio", "delay_p95"]}
            for h in per_holdout:
                a = h["slot_budget_results"][slot][oracle]
                b = h["slot_budget_results"][slot][other]
                deltas["recall"].append(a["weighted_attack_recall_no_backend_fail"] - b["weighted_attack_recall_no_backend_fail"])
                deltas["unnecessary"].append(a["unnecessary_mtd_count"] - b["unnecessary_mtd_count"])
                deltas["cost"].append(a["average_service_cost_per_step"] - b["average_service_cost_per_step"])
                deltas["served_ratio"].append(a["pred_expected_consequence_served_ratio"] - b["pred_expected_consequence_served_ratio"])
                deltas["delay_p95"].append(a["queue_delay_p95"] - b["queue_delay_p95"])
            merged["paired"].setdefault(slot, {})[other_name] = {
                metric: {
                    "mean": mean(vals),
                    "std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                    "raw_deltas": vals,
                }
                for metric, vals in deltas.items()
            }
    return merged


def make_hash_audit(asset_protocol_path: Path) -> Dict[str, Any]:
    protocol = json.loads(asset_protocol_path.read_text(encoding="utf-8"))
    checks = []
    for group_name in ["assets", "holdout_test_banks"]:
        group = protocol.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for key, meta in group.items():
            source = Path(meta["source_path"])
            exists = source.exists()
            current = sha256_file(source) if exists else None
            expected = meta.get("sha256")
            checks.append({
                "group": group_name,
                "key": key,
                "source_path": str(source),
                "exists": exists,
                "expected_sha256": expected,
                "current_sha256": current,
                "hash_match": bool(exists and current == expected),
            })
    return {
        "method": "case39_localretune_fallback_case14_hash_audit",
        "all_hashes_match_asset_protocol": all(c["hash_match"] for c in checks),
        "n_checks": len(checks),
        "checks": checks,
    }


def write_txt_summary(merged: Dict[str, Any], out_txt: Path, stage: str) -> None:
    lines = [f"stage={stage}", f"n_holdouts={merged['n_holdouts']}", "== merged_8_holdouts =="]
    for slot in ["1", "2"]:
        lines.append(f"-- slot_budget={slot} --")
        for method in METHODS:
            m = merged["merged_8_holdouts"][slot][method]
            lines.append(
                f"{method}: recall={m['mean_recall']:.6f}, unnecessary={m['mean_unnecessary']:.3f}, "
                f"cost={m['mean_cost']:.6f}, served_ratio={m['mean_served_ratio']:.6f}"
            )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged summary and fallback hash audit for protocol-compliant case39 local-retune rerun.")
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--asset_protocol", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage", default="case39_localretune_protocol_compliant_oracle_protected_ec")
    args = ap.parse_args()

    v1 = json.loads(Path(args.v1).read_text(encoding="utf-8"))
    v2 = json.loads(Path(args.v2).read_text(encoding="utf-8"))
    merged = merge_agg(v1, v2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": args.stage,
        **merged,
        "v1_path": str(Path(args.v1).resolve()),
        "v2_path": str(Path(args.v2).resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_txt_summary(summary, out_dir / "summary.txt", args.stage)

    audit = make_hash_audit(Path(args.asset_protocol))
    (out_dir / "fallback_case14_hash_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit_txt = [
        f"all_hashes_match_asset_protocol={audit['all_hashes_match_asset_protocol']}",
        f"n_checks={audit['n_checks']}",
    ]
    for c in audit["checks"]:
        audit_txt.append(f"[{c['group']}] {c['key']} hash_match={c['hash_match']} source={c['source_path']}")
    (out_dir / "fallback_case14_hash_audit.txt").write_text("\n".join(audit_txt) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "stage": args.stage,
        "all_hashes_match_asset_protocol": audit["all_hashes_match_asset_protocol"],
    }, indent=2))


if __name__ == "__main__":
    main()

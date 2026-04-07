#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def has_nan(x: Any) -> bool:
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, dict):
        return any(has_nan(v) for v in x.values())
    if isinstance(x, list):
        return any(has_nan(v) for v in x)
    return False


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_confirm_summary(name: str, data: dict, issues: List[str], notes: List[str]) -> int:
    per_holdout = data.get("per_holdout_results")
    n_holdouts = int(data.get("n_holdouts", -1))
    if not isinstance(per_holdout, list):
        issues.append(f"{name}: missing per_holdout_results")
        return 0
    if n_holdouts != len(per_holdout):
        issues.append(f"{name}: n_holdouts={n_holdouts} but len(per_holdout_results)={len(per_holdout)}")
    required_slots = {"1", "2"}
    required_policies = {"phase3_oracle_upgrade", "phase3_proposed", "best_threshold", "topk_expected_consequence"}
    for row in per_holdout:
        slots = set((row.get("slot_budget_results") or {}).keys())
        if not required_slots.issubset(slots):
            issues.append(f"{name}: holdout {row.get('tag')} missing slot budgets {sorted(required_slots - slots)}")
            continue
        for slot in required_slots:
            policies = set((row["slot_budget_results"].get(slot) or {}).keys())
            missing = required_policies - policies
            if missing:
                issues.append(f"{name}: holdout {row.get('tag')} slot {slot} missing policies {sorted(missing)}")
    if has_nan(data):
        issues.append(f"{name}: contains NaN")
    else:
        notes.append(f"{name}: no NaN in core summary")
    return len(per_holdout)


def check_merged_confirm(name: str, data: dict, expected_holdouts: int, issues: List[str], notes: List[str]) -> None:
    n_holdouts = int(data.get("n_holdouts", -1))
    holdout_tags = data.get("holdout_tags") or []
    if n_holdouts != expected_holdouts:
        issues.append(f"{name}: n_holdouts={n_holdouts} but expected {expected_holdouts} from source confirm summaries")
    if len(holdout_tags) != n_holdouts:
        issues.append(f"{name}: holdout_tags has len {len(holdout_tags)} but n_holdouts={n_holdouts}")
    slot_aggs = data.get("slot_budget_aggregates") or {}
    for slot in ("1", "2"):
        if slot not in slot_aggs:
            issues.append(f"{name}: missing slot_budget_aggregates[{slot}]")
            continue
        policy_stats = slot_aggs[slot].get("policy_stats") or {}
        for key in ("phase3_oracle_upgrade", "phase3_proposed", "best_threshold", "topk_expected_consequence"):
            if key not in policy_stats:
                issues.append(f"{name}: slot {slot} missing policy_stats[{key}]")
    if has_nan(data):
        issues.append(f"{name}: contains NaN")
    else:
        notes.append(f"{name}: no NaN in merged confirm summary")


def check_ablation_merged(name: str, data: dict, issues: List[str], notes: List[str]) -> None:
    n_holdouts = int(data.get("n_holdouts", -1))
    if n_holdouts <= 0:
        issues.append(f"{name}: invalid n_holdouts={n_holdouts}")
    slot_aggs = data.get("slot_budget_aggregates") or {}
    for slot in ("1", "2"):
        policy_stats = (slot_aggs.get(slot) or {}).get("policy_stats") or {}
        for key in ("phase3_reference", "oracle_fused_ec", "oracle_protected_ec"):
            if key not in policy_stats:
                issues.append(f"{name}: slot {slot} missing policy_stats[{key}]")
    if has_nan(data):
        issues.append(f"{name}: contains NaN")
    else:
        notes.append(f"{name}: no NaN in merged ablation summary")


def check_external_bundle(name: str, data: dict, expected_holdouts: int, issues: List[str], notes: List[str]) -> None:
    if int(data.get("n_confirm_holdouts", -1)) != expected_holdouts:
        issues.append(f"{name}: n_confirm_holdouts mismatch")
    rows = data.get("main_table_rows_full") or []
    if len(rows) != 2:
        issues.append(f"{name}: expected 2 main_table_rows_full entries, got {len(rows)}")
    if has_nan(data):
        issues.append(f"{name}: contains NaN")
    else:
        notes.append(f"{name}: no NaN in external baseline bundle")


def check_paper_bundle(name: str, data: dict, expected_holdouts: int, issues: List[str], notes: List[str]) -> None:
    if int(data.get("n_confirm_holdouts", -1)) != expected_holdouts:
        issues.append(f"{name}: n_confirm_holdouts mismatch")
    rows = data.get("main_table_rows") or []
    if len(rows) != 2:
        issues.append(f"{name}: expected 2 main_table_rows entries, got {len(rows)}")
    ab_rows = data.get("ablation_table_rows") or []
    if len(ab_rows) != 6:
        issues.append(f"{name}: expected 6 ablation_table_rows entries, got {len(ab_rows)}")
    if has_nan(data):
        issues.append(f"{name}: contains NaN")
    else:
        notes.append(f"{name}: no NaN in paper bundle summary")


def check_significance(name: str, data: dict, issues: List[str], notes: List[str]) -> None:
    bad = False
    for row in data.get("results", []):
        if int(row.get("n_holdouts", 0)) == 0:
            bad = True
            break
        if has_nan(row):
            bad = True
            break
    if bad:
        issues.append(f"{name}: invalid significance output (n_holdouts=0 and/or NaN present)")
    else:
        notes.append(f"{name}: significance output looks usable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm_inputs", nargs="+", required=True)
    parser.add_argument("--confirm_merged", required=True)
    parser.add_argument("--ablation_merged", required=True)
    parser.add_argument("--external_bundle", required=True)
    parser.add_argument("--paper_bundle", required=True)
    parser.add_argument("--significance", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    issues: List[str] = []
    notes: List[str] = []

    confirm_count = 0
    for idx, path in enumerate(args.confirm_inputs, start=1):
        data = load_json(path)
        confirm_count += check_confirm_summary(f"confirm_input_{idx}", data, issues, notes)

    confirm_merged = load_json(args.confirm_merged)
    check_merged_confirm("confirm_merged", confirm_merged, confirm_count, issues, notes)

    ablation_merged = load_json(args.ablation_merged)
    check_ablation_merged("ablation_merged", ablation_merged, issues, notes)

    external_bundle = load_json(args.external_bundle)
    check_external_bundle("external_bundle", external_bundle, confirm_count, issues, notes)

    paper_bundle = load_json(args.paper_bundle)
    check_paper_bundle("paper_bundle", paper_bundle, confirm_count, issues, notes)

    if args.significance:
        significance = load_json(args.significance)
        check_significance("significance", significance, issues, notes)

    status = "PASS" if not issues else "WARN"
    core_foundation_pass = len([x for x in issues if not x.startswith("significance:")]) == 0
    summary = {
        "method": "phase3_foundation_audit",
        "status": status,
        "core_foundation_pass": core_foundation_pass,
        "confirm_holdout_count": confirm_count,
        "issues": issues,
        "notes": notes,
        "recommended_next_step": (
            "Core summaries look consistent. Fix significance only, then proceed to system expansion."
            if core_foundation_pass else
            "Fix core summary inconsistencies before any larger-system expansion."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(out_path), "status": status, "core_foundation_pass": core_foundation_pass}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

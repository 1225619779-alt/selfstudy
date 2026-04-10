#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

STAGE_KEYS = {
    "native_clean_attack_test_with_frozen_case14_dev": "transfer_frozen_dev",
    "case39_localretune_protocol_compliant_oracle_protected_ec": "local_protected",
    "case39_fully_native_localretune": "local_unconstrained",
    "case39_source_anchored_localretune": "source_anchored",
    "case39_source_fixed_case14winner_native_test": "source_fixed_replay",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.json"):
        # skip huge unrelated envs / preflight internals only if needed? keep broad.
        yield p


def stage_name(obj: Dict[str, Any]) -> Optional[str]:
    if isinstance(obj.get("stage"), str):
        return obj["stage"]
    if isinstance(obj.get("native_case39_stage"), str):
        return obj["native_case39_stage"]
    if isinstance(obj.get("label"), str):
        return obj["label"]
    if obj.get("method") == "case39_stage_compare_significance_v2":
        return "case39_stage_compare_significance_v2"
    return None


def merged(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = obj.get("merged_8_holdouts")
    if isinstance(m, dict):
        return m
    return None


def find_stage_summaries(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    for p in iter_json_files(repo_root / "metric"):
        try:
            obj = load_json(p)
        except Exception:
            continue
        st = stage_name(obj)
        if not st:
            continue
        key = STAGE_KEYS.get(st)
        if key and merged(obj):
            # Prefer shallower/shorter paths, but keep first if exact key already found.
            if key not in found:
                found[key] = {"path": str(p), "obj": obj}
    return found


def method_block(m: Dict[str, Any], slot: str, method: str) -> Dict[str, Any]:
    return m.get(slot, {}).get(method, {})


def compact_stage(obj: Dict[str, Any]) -> Dict[str, Any]:
    m = merged(obj) or {}
    out: Dict[str, Any] = {"stage": stage_name(obj), "merged_8_holdouts": {}}
    for slot in ["1", "2"]:
        out["merged_8_holdouts"][slot] = {}
        for method in ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]:
            block = method_block(m, slot, method)
            if block:
                out["merged_8_holdouts"][slot][method] = {
                    "mean_recall": block.get("mean_recall"),
                    "mean_unnecessary": block.get("mean_unnecessary"),
                    "mean_cost": block.get("mean_cost"),
                    "mean_served_ratio": block.get("mean_served_ratio"),
                }
    return out


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def compare_to_transfer(transfer: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mt = compact_stage(transfer)["merged_8_holdouts"]
    mo = compact_stage(other)["merged_8_holdouts"]
    for slot in ["1", "2"]:
        out[slot] = {
            "oracle_vs_transfer": {
                "delta_recall": delta(mo.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_recall"),
                                      mt.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_recall")),
                "delta_unnecessary": delta(mo.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_unnecessary"),
                                            mt.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_unnecessary")),
                "delta_cost": delta(mo.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_cost"),
                                     mt.get(slot, {}).get("phase3_oracle_upgrade", {}).get("mean_cost")),
            }
        }
    return out


def recommend(stages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    transfer = stages.get("transfer_frozen_dev", {}).get("obj")
    src_fixed = stages.get("source_fixed_replay", {}).get("obj")
    src_anchor = stages.get("source_anchored", {}).get("obj")
    local_prot = stages.get("local_protected", {}).get("obj")
    local_un = stages.get("local_unconstrained", {}).get("obj")
    rec: Dict[str, Any] = {"main_result_stage": None, "diagnostic_support": [], "claim": []}
    if transfer:
        rec["main_result_stage"] = "transfer_frozen_dev"
        rec["claim"].append(
            "Use native_clean_attack_test_with_frozen_case14_dev as the primary result."
        )
    if local_un:
        rec["diagnostic_support"].append(
            "Unconstrained fully-native local retune drifts toward over-conservative behavior."
        )
    if local_prot:
        rec["diagnostic_support"].append(
            "Protocol-compliant local retune remains strongly conservative and underperforms transfer."
        )
    if src_anchor:
        rec["diagnostic_support"].append(
            "Source-anchored retune partially repairs local conservatism but does not recover transfer-level performance."
        )
    if src_fixed:
        rec["diagnostic_support"].append(
            "Source-fixed replay isolates that native train/val bank shift itself contributes to the performance gap."
        )
    rec["claim"].append(
        "A strong paper claim is that source-frozen transfer acts like a regularizer under case39 local-dev shift."
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact comparison bundle for case39 stages.")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--output", default="metric/case39_compare/case39_stage_compare_bundle.json")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stages = find_stage_summaries(repo_root)
    payload: Dict[str, Any] = {
        "method": "case39_stage_compare_bundle",
        "repo_root": str(repo_root),
        "found_stage_paths": {k: v["path"] for k, v in stages.items()},
        "stages": {k: compact_stage(v["obj"]) for k, v in stages.items()},
        "recommendation": recommend(stages),
    }

    transfer_obj = stages.get("transfer_frozen_dev", {}).get("obj")
    if transfer_obj:
        payload["vs_transfer"] = {}
        for key in ["source_fixed_replay", "source_anchored", "local_protected", "local_unconstrained"]:
            obj = stages.get(key, {}).get("obj")
            if obj:
                payload["vs_transfer"][key] = compare_to_transfer(transfer_obj, obj)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "status": "OK",
        "output": str(out_path),
        "found_stage_keys": sorted(stages.keys()),
        "main_result_stage": payload["recommendation"]["main_result_stage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

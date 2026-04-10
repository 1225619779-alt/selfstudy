#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_method(stage_obj: Dict[str, Any], slot: str, method: str) -> Dict[str, Any]:
    return stage_obj.get("merged_8_holdouts", {}).get(slot, {}).get(method, {})


def line(stage: str, slot: str, method: str, block: Dict[str, Any]) -> str:
    return (
        f"| {stage} | {slot} | {method} | "
        f"{block.get('mean_recall')} | {block.get('mean_unnecessary')} | {block.get('mean_cost')} | {block.get('mean_served_ratio')} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--bundle_json", default="metric/case39_compare/case39_stage_compare_bundle.json")
    ap.add_argument("--output_dir", default="metric/case39_final_bundle")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bundle = load_json(repo_root / args.bundle_json)
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    final_json = out_dir / "final_stage_bundle.json"
    final_md = out_dir / "final_stage_table.md"
    claim_md = out_dir / "claim_recommendation.md"

    with final_json.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    rows = [
        "| stage | slot_budget | method | mean_recall | mean_unnecessary | mean_cost | mean_served_ratio |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for stage_key, stage_obj in bundle.get("stages", {}).items():
        merged = stage_obj.get("merged_8_holdouts", {})
        for slot in ["1", "2"]:
            for method in ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]:
                block = merged.get(slot, {}).get(method)
                if block:
                    rows.append(line(stage_key, slot, method, block))
    final_md.write_text("\n".join(rows) + "\n", encoding="utf-8")

    rec = bundle.get("recommendation", {})
    claim_lines = [
        f"Main result stage: {rec.get('main_result_stage')}",
        "",
        "Recommended claim:",
    ]
    for x in rec.get("claim", []):
        claim_lines.append(f"- {x}")
    claim_lines.append("")
    claim_lines.append("Diagnostic support:")
    for x in rec.get("diagnostic_support", []):
        claim_lines.append(f"- {x}")
    claim_md.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_dir": str(out_dir),
        "files": [str(final_json), str(final_md), str(claim_md)],
        "main_result_stage": rec.get("main_result_stage"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

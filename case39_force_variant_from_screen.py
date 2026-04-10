#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Force a chosen winner variant from an existing oracle-family screen summary.")
    ap.add_argument("--input", required=True, help="Path to screen_train_val_summary.json")
    ap.add_argument("--variant", required=True, help="Variant name to force as winner, e.g. oracle_protected_ec")
    ap.add_argument("--output", required=True, help="Output projected summary path")
    args = ap.parse_args()

    inp = Path(args.input)
    data = json.loads(inp.read_text(encoding="utf-8"))

    variants = data.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("screen summary missing top-level 'variants' dict")
    if args.variant not in variants:
        raise ValueError(f"variant {args.variant!r} not found; available={sorted(variants.keys())}")

    old_sel = data.get("selection", {})
    payload = variants[args.variant]

    selection = dict(old_sel) if isinstance(old_sel, dict) else {}
    selection["winner_variant"] = args.variant
    selection["winner_payload"] = payload
    if isinstance(payload, dict):
        if "joint_val_delta_objective" in payload:
            selection["winner_joint_val_delta_objective"] = payload["joint_val_delta_objective"]
        if "joint_val_delta_recall" in payload:
            selection["winner_joint_val_delta_recall"] = payload["joint_val_delta_recall"]
    selection["projection_note"] = "Protocol-compliant projection from an already-computed oracle-family screen summary. No rescreening performed."
    selection["previous_winner_variant"] = old_sel.get("winner_variant") if isinstance(old_sel, dict) else None

    data["selection"] = selection
    data["projected_from_variant_search"] = True
    data["projected_variant"] = args.variant

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({
        "input": str(inp),
        "forced_variant": args.variant,
        "previous_winner": selection.get("previous_winner_variant"),
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()

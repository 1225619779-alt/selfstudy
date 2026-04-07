from __future__ import annotations

import argparse
import json

from phase3_oracle_ablation_core import run_phase3_oracle_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed oracle ablations (baseline/fused/protected) on a manifest without re-selecting a new winner.")
    parser.add_argument("--manifest", required=True, help="Manifest json to evaluate on (dev holdout or blind confirm manifest).")
    parser.add_argument("--dev_screen_summary", required=True, help="Dev screen summary from phase3_oracle_family.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["phase3_reference", "oracle_fused_ec", "oracle_protected_ec"],
        help="Variant names to evaluate. Default: phase3_reference oracle_fused_ec oracle_protected_ec",
    )
    args = parser.parse_args()

    result = run_phase3_oracle_ablation(
        manifest_path=args.manifest,
        dev_screen_summary_path=args.dev_screen_summary,
        output_dir=args.output,
        variant_names=args.variants,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

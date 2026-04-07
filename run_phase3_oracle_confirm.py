
from __future__ import annotations

import argparse
import json

from phase3_oracle_confirm_core import run_phase3_oracle_confirm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed oracle-upgraded phase3 winner on a blind confirm manifest.")
    parser.add_argument("--manifest", required=True, help="Blind confirm manifest json.")
    parser.add_argument("--dev_screen_summary", required=True, help="Dev screen summary from phase3_oracle_family.")
    parser.add_argument("--output", required=True, help="Output directory for confirm aggregate summary.")
    args = parser.parse_args()

    result = run_phase3_oracle_confirm(
        confirm_manifest_path=args.manifest,
        dev_screen_summary_path=args.dev_screen_summary,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

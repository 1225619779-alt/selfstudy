from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from phase3_overload_core import run_phase3_overload_experiment



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run overload-only phase3 upgrade family on frozen multi-holdout manifest."
    )
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument(
        "--output",
        "--output-dir",
        "--out_dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output root directory. Writes screen_train_val_summary.json and, unless --screen-only, multi_holdout/aggregate_summary.json",
    )
    p.add_argument("--screen-only", "--screen_only", action="store_true", dest="screen_only")
    return p.parse_args()



def main() -> int:
    ns = parse_args()
    os.chdir(os.path.expanduser(ns.workdir))
    result = run_phase3_overload_experiment(ns.manifest, ns.output_dir, screen_only=bool(ns.screen_only))
    payload = {
        "screen_summary_path": str(Path(result["screen_summary_path"]).resolve()),
        "screen_only": bool(ns.screen_only),
    }
    if not ns.screen_only:
        payload["aggregate_summary_path"] = str(Path(result["aggregate_summary_path"]).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

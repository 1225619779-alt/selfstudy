from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from phase3_oracle_family_core import run_phase3_oracle_family_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run oracle-upgraded phase3 family on frozen multi-holdout manifest.")
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="Output directory root.")
    p.add_argument("--out_dir", type=str, default=None, help="Alias for --output.")
    p.add_argument("--screen-only", action="store_true", dest="screen_only")
    p.add_argument("--screen_only", action="store_true", dest="screen_only")
    ns = p.parse_args()
    if ns.out_dir and not ns.output:
        ns.output = ns.out_dir
    if ns.out_dir:
        ns.output = ns.out_dir
    return ns


def main() -> int:
    ns = parse_args()
    os.chdir(os.path.expanduser(ns.workdir))
    result = run_phase3_oracle_family_experiment(ns.manifest, ns.output, screen_only=bool(ns.screen_only))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from phase3_state_adm_core import run_phase3_state_adm_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run state-conditional admission upgrade for phase3 on frozen multi-holdout manifest.")
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    os.chdir(os.path.expanduser(ns.workdir))
    result = run_phase3_state_adm_experiment(ns.manifest, ns.output)
    print(f"Saved state-adm + phase3-dispatch aggregate summary: {Path(ns.output).resolve()}")
    print(json.dumps({
        "output": str(Path(ns.output).resolve()),
        "slot_budget_aggregates_keys": list(result.get("slot_budget_aggregates", {}).keys()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

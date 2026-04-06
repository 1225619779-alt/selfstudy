from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run counterfactual-help-gain admission + phase3 dispatch on frozen multi-holdout manifest.')
    p.add_argument('--workdir', type=str, required=True)
    p.add_argument('--manifest', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    workdir = Path(os.path.expanduser(ns.workdir)).resolve()
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))

    from cfhelp_phase3_core import run_counterfactual_help_experiment

    result = run_counterfactual_help_experiment(str(ns.manifest), str(ns.output))
    print(f"Saved counterfactual-help + phase3-dispatch aggregate summary: {Path(ns.output).resolve()}")
    print(json.dumps({
        'output': str(Path(ns.output).resolve()),
        'slot_budget_aggregates_keys': list(result.get('slot_budget_aggregates', {}).keys()),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

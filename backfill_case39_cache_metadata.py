from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.run_metadata import attach_runtime_metadata


ROOT = Path(__file__).resolve().parent
CLEAN_VALID_FRONTEND = ROOT / "metric" / "case39" / "clean_alarm_cache_valid_mode_0_0.03_1.1.npy"
ATTACK_VALID_FRONTEND = ROOT / "metric" / "case39" / "attack_alarm_cache_valid_50_repo_compatible_mode_0_0.03_1.1.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill runtime metadata into case39 calibration caches.")
    p.add_argument("--path", nargs="+", required=True)
    return p.parse_args()


def infer_input_cache(target: Path) -> str | None:
    name = target.name
    if name.endswith("_clean.npy"):
        return str(CLEAN_VALID_FRONTEND)
    if name.endswith("_attack.npy"):
        return str(ATTACK_VALID_FRONTEND)
    return None


def main() -> None:
    args = parse_args()
    for raw in args.path:
        path = Path(raw)
        payload: Dict[str, Any] = np.load(path, allow_pickle=True).item()
        metadata = payload.setdefault("metadata", {})
        metadata["skip_backend"] = False
        attach_runtime_metadata(
            metadata,
            repo_root=str(ROOT),
            input_cache_path=infer_input_cache(path),
            output_cache_path=str(path),
            runner_name=Path(__file__).name,
        )
        np.save(path, payload, allow_pickle=True)
        print(f"Backfilled metadata: {path}")


if __name__ == "__main__":
    main()

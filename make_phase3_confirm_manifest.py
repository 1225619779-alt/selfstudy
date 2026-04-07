
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_FAMILIES = [
    {
        "family_tag": "cfA_frontloaded",
        "schedule": "clean:90;att-1-0.15:90;clean:45;att-2-0.25:90;clean:45;att-3-0.25:60;clean:120",
        "seeds": [20260511, 20260512],
        "offsets": [1020, 1080],
    },
    {
        "family_tag": "cfB_backloaded",
        "schedule": "clean:150;att-2-0.20:60;clean:30;att-3-0.35:90;clean:30;att-1-0.15:60;clean:120",
        "seeds": [20260521, 20260522],
        "offsets": [1140, 1200],
    },
]

FROZEN_REGIME = {
    "decision_step_group": 1,
    "busy_time_quantile": 0.65,
    "use_cost_budget": False,
    "cost_budget_window_steps": 20,
    "cost_budget_quantile": 0.60,
    "slot_budget_list": [1, 2],
    "max_wait_steps": 10,
}


def _run(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def _summary_path_from_npy(path: Path) -> Path:
    return path.with_suffix(".summary.json")


def build_manifest(workdir: Path, out_dir: Path) -> Dict[str, object]:
    banks_dir = out_dir / "banks"
    results_dir = out_dir / "results"
    banks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    holdouts = []
    for family in DEFAULT_FAMILIES:
        for idx, (seed, offset) in enumerate(zip(family["seeds"], family["offsets"])):
            tag = f"{family['family_tag']}_{idx}_seed{seed}_off{offset}"
            test_bank = banks_dir / f"mixed_bank_test_{tag}.npy"
            result_npy = results_dir / f"budget_scheduler_phase3_holdout_{tag}.npy"
            result_summary = _summary_path_from_npy(result_npy)
            holdouts.append(
                {
                    "tag": tag,
                    "family_tag": family["family_tag"],
                    "schedule": family["schedule"],
                    "seed_base": int(seed),
                    "start_offset": int(offset),
                    "test_bank": _rel(test_bank, workdir),
                    "result_npy": _rel(result_npy, workdir),
                    "result_summary": _rel(result_summary, workdir),
                }
            )

    return {
        "workdir": str(workdir.resolve()),
        "clean_bank": "metric/case14/metric_clean_alarm_scores_full.npy",
        "attack_bank": "metric/case14/metric_attack_alarm_scores_400.npy",
        "train_bank": "metric/case14/mixed_bank_fit.npy",
        "val_bank": "metric/case14/mixed_bank_eval.npy",
        "schedule": "confirm_multi_family_v1",
        "confirm_families": DEFAULT_FAMILIES,
        "frozen_regime": FROZEN_REGIME,
        "holdouts": holdouts,
    }


def generate_assets(manifest: Dict[str, object], workdir: Path, force: bool = False) -> None:
    for hold in manifest["holdouts"]:
        test_bank = workdir / hold["test_bank"]
        result_npy = workdir / hold["result_npy"]
        result_summary = workdir / hold["result_summary"]

        if force or not test_bank.exists():
            cmd = [
                sys.executable,
                "evaluation_mixed_timeline.py",
                "--tau_verify",
                "-1",
                "--schedule",
                str(hold["schedule"]),
                "--seed_base",
                str(hold["seed_base"]),
                "--start_offset",
                str(hold["start_offset"]),
                "--output",
                str(test_bank),
            ]
            _run(cmd, cwd=workdir)
        else:
            print(f"[skip] existing test bank: {test_bank}")

        if force or (not result_npy.exists()) or (not result_summary.exists()):
            cmd = [
                sys.executable,
                "evaluation_budget_scheduler_phase3_holdout.py",
                "--clean_bank",
                str(workdir / manifest["clean_bank"]),
                "--attack_bank",
                str(workdir / manifest["attack_bank"]),
                "--train_bank",
                str(workdir / manifest["train_bank"]),
                "--val_bank",
                str(workdir / manifest["val_bank"]),
                "--test_bank",
                str(test_bank),
                "--slot_budget_list",
                "1",
                "2",
                "--decision_step_group",
                str(manifest["frozen_regime"]["decision_step_group"]),
                "--busy_time_quantile",
                str(manifest["frozen_regime"]["busy_time_quantile"]),
                "--output",
                str(result_npy),
            ]
            _run(cmd, cwd=workdir)
        else:
            print(f"[skip] existing baseline holdout summary: {result_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 4-holdout blind confirm manifest and baseline phase3 holdout summaries.")
    parser.add_argument("--workdir", default=".", help="Repo root.")
    parser.add_argument("--output_dir", default="metric/case14/phase3_confirm_blind_v1", help="Directory for confirm manifest/banks/results.")
    parser.add_argument("--manifest_only", action="store_true", help="Only write manifest; do not generate banks/results.")
    parser.add_argument("--force", action="store_true", help="Regenerate banks/results even if files already exist.")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    out_dir = (workdir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(workdir=workdir, out_dir=out_dir)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.manifest_only:
        generate_assets(manifest=manifest, workdir=workdir, force=bool(args.force))

    print(json.dumps({"manifest_path": str(manifest_path), "output_dir": str(out_dir), "manifest_only": bool(args.manifest_only)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

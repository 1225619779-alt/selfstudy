from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")

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


def _default_bank_paths(case_name: str) -> Dict[str, str]:
    root = Path("metric") / case_name
    return {
        "clean_bank": str(root / "metric_clean_alarm_scores_full.npy"),
        "attack_bank": str(root / "metric_attack_alarm_scores_400.npy"),
        "train_bank": str(root / "mixed_bank_fit.npy"),
        "val_bank": str(root / "mixed_bank_eval.npy"),
    }


def build_manifest(
    workdir: Path,
    out_dir: Path,
    *,
    case_name: str,
    clean_bank: str | None,
    attack_bank: str | None,
    train_bank: str | None,
    val_bank: str | None,
) -> Dict[str, object]:
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

    banks = _default_bank_paths(case_name)
    return {
        "workdir": str(workdir.resolve()),
        "case_name": case_name,
        "clean_bank": clean_bank or banks["clean_bank"],
        "attack_bank": attack_bank or banks["attack_bank"],
        "train_bank": train_bank or banks["train_bank"],
        "val_bank": val_bank or banks["val_bank"],
        "schedule": "confirm_multi_family_v1",
        "confirm_families": DEFAULT_FAMILIES,
        "frozen_regime": FROZEN_REGIME,
        "holdouts": holdouts,
    }


def generate_assets(manifest: Dict[str, object], workdir: Path, force: bool = False, allow_raw_generation: bool = False) -> None:
    case_name = str(manifest.get("case_name", "case14"))
    for hold in manifest["holdouts"]:
        test_bank = workdir / hold["test_bank"]
        result_npy = workdir / hold["result_npy"]
        result_summary = workdir / hold["result_summary"]

        if force or not test_bank.exists():
            if case_name != "case14" and not allow_raw_generation:
                raise RuntimeError(
                    f"Missing staged test bank for {case_name}: {test_bank}. "
                    "Stage precomputed holdout banks first, or explicitly enable raw generation after wiring case39 raw support."
                )
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
                "--max_wait_steps",
                str(manifest["frozen_regime"]["max_wait_steps"]),
                "--output",
                str(result_npy),
            ]
            _run(cmd, cwd=workdir)
        else:
            print(f"[skip] existing baseline holdout summary: {result_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 4-holdout blind confirm manifest and baseline phase3 holdout summaries.")
    parser.add_argument("--workdir", default=".", help="Repo root.")
    parser.add_argument("--case_name", default=DEFAULT_CASE_NAME, help="Case tag used for default metric root.")
    parser.add_argument("--output_dir", default=None, help="Directory for confirm manifest/banks/results.")
    parser.add_argument("--clean_bank", default=None, help="Optional override for clean bank path (relative to workdir).")
    parser.add_argument("--attack_bank", default=None, help="Optional override for attack bank path (relative to workdir).")
    parser.add_argument("--train_bank", default=None, help="Optional override for train bank path (relative to workdir).")
    parser.add_argument("--val_bank", default=None, help="Optional override for val bank path (relative to workdir).")
    parser.add_argument("--manifest_only", action="store_true", help="Only write manifest; do not generate banks/results.")
    parser.add_argument("--force", action="store_true", help="Regenerate banks/results even if files already exist.")
    parser.add_argument("--allow_raw_generation", action="store_true", help="Allow non-case14 raw generation. Disabled by default for case39 pure migration.")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    output_dir = args.output_dir or f"metric/{args.case_name}/phase3_confirm_blind_v1"
    out_dir = (workdir / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        workdir=workdir,
        out_dir=out_dir,
        case_name=str(args.case_name),
        clean_bank=args.clean_bank,
        attack_bank=args.attack_bank,
        train_bank=args.train_bank,
        val_bank=args.val_bank,
    )
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.manifest_only:
        generate_assets(
            manifest=manifest,
            workdir=workdir,
            force=bool(args.force),
            allow_raw_generation=bool(args.allow_raw_generation),
        )

    print(json.dumps({
        "manifest_path": str(manifest_path),
        "output_dir": str(out_dir),
        "manifest_only": bool(args.manifest_only),
        "case_name": args.case_name,
        "n_holdouts": len(manifest["holdouts"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

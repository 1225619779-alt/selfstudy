from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Rerun holdout summaries for all holdouts in a manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--python", default="python")
    p.add_argument("--workdir", default=".")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.chdir(os.path.expanduser(args.workdir))
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frozen = manifest["frozen_regime"]
    clean_bank = manifest["clean_bank"]
    attack_bank = manifest["attack_bank"]
    train_bank = manifest["train_bank"]
    val_bank = manifest["val_bank"]

    for hold in manifest["holdouts"]:
        out_npy = Path(hold["result_npy"])
        out_sum = Path(hold["result_summary"])
        if out_sum.exists() and not args.overwrite:
            print(f"[skip] {hold['tag']} -> {out_sum}")
            continue
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            "evaluation_budget_scheduler_phase3_holdout.py",
            "--clean_bank", clean_bank,
            "--attack_bank", attack_bank,
            "--train_bank", train_bank,
            "--val_bank", val_bank,
            "--test_bank", hold["test_bank"],
            "--output", hold["result_npy"],
            "--slot_budget_list",
        ] + [str(x) for x in frozen["slot_budget_list"]] + [
            "--max_wait_steps", str(frozen["max_wait_steps"]),
            "--decision_step_group", str(frozen["decision_step_group"]),
            "--busy_time_quantile", str(frozen["busy_time_quantile"]),
        ]
        if bool(frozen.get("use_cost_budget", False)):
            cmd += [
                "--cost_budget_window_steps", str(frozen["cost_budget_window_steps"]),
                "--cost_budget_quantile", str(frozen["cost_budget_quantile"]),
            ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

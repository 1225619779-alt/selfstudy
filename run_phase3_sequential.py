#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class Job:
    name: str
    cmd: List[str]
    done_files: List[Path]


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def ensure_exists(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n  - " + "\n  - ".join(missing))


def append_log_header(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_job(job: Job, workdir: Path, log_dir: Path, skip_existing: bool) -> None:
    if skip_existing and job.done_files and all(p.exists() for p in job.done_files):
        print(f"[SKIP] {job.name} (output already exists)")
        return

    log_path = log_dir / f"{job.name}.log"
    cmd_str = quote_cmd(job.cmd)
    print("\n" + "=" * 90)
    print(f"[RUN] {job.name}")
    print(cmd_str)
    print(f"[LOG] {log_path}")
    print("=" * 90)

    append_log_header(log_path, "\n" + "=" * 90)
    append_log_header(log_path, f"[RUN] {job.name}")
    append_log_header(log_path, cmd_str)
    append_log_header(log_path, "=" * 90)

    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            job.cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        ret = proc.wait()

    if ret != 0:
        raise subprocess.CalledProcessError(ret, job.cmd)

    if job.done_files:
        missing = [str(p) for p in job.done_files if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Job '{job.name}' finished but expected output file(s) not found:\n  - "
                + "\n  - ".join(missing)
            )

    print(f"[DONE] {job.name}")


def build_jobs(args: argparse.Namespace, py: str) -> List[Job]:
    clean_bank = Path(args.clean_bank)
    attack_bank = Path(args.attack_bank)
    fit_bank = Path(args.fit_bank)
    eval_bank = Path(args.eval_bank)

    out_phase3 = Path(args.output_phase3)
    out_phase3_cost = Path(args.output_phase3_cost)
    out_sweep = Path(args.output_sweep)

    common = [
        "--clean_bank", str(clean_bank),
        "--attack_bank", str(attack_bank),
        "--fit_bank", str(fit_bank),
        "--eval_bank", str(eval_bank),
        "--slot_budget_list", *[str(x) for x in args.slot_budget_list],
        "--max_wait_steps", str(args.max_wait_steps),
        "--busy_time_quantile", str(args.busy_time_quantile),
    ]

    return [
        Job(
            name="phase3_no_cost",
            cmd=[py, "evaluation_budget_scheduler_phase3.py", *common, "--output", str(out_phase3)],
            done_files=[out_phase3, out_phase3.with_suffix(".summary.json")],
        ),
        Job(
            name="phase3_with_cost",
            cmd=[
                py, "evaluation_budget_scheduler_phase3.py", *common,
                "--use_cost_budget",
                "--cost_budget_window_steps", str(args.cost_budget_window_steps),
                "--cost_budget_quantile", str(args.cost_budget_quantile),
                "--output", str(out_phase3_cost),
            ],
            done_files=[out_phase3_cost, out_phase3_cost.with_suffix(".summary.json")],
        ),
        Job(
            name="phase3_regime_sweep",
            cmd=[
                py, "sweep_regimes_phase3.py",
                "--clean_bank", str(clean_bank),
                "--attack_bank", str(attack_bank),
                "--fit_bank", str(fit_bank),
                "--eval_bank", str(eval_bank),
                "--slot_budget_list", *[str(x) for x in args.slot_budget_list],
                "--decision_step_group_list", *[str(x) for x in args.decision_step_group_list],
                "--busy_time_quantile_list", *[str(x) for x in args.busy_time_quantile_list],
                "--use_cost_budget_modes", *args.use_cost_budget_modes,
                "--cost_budget_quantile_list", *[str(x) for x in args.cost_budget_quantile_list],
                "--output", str(out_sweep),
            ],
            done_files=[out_sweep],
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 3 experiments sequentially.")
    p.add_argument("--workdir", default=".", help="Repo root directory")
    p.add_argument("--python_bin", default=sys.executable, help="Python interpreter to use")
    p.add_argument("--clean_bank", default="metric/case14/metric_clean_alarm_scores_full.npy")
    p.add_argument("--attack_bank", default="metric/case14/metric_attack_alarm_scores_400.npy")
    p.add_argument("--fit_bank", default="metric/case14/mixed_bank_fit.npy")
    p.add_argument("--eval_bank", default="metric/case14/mixed_bank_eval.npy")
    p.add_argument("--slot_budget_list", nargs="+", type=int, default=[1, 2])
    p.add_argument("--max_wait_steps", type=int, default=10)
    p.add_argument("--busy_time_quantile", type=float, default=0.50)
    p.add_argument("--cost_budget_window_steps", type=int, default=20)
    p.add_argument("--cost_budget_quantile", type=float, default=0.60)
    p.add_argument("--decision_step_group_list", nargs="+", type=int, default=[1, 2])
    p.add_argument("--busy_time_quantile_list", nargs="+", type=float, default=[0.35, 0.50, 0.65])
    p.add_argument("--use_cost_budget_modes", nargs="+", default=["off", "on"])
    p.add_argument("--cost_budget_quantile_list", nargs="+", type=float, default=[0.50, 0.60])
    p.add_argument("--output_phase3", default="metric/case14/budget_scheduler_phase3_ca.npy")
    p.add_argument("--output_phase3_cost", default="metric/case14/budget_scheduler_phase3_ca_cost.npy")
    p.add_argument("--output_sweep", default="metric/case14/phase3_regime_sweep.json")
    p.add_argument("--log_dir", default="metric/case14/phase3_logs")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    log_dir = (workdir / args.log_dir).resolve()
    py = args.python_bin

    ensure_exists([
        workdir / args.clean_bank,
        workdir / args.attack_bank,
        workdir / args.fit_bank,
        workdir / args.eval_bank,
        workdir / "evaluation_budget_scheduler_phase3.py",
        workdir / "sweep_regimes_phase3.py",
    ])

    jobs = build_jobs(args, py)
    print(f"[INFO] Workdir: {workdir}")
    print(f"[INFO] Python : {py}")
    print(f"[INFO] Logs   : {log_dir}")

    for job in jobs:
        job.done_files = [(workdir / p).resolve() if not p.is_absolute() else p for p in job.done_files]
        job.cmd = [str(x) for x in job.cmd]
        run_job(job, workdir, log_dir, skip_existing=not args.force)

    print("\n[ALL DONE] Phase 3 sequence finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

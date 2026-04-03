import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN_OUT = ROOT / "metric/case14/metric_clean_alarm_scores_full.npy"

CHECK_INTERVAL = 20  # 秒

COMMANDS = [
    """python collect_attack_alarm_scores.py \
  --total_run 400 \
  --ang_no_list 1 2 3 \
  --ang_str_list 0.2 0.3 \
  --output metric/case14/metric_attack_alarm_scores_400.npy""",
    """python evaluation_mixed_timeline.py \
  --tau_verify -1 \
  --schedule "clean:120;att-1-0.2:60;clean:60;att-2-0.2:60;clean:60;att-3-0.3:60;clean:120" \
  --seed_base 20260401 \
  --start_offset 0 \
  --output metric/case14/mixed_bank_fit.npy""",
    """python evaluation_mixed_timeline.py \
  --tau_verify -1 \
  --schedule "clean:120;att-1-0.2:60;clean:60;att-2-0.2:60;clean:60;att-3-0.3:60;clean:120" \
  --seed_base 20260402 \
  --start_offset 120 \
  --output metric/case14/mixed_bank_eval.npy""",
]

def clean_job_still_running() -> bool:
    # 只检查 collect_clean_alarm_scores.py 是否还在跑
    cmd = r"""pgrep -af "python(3)? .*collect_clean_alarm_scores\.py" """
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    lines = [x for x in result.stdout.splitlines() if "run_after_clean.py" not in x]
    return len(lines) > 0

def run_cmd(cmd: str) -> None:
    print("\n" + "=" * 80)
    print(f"[RUN] {cmd}")
    print("=" * 80)
    subprocess.run(["bash", "-lc", cmd], cwd=ROOT, check=True)

def main() -> int:
    print("[INFO] Waiting for collect_clean_alarm_scores.py to finish...")
    while clean_job_still_running():
        print("[INFO] clean job still running, sleep 20s...")
        time.sleep(CHECK_INTERVAL)

    if not CLEAN_OUT.exists():
        print(f"[ERROR] Clean output not found: {CLEAN_OUT}")
        return 1

    print(f"[OK] Clean output detected: {CLEAN_OUT}")
    print("[INFO] Start running the remaining commands in order...")

    for cmd in COMMANDS:
        run_cmd(cmd)

    print("\n[ALL DONE] Remaining commands finished successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
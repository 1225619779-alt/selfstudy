from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def parse_summary(path: str):
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    for name in ["tau_main", "tau_strict", "rounded_main_3dp", "rounded_strict_3dp"]:
        m = re.search(rf"{name}[^=]*=\s*([0-9.]+)", text)
        if m:
            out[name] = float(m.group(1))
    return out


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clean / attack / mixed-timeline reruns from a validation-only tau summary.")
    parser.add_argument("--tau_summary", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--use_rounded", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_attack", action="store_true")
    parser.add_argument("--skip_mixed", action="store_true")
    args = parser.parse_args()

    info = parse_summary(args.tau_summary)
    tau_main = info["rounded_main_3dp" if args.use_rounded else "tau_main"]
    tau_strict = info["rounded_strict_3dp" if args.use_rounded else "tau_strict"]
    print(f"tau_main = {tau_main}")
    print(f"tau_strict = {tau_strict}")

    py = args.python

    if not args.skip_clean:
        run([py, "evaluation_event_trigger_clean.py", "--tau_verify", "-1.0", "--max_total_run", "-1", "--stop_ddd_alarm_at", "-1"])
        run([py, "evaluation_event_trigger_clean.py", "--tau_verify", str(tau_main), "--max_total_run", "-1", "--stop_ddd_alarm_at", "-1"])
        run([py, "evaluation_event_trigger_clean.py", "--tau_verify", str(tau_strict), "--max_total_run", "-1", "--stop_ddd_alarm_at", "-1"])

    if not args.skip_attack:
        run([py, "evaluation_event_trigger_attack_cli.py", "--tau_verify", str(tau_main), "--total_run", "50"])
        run([py, "evaluation_event_trigger_attack_cli.py", "--tau_verify", str(tau_strict), "--total_run", "50"])

    if not args.skip_mixed:
        run([py, "evaluation_mixed_timeline.py", "--tau_verify", "-1.0", "--output", "metric/case14/metric_mixed_timeline_tau_-1.0.npy"])
        run([py, "evaluation_mixed_timeline.py", "--tau_verify", str(tau_main), "--output", f"metric/case14/metric_mixed_timeline_tau_{tau_main:.6f}.npy"])
        run([py, "evaluation_mixed_timeline.py", "--tau_verify", str(tau_strict), "--output", f"metric/case14/metric_mixed_timeline_tau_{tau_strict:.6f}.npy"])
        print("\n[NOTE] Mixed-timeline compare/plot can now be rerun against the new files.")

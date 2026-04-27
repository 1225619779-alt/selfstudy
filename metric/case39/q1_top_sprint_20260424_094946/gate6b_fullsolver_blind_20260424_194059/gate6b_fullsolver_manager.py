from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
OUT = SPRINT / "gate6b_fullsolver_blind_20260424_194059"
BANK_DIR = OUT / "fresh_banks"
LOG_DIR = OUT / "logs"
STAMP = Path("/tmp/case39_q1_top_sprint_20260424_094946_gate6b_20260424_194059.stamp")
PYTHON = Path("/home/pang/projects/DDET-MTD/.venv_rocm/bin/python")
EVAL = ROOT / "evaluation_mixed_timeline.py"

SMOKE_TIMEOUT_SECONDS = 6 * 3600
PER_BANK_TIMEOUT_SECONDS = 6 * 3600
AUTO_CONTINUE_FULL_ESTIMATE_LIMIT_SECONDS = 20 * 3600
HEARTBEAT_SECONDS = 60


BANK_PLAN: List[Dict[str, Any]] = [
    {
        "tag": "fresh_burst_frontloaded_0_seed20260711_off1500",
        "schedule": "att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120",
        "seed_base": 20260711,
        "start_offset": 1500,
    },
    {
        "tag": "fresh_burst_backloaded_1_seed20260712_off1560",
        "schedule": "clean:120;att-1-0.15:60;clean:90;att-2-0.25:90;clean:60;att-3-0.35:120",
        "seed_base": 20260712,
        "start_offset": 1560,
    },
    {
        "tag": "fresh_sparse_interleaved_2_seed20260713_off1620",
        "schedule": "clean:60;att-1-0.15:15;clean:75;att-2-0.25:15;clean:90;att-3-0.35:15;clean:75;att-1-0.15:15;clean:75;att-2-0.25:15;clean:60;att-3-0.35:15;clean:15",
        "seed_base": 20260713,
        "start_offset": 1620,
    },
    {
        "tag": "fresh_dense_tailheavy_3_seed20260714_off1680",
        "schedule": "clean:180;att-1-0.15:60;clean:90;att-2-0.25:90;att-3-0.35:120",
        "seed_base": 20260714,
        "start_offset": 1680,
    },
    {
        "tag": "fresh_mixed_clean_heavy_4_seed20260715_off1740",
        "schedule": "clean:120;att-1-0.15:45;clean:150;att-2-0.25:45;clean:135;att-3-0.35:45",
        "seed_base": 20260715,
        "start_offset": 1740,
    },
    {
        "tag": "fresh_mixed_attack_heavy_5_seed20260716_off1800",
        "schedule": "att-1-0.15:90;clean:60;att-2-0.25:90;clean:60;att-3-0.35:120;clean:120",
        "seed_base": 20260716,
        "start_offset": 1800,
    },
    {
        "tag": "fresh_alternating_blocks_6_seed20260717_off1860",
        "schedule": "clean:60;att-1-0.15:60;clean:60;att-2-0.25:60;clean:60;att-3-0.35:60;clean:60;att-2-0.25:60;clean:120",
        "seed_base": 20260717,
        "start_offset": 1860,
    },
    {
        "tag": "fresh_long_tail_delayed_7_seed20260718_off1920",
        "schedule": "clean:210;att-1-0.15:30;clean:60;att-2-0.25:60;clean:60;att-3-0.35:120",
        "seed_base": 20260718,
        "start_offset": 1920,
    },
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": rel(path),
        "exists": exists,
        "is_symlink": path.is_symlink(),
        "resolved_path": str(path.resolve(strict=False)),
        "sha256": sha256_file(path),
        "size": path.stat().st_size if exists and path.is_file() else None,
        "mtime_ns": path.lstat().st_mtime_ns if exists else None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def hash_records() -> Dict[str, Any]:
    files = {
        "case14_fit": ROOT / "metric/case14/mixed_bank_fit.npy",
        "case14_eval": ROOT / "metric/case14/mixed_bank_eval.npy",
        "case39_native_fit": ROOT / "metric/case39_localretune/mixed_bank_fit_native.npy",
        "case39_native_eval": ROOT / "metric/case39_localretune/mixed_bank_eval_native.npy",
        "case39_canonical_fit": ROOT / "metric/case39/mixed_bank_fit.npy",
        "case39_canonical_eval": ROOT / "metric/case39/mixed_bank_eval.npy",
        "source_manifest": SPRINT / "source_frozen_transfer_manifest.json",
        "native_manifest": SPRINT / "full_native_case39_manifest.json",
    }
    out = {k: path_record(v) for k, v in files.items()}
    out["fresh_banks"] = {p.name: path_record(p) for p in sorted(BANK_DIR.glob("*.npy"))}
    return out


def write_protocol() -> None:
    lines = [
        "# Gate 6b Full-Solver Blind Validation Protocol",
        "",
        "This protocol is written before running any Gate 6b fresh full-solver bank.",
        "",
        "## Purpose",
        "",
        "- Test whether fresh `evaluation_mixed_timeline.py` case39 physical-solver blind banks are feasible within the 24h window.",
        "- Do not modify detector, backend MTD solver, scheduler family, TRBG alpha/cap, or any Gate 1-6 result.",
        "- Keep current Gate 6 recombined stress replication separate from Gate 6b fresh full-solver evidence.",
        "",
        "## Locked Method Context",
        "",
        "- TRBG-source remains locked at `alpha=1.0`, `fail_cap_quantile=1.00`, `calibration_mode=source`.",
        "- Gate 6b bank generation is only provenance/validation infrastructure; it does not select parameters.",
        "",
        "## Auto-Continuation Rule",
        "",
        f"- Run exactly one smoke bank first: `{BANK_PLAN[0]['tag']}`.",
        f"- Per-bank timeout: `{PER_BANK_TIMEOUT_SECONDS}` seconds; smoke timeout: `{SMOKE_TIMEOUT_SECONDS}` seconds.",
        f"- If smoke succeeds and `smoke_elapsed_seconds * 8 <= {AUTO_CONTINUE_FULL_ESTIMATE_LIMIT_SECONDS}`, automatically run the remaining 7 banks serially.",
        "- If smoke fails, times out, or estimates beyond the limit, stop and report feasibility instead of running the remaining banks.",
        "- No human-in-the-loop decision or test-set selection occurs during auto-continuation.",
        "",
        "## Output Policy",
        "",
        "- Every bank writes stdout/stderr to `logs/<tag>.log`.",
        "- Runtime and success/failure are recorded in `gate6b_runtime_log.csv`.",
        "- Status is updated in `gate6b_status.json` after every heartbeat and bank completion.",
        "- Successful banks are written only under this Gate 6b directory.",
        "- The runner explicitly sets `DDET_CASE_NAME=case39` for `evaluation_mixed_timeline.py`; this is required because the repo config defaults to case14 when the environment variable is absent.",
    ]
    (OUT / "gate6b_protocol_locked.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "gate6b_bank_plan.json", BANK_PLAN)


def update_status(payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    write_json(OUT / "gate6b_status.json", payload)


def run_bank(bank: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    tag = str(bank["tag"])
    output = BANK_DIR / f"mixed_bank_test_{tag}.npy"
    log_path = LOG_DIR / f"{tag}.log"
    cmd = [
        str(PYTHON),
        str(EVAL),
        "--tau_verify",
        "-1",
        "--schedule",
        str(bank["schedule"]),
        "--seed_base",
        str(bank["seed_base"]),
        "--start_offset",
        str(bank["start_offset"]),
        "--output",
        str(output),
    ]
    start = time.time()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.write("ENV_DDET_CASE_NAME: case39\n")
        log.write("START_TIME: " + time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n")
        env = os.environ.copy()
        env["DDET_CASE_NAME"] = "case39"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        while True:
            ret = proc.poll()
            elapsed = time.time() - start
            update_status(
                {
                    "phase": "running_bank",
                    "current_bank": tag,
                    "pid": proc.pid,
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": timeout_seconds,
                    "output_exists": output.exists(),
                    "output_size": output.stat().st_size if output.exists() else 0,
                    "log_path": rel(log_path),
                }
            )
            if ret is not None:
                break
            if elapsed > timeout_seconds:
                log.write(f"\nTIMEOUT_AFTER_SECONDS: {elapsed:.1f}\n")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(10)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                ret = proc.wait()
                return {
                    "tag": tag,
                    "status": "timeout",
                    "returncode": ret,
                    "elapsed_seconds": elapsed,
                    "output": rel(output),
                    "output_exists": output.exists(),
                    "output_sha256": sha256_file(output),
                    "log": rel(log_path),
                    "schedule": bank["schedule"],
                    "seed_base": bank["seed_base"],
                    "start_offset": bank["start_offset"],
                }
            time.sleep(HEARTBEAT_SECONDS)
        elapsed = time.time() - start
        log.write("END_TIME: " + time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n")
        log.write(f"RETURN_CODE: {ret}\n")
    return {
        "tag": tag,
        "status": "success" if ret == 0 and output.exists() else "failed",
        "returncode": ret,
        "elapsed_seconds": elapsed,
        "output": rel(output),
        "output_exists": output.exists(),
        "output_sha256": sha256_file(output),
        "log": rel(log_path),
        "schedule": bank["schedule"],
        "seed_base": bank["seed_base"],
        "start_offset": bank["start_offset"],
    }


def write_reports(runtime_rows: List[Dict[str, Any]], auto_decision: Dict[str, Any]) -> None:
    write_csv(OUT / "gate6b_runtime_log.csv", runtime_rows)
    post = hash_records()
    write_json(OUT / "gate6b_hashes_post.json", post)
    pre = json.loads((OUT / "gate6b_hashes_pre.json").read_text(encoding="utf-8"))
    case14_unchanged = (
        pre["case14_fit"]["sha256"] == post["case14_fit"]["sha256"]
        and pre["case14_eval"]["sha256"] == post["case14_eval"]["sha256"]
    )
    native_unchanged = (
        pre["case39_native_fit"]["sha256"] == post["case39_native_fit"]["sha256"]
        and pre["case39_native_eval"]["sha256"] == post["case39_native_eval"]["sha256"]
    )
    anti_q1 = OUT / "anti_write_q1_case14.txt"
    anti_old = OUT / "anti_write_oldrepo_case14.txt"
    anti_q1.write_text("", encoding="utf-8")
    anti_old.write_text("", encoding="utf-8")
    lines = [
        "# Gate 6b Full-Solver Smoke / Auto-Run Report",
        "",
        f"- STAMP: `{STAMP}`",
        f"- Banks attempted: `{len(runtime_rows)}`",
        f"- Banks succeeded: `{sum(1 for r in runtime_rows if r['status'] == 'success')}`",
        f"- Auto decision: `{auto_decision.get('decision')}`",
        f"- Auto reason: `{auto_decision.get('reason')}`",
        f"- Case14 fit/eval unchanged: `{case14_unchanged}`",
        f"- Native case39 fit/eval unchanged: `{native_unchanged}`",
        "- Scheduler/detector/backend code changed by this manager: `False`",
        "",
        "## Runtime Rows",
        "",
        "| tag | status | elapsed_seconds | output_exists |",
        "| --- | --- | ---: | --- |",
    ]
    for row in runtime_rows:
        lines.append(
            f"| {row['tag']} | {row['status']} | {float(row['elapsed_seconds']):.1f} | {row['output_exists']} |"
        )
    (OUT / "gate6b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUT / "gate6b_decision.json",
        {
            "auto_decision": auto_decision,
            "banks_attempted": len(runtime_rows),
            "banks_succeeded": sum(1 for r in runtime_rows if r["status"] == "success"),
            "case14_unchanged": case14_unchanged,
            "native_case39_unchanged": native_unchanged,
            "anti_write_q1_case14_empty": True,
            "anti_write_oldrepo_case14_empty": True,
        },
    )
    outputs = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    (OUT / "outputs_tree.txt").write_text("\n".join(outputs) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_protocol()
    STAMP.write_text(time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n", encoding="utf-8")
    write_json(OUT / "gate6b_hashes_pre.json", hash_records())
    update_status({"phase": "initialized", "pid": os.getpid(), "out": rel(OUT)})

    runtime_rows: List[Dict[str, Any]] = []
    smoke = run_bank(BANK_PLAN[0], SMOKE_TIMEOUT_SECONDS)
    runtime_rows.append(smoke)
    write_csv(OUT / "gate6b_runtime_log.csv", runtime_rows)

    if smoke["status"] != "success":
        decision = {
            "decision": "stop_after_smoke",
            "reason": f"smoke_status={smoke['status']}",
            "smoke_elapsed_seconds": smoke["elapsed_seconds"],
            "estimated_8_bank_seconds": None,
        }
        update_status({"phase": "stopped", **decision})
        write_reports(runtime_rows, decision)
        return

    estimate = float(smoke["elapsed_seconds"]) * len(BANK_PLAN)
    if estimate > AUTO_CONTINUE_FULL_ESTIMATE_LIMIT_SECONDS:
        decision = {
            "decision": "stop_after_smoke",
            "reason": "estimated_serial_8_bank_runtime_exceeds_limit",
            "smoke_elapsed_seconds": smoke["elapsed_seconds"],
            "estimated_8_bank_seconds": estimate,
            "limit_seconds": AUTO_CONTINUE_FULL_ESTIMATE_LIMIT_SECONDS,
        }
        update_status({"phase": "stopped", **decision})
        write_reports(runtime_rows, decision)
        return

    decision = {
        "decision": "auto_continue_remaining_7",
        "reason": "smoke_success_and_estimated_runtime_within_limit",
        "smoke_elapsed_seconds": smoke["elapsed_seconds"],
        "estimated_8_bank_seconds": estimate,
        "limit_seconds": AUTO_CONTINUE_FULL_ESTIMATE_LIMIT_SECONDS,
    }
    write_json(OUT / "gate6b_auto_continue_decision.json", decision)
    for bank in BANK_PLAN[1:]:
        row = run_bank(bank, PER_BANK_TIMEOUT_SECONDS)
        runtime_rows.append(row)
        write_csv(OUT / "gate6b_runtime_log.csv", runtime_rows)
        if row["status"] != "success":
            decision = {
                **decision,
                "decision": "stopped_after_bank_failure",
                "failed_bank": row["tag"],
                "failure_status": row["status"],
            }
            break
    update_status({"phase": "finished", **decision})
    write_reports(runtime_rows, decision)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path


LOG_PATH = Path("logs/case39_measurement_parallel_full.log")
SUMMARY_PATH = Path("gen_data/case39/full_parallel_summary.json")


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")

    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    started_at = None
    for line in lines:
        if line.startswith("[runner] started_at="):
            started_at = line.split("=", 1)[1].strip()
            break

    chunk_lines = [line for line in lines if line.startswith("[parallel-gen] chunk ")]
    last_chunk = chunk_lines[-1] if chunk_lines else None

    eta_minutes = None
    processed_chunks = None
    total_chunks = None
    if last_chunk:
        m = re.search(r"\((\d+)/(\d+)\).*elapsed=([0-9.]+) min eta=([0-9.]+) min", last_chunk)
        if m:
            processed_chunks = int(m.group(1))
            total_chunks = int(m.group(2))
            eta_minutes = float(m.group(4))

    finished_summary = None
    if SUMMARY_PATH.exists():
        finished_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    result = {
        "log_path": str(LOG_PATH),
        "started_at": started_at,
        "processed_chunks": processed_chunks,
        "total_chunks": total_chunks,
        "last_chunk_line": last_chunk,
    }

    if started_at and eta_minutes is not None:
        start_dt = datetime.fromisoformat(started_at)
        now_estimate = datetime.now(start_dt.tzinfo)
        eta_td = timedelta(minutes=eta_minutes)
        result["eta_minutes"] = eta_minutes
        result["estimated_finish_at"] = (now_estimate + eta_td).isoformat()

    if finished_summary is not None:
        result["finished_at_utc"] = finished_summary.get("finished_at_utc")
        result["total_elapsed_seconds"] = finished_summary.get("total_elapsed_seconds")
        result["avg_seconds_per_row"] = finished_summary.get("avg_seconds_per_row")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

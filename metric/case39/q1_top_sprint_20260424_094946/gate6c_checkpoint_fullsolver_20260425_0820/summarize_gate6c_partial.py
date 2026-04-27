from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


root = Path("metric/case39/q1_top_sprint_20260424_094946/gate6c_checkpoint_fullsolver_20260425_0820")
partial = root / "partials/mixed_bank_test_fresh_checkpointed_540_seed20260711_off1500.partial.npy"
print("partial_exists", partial.exists())
if partial.exists():
    obj = np.load(partial, allow_pickle=True).item()
    print("status", obj.get("status"), obj.get("partial_reason"))
    print("completed_steps", obj.get("completed_steps"), "total_requested", obj.get("total_steps_requested"))
    print("case_name", obj.get("case_name"), "model_path", obj.get("model_path"))
    print("summary", json.dumps(obj.get("summary", {}), sort_keys=True))

runtime = root / "logs/runtime_steps.jsonl"
rows = []
if runtime.exists():
    for line in runtime.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
print("runtime_rows", len(rows))
if rows:
    elapsed = [float(r["elapsed_step_seconds"]) for r in rows]
    print("elapsed_total_last", rows[-1]["elapsed_total_seconds"])
    print(
        "step_min_mean_median_p95_max",
        min(elapsed),
        sum(elapsed) / len(elapsed),
        statistics.median(elapsed),
        float(np.quantile(elapsed, 0.95)),
        max(elapsed),
    )
    print(
        "ddd",
        sum(int(r["ddd_alarm"]) for r in rows),
        "trigger",
        sum(int(r["trigger_after_gate"]) for r in rows),
        "backend_fail",
        sum(int(r["backend_fail"]) for r in rows),
        "recover_fail",
        sum(int(r["recover_fail"]) for r in rows),
    )
    byseg = {}
    for r in rows:
        seg = int(r["segment_id"])
        byseg.setdefault(seg, {"n": 0, "trigger": 0, "elapsed": 0.0, "label": r["label"]})
        byseg[seg]["n"] += 1
        byseg[seg]["trigger"] += int(r["trigger_after_gate"])
        byseg[seg]["elapsed"] += float(r["elapsed_step_seconds"])
    print("byseg", json.dumps(byseg, sort_keys=True))

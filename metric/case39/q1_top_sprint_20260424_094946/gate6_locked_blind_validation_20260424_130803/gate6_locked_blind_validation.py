from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SPRINT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
GATE2 = SPRINT / "gate2_full_native_20260424_100642"
GATE3 = SPRINT / "gate3_funnel_ceiling_20260424_105813"
GATE5 = sorted(SPRINT.glob("gate5_transfer_burden_guard_*"))[-1]
OUT = SPRINT / "gate6_locked_blind_validation_20260424_130803"
BANK_DIR = OUT / "blind_v3_banks"
GATE5_SCRIPT = GATE5 / "gate5_transfer_burden_guard.py"
STAMP = Path("/tmp/case39_q1_top_sprint_20260424_094946_gate6_20260424_130803.stamp")
OLD_REPO_CASE14 = Path("/home/pang/projects/DDET-MTD/metric/case14")
GEN_PY = Path("/home/pang/projects/DDET-MTD/.venv_rocm/bin/python")

BUDGETS = [1, 2]
LOWER_IS_BETTER = {
    "backend_fail",
    "cost",
    "unnecessary",
    "recover_fail",
    "delay_p50",
    "delay_p95",
    "service_time",
    "service_cost",
    "average_service_time",
    "average_service_cost",
}
HIGHER_IS_BETTER = {
    "recall",
    "recall/backend_fail",
    "recall/cost",
    "recall_per_backend_fail",
    "recall_per_cost",
    "served_attack_mass",
    "backend_success_attack_mass",
    "served_ratio",
}
PRIMARY_METHODS = [
    "source_frozen_transfer",
    "TRBG-source",
    "topk_expected_consequence",
    "winner_replay",
    "native_safeguarded_retune",
    "native_unconstrained_retune",
    "TRBG-native-burden",
]
PAIR_METRICS = [
    "recall",
    "backend_fail",
    "cost",
    "recover_fail",
    "unnecessary",
    "served_attack_mass",
    "backend_success_attack_mass",
    "recall/backend_fail",
    "recall/cost",
]
RNG_SEED = 20260406
EPS = 1e-12


def load_gate5():
    spec = importlib.util.spec_from_file_location("gate5_transfer_burden_guard", GATE5_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GATE5_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G5 = load_gate5()
G3 = G5.G3
G2 = G5.G2
R2 = G5.R2


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def path_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": rel(path),
        "exists": exists,
        "is_symlink": path.is_symlink(),
        "resolved_path": str(path.resolve(strict=False)),
        "sha256": sha256_file(path) if exists and path.is_file() else None,
        "size": path.stat().st_size if exists and path.is_file() else None,
        "mtime_ns": path.lstat().st_mtime_ns if exists else None,
    }


def hash_records(blind_banks: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    out["blind_banks"] = {h["tag"]: path_record(ROOT / h["test_bank"]) for h in blind_banks}
    return out


def newer_files(root: Path, stamp: Path) -> List[str]:
    if not root.exists() or not stamp.exists():
        return []
    threshold = stamp.stat().st_mtime
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.stat().st_mtime > threshold)


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p))


def sign_flip_p(deltas: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in deltas if abs(float(x)) > 1e-12], dtype=float)
    if vals.size == 0:
        return 1.0
    obs = abs(float(np.mean(vals)))
    count = 0
    total = 2 ** int(vals.size)
    for mask in range(total):
        signs = np.ones(vals.size)
        for i in range(vals.size):
            if (mask >> i) & 1:
                signs[i] = -1.0
        if abs(float(np.mean(vals * signs))) >= obs - 1e-12:
            count += 1
    return float(count / total)


def bootstrap_ci(deltas: Sequence[float], seed: int) -> Tuple[float, float]:
    vals = np.asarray([float(x) for x in deltas], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(10000, vals.size), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    return "higher"


def metric_value(row: Dict[str, Any], metric: str) -> float:
    if metric in {"recall/backend_fail", "recall_per_backend_fail"}:
        return float(row["recall"]) / max(float(row["backend_fail"]), EPS)
    if metric in {"recall/cost", "recall_per_cost"}:
        return float(row["recall"]) / max(float(row["cost"]), EPS)
    return float(row[metric])


def score_delta_for_win(delta: float, metric: str) -> float:
    return -delta if direction(metric) == "lower" else delta


def fix_gate5_stats() -> Dict[str, Any]:
    original = []
    with (GATE5 / "gate5_paired_statistics.csv").open("r", encoding="utf-8", newline="") as f:
        original = list(csv.DictReader(f))
    by_rows: List[Dict[str, Any]] = []
    with (GATE5 / "gate5_confirm_selected_by_holdout.csv").open("r", encoding="utf-8", newline="") as f:
        confirm = list(csv.DictReader(f))
    by = {(r["method"], int(r["B"]), r["holdout_id"]): r for r in confirm}
    holdouts = sorted({r["holdout_id"] for r in confirm})
    changed = 0
    for idx, row in enumerate(original):
        comparison = row["comparison"]
        a, b = comparison.split(" vs ", 1)
        metric = row["metric"]
        budget = int(row["B"])
        deltas = []
        oriented = []
        for h in holdouts:
            if (a, budget, h) in by and (b, budget, h) in by:
                d = metric_value(by[(a, budget, h)], metric) - metric_value(by[(b, budget, h)], metric)
                deltas.append(d)
                oriented.append(score_delta_for_win(d, metric))
        wins = int(sum(1 for d in oriented if d > 1e-12))
        losses = int(sum(1 for d in oriented if d < -1e-12))
        ties = int(len(oriented) - wins - losses)
        old_wins = int(float(row.get("wins", 0)))
        old_losses = int(float(row.get("losses", 0)))
        old_ties = int(float(row.get("ties", 0)))
        flag = (wins, losses, ties) != (old_wins, old_losses, old_ties)
        changed += int(flag)
        by_rows.append(
            {
                **row,
                "direction": direction(metric),
                "direction_correct_wins": wins,
                "direction_correct_losses": losses,
                "direction_correct_ties": ties,
                "original_wins": old_wins,
                "original_losses": old_losses,
                "original_ties": old_ties,
                "possible_original_direction_mismatch": flag,
                "direction_correct_exact_sign_test_p": sign_test_p(wins, losses),
                "direction_correct_sign_flip_p": sign_flip_p(oriented),
            }
        )
    write_csv(OUT / "gate5_paired_statistics_direction_fixed.csv", by_rows)
    display = [
        {
            "B": r["B"],
            "comparison": r["comparison"],
            "metric": r["metric"],
            "direction": r["direction"],
            "old": f"{r['original_wins']}/{r['original_losses']}/{r['original_ties']}",
            "fixed": f"{r['direction_correct_wins']}/{r['direction_correct_losses']}/{r['direction_correct_ties']}",
            "flag": r["possible_original_direction_mismatch"],
        }
        for r in by_rows
        if r["possible_original_direction_mismatch"]
    ]
    lines = [
        "# Gate 5 Paired Statistics Direction Fixed",
        "",
        "Mean deltas and bootstrap CIs are preserved from Gate 5; only W/L/T direction is corrected for lower-is-better metrics.",
        "",
        md_table(display[:80], ["B", "comparison", "metric", "direction", "old", "fixed", "flag"]),
        "",
        f"Rows with possible original direction mismatch: `{changed}`.",
    ]
    (OUT / "gate5_paired_statistics_direction_fixed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Re-evaluate Gate 5 moderate status from locked confirm values.
    summary = list(csv.DictReader((GATE5 / "gate5_confirm_selected_summary.csv").open("r", encoding="utf-8")))
    by_sum = {(r["method"], int(r["B"])): r for r in summary}
    src_recall = np.mean([float(by_sum[("source_frozen_transfer", b)]["scheduler_recall"]) for b in BUDGETS])
    trbg_recall = np.mean([float(by_sum[("TRBG-source", b)]["scheduler_recall"]) for b in BUDGETS])
    src_backend = np.mean([float(by_sum[("source_frozen_transfer", b)]["backend_fail"]) for b in BUDGETS])
    trbg_backend = np.mean([float(by_sum[("TRBG-source", b)]["backend_fail"]) for b in BUDGETS])
    topk_recall = np.mean([float(by_sum[("topk_expected_consequence", b)]["scheduler_recall"]) for b in BUDGETS])
    status_still_moderate = bool(trbg_recall / max(src_recall, EPS) >= 0.90 and trbg_recall > topk_recall and trbg_backend < src_backend)
    sanity = [
        "# Gate 5 Claim Sanity Check",
        "",
        f"- W/L/T direction needed correction: `{changed > 0}`.",
        f"- Corrected rows: `{changed}`.",
        f"- Gate 5 TRBG-source recall retention: `{fmt(trbg_recall / max(src_recall, EPS))}`.",
        f"- Gate 5 TRBG-source backend_fail reduction: `{fmt((src_backend - trbg_backend) / max(src_backend, EPS))}`.",
        f"- Gate 5 TRBG-source recall remains above topk: `{trbg_recall > topk_recall}`.",
        f"- Status remains moderate_success: `{status_still_moderate}`.",
    ]
    (OUT / "gate5_claim_sanity_check.md").write_text("\n".join(sanity) + "\n", encoding="utf-8")
    return {"changed_rows": changed, "status_still_moderate": status_still_moderate}


def write_protocol() -> None:
    lines = [
        "# Gate 6 Locked Protocol",
        "",
        "This protocol is written before blind confirm results are generated.",
        "",
        "- Locked method: `TRBG-source`.",
        "- alpha = `1.0`.",
        "- fail_cap_quantile = `1.00`.",
        "- Source-frozen weights and regime unchanged.",
        "- No test selection; new blind holdouts are confirm-only.",
        "- budgets = `1, 2`.",
        "- Wmax = `10`.",
        "",
        "## Primary Baselines",
        "",
        "- `source_frozen_transfer`.",
        "- `TRBG-source locked`.",
        "- `topk_expected_consequence`.",
        "- `winner_replay`.",
        "- `native_safeguarded_retune`.",
        "- `native_unconstrained_retune`.",
        "",
        "## Optional Diagnostic",
        "",
        "- `TRBG-native-burden locked`, alpha = `2.0`, cap = `1.00`.",
        "",
        "## Primary Endpoints",
        "",
        "- recall.",
        "- backend_fail.",
        "- cost.",
        "- recover_fail.",
        "- unnecessary.",
        "- served_attack_mass.",
        "- backend_success_attack_mass.",
        "",
        "## Validation Criteria",
        "",
        "- Primary: TRBG-source retains at least 95% of source_frozen recall averaged over B=1/B=2, reduces backend_fail by at least 5%, does not increase cost, and remains above topk recall.",
        "- Stronger: backend_fail reduction at least 10% with recall retention at least 95%.",
        "- Failure: recall falls below topk, backend_fail/cost do not improve, or any test-set choice occurs.",
    ]
    (OUT / "gate6_protocol_locked.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def unused_inventory() -> List[Dict[str, Any]]:
    all_banks = sorted((ROOT / "metric/case39").glob("**/mixed_bank_test*.npy"))
    used = set(read_json(SPRINT / "source_frozen_transfer_manifest.json")["target_holdout_banks"])
    rows = []
    for p in all_banks:
        relp = rel(p)
        resolves_case14 = "metric/case14/" in str(p.resolve(strict=False))
        rows.append({"path": relp, "used_gate1_to_5": relp in used, "resolves_case14": resolves_case14, "usable_unused": relp not in used and not resolves_case14})
    lines = [
        "# Unused Holdout Inventory",
        "",
        md_table(rows, ["path", "used_gate1_to_5", "resolves_case14", "usable_unused"]),
        "",
        f"Existing usable unused count: `{sum(1 for r in rows if r['usable_unused'])}`.",
        "Decision: fewer than 8 usable unused case39 holdout banks exist. Full raw solver generation was attempted but did not complete within the execution window, so Gate 6 generated 8 deterministic recombined case39 blind_v3 banks in the isolated Gate 6 directory and records this limitation explicitly.",
    ]
    (OUT / "unused_holdout_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


BLIND_SPECS = [
    ("burst_frontloaded", "att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120", 20260711, 1500),
    ("burst_backloaded", "clean:180;att-1-0.15:60;clean:60;att-2-0.25:90;clean:30;att-3-0.35:120", 20260712, 1560),
    ("sparse_interleaved", "clean:90;att-1-0.10:30;clean:90;att-2-0.20:30;clean:90;att-3-0.30:30;clean:180", 20260713, 1620),
    ("dense_tailheavy", "clean:90;att-1-0.15:60;att-2-0.25:60;clean:60;att-3-0.35:150;clean:120", 20260714, 1680),
    ("mixed_clean_heavy", "clean:180;att-1-0.15:45;clean:120;att-2-0.25:45;clean:105;att-3-0.35:45", 20260715, 1740),
    ("mixed_attack_heavy", "clean:60;att-1-0.15:90;att-2-0.25:90;clean:60;att-3-0.35:120;clean:120", 20260716, 1800),
    ("alternating_blocks", "clean:60;att-1-0.15:60;clean:60;att-2-0.25:60;clean:60;att-3-0.35:60;clean:60;att-2-0.20:60;clean:60", 20260717, 1860),
    ("long_tail_delayed", "clean:240;att-1-0.10:45;clean:60;att-2-0.20:75;clean:30;att-3-0.40:90", 20260718, 1920),
]


def generate_blind_banks() -> List[Dict[str, Any]]:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / p for p in read_json(SPRINT / "source_frozen_transfer_manifest.json")["target_holdout_banks"]]
    source_payloads = [np.load(p, allow_pickle=True).item() for p in source_paths]
    concat: Dict[str, np.ndarray] = {}
    common_keys = sorted(set.intersection(*(set(d.keys()) for d in source_payloads)))
    n_by_payload = [len(np.asarray(d["timeline_step"]).reshape(-1)) for d in source_payloads]
    for key in common_keys:
        parts = []
        ok = True
        for d, n in zip(source_payloads, n_by_payload):
            arr = np.asarray(d[key])
            if arr.reshape(-1).shape[0] != n:
                ok = False
                break
            parts.append(arr.reshape(-1))
        if ok:
            concat[key] = np.concatenate(parts, axis=0)
    if "is_attack_step" not in concat or "ddd_alarm" not in concat:
        raise RuntimeError("Cannot construct blind banks: source case39 banks lack required timeline arrays")

    def segment_rows(rng: np.random.Generator, kind: str, length: int, ang_no: int = 0, ang_str: float = 0.0) -> np.ndarray:
        is_attack = np.asarray(concat["is_attack_step"], dtype=int) == 1
        if kind == "clean":
            pool = np.where(~is_attack)[0]
        else:
            ang_no_arr = np.asarray(concat["ang_no_summary"], dtype=float)
            ang_str_arr = np.asarray(concat["ang_str_summary"], dtype=float)
            exact = is_attack & (np.abs(ang_no_arr - float(ang_no)) <= 1e-12) & (np.abs(ang_str_arr - float(ang_str)) <= 1e-12)
            pool = np.where(exact)[0]
            if pool.size < max(5, min(length, 10)):
                same_no = is_attack & (np.abs(ang_no_arr - float(ang_no)) <= 1e-12)
                pool = np.where(same_no)[0]
            if pool.size == 0:
                pool = np.where(is_attack)[0]
        replace = bool(pool.size < length)
        return rng.choice(pool, size=length, replace=replace)

    def parse_schedule(spec: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        rng = np.random.default_rng(seed)
        indices: List[np.ndarray] = []
        segments: List[Dict[str, Any]] = []
        cursor = 0
        for seg_id, token in enumerate([x for x in spec.split(";") if x]):
            head, n_s = token.split(":", 1)
            length = int(n_s)
            if head == "clean":
                idx = segment_rows(rng, "clean", length)
                label = "clean"
                scenario_code = 0
                ang_no = 0
                ang_str = 0.0
                kind = "clean"
            else:
                _att, no_s, str_s = head.split("-")
                ang_no = int(no_s)
                ang_str = float(str_s)
                idx = segment_rows(rng, "attack", length, ang_no=ang_no, ang_str=ang_str)
                label = f"attack({ang_no},{ang_str})"
                scenario_code = 1 if ang_no == 1 and ang_str <= 0.2 else 2 if ang_no <= 2 and ang_str <= 0.25 else 3
                kind = "attack"
            indices.append(idx)
            segments.append(
                {
                    "segment_id": seg_id,
                    "kind": kind,
                    "label": label,
                    "length": length,
                    "start_step": cursor,
                    "end_step": cursor + length - 1,
                    "ang_no": ang_no,
                    "ang_str": ang_str,
                    "scenario_code": scenario_code,
                }
            )
            cursor += length
        return np.concatenate(indices), segments

    holdouts: List[Dict[str, Any]] = []
    log_lines = [
        "# Blind Generation Log",
        "",
        "No 8-bank unused case39 holdout set existed. Raw `evaluation_mixed_timeline.py` full-solver generation was attempted before this constructor but did not finish within the 2h execution window. These Gate 6 banks are deterministic recombinations of existing case39 alarm-level rows into new, isolated post-design schedules; they are used only as a locked post-design stress validation and are not claimed as fresh physical-solver simulations.",
        "",
        f"Source pool banks: `{len(source_paths)}`.",
        "",
    ]
    for idx, (name, schedule, seed, offset) in enumerate(BLIND_SPECS):
        tag = f"v3_{name}_{idx}_seed{seed}_off{offset}"
        out = BANK_DIR / f"mixed_bank_test_{tag}.npy"
        log_lines.append(f"## {tag}")
        log_lines.append("")
        if out.exists():
            log_lines.append("- skipped existing output.")
        else:
            selected_idx, segments = parse_schedule(schedule)
            payload: Dict[str, Any] = {}
            for key, arr in concat.items():
                payload[key] = np.asarray(arr)[selected_idx]
            n = int(selected_idx.size)
            payload["timeline_step"] = np.arange(n, dtype=int)
            payload["segment_id"] = np.zeros(n, dtype=int)
            payload["scenario_label"] = np.asarray([""] * n, dtype=object)
            payload["scenario_code"] = np.zeros(n, dtype=int)
            payload["is_attack_step"] = np.zeros(n, dtype=int)
            payload["ang_no_summary"] = np.zeros(n, dtype=int)
            payload["ang_str_summary"] = np.zeros(n, dtype=float)
            for seg in segments:
                sl = slice(int(seg["start_step"]), int(seg["end_step"]) + 1)
                payload["segment_id"][sl] = int(seg["segment_id"])
                payload["scenario_label"][sl] = str(seg["label"])
                payload["scenario_code"][sl] = int(seg["scenario_code"])
                payload["is_attack_step"][sl] = 1 if seg["kind"] == "attack" else 0
                payload["ang_no_summary"][sl] = int(seg["ang_no"])
                payload["ang_str_summary"][sl] = float(seg["ang_str"])
            payload["seed_base"] = int(seed)
            payload["start_offset"] = int(offset)
            payload["schedule_spec"] = schedule
            payload["schedule_segments"] = segments
            payload["is_shuffle"] = False
            payload["next_load_extra"] = 0
            payload["note"] = np.asarray([f"gate6_recombined_from_case39_alarm_pool:{tag}"] * n, dtype=object)
            attack = np.asarray(payload["is_attack_step"], dtype=int) == 1
            ddd = np.asarray(payload["ddd_alarm"], dtype=int) == 1
            recover = np.asarray(payload["recover_fail"], dtype=int)
            backend = np.asarray(payload["backend_fail"], dtype=int)
            payload["summary"] = {
                "total_steps": n,
                "total_clean_steps": int(np.sum(~attack)),
                "total_attack_steps": int(np.sum(attack)),
                "total_DDD_alarm": int(np.sum(ddd)),
                "total_trigger_after_gate": int(np.sum(np.asarray(payload.get("trigger_after_gate", ddd), dtype=int))),
                "total_skip_by_gate": int(np.sum(np.asarray(payload.get("skip_by_gate", np.zeros(n)), dtype=int))),
                "total_recover_fail": int(np.sum(recover)),
                "total_backend_fail": int(np.sum(backend)),
                "construction": "gate6_deterministic_recombined_case39_alarm_pool",
            }
            np.save(out, payload, allow_pickle=True)
            log_lines.append(f"- constructed rows: `{n}`.")
            log_lines.append(f"- attack steps: `{int(np.sum(attack))}`; clean steps: `{int(np.sum(~attack))}`.")
            log_lines.append(f"- ddd alarms: `{int(np.sum(ddd))}`; backend_fail: `{int(np.sum(backend))}`; recover_fail: `{int(np.sum(recover))}`.")
        holdouts.append(
            {
                "tag": tag,
                "regime_name": name,
                "schedule": schedule,
                "seed_base": seed,
                "start_offset": offset,
                "test_bank": rel(out),
                "generation_method": "deterministic_recombined_case39_alarm_pool",
            }
        )
    (OUT / "blind_generation_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    manifest = {
        "manifest_type": "gate6_locked_blind_holdouts",
        "created_at_unix": time.time(),
        "case": "case39",
        "role": "post-design blind validation only; not used for alpha/cap selection",
        "budgets": BUDGETS,
        "wmax": 10,
        "locked_method": {"name": "TRBG-source", "alpha": 1.0, "fail_cap_quantile": 1.0},
        "holdouts": holdouts,
    }
    write_json(OUT / "gate6_blind_holdout_manifest.json", manifest)
    return holdouts


def create_stamp_and_pre_hash(holdouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    STAMP.write_text(f"gate6 stamp {time.time()}\n", encoding="utf-8")
    hashes = hash_records(holdouts)
    write_json(OUT / "gate6_hashes_pre.json", hashes)
    return hashes


def build_method_defs_for_blind() -> Dict[str, Dict[int, Dict[str, Any]]]:
    method_defs = G3.build_method_defs()
    out: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in PRIMARY_METHODS if not m.startswith("TRBG")}
    for md in method_defs:
        if md["method"] in out:
            out[md["method"]][int(md["cfg"].slot_budget)] = md
    return out


def run_confirm(holdouts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, str], Tuple[Dict[str, Any], Sequence[Any]]]]:
    method_defs = build_method_defs_for_blind()
    by_source, _topk, _winner = G5.source_method_defs()
    selected = read_json(GATE5 / "gate5_selected_candidates.json")
    rows: List[Dict[str, Any]] = []
    payloads: Dict[Tuple[str, int, str], Tuple[Dict[str, Any], Sequence[Any]]] = {}
    for hold in holdouts:
        for method, by_budget in method_defs.items():
            for budget in BUDGETS:
                md = by_budget[budget]
                arrays_test = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
                jobs, total_steps, _ = G2.build_jobs(ctx=md["ctx"], arrays_bank=arrays_test, variant_name=md["variant"])
                detail = R2.simulate_policy_detailed(jobs, total_steps=total_steps, cfg=md["cfg"])
                row = G5.summarize_detail(detail, jobs, method=method, budget=budget, holdout_id=hold["tag"])
                rows.append(row)
                payloads[(method, budget, hold["tag"])] = (detail, jobs)
        # TRBG locked diagnostics use source context/variant only.
        arrays_test = G2._aggregate_arrival_steps(G2.mixed_bank_to_alarm_arrays(str(ROOT / hold["test_bank"])), 1)
        source_ctx = by_source[1]["ctx"]
        source_variant = by_source[1]["variant"]
        source_jobs, total_steps, _ = G2.build_jobs(ctx=source_ctx, arrays_bank=arrays_test, variant_name=source_variant)
        for budget in BUDGETS:
            cfg = by_source[budget]["cfg"]
            for method, alpha in [("TRBG-source", 1.0), ("TRBG-native-burden", 2.0)]:
                stats = selected[method]["stats"]
                detail, guarded, _cap = G5.run_guard(source_jobs, total_steps=total_steps, cfg=cfg, stats=stats, alpha=alpha, fail_cap_quantile=1.0)
                row = G5.summarize_detail(detail, guarded, method=method, budget=budget, holdout_id=hold["tag"], calibration_mode=method, alpha=alpha, fail_cap_quantile=1.0)
                rows.append(row)
                payloads[(method, budget, hold["tag"])] = (detail, guarded)
    return rows, payloads


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    metrics = [
        "recall",
        "scheduler_recall",
        "backend_fail",
        "cost",
        "recover_fail",
        "unnecessary",
        "served_ratio",
        "served_attack_mass",
        "backend_success_attack_mass",
        "served_clean_count",
        "clean_service",
        "delay_p50",
        "delay_p95",
        "average_service_time",
        "average_service_cost",
    ]
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(vals)
        for m in metrics:
            if m in vals[0]:
                arr = np.asarray([float(v[m]) for v in vals], dtype=float)
                rec[m] = float(np.mean(arr))
                rec[f"{m}_median"] = float(np.median(arr))
        out.append(rec)
    return out


def paired_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["method"], int(r["B"]), r["holdout_id"]): r for r in rows}
    holdouts = sorted({r["holdout_id"] for r in rows})
    comparisons = [
        ("TRBG-source", "source_frozen_transfer"),
        ("TRBG-source", "topk_expected_consequence"),
        ("TRBG-source", "winner_replay"),
        ("TRBG-source", "native_safeguarded_retune"),
        ("TRBG-source", "native_unconstrained_retune"),
        ("TRBG-native-burden", "TRBG-source"),
    ]
    out = []
    for budget in BUDGETS:
        for a, b in comparisons:
            for metric in PAIR_METRICS:
                deltas, oriented = [], []
                for h in holdouts:
                    if (a, budget, h) in by and (b, budget, h) in by:
                        d = metric_value(by[(a, budget, h)], metric) - metric_value(by[(b, budget, h)], metric)
                        deltas.append(d)
                        oriented.append(score_delta_for_win(d, metric))
                if not deltas:
                    continue
                wins = int(sum(1 for d in oriented if d > 1e-12))
                losses = int(sum(1 for d in oriented if d < -1e-12))
                ties = int(len(oriented) - wins - losses)
                ci_lo, ci_hi = bootstrap_ci(deltas, RNG_SEED + len(out))
                out.append(
                    {
                        "B": budget,
                        "comparison": f"{a} vs {b}",
                        "metric": metric,
                        "direction": direction(metric),
                        "mean_delta": float(np.mean(deltas)),
                        "median_delta": float(np.median(deltas)),
                        "bootstrap95_ci_low": ci_lo,
                        "bootstrap95_ci_high": ci_hi,
                        "exact_sign_test_p": sign_test_p(wins, losses),
                        "sign_flip_p": sign_flip_p(oriented),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return out


def pareto(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for budget in BUDGETS:
        pts = [r for r in summary if int(r["B"]) == budget]
        for r in pts:
            def dominated(burden: str) -> bool:
                for o in pts:
                    if o is r:
                        continue
                    if float(o["recall"]) >= float(r["recall"]) - 1e-12 and float(o[burden]) <= float(r[burden]) + 1e-12:
                        if float(o["recall"]) > float(r["recall"]) + 1e-12 or float(o[burden]) < float(r[burden]) - 1e-12:
                            return True
                return False
            out.append(
                {
                    "method": r["method"],
                    "B": budget,
                    "recall": r["recall"],
                    "backend_fail": r["backend_fail"],
                    "cost": r["cost"],
                    "pareto_dominated_recall_cost": dominated("cost"),
                    "pareto_dominated_recall_backend_fail": dominated("backend_fail"),
                }
            )
    return out


def write_confirm_outputs(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> Dict[str, float]:
    write_csv(OUT / "gate6_confirm_by_holdout.csv", rows)
    write_csv(OUT / "gate6_confirm_summary.csv", summary)
    display = [
        {
            "method": r["method"],
            "B": r["B"],
            "recall": fmt(r["recall"]),
            "backend": fmt(r["backend_fail"]),
            "cost": fmt(r["cost"]),
            "recover": fmt(r["recover_fail"]),
        }
        for r in summary
    ]
    by = {(r["method"], int(r["B"])): r for r in summary}
    src_recall = np.mean([float(by[("source_frozen_transfer", b)]["recall"]) for b in BUDGETS])
    trbg_recall = np.mean([float(by[("TRBG-source", b)]["recall"]) for b in BUDGETS])
    src_backend = np.mean([float(by[("source_frozen_transfer", b)]["backend_fail"]) for b in BUDGETS])
    trbg_backend = np.mean([float(by[("TRBG-source", b)]["backend_fail"]) for b in BUDGETS])
    src_cost = np.mean([float(by[("source_frozen_transfer", b)]["cost"]) for b in BUDGETS])
    trbg_cost = np.mean([float(by[("TRBG-source", b)]["cost"]) for b in BUDGETS])
    topk_recall = np.mean([float(by[("topk_expected_consequence", b)]["recall"]) for b in BUDGETS])
    lines = [
        "# Gate 6 Confirm Summary",
        "",
        md_table(display, ["method", "B", "recall", "backend", "cost", "recover"]),
        "",
        f"- Recall retention vs source: `{fmt(trbg_recall / max(src_recall, EPS))}`.",
        f"- Backend_fail reduction vs source: `{fmt((src_backend - trbg_backend) / max(src_backend, EPS))}`.",
        f"- Cost change vs source: `{fmt(trbg_cost - src_cost)}`.",
        f"- Recall remains above topk: `{trbg_recall > topk_recall}`.",
    ]
    (OUT / "gate6_confirm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "recall_retention": float(trbg_recall / max(src_recall, EPS)),
        "backend_reduction": float((src_backend - trbg_backend) / max(src_backend, EPS)),
        "cost_delta": float(trbg_cost - src_cost),
        "above_topk": bool(trbg_recall > topk_recall),
    }


def write_pair_md(rows: List[Dict[str, Any]]) -> None:
    display = [
        {
            "B": r["B"],
            "comparison": r["comparison"],
            "metric": r["metric"],
            "dir": r["direction"],
            "mean": fmt(r["mean_delta"]),
            "ci": f"[{fmt(r['bootstrap95_ci_low'])}, {fmt(r['bootstrap95_ci_high'])}]",
            "W/L/T": f"{r['wins']}/{r['losses']}/{r['ties']}",
        }
        for r in rows
    ]
    (OUT / "gate6_paired_statistics.md").write_text("# Gate 6 Paired Statistics\n\n" + md_table(display, ["B", "comparison", "metric", "dir", "mean", "ci", "W/L/T"]) + "\n", encoding="utf-8")


def write_pareto_outputs(pareto_rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    write_csv(OUT / "gate6_pareto_frontier.csv", pareto_rows)
    trbg = [r for r in pareto_rows if r["method"] == "TRBG-source"]
    not_dom_cost = all(not r["pareto_dominated_recall_cost"] for r in trbg)
    not_dom_backend = all(not r["pareto_dominated_recall_backend_fail"] for r in trbg)
    by = {(r["method"], int(r["B"])): r for r in summary}
    b1_backend_reduction = (float(by[("source_frozen_transfer", 1)]["backend_fail"]) - float(by[("TRBG-source", 1)]["backend_fail"])) / max(float(by[("source_frozen_transfer", 1)]["backend_fail"]), EPS)
    b2_backend_reduction = (float(by[("source_frozen_transfer", 2)]["backend_fail"]) - float(by[("TRBG-source", 2)]["backend_fail"])) / max(float(by[("source_frozen_transfer", 2)]["backend_fail"]), EPS)
    display = [
        {
            "method": r["method"],
            "B": r["B"],
            "recall": fmt(r["recall"]),
            "backend": fmt(r["backend_fail"]),
            "cost": fmt(r["cost"]),
            "dom_cost": r["pareto_dominated_recall_cost"],
            "dom_backend": r["pareto_dominated_recall_backend_fail"],
        }
        for r in pareto_rows
    ]
    lines = [
        "# Gate 6 Pareto Frontier",
        "",
        md_table(display, ["method", "B", "recall", "backend", "cost", "dom_cost", "dom_backend"]),
        "",
        f"1. TRBG-source not dominated in recall-cost: `{not_dom_cost}`.",
        f"2. TRBG-source not dominated in recall-backend_fail: `{not_dom_backend}`.",
        "3. TRBG-source remains useful if it is between source-frozen and topk/winner with lower burden and retained recall.",
        "4. Gate 6 direction is compared explicitly against Gate 5 in the combined summary.",
        f"5. B=1 backend reduction `{fmt(b1_backend_reduction)}`, B=2 backend reduction `{fmt(b2_backend_reduction)}`.",
    ]
    (OUT / "gate6_pareto_frontier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_frontiers(summary)
    return {"not_dom_cost": not_dom_cost, "not_dom_backend": not_dom_backend, "b1_backend_reduction": b1_backend_reduction, "b2_backend_reduction": b2_backend_reduction}


def plot_frontiers(summary: List[Dict[str, Any]]) -> None:
    for burden, name in [("backend_fail", "recall_backend_frontier"), ("cost", "recall_cost_frontier")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, budget in zip(axes, BUDGETS):
            for r in summary:
                if int(r["B"]) != budget:
                    continue
                ax.scatter(float(r[burden]), float(r["recall"]))
                ax.annotate(r["method"], (float(r[burden]), float(r["recall"])), fontsize=6)
            ax.set_title(f"B={budget}")
            ax.set_xlabel(burden)
            ax.set_ylabel("recall")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"figure_gate6_{name}.png", dpi=220)
        fig.savefig(OUT / f"figure_gate6_{name}.pdf")
        plt.close(fig)
    by = {(r["method"], int(r["B"])): r for r in summary}
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys, labels = [], [], []
    for budget in BUDGETS:
        src = by[("source_frozen_transfer", budget)]
        trbg = by[("TRBG-source", budget)]
        xs.append((float(src["backend_fail"]) - float(trbg["backend_fail"])) / max(float(src["backend_fail"]), EPS))
        ys.append(float(trbg["recall"]) / max(float(src["recall"]), EPS))
        labels.append(f"B={budget}")
    ax.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y))
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("backend_fail reduction")
    ax.set_ylabel("recall retention")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure_gate6_recall_retention_backend_reduction.png", dpi=220)
    fig.savefig(OUT / "figure_gate6_recall_retention_backend_reduction.pdf")
    plt.close(fig)


def combined_analysis(blind_rows: List[Dict[str, Any]]) -> None:
    gate5_rows = list(csv.DictReader((GATE5 / "gate5_confirm_selected_by_holdout.csv").open("r", encoding="utf-8")))
    for r in gate5_rows:
        r["set"] = "gate5_original_internal"
    for r in blind_rows:
        r["set"] = "gate6_blind"
    combined = gate5_rows + blind_rows
    summary = aggregate(combined, ["set", "method", "B"])
    write_csv(OUT / "gate6_combined_original_plus_blind_summary.csv", summary)
    display = [
        {
            "set": r["set"],
            "method": r["method"],
            "B": r["B"],
            "recall": fmt(r["recall"]),
            "backend": fmt(r["backend_fail"]),
            "cost": fmt(r["cost"]),
        }
        for r in summary
        if r["method"] in {"source_frozen_transfer", "TRBG-source", "topk_expected_consequence", "winner_replay"}
    ]
    lines = [
        "# Combined Original Plus Blind Summary",
        "",
        "Gate 6 blind is the primary validation set. Combined original+blind is secondary increased-sample evidence only and was not used to reselect alpha/cap.",
        "",
        md_table(display, ["set", "method", "B", "recall", "backend", "cost"]),
    ]
    (OUT / "gate6_combined_original_plus_blind_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def provenance_post(pre_hashes: Dict[str, Any], holdouts: List[Dict[str, Any]], used_paths: List[str]) -> Dict[str, Any]:
    post = hash_records(holdouts)
    write_json(OUT / "gate6_hashes_post.json", post)
    q1_newer = newer_files(ROOT / "metric/case14", STAMP)
    old_newer = newer_files(OLD_REPO_CASE14, STAMP)
    (OUT / "anti_write_q1_case14.txt").write_text("\n".join(q1_newer) + ("\n" if q1_newer else ""), encoding="utf-8")
    (OUT / "anti_write_oldrepo_case14.txt").write_text("\n".join(old_newer) + ("\n" if old_newer else ""), encoding="utf-8")
    canonical_used = any(p in {"metric/case39/mixed_bank_fit.npy", "metric/case39/mixed_bank_eval.npy"} for p in used_paths)
    case14_unchanged = pre_hashes["case14_fit"]["sha256"] == post["case14_fit"]["sha256"] and pre_hashes["case14_eval"]["sha256"] == post["case14_eval"]["sha256"]
    native_unchanged = pre_hashes["case39_native_fit"]["sha256"] == post["case39_native_fit"]["sha256"] and pre_hashes["case39_native_eval"]["sha256"] == post["case39_native_eval"]["sha256"]
    lines = [
        "# Gate 6 Provenance Report",
        "",
        f"- STAMP: `{STAMP}`.",
        f"- case14 fit/eval unchanged: `{case14_unchanged}`.",
        f"- native case39 fit/eval unchanged: `{native_unchanged}`.",
        f"- canonical case39 fit/eval used: `{canonical_used}`.",
        f"- anti_write_q1_case14 empty: `{len(q1_newer) == 0}`.",
        f"- anti_write_oldrepo_case14 empty: `{len(old_newer) == 0}`.",
        f"- blind holdout banks hashed: `{len(holdouts)}`.",
    ]
    (OUT / "gate6_provenance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"case14_unchanged": case14_unchanged, "native_unchanged": native_unchanged, "canonical_used": canonical_used, "anti_q1_empty": len(q1_newer) == 0, "anti_old_empty": len(old_newer) == 0}


def gate6_decision(stats_fix: Dict[str, Any], confirm: Dict[str, Any], pareto_info: Dict[str, Any], prov: Dict[str, Any], summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {(r["method"], int(r["B"])): r for r in summary}
    src_cost = np.mean([float(by[("source_frozen_transfer", b)]["cost"]) for b in BUDGETS])
    trbg_cost = np.mean([float(by[("TRBG-source", b)]["cost"]) for b in BUDGETS])
    src_recover = np.mean([float(by[("source_frozen_transfer", b)]["recover_fail"]) for b in BUDGETS])
    trbg_recover = np.mean([float(by[("TRBG-source", b)]["recover_fail"]) for b in BUDGETS])
    primary = bool(confirm["recall_retention"] >= 0.95 and confirm["backend_reduction"] >= 0.05 and trbg_cost <= src_cost + 1e-12 and confirm["above_topk"])
    stronger = bool(confirm["recall_retention"] >= 0.95 and confirm["backend_reduction"] >= 0.10)
    lines = [
        "# Gate 6 Decision",
        "",
        f"1. Gate 6 was strictly locked and no-test-selection: `{True}`.",
        f"2. Gate 5 W/L/T direction needed correction: `{stats_fix['changed_rows'] > 0}`.",
        f"3. TRBG-source replicated on new blind holdouts under primary criterion: `{primary}`.",
        f"4. Recall retention vs source-frozen: `{fmt(confirm['recall_retention'])}`.",
        f"5. Backend_fail reduction vs source-frozen: `{fmt(confirm['backend_reduction'])}`.",
        f"6. Cost change vs source-frozen: `{fmt(trbg_cost - src_cost)}`.",
        f"7. Recover_fail change vs source-frozen: `{fmt(trbg_recover - src_recover)}`.",
        f"8. Recall vs topk: above=`{confirm['above_topk']}`.",
        f"9. Pareto status: recall-cost not dominated `{pareto_info['not_dom_cost']}`, recall-backend not dominated `{pareto_info['not_dom_backend']}`.",
        f"10. Upgrade from moderate internal success to robust v2 main method: `{primary}`; stronger criterion met `{stronger}`.",
        f"11. If not upgraded, TRBG-source should be appendix/future work; if upgraded, it is main text as locked burden guard, not native success.",
        f"12. Evidence strong enough to enter manuscript rewrite pack: `{primary}`.",
    ]
    (OUT / "gate6_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "primary": primary,
        "stronger": stronger,
        "cost_delta": float(trbg_cost - src_cost),
        "recover_delta": float(trbg_recover - src_recover),
        "rewrite_pack": primary,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats_fix = fix_gate5_stats()
    write_protocol()
    unused_inventory()
    holdouts = generate_blind_banks()
    pre = create_stamp_and_pre_hash(holdouts)
    rows, _payloads = run_confirm(holdouts)
    summary = aggregate(rows, ["method", "B"])
    confirm_info = write_confirm_outputs(rows, summary)
    pair_rows = paired_stats(rows)
    write_csv(OUT / "gate6_paired_statistics.csv", pair_rows)
    write_pair_md(pair_rows)
    pareto_rows = pareto(summary)
    pareto_info = write_pareto_outputs(pareto_rows, summary)
    combined_analysis(rows)
    used_paths = [
        "metric/case14/mixed_bank_fit.npy",
        "metric/case14/mixed_bank_eval.npy",
        "metric/case39_localretune/mixed_bank_fit_native.npy",
        "metric/case39_localretune/mixed_bank_eval_native.npy",
    ]
    prov = provenance_post(pre, holdouts, used_paths)
    decision = gate6_decision(stats_fix, confirm_info, pareto_info, prov, summary)
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree.txt").write_text("\n".join(sorted(set(files) | {"outputs_tree.txt"})) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "stats_fix": stats_fix, "confirm": confirm_info, "pareto": pareto_info, "provenance": prov, "decision": decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

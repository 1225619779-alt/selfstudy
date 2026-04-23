from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent
CAND_DIR = ROOT / "metric" / "case39" / "backend_calibration_valid_grid" / "candidates"
REPORT_MD = ROOT / "reports" / "case39_backend_calibration_consistency.md"
REPORT_JSON = ROOT / "metric" / "case39" / "backend_calibration_consistency.json"
REPORT_CSV = ROOT / "metric" / "case39" / "backend_calibration_consistency.csv"


def parse_candidate_name(name: str) -> Dict[str, Any]:
    stem = Path(name).stem
    parts = stem.split("_")
    return {
        "x_facts_ratio": float(parts[1]),
        "varrho": float(parts[3]),
        "upper_scale": float(parts[5]),
        "multi_run_no": int(parts[7]),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def boolish(v: Any) -> bool:
    return bool(v)


def main() -> None:
    json_files = sorted(Path(p) for p in glob.glob(str(CAND_DIR / "*.json")))
    rows: List[Dict[str, Any]] = []

    for path in json_files:
        payload = load_json(path)
        cand = parse_candidate_name(path.name)
        clean_meta = payload.get("clean_metadata", {})
        attack_meta = payload.get("attack_metadata", {})
        consistency = payload.get("consistency", {})
        clean_main = payload["clean"]["proposed_main"]
        clean_strict = payload["clean"]["proposed_strict"]
        attack_main = payload["attack"]["proposed_main"]
        attack_strict = payload["attack"]["proposed_strict"]

        row = {
            "candidate": path.name,
            "x_facts_ratio": cand["x_facts_ratio"],
            "varrho": cand["varrho"],
            "upper_scale": cand["upper_scale"],
            "multi_run_no": cand["multi_run_no"],
            "clean_alarm_count_main": clean_main["clean_alarm_count"],
            "clean_trigger_count_main": clean_main["clean_trigger_count"],
            "clean_trigger_rate_main": clean_main["clean_trigger_rate"],
            "clean_fail_count_main": clean_main["fail_count"],
            "clean_fail_per_alarm_main": clean_main["fail_per_alarm"],
            "clean_fail_per_trigger_main": clean_main["fail_per_trigger"],
            "clean_alarm_count_strict": clean_strict["clean_alarm_count"],
            "clean_trigger_count_strict": clean_strict["clean_trigger_count"],
            "attack_alarm_count_main": attack_main["attack_alarm_count"],
            "attack_trigger_count_main": attack_main["attack_trigger_count"],
            "attack_overall_arr_main": attack_main["overall_arr"],
            "attack_alarm_count_strict": attack_strict["attack_alarm_count"],
            "attack_trigger_count_strict": attack_strict["attack_trigger_count"],
            "attack_overall_arr_strict": attack_strict["overall_arr"],
            "clean_meta_matches_candidate": boolish(consistency.get("clean_matches_candidate")),
            "attack_meta_matches_candidate": boolish(consistency.get("attack_matches_candidate")),
            "split_matches": boolish(consistency.get("split_matches")),
            "case_matches": boolish(consistency.get("case_matches")),
            "seed_matches": boolish(consistency.get("seed_matches")),
            "clean_git_head": clean_meta.get("git_head", ""),
            "attack_git_head": attack_meta.get("git_head", ""),
            "recover_input_mode": attack_meta.get("recover_input_mode", ""),
            "next_load_mode_clean": clean_meta.get("next_load_mode", ""),
            "next_load_modes_attack": ",".join(attack_meta.get("next_load_modes", [])),
            "clean_input_path": clean_meta.get("input_cache_fingerprint", {}).get("path", ""),
            "attack_input_path": attack_meta.get("input_cache_fingerprint", {}).get("path", ""),
        }
        rows.append(row)

    invariance = {
        "clean_alarm_count_main_unique": sorted({row["clean_alarm_count_main"] for row in rows}),
        "clean_trigger_count_main_unique": sorted({row["clean_trigger_count_main"] for row in rows}),
        "clean_alarm_count_strict_unique": sorted({row["clean_alarm_count_strict"] for row in rows}),
        "clean_trigger_count_strict_unique": sorted({row["clean_trigger_count_strict"] for row in rows}),
        "attack_alarm_count_main_unique": sorted({row["attack_alarm_count_main"] for row in rows}),
        "attack_trigger_count_main_unique": sorted({row["attack_trigger_count_main"] for row in rows}),
        "attack_overall_arr_main_unique": sorted({row["attack_overall_arr_main"] for row in rows}),
        "attack_alarm_count_strict_unique": sorted({row["attack_alarm_count_strict"] for row in rows}),
        "attack_trigger_count_strict_unique": sorted({row["attack_trigger_count_strict"] for row in rows}),
        "attack_overall_arr_strict_unique": sorted({row["attack_overall_arr_strict"] for row in rows}),
    }

    mismatches = [
        row["candidate"]
        for row in rows
        if not (
            row["clean_meta_matches_candidate"]
            and row["attack_meta_matches_candidate"]
            and row["split_matches"]
            and row["case_matches"]
            and row["seed_matches"]
        )
    ]

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "candidate_count": len(rows),
                "rows": rows,
                "invariance": invariance,
                "metadata_mismatch_candidates": mismatches,
            },
            fh,
            ensure_ascii=True,
            indent=2,
        )

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Case39 Backend Calibration Consistency",
        "",
        f"- completed_candidate_count: `{len(rows)}`",
        f"- metadata_mismatch_count: `{len(mismatches)}`",
        "",
        "## Invariance",
    ]
    for key, values in invariance.items():
        lines.append(f"- {key}: `{values}`")
    lines.extend(
        [
            "",
            "## Metadata Mismatch Candidates",
            "- none" if not mismatches else "",
        ]
    )
    if mismatches:
        lines.extend(f"- `{name}`" for name in mismatches)
    lines.extend(
        [
            "",
            "## Best Current Candidate By clean fail/alarm",
        ]
    )
    if rows:
        best = min(rows, key=lambda r: (r["clean_fail_per_alarm_main"], r["clean_fail_per_trigger_main"]))
        lines.extend(
            [
                f"- candidate: `{best['candidate']}`",
                f"- clean main fail/alarm: `{best['clean_fail_per_alarm_main']:.12f}`",
                f"- clean main fail/trigger: `{best['clean_fail_per_trigger_main']:.12f}`",
                f"- clean main trigger_count: `{best['clean_trigger_count_main']}`",
                f"- attack main ARR: `{best['attack_overall_arr_main']:.12f}`",
            ]
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_CSV}")


if __name__ == "__main__":
    main()

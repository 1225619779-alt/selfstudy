from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROTECTED_GROUPS = ["(2,0.3)", "(3,0.2)", "(3,0.3)"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate validation-only backend calibration candidate summaries.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--reference-main-fail-per-alarm", type=float, default=0.9023198168498168)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def safe_num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def min_protected_arr(payload: Dict[str, Any], regime: str) -> float:
    vals = payload["attack"][regime]["protected_group_arr"]
    xs = [safe_num(vals.get(k, float("nan"))) for k in PROTECTED_GROUPS]
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return float("nan")
    return float(min(xs))


def flatten_candidate(path: Path, payload: Dict[str, Any], reference_main_fail: float) -> Dict[str, Any]:
    cand = payload["candidate"]
    clean = payload["clean"]
    attack = payload["attack"]
    main_fail = safe_num(clean["proposed_main"]["fail_per_alarm"])
    rel_drop = (
        float((reference_main_fail - main_fail) / reference_main_fail)
        if np.isfinite(reference_main_fail) and reference_main_fail > 0 and np.isfinite(main_fail)
        else float("nan")
    )
    row = {
        "candidate_file": str(path),
        "x_facts_ratio": safe_num(cand["x_facts_ratio"]),
        "varrho": safe_num(cand["varrho"]),
        "upper_scale": safe_num(cand["upper_scale"]),
        "multi_run_no": int(cand["multi_run_no"]),
        "tau_main_exact": safe_num(payload["tau_main_exact"]),
        "tau_strict_exact": safe_num(payload["tau_strict_exact"]),
        "proposed_main_clean_fail_per_alarm": main_fail,
        "proposed_main_clean_fail_per_trigger": safe_num(clean["proposed_main"]["fail_per_trigger"]),
        "proposed_main_clean_stage_two_time_per_alarm": safe_num(clean["proposed_main"]["stage_two_time_per_alarm"]),
        "proposed_main_clean_delta_cost_two_per_alarm": safe_num(clean["proposed_main"]["delta_cost_two_per_alarm"]),
        "proposed_strict_clean_fail_per_alarm": safe_num(clean["proposed_strict"]["fail_per_alarm"]),
        "proposed_strict_clean_fail_per_trigger": safe_num(clean["proposed_strict"]["fail_per_trigger"]),
        "proposed_strict_clean_stage_two_time_per_alarm": safe_num(clean["proposed_strict"]["stage_two_time_per_alarm"]),
        "proposed_strict_clean_delta_cost_two_per_alarm": safe_num(clean["proposed_strict"]["delta_cost_two_per_alarm"]),
        "detector_main_clean_fail_per_alarm": safe_num(clean["detector_main_budget"]["fail_per_alarm"]),
        "detector_main_clean_stage_two_time_per_alarm": safe_num(clean["detector_main_budget"]["stage_two_time_per_alarm"]),
        "detector_main_clean_delta_cost_two_per_alarm": safe_num(clean["detector_main_budget"]["delta_cost_two_per_alarm"]),
        "proposed_main_attack_overall_arr": safe_num(attack["proposed_main"]["overall_arr"]),
        "proposed_strict_attack_overall_arr": safe_num(attack["proposed_strict"]["overall_arr"]),
        "proposed_main_protected_min_arr": min_protected_arr(payload, "proposed_main"),
        "proposed_strict_protected_min_arr": min_protected_arr(payload, "proposed_strict"),
        "proposed_main_attack_backend_metric_fail_rate": safe_num(attack["proposed_main"]["backend_metric_fail_rate_among_triggers"]),
        "proposed_main_attack_backend_mtd_fail_rate": safe_num(attack["proposed_main"]["backend_mtd_fail_rate_among_triggers"]),
        "proposed_strict_attack_backend_metric_fail_rate": safe_num(attack["proposed_strict"]["backend_metric_fail_rate_among_triggers"]),
        "proposed_strict_attack_backend_mtd_fail_rate": safe_num(attack["proposed_strict"]["backend_mtd_fail_rate_among_triggers"]),
        "detector_main_attack_overall_arr": safe_num(attack["detector_main_budget"]["overall_arr"]),
        "detector_main_attack_backend_metric_fail_rate": safe_num(attack["detector_main_budget"]["backend_metric_fail_rate_among_triggers"]),
        "relative_main_fail_drop_vs_reference": rel_drop,
    }
    return row


def row_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        safe_num(row["proposed_main_clean_fail_per_alarm"]),
        safe_num(row["proposed_main_clean_fail_per_trigger"]),
        safe_num(row["proposed_main_clean_stage_two_time_per_alarm"]),
        safe_num(row["proposed_main_clean_delta_cost_two_per_alarm"]),
        safe_num(row["proposed_strict_clean_fail_per_alarm"]),
    )


def decision_for(best: Dict[str, Any]) -> str:
    main_fail = safe_num(best["proposed_main_clean_fail_per_alarm"])
    rel_drop = safe_num(best["relative_main_fail_drop_vs_reference"])
    if np.isfinite(main_fail) and (main_fail < 0.5 or rel_drop >= 0.40):
        return "use case39 as larger benchmark"
    if np.isfinite(main_fail) and main_fail < 0.70:
        return "use case39 as weak larger benchmark"
    return "use case39 as stress-test / limitation evidence"


def write_md(path: Path, rows: List[Dict[str, Any]], best: Dict[str, Any], top_rows: List[Dict[str, Any]], decision: str) -> None:
    lines: List[str] = []
    lines.append("# Case39 Backend Calibration Validation Grid")
    lines.append("")
    lines.append(f"- candidate_count: `{len(rows)}`")
    lines.append(f"- recommendation: `{decision}`")
    lines.append("")
    lines.append("## Best Candidate")
    lines.append("")
    for k in [
        "x_facts_ratio",
        "varrho",
        "upper_scale",
        "multi_run_no",
        "proposed_main_clean_fail_per_alarm",
        "proposed_main_clean_fail_per_trigger",
        "proposed_main_clean_stage_two_time_per_alarm",
        "proposed_main_clean_delta_cost_two_per_alarm",
        "proposed_strict_clean_fail_per_alarm",
        "proposed_main_attack_overall_arr",
        "proposed_main_protected_min_arr",
        "proposed_main_attack_backend_metric_fail_rate",
        "proposed_main_attack_backend_mtd_fail_rate",
        "relative_main_fail_drop_vs_reference",
    ]:
        lines.append(f"- {k}: `{best[k]}`")
    lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    for i, row in enumerate(top_rows, start=1):
        lines.append(f"### {i}")
        lines.append("")
        lines.append(f"- x_facts_ratio: `{row['x_facts_ratio']}`")
        lines.append(f"- varrho: `{row['varrho']}`")
        lines.append(f"- upper_scale: `{row['upper_scale']}`")
        lines.append(f"- multi_run_no: `{row['multi_run_no']}`")
        lines.append(f"- proposed_main_clean_fail_per_alarm: `{row['proposed_main_clean_fail_per_alarm']}`")
        lines.append(f"- proposed_main_clean_stage_two_time_per_alarm: `{row['proposed_main_clean_stage_two_time_per_alarm']}`")
        lines.append(f"- proposed_main_clean_delta_cost_two_per_alarm: `{row['proposed_main_clean_delta_cost_two_per_alarm']}`")
        lines.append(f"- proposed_main_attack_overall_arr: `{row['proposed_main_attack_overall_arr']}`")
        lines.append(f"- proposed_main_protected_min_arr: `{row['proposed_main_protected_min_arr']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    paths = sorted(in_dir.glob("*.json"))
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(flatten_candidate(path, payload, float(args.reference_main_fail_per_alarm)))
    if not rows:
        raise RuntimeError(f"No candidate json files found in {in_dir}")

    rows.sort(key=row_sort_key)
    best = rows[0]
    top_rows = rows[: max(1, int(args.top_k))]
    decision = decision_for(best)

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    headers = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "candidate_count": len(rows),
        "decision": decision,
        "best_candidate": best,
        "top_candidates": top_rows,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    write_md(out_md, rows, best, top_rows, decision)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved MD: {out_md}")


if __name__ == "__main__":
    main()

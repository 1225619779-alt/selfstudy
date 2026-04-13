import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
OUT = ROOT / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

CASE14_CONFIRM = ROOT / "metric/case14/phase3_confirm_combined_v1_v2/aggregate_summary_merged.json"
CASE14_ABLATION = ROOT / "metric/case14/phase3_oracle_ablation_merged/aggregate_summary.json"
CASE14_EXTERNAL = ROOT / "metric/case14/phase3_external_baseline_bundle/external_baseline_bundle_summary.json"
CASE14_PAPER = ROOT / "metric/case14/phase3_paper_bundle/paper_bundle_summary.json"
CASE14_SIG = ROOT / "metric/case14/phase3_significance_v3/significance_summary.json"
CASE14_FOUNDATION = ROOT / "metric/case14/phase3_foundation_audit/foundation_audit_summary.json"
CASE14_RECOMPUTE = ROOT / "metric/case14/phase3_recompute_guard/recompute_guard_summary.json"
CASE14_REPRO_V1 = ROOT / "metric/case14/phase3_repro_spotcheck_v1/repro_spotcheck_summary.json"
CASE14_REPRO_V2 = ROOT / "metric/case14/phase3_repro_spotcheck_v2/repro_spotcheck_summary.json"
CASE14_IMPORT_MANIFEST = ROOT / "metric/case14/case14_paper_evidence_import_manifest.json"

CASE39_TRANSFER = ROOT / "metric/case39/postrun_audits/20260409_231456/summary.json"
CASE39_SOURCE_FIXED = ROOT / "metric/case39_source_fixed_replay/postrun_bundle/summary.json"
CASE39_SOURCE_ANCHOR = ROOT / "metric/case39_source_anchor/postrun_bundle/summary.json"
CASE39_LOCAL_PROTECTED = ROOT / "metric/case39_localretune_protectedec/postrun_bundle/summary.json"
CASE39_LOCAL_UNCONSTRAINED = ROOT / "metric/case39_localretune/postrun_bundle/summary.json"
CASE39_STAGE_COMPARE = ROOT / "metric/case39_compare/stage_compare_significance_v2.json"
CASE39_MEASURE_AUDIT = ROOT / "metric/case39/preflight/case39_measure_v2_audit_128_144.json"
CASE39_MEASURE_BENCH = ROOT / "gen_data/case39_bench_parallel_v2_256/parallel_measure_report.json"

METRIC_MAP_CASE14 = {
    "recall": "weighted_attack_recall_no_backend_fail",
    "unnecessary": "unnecessary_mtd_count",
    "delay_p95": "queue_delay_p95",
    "cost": "average_service_cost_per_step",
    "served_ratio": "pred_expected_consequence_served_ratio",
}

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten_dict(v, p))
            elif isinstance(v, list):
                if v and all(isinstance(x, dict) for x in v):
                    for i, item in enumerate(v):
                        out.update(flatten_dict(item, f"{p}[{i}]"))
                else:
                    out[p] = json.dumps(v, ensure_ascii=False)
            else:
                out[p] = v
    else:
        out[prefix or "value"] = obj
    return out

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_md_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            vals = []
            for c in columns:
                v = row.get(c, "")
                vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def extract_case14_policy_table(confirm_obj: dict) -> list[dict[str, Any]]:
    label_map = {
        "phase3_oracle_upgrade": "oracle_protected_ec",
        "phase3_proposed": "phase3_proposed",
        "best_threshold": "best_threshold",
        "topk_expected_consequence": "topk_expected_consequence",
    }
    rows = []
    for slot_key, slot_obj in sorted(confirm_obj["slot_budget_aggregates"].items(), key=lambda kv: int(kv[0])):
        slot = int(slot_key)
        policy_stats = slot_obj.get("policy_stats", {})
        for raw_name, display_name in label_map.items():
            if raw_name not in policy_stats:
                continue
            child = policy_stats[raw_name]
            row = {
                "slot_budget": slot,
                "method": display_name,
            }
            for out_name, json_name in METRIC_MAP_CASE14.items():
                row[f"{out_name}_mean"] = safe_get(child, json_name, "mean")
                row[f"{out_name}_std"] = safe_get(child, json_name, "std")
            rows.append(row)
    return rows

def extract_case14_paired_table(confirm_obj: dict) -> list[dict[str, Any]]:
    rows = []
    for slot_key, slot_obj in sorted(confirm_obj["slot_budget_aggregates"].items(), key=lambda kv: int(kv[0])):
        slot = int(slot_key)
        for comp_name, comp_obj in slot_obj.get("paired_stats", {}).items():
            row = {
                "slot_budget": slot,
                "comparison": comp_name,
                "mean_delta_recall": safe_get(comp_obj, "delta_recall", "mean"),
                "mean_delta_unnecessary": safe_get(comp_obj, "delta_unnecessary", "mean"),
                "mean_delta_delay_p95": safe_get(comp_obj, "delta_delay_p95", "mean"),
                "mean_delta_cost": safe_get(comp_obj, "delta_cost_per_step", "mean"),
                "mean_delta_served_ratio": safe_get(comp_obj, "delta_pred_expected_consequence_served_ratio", "mean"),
                "wins_on_recall": comp_obj.get("wins_on_recall"),
                "ties_on_recall": comp_obj.get("ties_on_recall"),
                "lower_unnecessary": comp_obj.get("lower_unnecessary"),
                "lower_delay": comp_obj.get("lower_delay"),
                "lower_cost": comp_obj.get("lower_cost"),
                "higher_served_ratio": comp_obj.get("higher_served_ratio"),
            }
            rows.append(row)
    return rows

def match_significance_to_comparison(sig_obj: dict, paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in paired_rows:
        grouped.setdefault(int(r["slot_budget"]), []).append(r)

    out = []
    for idx, result in enumerate(sig_obj.get("results", [])):
        slot = int(result["slot_budget"])
        recall = safe_get(result, "metrics", "recall", "mean_delta")
        unnecessary = safe_get(result, "metrics", "unnecessary", "mean_delta")
        cost = safe_get(result, "metrics", "cost", "mean_delta")

        matched = None
        for cand in grouped.get(slot, []):
            ok = (
                abs((cand["mean_delta_recall"] or 0) - (recall or 0)) < 1e-12
                and abs((cand["mean_delta_unnecessary"] or 0) - (unnecessary or 0)) < 1e-12
                and abs((cand["mean_delta_cost"] or 0) - (cost or 0)) < 1e-12
            )
            if ok:
                matched = cand["comparison"]
                break

        out.append({
            "result_index": idx,
            "slot_budget": slot,
            "comparison": matched,
            "n_holdouts": result.get("n_holdouts"),
            "recall_mean_delta": recall,
            "recall_ci95_low": safe_get(result, "metrics", "recall", "ci95_low"),
            "recall_ci95_high": safe_get(result, "metrics", "recall", "ci95_high"),
            "unnecessary_mean_delta": unnecessary,
            "unnecessary_ci95_low": safe_get(result, "metrics", "unnecessary", "ci95_low"),
            "unnecessary_ci95_high": safe_get(result, "metrics", "unnecessary", "ci95_high"),
            "cost_mean_delta": cost,
            "cost_ci95_low": safe_get(result, "metrics", "cost", "ci95_low"),
            "cost_ci95_high": safe_get(result, "metrics", "cost", "ci95_high"),
            "delay_mean_delta": safe_get(result, "metrics", "delay", "mean_delta"),
            "delay_ci95_low": safe_get(result, "metrics", "delay", "ci95_low"),
            "delay_ci95_high": safe_get(result, "metrics", "delay", "ci95_high"),
            "served_ratio_mean_delta": safe_get(result, "metrics", "served_ratio", "mean_delta"),
            "served_ratio_ci95_low": safe_get(result, "metrics", "served_ratio", "ci95_low"),
            "served_ratio_ci95_high": safe_get(result, "metrics", "served_ratio", "ci95_high"),
        })
    return out

def dump_list_of_dicts(obj: dict, key: str, out_name: str) -> None:
    val = obj.get(key)
    if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
        rows = [flatten_dict(x) for x in val]
        write_csv(OUT / out_name, rows)

def extract_case14_ablation(ablation_obj: dict) -> list[dict[str, Any]]:
    rows = []
    slot_aggs = ablation_obj.get("slot_budget_aggregates", {})
    methods = ["phase3_reference", "oracle_fused_ec", "oracle_protected_ec", "phase3_oracle_upgrade"]
    for slot_key, slot_obj in sorted(slot_aggs.items(), key=lambda kv: int(kv[0])):
        slot = int(slot_key)
        policy_stats = slot_obj.get("policy_stats", {})
        for method in methods:
            if method not in policy_stats:
                continue
            child = policy_stats[method]
            row = {"slot_budget": slot, "method": method}
            for out_name, json_name in METRIC_MAP_CASE14.items():
                row[f"{out_name}_mean"] = safe_get(child, json_name, "mean")
                row[f"{out_name}_std"] = safe_get(child, json_name, "std")
            rows.append(row)
    return rows

def extract_case39_slot_metrics(summary_obj: dict, slot: int, method: str) -> dict[str, Any]:
    slot_obj = summary_obj.get("merged_8_holdouts", {}).get(str(slot), {})
    child = slot_obj.get(method, {})
    return {
        "recall": child.get("mean_recall"),
        "unnecessary": child.get("mean_unnecessary"),
        "cost": child.get("mean_cost"),
        "served_ratio": child.get("mean_served_ratio"),
    }

def build_case39_stage_rows() -> list[dict[str, Any]]:
    stage_defs = [
        ("transfer_frozen_dev", "main_result", CASE39_TRANSFER),
        ("source_fixed_replay", "mechanism_isolation", CASE39_SOURCE_FIXED),
        ("source_anchor", "repair_attempt", CASE39_SOURCE_ANCHOR),
        ("local_protected", "protocol_internal_negative_control", CASE39_LOCAL_PROTECTED),
        ("local_unconstrained", "stress_test_out_of_protocol", CASE39_LOCAL_UNCONSTRAINED),
    ]

    stage_rows = []
    loaded = {}
    for stage_name, role, path in stage_defs:
        obj = load_json(path)
        loaded[stage_name] = obj
        for slot in (1, 2):
            oracle = extract_case39_slot_metrics(obj, slot, "phase3_oracle_upgrade")
            phase3 = extract_case39_slot_metrics(obj, slot, "phase3_proposed")
            topk = extract_case39_slot_metrics(obj, slot, "topk_expected_consequence")
            stage_rows.append({
                "stage": stage_name,
                "role": role,
                "slot_budget": slot,
                "oracle_recall": oracle["recall"],
                "oracle_unnecessary": oracle["unnecessary"],
                "oracle_cost": oracle["cost"],
                "oracle_served_ratio": oracle["served_ratio"],
                "phase3_recall": phase3["recall"],
                "phase3_unnecessary": phase3["unnecessary"],
                "phase3_cost": phase3["cost"],
                "phase3_served_ratio": phase3["served_ratio"],
                "topk_recall": topk["recall"],
                "topk_unnecessary": topk["unnecessary"],
                "topk_cost": topk["cost"],
                "topk_served_ratio": topk["served_ratio"],
                "stage_meta": obj.get("stage"),
                "label_meta": obj.get("label"),
                "reference_label": obj.get("reference_label"),
                "native_case39_stage": obj.get("native_case39_stage"),
            })

    transfer_lookup = {
        (r["slot_budget"]): r for r in stage_rows if r["stage"] == "transfer_frozen_dev"
    }
    for r in stage_rows:
        base = transfer_lookup[r["slot_budget"]]
        r["delta_vs_transfer_recall"] = None if r["oracle_recall"] is None else base["oracle_recall"] - r["oracle_recall"]
        r["delta_vs_transfer_unnecessary"] = None if r["oracle_unnecessary"] is None else base["oracle_unnecessary"] - r["oracle_unnecessary"]
        r["delta_vs_transfer_cost"] = None if r["oracle_cost"] is None else base["oracle_cost"] - r["oracle_cost"]
        r["delta_vs_transfer_served_ratio"] = None if r["oracle_served_ratio"] is None else base["oracle_served_ratio"] - r["oracle_served_ratio"]

    return stage_rows

def build_case39_measure_support() -> list[dict[str, Any]]:
    audit = load_json(CASE39_MEASURE_AUDIT)
    bench = load_json(CASE39_MEASURE_BENCH)
    return [{
        "success_exact_equal": safe_get(audit, "agreement", "success_exact_equal"),
        "z_max_abs_diff": safe_get(audit, "agreement", "z_max_abs_diff"),
        "v_max_abs_diff": safe_get(audit, "agreement", "v_max_abs_diff"),
        "seq_sec_per_iter_min": safe_get(audit, "sequential_runtime", "sec_per_iter_min"),
        "seq_sec_per_iter_mean": safe_get(audit, "sequential_runtime", "sec_per_iter_mean"),
        "seq_sec_per_iter_max": safe_get(audit, "sequential_runtime", "sec_per_iter_max"),
        "parallel_sec_per_iter_effective": bench.get("sec_per_iter_effective"),
    }]

def build_reliability_rows() -> list[dict[str, Any]]:
    foundation = load_json(CASE14_FOUNDATION)
    recompute = load_json(CASE14_RECOMPUTE)
    repro_v1 = load_json(CASE14_REPRO_V1)
    repro_v2 = load_json(CASE14_REPRO_V2)

    return [
        {
            "check": "foundation_audit",
            "status": foundation.get("status"),
            "detail_1": foundation.get("core_foundation_pass"),
            "detail_2": foundation.get("confirm_holdout_count"),
        },
        {
            "check": "recompute_guard",
            "status": recompute.get("status"),
            "detail_1": recompute.get("n_recomputed_holdouts"),
            "detail_2": safe_get(recompute, "checks", "confirm_merged", "max_abs_error"),
        },
        {
            "check": "repro_spotcheck_v1",
            "status": repro_v1.get("status"),
            "detail_1": repro_v1.get("n_selected_holdouts"),
            "detail_2": repro_v1.get("max_abs_error"),
        },
        {
            "check": "repro_spotcheck_v2",
            "status": repro_v2.get("status"),
            "detail_1": repro_v2.get("n_selected_holdouts"),
            "detail_2": repro_v2.get("max_abs_error"),
        },
    ]

def write_figure_plan() -> None:
    md = """# Figure and Table Plan

## Main-text figures
1. Figure 1: Online decision pipeline
   - Source: diagram drawn from method description
   - Message: detector -> verification gate -> queue/server -> backend MTD -> metrics

2. Figure 2: Case14 main blind-confirm comparison
   - Source: `case14_confirm_main.csv`
   - Methods: oracle_protected_ec, phase3_proposed, best_threshold, topk_expected_consequence
   - Metrics: recall, unnecessary, cost for slot1 and slot2

3. Figure 3: Case14 ablation / external baseline
   - Sources: `case14_ablation.csv`, `case14_external_main_table_rows_full.csv`
   - Message: fused vs protected; external static and aggressive baselines

4. Figure 4: Case39 five-stage ladder
   - Source: `case39_stage_ladder.csv`
   - Stages: transfer_frozen_dev, source_fixed_replay, source_anchor, local_protected, local_unconstrained
   - Metrics: oracle recall, oracle unnecessary, oracle cost

5. Figure 5: Trustworthiness and runtime
   - Sources: `case14_reliability_summary.csv`, `case39_measure_support.csv`
   - Message: PASS/PASS/PASS + exact-match audit + runtime benchmark

## Main-text tables
1. Table 1: Protocol and metrics dictionary
2. Table 2: Case14 main confirm table
3. Table 3: Case14 ablation / external baseline table
4. Table 4: Case39 stage ladder table

## Supplementary tables
- Significance CI table
- Family breakdown tables
- Import / provenance manifest summary
"""
    (OUT / "figure_plan.md").write_text(md, encoding="utf-8")

def main() -> None:
    confirm = load_json(CASE14_CONFIRM)
    ablation = load_json(CASE14_ABLATION)
    external = load_json(CASE14_EXTERNAL)
    paper = load_json(CASE14_PAPER)
    sig = load_json(CASE14_SIG)
    import_manifest = load_json(CASE14_IMPORT_MANIFEST)
    stage_compare = load_json(CASE39_STAGE_COMPARE)

    case14_main = extract_case14_policy_table(confirm)
    case14_paired = extract_case14_paired_table(confirm)
    case14_sig = match_significance_to_comparison(sig, case14_paired)
    case14_ablation = extract_case14_ablation(ablation)
    case39_stage_rows = build_case39_stage_rows()
    case39_measure = build_case39_measure_support()
    reliability = build_reliability_rows()

    write_csv(OUT / "case14_confirm_main.csv", case14_main)
    write_csv(OUT / "case14_confirm_paired_deltas.csv", case14_paired)
    write_csv(OUT / "case14_significance_matched.csv", case14_sig)
    write_csv(OUT / "case14_ablation.csv", case14_ablation)
    write_csv(OUT / "case14_reliability_summary.csv", reliability)
    write_csv(OUT / "case39_stage_ladder.csv", case39_stage_rows)
    write_csv(OUT / "case39_measure_support.csv", case39_measure)

    dump_list_of_dicts(external, "main_table_rows_full", "case14_external_main_table_rows_full.csv")
    dump_list_of_dicts(external, "family_breakdown_rows", "case14_external_family_breakdown_rows.csv")
    dump_list_of_dicts(paper, "main_table_rows", "case14_paper_main_table_rows.csv")
    dump_list_of_dicts(paper, "ablation_table_rows", "case14_paper_ablation_table_rows.csv")

    write_md_table(
        OUT / "case14_confirm_main.md",
        case14_main,
        ["slot_budget", "method", "recall_mean", "unnecessary_mean", "cost_mean", "delay_p95_mean", "served_ratio_mean"],
    )
    write_md_table(
        OUT / "case39_stage_ladder.md",
        case39_stage_rows,
        ["stage", "role", "slot_budget", "oracle_recall", "oracle_unnecessary", "oracle_cost", "delta_vs_transfer_recall"],
    )
    write_md_table(
        OUT / "case14_reliability_summary.md",
        reliability,
        ["check", "status", "detail_1", "detail_2"],
    )

    # Provenance snapshot
    provenance_rows = [{
        "confirm_source_paths_json": json.dumps(confirm.get("source_paths", []), ensure_ascii=False),
        "import_copied_count": len(import_manifest.get("copied", [])),
        "import_conflict_count": len(import_manifest.get("conflicts", [])),
        "import_missing_source_count": len(import_manifest.get("missing_sources", [])),
        "case39_stage_compare_keys": ",".join(sorted(stage_compare.get("cross_stage_merged_deltas", {}).keys())),
    }]
    write_csv(OUT / "provenance_snapshot.csv", provenance_rows)

    write_figure_plan()

    print("wrote files under:", OUT)
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(p.name)

if __name__ == "__main__":
    main()
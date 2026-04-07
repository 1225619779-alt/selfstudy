
#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

CORE_METRICS = [
    "weighted_attack_recall_no_backend_fail",
    "unnecessary_mtd_count",
    "queue_delay_p95",
    "average_service_cost_per_step",
    "pred_expected_consequence_served_ratio",
]

POLICIES = [
    "phase3_oracle_upgrade",
    "phase3_proposed",
    "best_threshold",
    "topk_expected_consequence",
]

PAIRS = {
    "oracle_vs_phase3": ("phase3_oracle_upgrade", "phase3_proposed"),
    "oracle_vs_best_threshold": ("phase3_oracle_upgrade", "best_threshold"),
    "oracle_vs_topk_expected": ("phase3_oracle_upgrade", "topk_expected_consequence"),
}

EXT_KEY_MAP = {
    "oracle_recall_mean": ("phase3_oracle_upgrade", "weighted_attack_recall_no_backend_fail"),
    "oracle_unnecessary_mean": ("phase3_oracle_upgrade", "unnecessary_mtd_count"),
    "oracle_delay_mean": ("phase3_oracle_upgrade", "queue_delay_p95"),
    "oracle_cost_mean": ("phase3_oracle_upgrade", "average_service_cost_per_step"),
    "phase3_recall_mean": ("phase3_proposed", "weighted_attack_recall_no_backend_fail"),
    "phase3_unnecessary_mean": ("phase3_proposed", "unnecessary_mtd_count"),
    "phase3_delay_mean": ("phase3_proposed", "queue_delay_p95"),
    "phase3_cost_mean": ("phase3_proposed", "average_service_cost_per_step"),
    "historical_recall_mean": ("best_threshold", "weighted_attack_recall_no_backend_fail"),
    "historical_unnecessary_mean": ("best_threshold", "unnecessary_mtd_count"),
    "historical_delay_mean": ("best_threshold", "queue_delay_p95"),
    "historical_cost_mean": ("best_threshold", "average_service_cost_per_step"),
    "aggressive_recall_mean": ("topk_expected_consequence", "weighted_attack_recall_no_backend_fail"),
    "aggressive_unnecessary_mean": ("topk_expected_consequence", "unnecessary_mtd_count"),
    "aggressive_delay_mean": ("topk_expected_consequence", "queue_delay_p95"),
    "aggressive_cost_mean": ("topk_expected_consequence", "average_service_cost_per_step"),
}

PAPER_KEY_MAP = {
    "oracle_recall_mean": ("phase3_oracle_upgrade", "weighted_attack_recall_no_backend_fail"),
    "phase3_recall_mean": ("phase3_proposed", "weighted_attack_recall_no_backend_fail"),
    "delta_recall_mean": ("oracle_vs_phase3", "delta_recall"),
    "oracle_unnecessary_mean": ("phase3_oracle_upgrade", "unnecessary_mtd_count"),
    "phase3_unnecessary_mean": ("phase3_proposed", "unnecessary_mtd_count"),
    "delta_unnecessary_mean": ("oracle_vs_phase3", "delta_unnecessary"),
    "oracle_cost_mean": ("phase3_oracle_upgrade", "average_service_cost_per_step"),
    "phase3_cost_mean": ("phase3_proposed", "average_service_cost_per_step"),
    "delta_cost_mean": ("oracle_vs_phase3", "delta_cost_per_step"),
    "oracle_delay_mean": ("phase3_oracle_upgrade", "queue_delay_p95"),
    "phase3_delay_mean": ("phase3_proposed", "queue_delay_p95"),
    "delta_delay_mean": ("oracle_vs_phase3", "delta_delay_p95"),
    "wins_on_recall_vs_phase3": ("oracle_vs_phase3", "wins_on_recall"),
    "ties_on_recall_vs_phase3": ("oracle_vs_phase3", "ties_on_recall"),
    "lower_unnecessary_vs_phase3": ("oracle_vs_phase3", "lower_unnecessary"),
    "wins_vs_best_threshold": ("oracle_vs_best_threshold", "wins_on_recall"),
}

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")

def std(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))

def collect_source_holdouts(source_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    holdouts: Dict[str, Dict[str, Any]] = {}
    for path in source_paths:
        doc = load_json(path)
        for entry in doc["per_holdout_results"]:
            tag = entry["tag"]
            if tag in holdouts:
                raise ValueError(f"Duplicate holdout tag across sources: {tag}")
            holdouts[tag] = entry
    return holdouts

def recompute_from_sources(source_paths: List[str]) -> Dict[str, Any]:
    holdouts = collect_source_holdouts(source_paths)
    slot_out: Dict[str, Any] = {}
    for slot in ("1", "2"):
        policy_stats: Dict[str, Any] = {}
        per_policy_values: Dict[str, Dict[str, List[float]]] = {}
        for policy in POLICIES:
            per_policy_values[policy] = {metric: [] for metric in CORE_METRICS}

        best_threshold_freq: Dict[str, int] = {}
        for tag, entry in holdouts.items():
            slot_res = entry["slot_budget_results"][slot]
            best_name = slot_res.get("best_threshold_name")
            if best_name:
                best_threshold_freq[best_name] = best_threshold_freq.get(best_name, 0) + 1
            for policy in POLICIES:
                if policy not in slot_res:
                    raise KeyError(f"Missing policy {policy} in holdout {tag}, slot {slot}")
                pol = slot_res[policy]
                for metric in CORE_METRICS:
                    per_policy_values[policy][metric].append(float(pol[metric]))

        for policy in POLICIES:
            policy_stats[policy] = {}
            for metric in CORE_METRICS:
                vals = per_policy_values[policy][metric]
                policy_stats[policy][metric] = {
                    "mean": mean(vals),
                    "std": std(vals),
                    "min": min(vals),
                    "max": max(vals),
                }

        paired_stats: Dict[str, Any] = {}
        lower_better_metrics = {
            "delta_unnecessary": "unnecessary_mtd_count",
            "delta_delay_p95": "queue_delay_p95",
            "delta_cost_per_step": "average_service_cost_per_step",
        }
        higher_better_metrics = {
            "delta_recall": "weighted_attack_recall_no_backend_fail",
            "delta_pred_expected_consequence_served_ratio": "pred_expected_consequence_served_ratio",
        }

        for pair_name, (lhs, rhs) in PAIRS.items():
            deltas: Dict[str, List[float]] = {
                "delta_recall": [],
                "delta_unnecessary": [],
                "delta_delay_p95": [],
                "delta_cost_per_step": [],
                "delta_pred_expected_consequence_served_ratio": [],
            }
            wins_on_recall = ties_on_recall = lower_unnecessary = lower_delay = lower_cost = higher_served_ratio = 0

            for tag, entry in holdouts.items():
                slot_res = entry["slot_budget_results"][slot]
                l = slot_res[lhs]
                r = slot_res[rhs]

                dr = float(l["weighted_attack_recall_no_backend_fail"]) - float(r["weighted_attack_recall_no_backend_fail"])
                du = float(l["unnecessary_mtd_count"]) - float(r["unnecessary_mtd_count"])
                dd = float(l["queue_delay_p95"]) - float(r["queue_delay_p95"])
                dc = float(l["average_service_cost_per_step"]) - float(r["average_service_cost_per_step"])
                ds = float(l["pred_expected_consequence_served_ratio"]) - float(r["pred_expected_consequence_served_ratio"])

                deltas["delta_recall"].append(dr)
                deltas["delta_unnecessary"].append(du)
                deltas["delta_delay_p95"].append(dd)
                deltas["delta_cost_per_step"].append(dc)
                deltas["delta_pred_expected_consequence_served_ratio"].append(ds)

                if dr > 0:
                    wins_on_recall += 1
                elif dr == 0:
                    ties_on_recall += 1
                if du < 0:
                    lower_unnecessary += 1
                if dd < 0:
                    lower_delay += 1
                if dc < 0:
                    lower_cost += 1
                if ds > 0:
                    higher_served_ratio += 1

            paired_stats[pair_name] = {
                "delta_recall": {
                    "mean": mean(deltas["delta_recall"]),
                    "std": std(deltas["delta_recall"]),
                    "min": min(deltas["delta_recall"]),
                    "max": max(deltas["delta_recall"]),
                },
                "delta_unnecessary": {
                    "mean": mean(deltas["delta_unnecessary"]),
                    "std": std(deltas["delta_unnecessary"]),
                    "min": min(deltas["delta_unnecessary"]),
                    "max": max(deltas["delta_unnecessary"]),
                },
                "delta_delay_p95": {
                    "mean": mean(deltas["delta_delay_p95"]),
                    "std": std(deltas["delta_delay_p95"]),
                    "min": min(deltas["delta_delay_p95"]),
                    "max": max(deltas["delta_delay_p95"]),
                },
                "delta_cost_per_step": {
                    "mean": mean(deltas["delta_cost_per_step"]),
                    "std": std(deltas["delta_cost_per_step"]),
                    "min": min(deltas["delta_cost_per_step"]),
                    "max": max(deltas["delta_cost_per_step"]),
                },
                "delta_pred_expected_consequence_served_ratio": {
                    "mean": mean(deltas["delta_pred_expected_consequence_served_ratio"]),
                    "std": std(deltas["delta_pred_expected_consequence_served_ratio"]),
                    "min": min(deltas["delta_pred_expected_consequence_served_ratio"]),
                    "max": max(deltas["delta_pred_expected_consequence_served_ratio"]),
                },
                "wins_on_recall": wins_on_recall,
                "ties_on_recall": ties_on_recall,
                "lower_unnecessary": lower_unnecessary,
                "lower_delay": lower_delay,
                "lower_cost": lower_cost,
                "higher_served_ratio": higher_served_ratio,
            }

        slot_out[slot] = {
            "policy_stats": policy_stats,
            "paired_stats": paired_stats,
            "best_threshold_frequency": best_threshold_freq,
        }

    return {
        "n_holdouts": len(holdouts),
        "holdout_tags": sorted(holdouts.keys()),
        "slot_budget_aggregates": slot_out,
    }

def compare_value(a: float, b: float, tol: float = 1e-9) -> float:
    return abs(float(a) - float(b))

def compare_dict_value(record: Dict[str, Any], path: List[str]) -> Any:
    cur = record
    for key in path:
        cur = cur[key]
    return cur

def audit_merged(recomputed: Dict[str, Any], merged: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    max_abs_error = 0.0
    if recomputed["n_holdouts"] != merged["n_holdouts"]:
        issues.append(f"n_holdouts mismatch: recomputed={recomputed['n_holdouts']} merged={merged['n_holdouts']}")
    if sorted(recomputed["holdout_tags"]) != sorted(merged["holdout_tags"]):
        issues.append("holdout_tags mismatch")

    for slot in ("1", "2"):
        rslot = recomputed["slot_budget_aggregates"][slot]
        mslot = merged["slot_budget_aggregates"][slot]

        for policy in POLICIES:
            for metric in CORE_METRICS:
                rv = rslot["policy_stats"][policy][metric]["mean"]
                mv = mslot["policy_stats"][policy][metric]["mean"]
                err = compare_value(rv, mv)
                max_abs_error = max(max_abs_error, err)
                if err > 1e-9:
                    issues.append(f"merged mean mismatch slot={slot} policy={policy} metric={metric}: {rv} vs {mv}")

        for pair in PAIRS.keys():
            for metric in [
                "delta_recall", "delta_unnecessary", "delta_delay_p95",
                "delta_cost_per_step", "delta_pred_expected_consequence_served_ratio"
            ]:
                rv = rslot["paired_stats"][pair][metric]["mean"]
                mv = mslot["paired_stats"][pair][metric]["mean"]
                err = compare_value(rv, mv)
                max_abs_error = max(max_abs_error, err)
                if err > 1e-9:
                    issues.append(f"merged paired mean mismatch slot={slot} pair={pair} metric={metric}: {rv} vs {mv}")
            for count_key in ["wins_on_recall", "ties_on_recall", "lower_unnecessary", "lower_delay", "lower_cost", "higher_served_ratio"]:
                rv = rslot["paired_stats"][pair][count_key]
                mv = mslot["paired_stats"][pair][count_key]
                if rv != mv:
                    issues.append(f"merged paired count mismatch slot={slot} pair={pair} key={count_key}: {rv} vs {mv}")

        if rslot["best_threshold_frequency"] != mslot.get("best_threshold_frequency", {}):
            issues.append(f"best_threshold_frequency mismatch slot={slot}")

    return {
        "status": "PASS" if not issues else "FAIL",
        "max_abs_error": max_abs_error,
        "issues": issues,
    }

def rows_by_slot(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        out[str(row["slot_budget"])] = row
    return out

def audit_external_bundle(recomputed: Dict[str, Any], external: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    max_abs_error = 0.0
    rows = rows_by_slot(external["main_table_rows_full"])
    for slot in ("1", "2"):
        row = rows[slot]
        rslot = recomputed["slot_budget_aggregates"][slot]
        for out_key, mapping in EXT_KEY_MAP.items():
            policy, metric = mapping
            rv = rslot["policy_stats"][policy][metric]["mean"]
            mv = row[out_key]
            err = compare_value(rv, mv)
            max_abs_error = max(max_abs_error, err)
            if err > 1e-9:
                issues.append(f"external bundle mismatch slot={slot} key={out_key}: {rv} vs {mv}")
    return {
        "status": "PASS" if not issues else "FAIL",
        "max_abs_error": max_abs_error,
        "issues": issues,
    }

def audit_paper_bundle(recomputed: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    max_abs_error = 0.0
    rows = rows_by_slot(paper["main_table_rows"])
    for slot in ("1", "2"):
        row = rows[slot]
        rslot = recomputed["slot_budget_aggregates"][slot]
        for out_key, mapping in PAPER_KEY_MAP.items():
            group, metric = mapping
            if group in rslot["policy_stats"]:
                rv = rslot["policy_stats"][group][metric]["mean"]
            else:
                val = rslot["paired_stats"][group][metric]
                rv = val["mean"] if isinstance(val, dict) else val
            mv = row[out_key]
            err = compare_value(rv, mv)
            max_abs_error = max(max_abs_error, err)
            if err > 1e-9:
                issues.append(f"paper bundle mismatch slot={slot} key={out_key}: {rv} vs {mv}")
    return {
        "status": "PASS" if not issues else "FAIL",
        "max_abs_error": max_abs_error,
        "issues": issues,
    }

def audit_ablation_vs_winner(ablation: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    max_abs_error = 0.0
    # Compare oracle_protected_ec rows in ablation against paper main_table oracle stats.
    abl_rows = {}
    for slot, slot_payload in ablation["slot_budget_aggregates"].items():
        abl_rows[str(slot)] = slot_payload["policy_stats"]["oracle_protected_ec"]
    paper_rows = rows_by_slot(paper["ablation_table_rows"])
    for slot in ("1", "2"):
        row = None
        for candidate in paper["ablation_table_rows"]:
            if str(candidate["slot_budget"]) == slot and candidate["policy"] == "oracle_protected_ec":
                row = candidate
                break
        if row is None:
            issues.append(f"missing oracle_protected_ec ablation row for slot {slot}")
            continue
        mapping = {
            "recall_mean": "weighted_attack_recall_no_backend_fail",
            "unnecessary_mean": "unnecessary_mtd_count",
            "delay_mean": "queue_delay_p95",
            "cost_mean": "average_service_cost_per_step",
            "served_ratio_mean": "pred_expected_consequence_served_ratio",
        }
        for out_key, metric in mapping.items():
            rv = abl_rows[slot][metric]["mean"]
            mv = row[out_key]
            err = compare_value(rv, mv)
            max_abs_error = max(max_abs_error, err)
            if err > 1e-9:
                issues.append(f"ablation mismatch slot={slot} key={out_key}: {rv} vs {mv}")
    return {
        "status": "PASS" if not issues else "FAIL",
        "max_abs_error": max_abs_error,
        "issues": issues,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically recompute confirm summaries from source per-holdout JSONs.")
    parser.add_argument("--confirm_inputs", nargs="+", required=True, help="Source confirm aggregate_summary.json files (v1, v2, ...)")
    parser.add_argument("--confirm_merged", required=True, help="Merged confirm summary to audit")
    parser.add_argument("--external_bundle", required=False, help="External baseline bundle summary")
    parser.add_argument("--paper_bundle", required=False, help="Paper bundle summary")
    parser.add_argument("--ablation_merged", required=False, help="Merged ablation summary")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    recomputed = recompute_from_sources(args.confirm_inputs)
    merged = load_json(args.confirm_merged)

    report: Dict[str, Any] = {
        "method": "phase3_recompute_guard",
        "source_confirm": args.confirm_inputs,
        "n_recomputed_holdouts": recomputed["n_holdouts"],
        "holdout_tags": recomputed["holdout_tags"],
        "checks": {},
    }
    report["checks"]["confirm_merged"] = audit_merged(recomputed, merged)

    if args.external_bundle:
        external = load_json(args.external_bundle)
        report["checks"]["external_bundle"] = audit_external_bundle(recomputed, external)

    if args.paper_bundle:
        paper = load_json(args.paper_bundle)
        report["checks"]["paper_bundle"] = audit_paper_bundle(recomputed, paper)
        if args.ablation_merged:
            ablation = load_json(args.ablation_merged)
            report["checks"]["ablation_vs_paper"] = audit_ablation_vs_winner(ablation, paper)

    any_fail = any(v["status"] != "PASS" for v in report["checks"].values())
    report["status"] = "PASS" if not any_fail else "FAIL"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"status": report["status"], "output": str(output_path)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

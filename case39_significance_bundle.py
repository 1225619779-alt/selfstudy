from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

POLICIES = ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence", "best_threshold"]
METRICS = {
    "recall": ("weighted_attack_recall_no_backend_fail", +1),
    "unnecessary": ("unnecessary_mtd_count", -1),
    "cost": ("average_service_cost_per_step", -1),
    "delay_p95": ("queue_delay_p95", -1),
    "served_ratio": ("pred_expected_consequence_served_ratio", +1),
}
COMPARISONS = [
    ("phase3_oracle_upgrade", "phase3_proposed", "oracle_vs_phase3"),
    ("phase3_oracle_upgrade", "topk_expected_consequence", "oracle_vs_topk_expected"),
    ("phase3_oracle_upgrade", "best_threshold", "oracle_vs_best_threshold"),
]


def load_json(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gather_holdouts(v1: Dict[str, object], v2: Dict[str, object]) -> List[Dict[str, object]]:
    return list(v1.get("per_holdout_results", [])) + list(v2.get("per_holdout_results", []))


def merged_summary(holdouts: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"merged_8_holdouts": {}}
    for slot in ["1", "2"]:
        slot_payload: Dict[str, object] = {}
        for pol in POLICIES[:-1]:
            recalls = [float(h["slot_budget_results"][slot][pol]["weighted_attack_recall_no_backend_fail"]) for h in holdouts]
            unnecessary = [float(h["slot_budget_results"][slot][pol]["unnecessary_mtd_count"]) for h in holdouts]
            cost = [float(h["slot_budget_results"][slot][pol]["average_service_cost_per_step"]) for h in holdouts]
            served = [float(h["slot_budget_results"][slot][pol]["pred_expected_consequence_served_ratio"]) for h in holdouts]
            slot_payload[pol] = {
                "mean_recall": float(np.mean(recalls)),
                "mean_unnecessary": float(np.mean(unnecessary)),
                "mean_cost": float(np.mean(cost)),
                "mean_served_ratio": float(np.mean(served)),
            }
        out["merged_8_holdouts"][slot] = slot_payload
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo, med, hi = np.quantile(means, [0.025, 0.5, 0.975])
    return float(lo), float(med), float(hi)


def exact_sign_test(x: np.ndarray) -> Dict[str, object]:
    nz = x[x != 0]
    n = int(nz.size)
    k = int(np.sum(nz > 0))
    if n == 0:
        return {"n_nonzero": 0, "n_positive": 0, "p_two_sided": 1.0}
    # exact two-sided binomial p-value for p=0.5
    from math import comb
    probs = [comb(n, i) / (2 ** n) for i in range(n + 1)]
    p_obs = probs[k]
    p_two = sum(p for p in probs if p <= p_obs + 1e-15)
    return {"n_nonzero": n, "n_positive": k, "p_two_sided": float(min(1.0, p_two))}


def exact_signflip_permutation_mean_test(x: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return {"n": 0, "observed_mean": float("nan"), "p_two_sided": 1.0}
    obs = abs(float(np.mean(x)))
    cnt = 0
    ge = 0
    # exact enumeration over sign flips; n=8 so 256 only
    for bits in itertools.product([-1.0, 1.0], repeat=n):
        cnt += 1
        m = abs(float(np.mean(x * np.asarray(bits))))
        if m >= obs - 1e-15:
            ge += 1
    return {"n": n, "observed_mean": float(np.mean(x)), "p_two_sided": float(ge / cnt)}


def paired_stats(holdouts: List[Dict[str, object]], n_boot: int, seed: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for slot in ["1", "2"]:
        out[slot] = {}
        for a, b, tag in COMPARISONS:
            cmp_payload: Dict[str, object] = {}
            for m_name, (src_key, _) in METRICS.items():
                da = np.asarray([float(h["slot_budget_results"][slot][a][src_key]) for h in holdouts], dtype=float)
                db = np.asarray([float(h["slot_budget_results"][slot][b][src_key]) for h in holdouts], dtype=float)
                delta = da - db
                lo, med, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed + hash((slot, tag, m_name)) % 100000)
                cmp_payload[m_name] = {
                    "delta_mean": float(np.mean(delta)),
                    "delta_std": float(np.std(delta)),
                    "min": float(np.min(delta)),
                    "max": float(np.max(delta)),
                    "bootstrap_ci95": {"lo": lo, "median": med, "hi": hi},
                    "sign_test": exact_sign_test(delta),
                    "signflip_permutation_mean_test": exact_signflip_permutation_mean_test(delta),
                    "raw_deltas": delta.tolist(),
                }
            out[slot][tag] = cmp_payload
    return out


def compare_to_reference(stage_summary: Dict[str, object], ref_summary: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for slot in ["1", "2"]:
        out[slot] = {}
        for pol in ["phase3_oracle_upgrade", "phase3_proposed", "topk_expected_consequence"]:
            cur = stage_summary["merged_8_holdouts"][slot][pol]
            ref = ref_summary["merged_8_holdouts"][slot][pol]
            out[slot][pol] = {
                "delta_mean_recall": float(cur["mean_recall"] - ref["mean_recall"]),
                "delta_mean_unnecessary": float(cur["mean_unnecessary"] - ref["mean_unnecessary"]),
                "delta_mean_cost": float(cur["mean_cost"] - ref["mean_cost"]),
                "delta_mean_served_ratio": float(cur["mean_served_ratio"] - ref["mean_served_ratio"]),
            }
    return out


def to_text(bundle: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"label={bundle['label']}")
    lines.append(f"n_holdouts={bundle['n_holdouts']}")
    if "reference_label" in bundle:
        lines.append(f"reference_label={bundle['reference_label']}")
    lines.append("== merged_8_holdouts ==")
    for slot in ["1", "2"]:
        lines.append(f"-- slot_budget={slot} --")
        for pol, stats in bundle["merged_8_holdouts"][slot].items():
            lines.append(
                f"{pol}: recall={stats['mean_recall']:.6f}, unnecessary={stats['mean_unnecessary']:.3f}, cost={stats['mean_cost']:.6f}, served_ratio={stats['mean_served_ratio']:.6f}"
            )
    lines.append("== paired_significance ==")
    for slot in ["1", "2"]:
        lines.append(f"-- slot_budget={slot} --")
        for tag, metrics in bundle["paired_significance"][slot].items():
            dr = metrics["recall"]
            du = metrics["unnecessary"]
            dc = metrics["cost"]
            lines.append(
                f"{tag}: dRecall={dr['delta_mean']:.6f} CI95=[{dr['bootstrap_ci95']['lo']:.6f},{dr['bootstrap_ci95']['hi']:.6f}] p_perm={dr['signflip_permutation_mean_test']['p_two_sided']:.4f}; "
                f"dUnnecessary={du['delta_mean']:.3f} CI95=[{du['bootstrap_ci95']['lo']:.3f},{du['bootstrap_ci95']['hi']:.3f}] p_perm={du['signflip_permutation_mean_test']['p_two_sided']:.4f}; "
                f"dCost={dc['delta_mean']:.6f} CI95=[{dc['bootstrap_ci95']['lo']:.6f},{dc['bootstrap_ci95']['hi']:.6f}] p_perm={dc['signflip_permutation_mean_test']['p_two_sided']:.4f}"
            )
    if "vs_reference" in bundle:
        lines.append("== delta_vs_reference ==")
        for slot in ["1", "2"]:
            lines.append(f"-- slot_budget={slot} --")
            for pol, stats in bundle["vs_reference"][slot].items():
                lines.append(
                    f"{pol}: dRecall={stats['delta_mean_recall']:.6f}, dUnnecessary={stats['delta_mean_unnecessary']:.3f}, dCost={stats['delta_mean_cost']:.6f}, dServed={stats['delta_mean_served_ratio']:.6f}"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Build paired significance bundle from case39 oracle confirm aggregates.")
    p.add_argument("--v1", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--reference_summary", default=None)
    p.add_argument("--reference_label", default=None)
    p.add_argument("--n_boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260409)
    args = p.parse_args()

    v1 = load_json(args.v1)
    v2 = load_json(args.v2)
    holdouts = gather_holdouts(v1, v2)
    merged = merged_summary(holdouts)
    paired = paired_stats(holdouts, n_boot=args.n_boot, seed=args.seed)

    bundle: Dict[str, object] = {
        "label": args.label,
        "stage": v1.get("method", "unknown"),
        "n_holdouts": len(holdouts),
        **merged,
        "paired_significance": paired,
        "sources": {"v1": str(Path(args.v1).resolve()), "v2": str(Path(args.v2).resolve())},
    }

    if args.reference_summary:
        ref = load_json(args.reference_summary)
        bundle["reference_label"] = args.reference_label or str(args.reference_summary)
        bundle["vs_reference"] = compare_to_reference(merged, ref)
        bundle["reference_summary_path"] = str(Path(args.reference_summary).resolve())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "summary.json"
    txt = out_dir / "summary.txt"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    txt.write_text(to_text(bundle), encoding="utf-8")
    print(json.dumps({"summary_json": str(js.resolve()), "summary_txt": str(txt.resolve())}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

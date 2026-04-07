#!/usr/bin/env python3
"""Recompute paired significance statistics for merged confirm summaries.

This script is intentionally self-contained and only depends on the Python
standard library so it can be dropped into the repo root and run directly.

Expected input format: the merged confirm JSON produced by
merge_phase3_confirm_summaries.py, containing at least:
  - per_holdout_results: list
  - each holdout has slot_budget_results
  - each slot entry has metrics for:
      phase3_oracle_upgrade, phase3_proposed, best_threshold,
      topk_expected_consequence

Outputs a JSON with paired mean deltas, bootstrap 95% CI, and exact binomial
sign tests using non-tied holdouts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

POLICY_KEY_MAP = {
    "phase3_oracle_upgrade": "phase3_oracle_upgrade",
    "phase3_proposed": "phase3_proposed",
    "best_threshold": "best_threshold",
    "topk_expected_consequence": "topk_expected_consequence",
}

METRIC_KEY_MAP = {
    "recall": "weighted_attack_recall_no_backend_fail",
    "unnecessary": "unnecessary_mtd_count",
    "delay": "queue_delay_p95",
    "cost": "average_service_cost_per_step",
    "served_ratio": "pred_expected_consequence_served_ratio",
}

# For sign tests, True means higher is better; False means lower is better.
METRIC_HIGHER_IS_BETTER = {
    "recall": True,
    "unnecessary": False,
    "delay": False,
    "cost": False,
    "served_ratio": True,
}

COMPARES: List[Tuple[str, str]] = [
    ("phase3_oracle_upgrade", "phase3_proposed"),
    ("phase3_oracle_upgrade", "best_threshold"),
    ("phase3_oracle_upgrade", "topk_expected_consequence"),
]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _quantile(sorted_xs: List[float], q: float) -> float:
    if not sorted_xs:
        return math.nan
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def _bootstrap_ci(deltas: List[float], n_boot: int, seed: int) -> Tuple[float, float]:
    if not deltas:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(deltas)
    samples = []
    for _ in range(n_boot):
        resampled = [deltas[rng.randrange(n)] for _ in range(n)]
        samples.append(_mean(resampled))
    samples.sort()
    return _quantile(samples, 0.025), _quantile(samples, 0.975)


def _binom_tail_prob(n: int, k: int) -> float:
    """P[X >= k] for X ~ Binomial(n, 0.5)."""
    if n <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i)
    return total / (2 ** n)


def _sign_test(wins: int, losses: int) -> Tuple[float, float]:
    """Return (two_sided_p, one_sided_p) using exact binomial on non-ties."""
    n = wins + losses
    if n <= 0:
        return 1.0, 1.0
    one_sided = _binom_tail_prob(n, wins)
    # Two-sided exact sign test as 2 * min(P[X>=wins], P[X<=wins]) capped at 1.
    lower_tail = 0.0
    for i in range(0, wins + 1):
        lower_tail += math.comb(n, i)
    lower_tail /= (2 ** n)
    two_sided = min(1.0, 2.0 * min(one_sided, lower_tail))
    return two_sided, one_sided


def _collect_metric_deltas(data: dict, slot_budget: str, lhs: str, rhs: str, metric_name: str) -> Tuple[List[float], int, int, int]:
    metric_key = METRIC_KEY_MAP[metric_name]
    higher_is_better = METRIC_HIGHER_IS_BETTER[metric_name]

    deltas: List[float] = []
    wins = 0
    losses = 0
    ties = 0

    for holdout in data.get("per_holdout_results", []):
        slot_results = holdout.get("slot_budget_results", {}).get(slot_budget)
        if not slot_results:
            continue
        lhs_row = slot_results.get(lhs)
        rhs_row = slot_results.get(rhs)
        if lhs_row is None or rhs_row is None:
            continue
        if metric_key not in lhs_row or metric_key not in rhs_row:
            continue
        lhs_val = lhs_row[metric_key]
        rhs_val = rhs_row[metric_key]
        if lhs_val is None or rhs_val is None:
            continue
        delta = float(lhs_val) - float(rhs_val)
        deltas.append(delta)
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            ties += 1
        else:
            lhs_better = delta > 0 if higher_is_better else delta < 0
            if lhs_better:
                wins += 1
            else:
                losses += 1
    return deltas, wins, losses, ties


def build_summary(data: dict, n_boot: int, seed: int) -> dict:
    results = []
    slot_keys = sorted((data.get("slot_budget_aggregates") or {}).keys(), key=lambda x: int(x))
    if not slot_keys:
        # fall back to scanning per_holdout_results
        seen = set()
        for holdout in data.get("per_holdout_results", []):
            seen.update(holdout.get("slot_budget_results", {}).keys())
        slot_keys = sorted(seen, key=lambda x: int(x))

    for slot_budget in slot_keys:
        for lhs, rhs in COMPARES:
            metrics_summary = {}
            n_holdouts_ref = None
            for metric_name in METRIC_KEY_MAP:
                deltas, wins, losses, ties = _collect_metric_deltas(data, slot_budget, lhs, rhs, metric_name)
                if n_holdouts_ref is None:
                    n_holdouts_ref = len(deltas)
                ci_low, ci_high = _bootstrap_ci(deltas, n_boot=n_boot, seed=seed + int(slot_budget) * 1000 + hash((lhs, rhs, metric_name)) % 997)
                two_sided, one_sided = _sign_test(wins, losses)
                metrics_summary[metric_name] = {
                    "mean_delta": _mean(deltas),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "sign_test_p_two_sided": two_sided,
                    "sign_test_p_one_sided": one_sided,
                }
            results.append({
                "slot_budget": int(slot_budget),
                "lhs": lhs,
                "rhs": rhs,
                "n_holdouts": int(n_holdouts_ref or 0),
                "metrics": metrics_summary,
            })

    return {
        "method": "phase3_significance_stats_v2",
        "source_confirm": "merged confirm summary",
        "n_boot": n_boot,
        "seed": seed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True, help="Path to merged confirm aggregate_summary_merged.json")
    parser.add_argument("--output_dir", required=True, help="Directory to write significance_summary.json")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260407)
    args = parser.parse_args()

    confirm_path = Path(args.confirm)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "significance_summary.json"

    with confirm_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = build_summary(data, n_boot=args.n_boot, seed=args.seed)
    summary["source_confirm"] = str(confirm_path)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output": str(out_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

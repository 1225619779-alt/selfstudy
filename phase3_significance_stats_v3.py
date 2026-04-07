#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

METRIC_KEY_MAP = {
    "recall": "weighted_attack_recall_no_backend_fail",
    "unnecessary": "unnecessary_mtd_count",
    "delay": "queue_delay_p95",
    "cost": "average_service_cost_per_step",
    "served_ratio": "pred_expected_consequence_served_ratio",
}

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
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    return _quantile(samples, 0.025), _quantile(samples, 0.975)


def _binom_tail_prob(n: int, k: int) -> float:
    if n <= 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def _sign_test(wins: int, losses: int) -> Tuple[float, float]:
    n = wins + losses
    if n <= 0:
        return 1.0, 1.0
    one_sided = _binom_tail_prob(n, wins)
    lower_tail = sum(math.comb(n, i) for i in range(0, wins + 1)) / (2 ** n)
    two_sided = min(1.0, 2.0 * min(one_sided, lower_tail))
    return two_sided, one_sided


def _load_per_holdout_results(paths: List[Path]) -> Tuple[List[dict], List[str], List[str]]:
    merged: Dict[str, dict] = {}
    source_paths: List[str] = []
    warnings: List[str] = []
    for path in paths:
        source_paths.append(str(path))
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        per_holdout = data.get("per_holdout_results")
        if not per_holdout:
            warnings.append(f"{path}: missing per_holdout_results, skipped")
            continue
        for row in per_holdout:
            tag = row.get("tag")
            if not tag:
                warnings.append(f"{path}: found holdout without tag, skipped")
                continue
            merged[tag] = row
    return list(merged.values()), source_paths, warnings


def _collect_metric_deltas(per_holdouts: List[dict], slot_budget: str, lhs: str, rhs: str, metric_name: str) -> Tuple[List[float], int, int, int]:
    metric_key = METRIC_KEY_MAP[metric_name]
    higher_is_better = METRIC_HIGHER_IS_BETTER[metric_name]
    deltas: List[float] = []
    wins = losses = ties = 0
    for holdout in per_holdouts:
        slot_results = (holdout.get("slot_budget_results") or {}).get(slot_budget)
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
        lhs_val = float(lhs_val)
        rhs_val = float(rhs_val)
        if math.isnan(lhs_val) or math.isnan(rhs_val):
            continue
        delta = lhs_val - rhs_val
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


def build_summary(per_holdouts: List[dict], source_paths: List[str], warnings: List[str], n_boot: int, seed: int) -> dict:
    slot_keys = sorted({str(k) for holdout in per_holdouts for k in (holdout.get("slot_budget_results") or {}).keys()}, key=lambda x: int(x))
    results = []
    for slot_budget in slot_keys:
        for lhs, rhs in COMPARES:
            metrics_summary = {}
            n_holdouts_ref = None
            for metric_name in METRIC_KEY_MAP:
                deltas, wins, losses, ties = _collect_metric_deltas(per_holdouts, slot_budget, lhs, rhs, metric_name)
                if n_holdouts_ref is None:
                    n_holdouts_ref = len(deltas)
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    n_boot=n_boot,
                    seed=seed + int(slot_budget) * 1000 + sum(ord(c) for c in f"{lhs}|{rhs}|{metric_name}"),
                )
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
            results.append(
                {
                    "slot_budget": int(slot_budget),
                    "lhs": lhs,
                    "rhs": rhs,
                    "n_holdouts": int(n_holdouts_ref or 0),
                    "metrics": metrics_summary,
                }
            )
    return {
        "method": "phase3_significance_stats_v3",
        "source_confirm": source_paths,
        "n_boot": n_boot,
        "seed": seed,
        "total_unique_holdouts": len(per_holdouts),
        "warnings": warnings,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more confirm aggregate_summary.json files WITH per_holdout_results (e.g. v1 and v2 confirm summaries)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260407)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "significance_summary.json"

    per_holdouts, source_paths, warnings = _load_per_holdout_results(input_paths)
    if not per_holdouts:
        raise SystemExit("No per_holdout_results found in any input. Pass v1/v2 confirm aggregate summaries, not only the merged summary.")

    summary = build_summary(per_holdouts, source_paths, warnings, n_boot=args.n_boot, seed=args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(out_path), "total_unique_holdouts": len(per_holdouts)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

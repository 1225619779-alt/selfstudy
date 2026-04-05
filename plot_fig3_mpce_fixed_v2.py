#!/usr/bin/env python3
"""Fixed Fig. 3 generator for the MPCE manuscript.

What this version fixes:
1) clean score files where `score_phys_l2` is a nested dict/list/scalar mix;
2) attack score files that contain metadata keys mixed with per-group scores;
3) accidental plotting of metadata like total_run/seed_base/ddd_threshold;
4) extreme outliers dominating the y-axis.

Usage example:
python plot_fig3_mpce_fixed_v2.py \
  --clean_scores metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_scores metric/case14/metric_attack_alarm_scores_200.npy \
  --tau_main 0.03318 \
  --out_pdf metric/case14/plots_mpce_final/fig3_verify_score_distribution_fixed_v2.pdf \
  --out_png metric/case14/plots_mpce_final/fig3_verify_score_distribution_fixed_v2.png
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

GROUP_ORDER = [(1, 0.2), (1, 0.3), (2, 0.2), (2, 0.3), (3, 0.2), (3, 0.3)]
META_TOKENS = {
    "total_run", "seed", "threshold", "tau", "mean", "median", "count",
    "alarm", "arr", "front_end", "backend", "quantile", "ddd", "shuffle",
    "mag_no", "mag_str", "ang_no", "ang_str", "next_load", "mode",
    "group_key", "recover_fail", "recover_time", "false_alarm", "clean_sample",
    "idx", "start_offset", "schedule", "timeline", "metric", "summary"
}


def _as_mapping(x: Any) -> Optional[Mapping[str, Any]]:
    return x if isinstance(x, Mapping) else None


def _is_meta_key(k: Any) -> bool:
    ks = str(k).lower()
    return any(tok in ks for tok in META_TOKENS)


def _to_numeric_array(x: Any, allow_scalar: bool = True) -> Optional[np.ndarray]:
    """Try to convert x to a 1D finite float array.

    Returns None if conversion is impossible or empty.
    """
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        if not allow_scalar:
            return None
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _flatten_nested_numeric(x: Any, *, skip_meta_keys: bool = True) -> np.ndarray:
    """Recursively collect numeric leaves from nested dict/list/array structures.

    This is robust to historical score files where a score field may be stored as:
    - dict(idx -> scalar)
    - dict(group -> dict(idx -> scalar))
    - list/tuple of scalars or arrays
    - numpy arrays
    """
    chunks = []

    def walk(obj: Any) -> None:
        if obj is None:
            return

        mp = _as_mapping(obj)
        if mp is not None:
            for k, v in mp.items():
                if skip_meta_keys and _is_meta_key(k):
                    continue
                walk(v)
            return

        # Sequence but not string/bytes
        if isinstance(obj, (list, tuple, set)):
            for it in obj:
                walk(it)
            return

        # numpy array cases
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for it in obj.reshape(-1):
                    walk(it)
                return
            arr = _to_numeric_array(obj, allow_scalar=True)
            if arr is not None:
                chunks.append(arr)
            return

        # scalar or scalar-like
        arr = _to_numeric_array(obj, allow_scalar=True)
        if arr is not None:
            chunks.append(arr)

    walk(x)
    if not chunks:
        return np.array([], dtype=float)
    out = np.concatenate(chunks, axis=0)
    out = out[np.isfinite(out)]
    return out


def _parse_group_key(k: Any) -> Optional[Tuple[int, float]]:
    if isinstance(k, tuple) and len(k) == 2:
        try:
            out = (int(k[0]), float(k[1]))
            return out if out in GROUP_ORDER else None
        except Exception:
            return None

    ks = str(k).strip()

    if ks.startswith("(") and ks.endswith(")"):
        try:
            lit = ast.literal_eval(ks)
            if isinstance(lit, tuple) and len(lit) == 2:
                out = (int(lit[0]), float(lit[1]))
                return out if out in GROUP_ORDER else None
        except Exception:
            pass

    # Accept forms like: 1_0.2, group_1_0.2, k1_e0.2
    m = re.search(r"(\d)\D+([0-9]+\.?[0-9]*)", ks)
    if m:
        try:
            out = (int(m.group(1)), float(m.group(2)))
            return out if out in GROUP_ORDER else None
        except Exception:
            return None
    return None


def _extract_clean_scores(obj: Any) -> np.ndarray:
    # Direct array first
    arr = _to_numeric_array(obj, allow_scalar=False)
    if arr is not None:
        return arr

    mp = _as_mapping(obj)
    if mp is None:
        raise ValueError("Cannot parse clean-score file.")

    preferred_keys = [
        "score_phys_l2", "verify_score_clean_false_alarm", "verify_score",
        "scores", "clean_scores", "clean_false_alarm_scores"
    ]
    for key in preferred_keys:
        if key in mp:
            arr = _flatten_nested_numeric(mp[key], skip_meta_keys=True)
            if arr.size:
                return arr

    # Fallback: search non-metadata top-level entries
    for k, v in mp.items():
        if _is_meta_key(k):
            continue
        arr = _flatten_nested_numeric(v, skip_meta_keys=True)
        if arr.size:
            return arr

    raise KeyError(f"Could not find clean-score array. Keys={list(mp.keys())}")


def _extract_attack_group_scores(obj: Any) -> Dict[Tuple[int, float], np.ndarray]:
    mp = _as_mapping(obj)
    if mp is None:
        raise ValueError("Attack-score file is not a mapping.")

    candidate_maps: Sequence[Mapping[str, Any]] = [mp]
    for key in [
        "verify_score_attack_true_alarm", "score_phys_l2", "verify_score",
        "scores", "group_scores", "attack_scores"
    ]:
        sub = mp.get(key)
        if isinstance(sub, Mapping):
            candidate_maps = [sub, *candidate_maps]
            break

    out: Dict[Tuple[int, float], np.ndarray] = {}
    for cmap in candidate_maps:
        tmp: Dict[Tuple[int, float], np.ndarray] = {}
        for k, v in cmap.items():
            g = _parse_group_key(k)
            if g is None:
                continue
            arr = _flatten_nested_numeric(v, skip_meta_keys=True)
            if arr.size:
                tmp[g] = arr
        if tmp:
            out.update(tmp)
            if all(g in out for g in GROUP_ORDER):
                break

    missing = [g for g in GROUP_ORDER if g not in out]
    if missing:
        raise KeyError(
            "Could not recover raw attack-score arrays for groups: " + ", ".join(map(str, missing))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Fig. 3 generator for MPCE paper.")
    parser.add_argument("--clean_scores", default="metric/case14/metric_clean_alarm_scores_full.npy")
    parser.add_argument("--attack_scores", default="metric/case14/metric_attack_alarm_scores_200.npy")
    parser.add_argument("--tau_main", type=float, default=0.03318)
    parser.add_argument("--out_pdf", default="metric/case14/plots_mpce_final/fig3_verify_score_distribution_fixed_v2.pdf")
    parser.add_argument("--out_png", default="metric/case14/plots_mpce_final/fig3_verify_score_distribution_fixed_v2.png")
    parser.add_argument(
        "--ylim_max",
        type=float,
        default=None,
        help="Optional hard upper y-limit. If omitted, use a percentile-based cap.",
    )
    args = parser.parse_args()

    clean_obj = np.load(args.clean_scores, allow_pickle=True)
    if isinstance(clean_obj, np.ndarray) and clean_obj.shape == ():
        clean_obj = clean_obj.item()
    clean_scores = _extract_clean_scores(clean_obj)

    attack_obj = np.load(args.attack_scores, allow_pickle=True)
    if isinstance(attack_obj, np.ndarray) and attack_obj.shape == ():
        attack_obj = attack_obj.item()
    attack_by_group = _extract_attack_group_scores(attack_obj)

    labels = ["clean"] + [f"({k},{e:.1f})" for k, e in GROUP_ORDER]
    data = [clean_scores] + [attack_by_group[g] for g in GROUP_ORDER]

    all_vals = np.concatenate(data)
    q995 = float(np.quantile(all_vals, 0.995))
    upper = args.ylim_max if args.ylim_max is not None else max(q995 * 1.10, args.tau_main * 2.0)

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(args.tau_main, linestyle="--", linewidth=1.5, label=fr"Main OP $\tau$={args.tau_main:.5f}")
    ax.set_ylabel(r"Verification score $\|c^{\mathrm{rec}}_{t,-\mathrm{ref}}\|_2$")
    ax.set_xlabel("")
    ax.set_ylim(0.0, upper)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.out_pdf}")
    print(f"Saved: {args.out_png}")
    print("Group medians:")
    for lab, arr in zip(labels, data):
        print(f"  {lab}: median={float(np.median(arr)):.6f}")
    print(f"Auto y-limit upper bound: {upper:.6f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 3 (verify-score distribution) for the v2 paper worldline."
    )
    parser.add_argument(
        "--clean_scores",
        type=str,
        default="metric/case14/metric_clean_alarm_scores_full.npy",
        help="Path to clean score file.",
    )
    parser.add_argument(
        "--attack_scores",
        type=str,
        default="metric/case14/metric_attack_alarm_scores_200.npy",
        help="Path to attack score file.",
    )
    parser.add_argument("--tau_main", type=float, default=0.033183758162)
    parser.add_argument("--tau_strict", type=float, default=0.036751313717)
    parser.add_argument(
        "--show_strict",
        action="store_true",
        help="Show the sensitivity threshold line as well. Recommended for appendix/sensitivity only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_paper_v2",
        help="Directory to save the new Fig. 3 files.",
    )
    return parser.parse_args()


def _flatten_numeric(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def _label_from_key(key: str) -> str:
    key = str(key)
    m = re.search(r"\((\d+)\s*,\s*([0-9.]+)\)", key)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    m = re.search(r"(\d+)_(0\.[0-9]+)", key)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    return key


def _extract_group_arrays(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = _label_from_key(k)
            arr = _flatten_numeric(v)
            if arr.size > 0:
                out[label] = arr
            nested = _extract_group_arrays(v, prefix=label)
            for nk, nv in nested.items():
                if nv.size > 0 and nk not in out:
                    out[nk] = nv
    elif isinstance(obj, (list, tuple)):
        arr = _flatten_numeric(obj)
        if arr.size > 0 and prefix:
            out[prefix] = arr
    return out


def _load_clean(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype != object:
        return _flatten_numeric(data)
    try:
        obj = data.item()
    except Exception:
        return _flatten_numeric(data)
    if isinstance(obj, dict):
        preferred = [
            "scores",
            "verify_score_clean_false_alarm",
            "clean_scores",
            "metric_clean_alarm_scores_full",
        ]
        for key in preferred:
            if key in obj:
                arr = _flatten_numeric(obj[key])
                if arr.size > 0:
                    return arr
        for _, v in obj.items():
            arr = _flatten_numeric(v)
            if arr.size > 0:
                return arr
    return np.array([], dtype=float)


def _load_attack_groups(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype != object:
        arr = _flatten_numeric(data)
        return {"attack": arr} if arr.size > 0 else {}
    try:
        obj = data.item()
    except Exception:
        arr = _flatten_numeric(data)
        return {"attack": arr} if arr.size > 0 else {}

    if isinstance(obj, dict):
        # First try keys that look like group dictionaries.
        for key in ["scores_by_group", "group_scores", "attack_scores_by_group"]:
            if key in obj and isinstance(obj[key], dict):
                extracted = _extract_group_arrays(obj[key])
                if extracted:
                    return extracted
        extracted = _extract_group_arrays(obj)
        if extracted:
            ordered = {}
            for wanted in ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]:
                if wanted in extracted:
                    ordered[wanted] = extracted[wanted]
            for k, v in extracted.items():
                if k not in ordered:
                    ordered[k] = v
            return ordered
    return {}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = _load_clean(args.clean_scores)
    attack_groups = _load_attack_groups(args.attack_scores)
    if clean.size == 0:
        raise RuntimeError(f"Could not parse clean scores from {args.clean_scores}")
    if not attack_groups:
        raise RuntimeError(f"Could not parse grouped attack scores from {args.attack_scores}")

    labels = ["clean"] + list(attack_groups.keys())
    data = [clean] + [attack_groups[k] for k in attack_groups.keys()]

    plt.figure(figsize=(8.0, 4.8))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(r"Verification score $\|c^{rec}_{t,-ref}\|_2$")
    plt.title("Per-group verify-score distribution (v2)")
    plt.axhline(args.tau_main, linestyle="--", linewidth=1.2, label=f"Main OP τ={args.tau_main:.3f}")
    if args.show_strict:
        plt.axhline(args.tau_strict, linestyle="-.", linewidth=1.2, label=f"Sensitivity τ={args.tau_strict:.3f}")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    stem = "fig3_verify_score_distribution_v2"
    if args.show_strict:
        stem += "_with_sensitivity"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()

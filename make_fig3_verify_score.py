from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

GROUP_RE = re.compile(r"\((\d+)\s*,\s*([0-9.]+)\)")
ALT_GROUP_RE = re.compile(r"(?:^|_)([123])_(0\.[0-9]+)(?:$|_)")
PREFERRED_SCORE_KEYS = [
    "scores",
    "verify_score",
    "verify_scores",
    "verify_score_attack_true_alarm",
    "verify_score_clean_false_alarm",
    "clean_scores",
    "attack_scores",
]
ORDER = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final paper Fig. 3 (verify-score distribution) with robust group parsing."
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
    parser.add_argument("--tau_main", type=float, default=0.03318)
    parser.add_argument("--tau_strict", type=float, default=0.03675)
    parser.add_argument(
        "--show_strict",
        action="store_true",
        help="Show sensitivity threshold line too. Use mainly for appendix/sensitivity figure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_paper_v3",
        help="Directory to save the final Fig. 3 files.",
    )
    return parser.parse_args()


def _flatten_numeric(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _normalize_group_label(key: str) -> str | None:
    key = str(key)
    m = GROUP_RE.search(key)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    m = ALT_GROUP_RE.search(key)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    return None


def _extract_scores_from_value(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        for k in PREFERRED_SCORE_KEYS:
            if k in v:
                arr = _flatten_numeric(v[k])
                if arr.size > 0:
                    return arr
    return _flatten_numeric(v)


def _load_obj(path: str) -> Any:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype != object:
        return data
    try:
        return data.item()
    except Exception:
        return data


def _load_clean(path: str) -> np.ndarray:
    obj = _load_obj(path)
    if isinstance(obj, dict):
        for key in PREFERRED_SCORE_KEYS:
            if key in obj:
                arr = _flatten_numeric(obj[key])
                if arr.size > 0:
                    return arr
        for v in obj.values():
            arr = _flatten_numeric(v)
            if arr.size > 0:
                return arr
    return _flatten_numeric(obj)


def _collect_group_arrays(obj: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = _normalize_group_label(k)
            if label is not None:
                arr = _extract_scores_from_value(v)
                if arr.size > 0:
                    out[label] = arr
                    continue
            # recurse only into nested dict/list structures; ignore scalars/meta arrays under non-group keys
            if isinstance(v, (dict, list, tuple)):
                _collect_group_arrays(v, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                _collect_group_arrays(item, out)


def _load_attack_groups(path: str) -> dict[str, np.ndarray]:
    obj = _load_obj(path)
    out: dict[str, np.ndarray] = {}
    _collect_group_arrays(obj, out)
    ordered: dict[str, np.ndarray] = {}
    for key in ORDER:
        if key in out:
            ordered[key] = out[key]
    # keep any extra recognized groups after the canonical ones
    for k, v in out.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = _load_clean(args.clean_scores)
    attack_groups = _load_attack_groups(args.attack_scores)
    if clean.size == 0:
        raise RuntimeError(f"Could not parse clean scores from {args.clean_scores}")
    if not attack_groups:
        raise RuntimeError(
            f"Could not parse grouped attack scores from {args.attack_scores}. "
            "Expected group-like keys such as '(1,0.2)' or '1_0.2'."
        )

    labels = ["clean"] + list(attack_groups.keys())
    data = [clean] + [attack_groups[k] for k in attack_groups.keys()]

    all_vals = np.concatenate([d for d in data if d.size > 0])
    ymax = float(np.quantile(all_vals, 0.995)) * 1.08
    ymax = max(ymax, args.tau_main * 1.6)

    plt.figure(figsize=(7.4, 3.9))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(r"Verification score $\|c^{rec}_{t,-ref}\|_2$")
    plt.axhline(args.tau_main, linestyle="--", linewidth=1.2, label=f"Main OP $\\tau$={args.tau_main:.5f}")
    if args.show_strict:
        plt.axhline(args.tau_strict, linestyle="-.", linewidth=1.2, label=f"Sensitivity $\\tau$={args.tau_strict:.5f}")
    plt.ylim(bottom=min(0.0, float(np.min(clean)) * 1.05), top=ymax)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    stem = "fig3_verify_score_distribution_v3"
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

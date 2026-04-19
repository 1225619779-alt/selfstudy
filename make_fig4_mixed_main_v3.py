from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from paper_worldline import BASELINE_MIXED_METRIC, MAIN_MIXED_METRIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final paper Fig. 4 (mixed-timeline cumulative burden at main operating point)."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=BASELINE_MIXED_METRIC,
        help="Path to baseline mixed-timeline metric file.",
    )
    parser.add_argument(
        "--gated",
        type=str,
        default=MAIN_MIXED_METRIC,
        help="Path to gated mixed-timeline metric file (main operating point).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_paper_v3",
        help="Directory to save Fig. 4 files.",
    )
    return parser.parse_args()


def _as_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_int(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


def _load_metric(path: str) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_metric(args.baseline)
    gate = _load_metric(args.gated)

    t = _as_int(base["timeline_step"])
    base_time = _as_float(base["cumulative_stage_time"])
    base_cost = _as_float(base["cumulative_delta_cost"])
    gate_time = _as_float(gate["cumulative_stage_time"])
    gate_cost = _as_float(gate["cumulative_delta_cost"])

    plt.figure(figsize=(7.8, 3.6))
    ax = plt.gca()
    ax.plot(t, base_time, label="baseline cumulative time")
    ax.plot(t, gate_time, label="gated cumulative time")
    ax.set_xlabel("Timeline step")
    ax.set_ylabel("Cumulative defense time")
    ax2 = ax.twinx()
    ax2.plot(t, base_cost, linestyle="--", label="baseline cumulative cost")
    ax2.plot(t, gate_cost, linestyle="--", label="gated cumulative cost")
    ax2.set_ylabel("Cumulative delta cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    plt.tight_layout()

    png_path = out_dir / "fig4_mixed_cumulative_main_v3.png"
    pdf_path = out_dir / "fig4_mixed_cumulative_main_v3.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()


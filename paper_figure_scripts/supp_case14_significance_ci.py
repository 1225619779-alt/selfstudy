from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import add_panel_label, apply_ieee_style, find_repo_root, load_csv, output_dir, require_columns, save_figure

COMPARE_ORDER = [
    "oracle_vs_phase3_proposed",
    "oracle_vs_best_threshold",
    "oracle_vs_topk_expected_consequence",
]
COMPARE_DISPLAY = {
    "oracle_vs_phase3_proposed": "Safeguarded vs incumbent",
    "oracle_vs_best_threshold": "Safeguarded vs static",
    "oracle_vs_topk_expected_consequence": "Safeguarded vs top-k",
}
METRICS = [
    ("recall", "Recall delta"),
    ("unnecessary", "Unnecessary delta"),
    ("cost", "Cost delta"),
]


def build_figure(repo_root: Path) -> None:
    apply_ieee_style()
    out_dir = output_dir(repo_root)
    df = load_csv(repo_root, "case14_significance_matched.csv")
    require_columns(df, ["comparison", "slot_budget", "recall_mean_delta", "unnecessary_mean_delta", "cost_mean_delta"], "case14_significance_matched.csv")
    df = df[df["comparison"].isin(COMPARE_ORDER)].copy()
    df["comparison"] = pd.Categorical(df["comparison"], categories=COMPARE_ORDER, ordered=True)
    df = df.sort_values(["comparison", "slot_budget"])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    y = np.arange(len(COMPARE_ORDER))[::-1]
    offsets = {1: 0.10, 2: -0.10}
    markers = {1: "o", 2: "s"}

    for ax, (metric, title) in zip(axes, METRICS):
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        for slot in [1, 2]:
            slot_df = df[df["slot_budget"] == slot].set_index("comparison").reindex(COMPARE_ORDER)
            mean = slot_df[f"{metric}_mean_delta"].to_numpy()
            lo = slot_df[f"{metric}_ci95_low"].to_numpy()
            hi = slot_df[f"{metric}_ci95_high"].to_numpy()
            ax.errorbar(mean, y + offsets[slot], xerr=[mean - lo, hi - mean], fmt=markers[slot], ms=6.5, capsize=3, linewidth=1.2, label=f"budget {slot}")
        ax.set_title(title, pad=6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([COMPARE_DISPLAY[c] for c in COMPARE_ORDER])
            add_panel_label(ax, "(s1)", x=-0.23, y=1.03)
        else:
            ax.set_yticklabels([])

    axes[0].legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.92, bottom=0.16, wspace=0.12)
    save_figure(fig, out_dir, "supp_case14_significance_ci")
    plt.close(fig)


if __name__ == "__main__":
    build_figure(find_repo_root())

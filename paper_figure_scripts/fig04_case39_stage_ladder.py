from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import MARKERS, LINESTYLES, STAGE_ORDER, add_panel_label, apply_ieee_style, find_repo_root, load_csv, output_dir, pretty_stage, require_columns, save_figure

METRICS = [
    ("oracle_recall", r"$R_{atk}$"),
    ("oracle_unnecessary", r"$N_{unnec}$"),
    ("oracle_cost", r"$C_{avg}$"),
]


def build_figure(repo_root: Path) -> None:
    apply_ieee_style()
    out_dir = output_dir(repo_root)
    df = load_csv(repo_root, "case39_stage_ladder.csv")
    require_columns(df, ["stage", "slot_budget", "oracle_recall", "oracle_unnecessary", "oracle_cost"], "case39_stage_ladder.csv")

    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df = df.sort_values(["slot_budget", "stage"])

    fig, axes = plt.subplots(3, 1, figsize=(6.9, 4.0), sharex=True)
    x = np.arange(len(STAGE_ORDER))
    labels = [pretty_stage(s) for s in STAGE_ORDER]

    for ax, (metric, ylabel) in zip(axes, METRICS):
        ax.axvspan(-0.35, 0.35, alpha=0.10, color="gray")
        for slot in [1, 2]:
            slot_df = df[df["slot_budget"] == slot].sort_values("stage")
            y = slot_df[metric].to_numpy()
            ax.plot(x, y, marker=MARKERS[slot], linestyle=LINESTYLES[slot], linewidth=1.6, markersize=6.5, label=f"budget = {slot}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.99, 1.03))
    axes[0].text(0.03, 0.90, "shaded = recommended main result", transform=axes[0].transAxes, ha="left", va="top", fontsize=7.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    add_panel_label(axes[0], "(d)", x=-0.14, y=1.05)

    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.14, hspace=0.08)
    save_figure(fig, out_dir, "fig04_case39_stage_ladder")
    plt.close(fig)


if __name__ == "__main__":
    build_figure(find_repo_root())

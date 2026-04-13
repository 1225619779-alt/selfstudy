from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import add_panel_label, apply_ieee_style, bar_hatch, budget_badge, find_repo_root, load_csv, output_dir, pretty_method, require_columns, save_figure, style_bar

METHOD_ORDER = ["oracle_protected_ec", "phase3_proposed", "best_threshold", "topk_expected_consequence"]
METRICS = [
    ("recall", r"Attack recall $R_{atk}$"),
    ("unnecessary", r"Unnecessary actions $N_{unnec}$"),
    ("cost", r"Service cost $C_{avg}$"),
]


def build_figure(repo_root: Path) -> None:
    apply_ieee_style()
    out_dir = output_dir(repo_root)
    df = load_csv(repo_root, "case14_confirm_main.csv")
    require_columns(df, ["slot_budget", "method", "recall_mean", "unnecessary_mean", "cost_mean"], "case14_confirm_main.csv")
    df = df[df["method"].isin(METHOD_ORDER)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.2), sharex="col")
    x = np.arange(len(METHOD_ORDER))

    for row, slot in enumerate([1, 2]):
        slot_df = df[df["slot_budget"] == slot].copy()
        slot_df["method"] = pd.Categorical(slot_df["method"], categories=METHOD_ORDER, ordered=True)
        slot_df = slot_df.sort_values("method")

        for col, (metric, title) in enumerate(METRICS):
            ax = axes[row, col]
            means = slot_df[f"{metric}_mean"].to_numpy()
            stds = slot_df[f"{metric}_std"].to_numpy() if f"{metric}_std" in slot_df.columns else None
            bars = ax.bar(x, means, yerr=stds, width=0.64, color="white", edgecolor="black", linewidth=1.2)
            for method, bar in zip(METHOD_ORDER, bars):
                style_bar(bar, bar_hatch(method))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title, pad=6)
            if col == 0:
                ax.set_ylabel(r"$R_{atk}$")
                budget_badge(ax, f"Budget {slot}")
            elif col == 1:
                ax.set_ylabel(r"$N_{unnec}$")
            else:
                ax.set_ylabel(r"$C_{avg}$")
            ax.set_xticks(x)
            if row == 1:
                labels = [pretty_method(m) for m in METHOD_ORDER]
                ax.set_xticklabels(labels)
            else:
                ax.set_xticklabels([])
            ymax = float(np.nanmax(means if stds is None else means + stds)) if len(means) else 1.0
            ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
            if row == 0 and col == 0:
                add_panel_label(ax, "(b)", x=-0.17, y=1.07)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.12, wspace=0.27, hspace=0.09)
    save_figure(fig, out_dir, "fig02_case14_main_blind_confirm")
    plt.close(fig)


if __name__ == "__main__":
    build_figure(find_repo_root())

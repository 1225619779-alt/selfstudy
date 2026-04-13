from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

METHOD_DISPLAY = {
    "phase3_proposed": "Incumbent",
    "oracle_protected_ec": "Safeguarded",
    "phase3_oracle_upgrade": "Safeguarded",
    "phase3_reference": "Incumbent",
    "oracle_fused_ec": "Unprotected",
    "best_threshold": "Static",
    "topk_expected_consequence": "Top-k",
}

STAGE_DISPLAY = {
    "transfer_frozen_dev": "Source-frozen\nselection",
    "source_fixed_replay": "Winner\nreplay",
    "source_anchor": "Anchored\nretune",
    "local_protected": "Native\nsafeguarded",
    "local_unconstrained": "Native\nunconstrained",
}

STAGE_ORDER = [
    "transfer_frozen_dev",
    "source_fixed_replay",
    "source_anchor",
    "local_protected",
    "local_unconstrained",
]

BAR_HATCHES = {
    "oracle_protected_ec": "",
    "phase3_oracle_upgrade": "",
    "phase3_proposed": "//",
    "phase3_reference": "",
    "oracle_fused_ec": "//",
    "best_threshold": "xx",
    "topk_expected_consequence": "..",
}

MARKERS = {1: "o", 2: "s"}
LINESTYLES = {1: "-", 2: "--"}


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path.cwd()
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "paper_tables").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root containing 'paper_tables'.")


def data_dir(repo_root: Path) -> Path:
    return repo_root / "paper_tables"


def output_dir(repo_root: Path) -> Path:
    out = repo_root / "paper_figures_v6"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_csv(repo_root: Path, filename: str) -> pd.DataFrame:
    path = data_dir(repo_root) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.23,
            "errorbar.capsize": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"wrote: {pdf_path}")
    print(f"wrote: {png_path}")


def pretty_method(name: str) -> str:
    return METHOD_DISPLAY.get(name, name.replace("_", " "))


def pretty_stage(name: str) -> str:
    return STAGE_DISPLAY.get(name, name.replace("_", " "))


def require_columns(df: pd.DataFrame, columns: Sequence[str], file_label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{file_label} missing required columns: {missing}")


def dedupe_ablation_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].replace({"phase3_oracle_upgrade": "oracle_protected_ec"})
    numeric_cols = [c for c in out.columns if c.endswith("_mean") or c.endswith("_std")]
    if numeric_cols:
        out = out.groupby(["slot_budget", "method"], as_index=False)[numeric_cols].mean(numeric_only=True)
    return out


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def bar_hatch(method_name: str) -> str:
    return BAR_HATCHES.get(method_name, "")


def budget_badge(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.04,
        0.95,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fill": True, "facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
    )


def style_bar(bar, hatch: str) -> None:
    bar.set_fill(False)
    bar.set_linewidth(1.2)
    if hatch:
        bar.set_hatch(hatch)


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out

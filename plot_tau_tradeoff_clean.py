from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

DEFAULT_SUMMARY_CSV = "metric/case14/clean_main_table.csv"

METRICS = [
    ("unnecessary_mtd_deployment_rate", "Unnecessary MTD deployment rate"),
    ("backend_failure_rate_per_false_alarm", "Backend failure rate per false alarm"),
    ("mean_stage_i_defense_time_per_false_alarm", "Mean stage-I defense time per false alarm"),
    ("mean_stage_ii_defense_time_per_false_alarm", "Mean stage-II defense time per false alarm"),
    ("mean_stage_i_incremental_operating_cost_per_false_alarm", "Mean stage-I incremental operating cost per false alarm"),
    ("mean_stage_ii_incremental_operating_cost_per_false_alarm", "Mean stage-II incremental operating cost per false alarm"),
]

REDUCTION_METRICS = [
    ("red_unnecessary_mtd_pct", "Unnecessary MTD deployment rate reduction (%)"),
    ("red_failure_pct", "Backend failure rate reduction (%)"),
    ("red_stage_i_time_pct", "Stage-I defense time reduction (%)"),
    ("red_stage_ii_time_pct", "Stage-II defense time reduction (%)"),
    ("red_stage_i_cost_pct", "Stage-I incremental operating cost reduction (%)"),
    ("red_stage_ii_cost_pct", "Stage-II incremental operating cost reduction (%)"),
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot clean tau trade-off figures from clean_main_table.csv")
    parser.add_argument("--summary_csv", type=str, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output_dir", type=str, default="metric/case14/plots_clean")
    return parser.parse_args()



def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")



def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))



def xlabels(rows: List[Dict[str, str]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        labels.append(f"{row['row_label']}\n(tau={row['tau_label']})")
    return labels



def plot_single_metric(rows: List[Dict[str, str]], key: str, title: str, output_path: Path) -> None:
    xs = list(range(len(rows)))
    ys = [safe_float(row[key]) for row in rows]
    labs = xlabels(rows)

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs, labs)
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    for x, y in zip(xs, ys):
        if not math.isnan(y):
            plt.annotate(f"{y:.6f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_reduction_summary(rows: List[Dict[str, str]], output_path: Path) -> None:
    xs = list(range(len(rows)))
    labs = xlabels(rows)
    plt.figure(figsize=(8, 5))
    for key, title in REDUCTION_METRICS:
        ys = [safe_float(row[key]) for row in rows]
        plt.plot(xs, ys, marker="o", label=title)
    plt.xticks(xs, labs)
    plt.ylabel("Reduction (%) vs baseline")
    plt.title("Clean trade-off summary")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()



def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, title in METRICS:
        filename = key + ".png"
        plot_single_metric(rows, key, title, out_dir / filename)

    plot_reduction_summary(rows, out_dir / "clean_tradeoff_reduction_summary.png")

    print("Saved figures to:")
    print(out_dir)


if __name__ == "__main__":
    main()

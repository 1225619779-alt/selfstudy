from __future__ import annotations

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final paper Fig. 2 (clean trade-off summary)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_paper_v3",
        help="Directory to save Fig. 2 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "label": "Baseline\n($\\tau$=-1.0)",
            "udr": 0.16792479703745905,
            "fail": 0.10517387616624257,
            "t1": 0.38944895071332186,
            "t2": 2.344803928014482,
            "c1": 29.604552968633822,
            "c2": 4.16184252907397,
        },
        {
            "label": "Main OP\n($\\tau$=0.03318)",
            "udr": 0.01167924797037459,
            "fail": 0.021204410517387615,
            "t1": 0.06885962106091578,
            "t2": 0.27059326463073263,
            "c1": 1.5961073868816458,
            "c2": 0.4145948330099888,
        },
        {
            "label": "Sensitivity\n($\\tau$=0.03675)",
            "udr": 0.01025494943740208,
            "fail": 0.017811704834605598,
            "t1": 0.08096210121400686,
            "t2": 0.3987950906599852,
            "c1": 1.342565001717647,
            "c2": 0.29332055708987204,
        },
    ]

    baseline = rows[0]
    reduction_specs = [
        ("udr", "Unnecessary MTD deployment rate reduction (%)"),
        ("fail", "Backend failure rate reduction (%)"),
        ("t1", "Stage-I defense time reduction (%)"),
        ("t2", "Stage-II defense time reduction (%)"),
        ("c1", "Stage-I incremental cost reduction (%)"),
        ("c2", "Stage-II incremental cost reduction (%)"),
    ]

    xs = list(range(len(rows)))
    labels = [r["label"] for r in rows]

    plt.figure(figsize=(7.8, 4.7))
    reductions_for_json: dict[str, list[float]] = {}
    for key, title in reduction_specs:
        ys = []
        for row in rows:
            if row is baseline:
                ys.append(0.0)
            else:
                ys.append(100.0 * (baseline[key] - row[key]) / baseline[key])
        reductions_for_json[key] = ys
        plt.plot(xs, ys, marker="o", label=title)

    plt.xticks(xs, labels)
    plt.ylabel("Change (%) vs baseline")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    png_path = out_dir / "fig2_clean_tradeoff_summary_v3.png"
    pdf_path = out_dir / "fig2_clean_tradeoff_summary_v3.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    summary = {
        "rows": rows,
        "changes_pct": reductions_for_json,
    }
    with open(out_dir / "fig2_clean_tradeoff_summary_v3.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(png_path)
    print(pdf_path)
    print(out_dir / "fig2_clean_tradeoff_summary_v3.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mixed-timeline case study results.")
    parser.add_argument(
        "--metric",
        type=str,
        default="metric/case14/metric_mixed_timeline_tau_0.021.npy",
        help="Path to the mixed timeline metric .npy file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_mixed_timeline",
        help="Directory to save the figures.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def as_int(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


def _sanitize_for_filename(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(";", "_")
        .replace(":", "_")
    )


def main() -> None:
    args = parse_args()
    metric_path = args.metric
    output_dir = args.output_dir
    ensure_dir(output_dir)

    data = np.load(metric_path, allow_pickle=True).item()

    t = as_int(data["timeline_step"])
    scenario_code = as_int(data["scenario_code"])
    scenario_label = list(data["scenario_label"])
    ddd_alarm = as_int(data["ddd_alarm"])
    trigger = as_int(data["trigger_after_gate"])
    verify_score = as_float(data["verify_score"])
    cum_time = as_float(data["cumulative_stage_time"])
    cum_cost = as_float(data["cumulative_delta_cost"])
    tau_verify = float(data["tau_verify"])
    schedule_segments: List[Dict[str, Any]] = list(data.get("schedule_segments", []))
    summary: Dict[str, Any] = dict(data.get("summary", {}))
    scenario_breakdown: Dict[str, Any] = dict(data.get("scenario_breakdown", {}))

    scenario_map = {0: "clean", 1: "weak", 2: "medium", 3: "strong"}

    # Main 4-row timeline figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    # Light background shading for attack segments
    for seg in schedule_segments:
        if seg["kind"] != "clean":
            for ax in axes:
                ax.axvspan(seg["start_step"] - 0.5, seg["end_step"] + 0.5, alpha=0.08)

    # Row 1: scenario code timeline
    axes[0].step(t, scenario_code, where="mid")
    axes[0].set_ylabel("Scenario")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels([scenario_map[i] for i in [0, 1, 2, 3]])
    axes[0].set_title("Mixed clean/attack timeline")

    # Row 2: verify score + threshold
    axes[1].plot(t, verify_score, linewidth=1.0)
    axes[1].axhline(tau_verify, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("verify score")

    # Row 3: DDD alarm / backend trigger
    axes[2].step(t, ddd_alarm, where="mid", label="DDD alarm")
    axes[2].step(t, trigger, where="mid", label="trigger after gate")
    axes[2].set_ylabel("Binary events")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="upper right")

    # Row 4: cumulative time / cumulative cost
    ax4 = axes[3]
    ax4.plot(t, cum_time, label="cumulative defense time")
    ax4.set_ylabel("cum. time")
    ax4.set_xlabel("Timeline step")
    ax4b = ax4.twinx()
    ax4b.plot(t, cum_cost, linestyle="--", label="cumulative extra cost")
    ax4b.set_ylabel("cum. cost")

    # Combined legend for row 4
    handles1, labels1 = ax4.get_legend_handles_labels()
    handles2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    # Add segment labels on the top row
    ymax = np.nanmax(scenario_code) if len(scenario_code) else 0
    for seg in schedule_segments:
        center = 0.5 * (seg["start_step"] + seg["end_step"])
        axes[0].text(center, ymax + 0.25, seg["label"], ha="center", va="bottom", fontsize=8)

    base_name = _sanitize_for_filename(os.path.splitext(os.path.basename(metric_path))[0])
    out_png = os.path.join(output_dir, f"{base_name}_timeline.png")
    out_pdf = os.path.join(output_dir, f"{base_name}_timeline.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Compact burden-only figure for paper appendix / zoomed usage
    fig2, ax = plt.subplots(figsize=(12, 3.6), constrained_layout=True)
    ax.plot(t, cum_time, label="cumulative defense time")
    ax2 = ax.twinx()
    ax2.plot(t, cum_cost, linestyle="--", label="cumulative extra cost")
    ax.set_xlabel("Timeline step")
    ax.set_ylabel("cum. time")
    ax2.set_ylabel("cum. cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    out_png2 = os.path.join(output_dir, f"{base_name}_burden.png")
    out_pdf2 = os.path.join(output_dir, f"{base_name}_burden.pdf")
    fig2.savefig(out_png2, dpi=300, bbox_inches="tight")
    fig2.savefig(out_pdf2, bbox_inches="tight")
    plt.close(fig2)

    # Summary text
    out_txt = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("==== Mixed timeline summary ====\n")
        f.write(f"metric_file: {metric_path}\n")
        f.write(f"tau_verify: {tau_verify}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n---- Scenario breakdown ----\n")
        for key, value in scenario_breakdown.items():
            f.write(f"{key}: {value}\n")
        f.write("\n---- Segment meta ----\n")
        for seg in schedule_segments:
            f.write(f"{seg}\n")

    print("Saved figures to:", output_dir)
    print("Saved summary to:", out_txt)
    print("total_steps =", int(summary.get("total_steps", len(t))))
    print("total_DDD_alarm =", int(summary.get("total_DDD_alarm", np.sum(ddd_alarm))))
    print("total_trigger_after_gate =", int(summary.get("total_trigger_after_gate", np.sum(trigger))))


if __name__ == "__main__":
    main()

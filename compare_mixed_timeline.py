from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired mixed-timeline runs, typically baseline (tau=-1.0) vs gated (tau>0)."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="metric/case14/metric_mixed_timeline_tau_-1.0.npy",
        help="Path to the baseline mixed-timeline metric (.npy).",
    )
    parser.add_argument(
        "--gated",
        type=str,
        default="metric/case14/metric_mixed_timeline_tau_0.021.npy",
        help="Path to the gated mixed-timeline metric (.npy).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric/case14/plots_mixed_timeline_compare",
        help="Directory for comparison figures and summary.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_int(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


def as_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def load_metric(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def check_same_timeline(base: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, bool]:
    checks = {
        "same_schedule_spec": str(base.get("schedule_spec")) == str(gate.get("schedule_spec")),
        "same_total_steps": int(base["summary"]["total_steps"]) == int(gate["summary"]["total_steps"]),
        "same_timeline_step": np.array_equal(as_int(base["timeline_step"]), as_int(gate["timeline_step"])),
        "same_idx_summary": np.array_equal(as_int(base["idx_summary"]), as_int(gate["idx_summary"])),
        "same_scenario_label": np.array_equal(np.asarray(base["scenario_label"], dtype=object), np.asarray(gate["scenario_label"], dtype=object)),
        "same_is_attack_step": np.array_equal(as_int(base["is_attack_step"]), as_int(gate["is_attack_step"])),
        "same_seed_base": int(base.get("seed_base", -999999)) == int(gate.get("seed_base", -888888)),
        "same_start_offset": int(base.get("start_offset", -999999)) == int(gate.get("start_offset", -888888)),
    }
    return checks


def summarize_one(data: Dict[str, Any], mask: np.ndarray, label: str) -> Dict[str, float]:
    ddd_alarm = as_int(data["ddd_alarm"])
    trigger = as_int(data["trigger_after_gate"])
    skip = as_int(data["skip_by_gate"])
    recover_fail = as_int(data["recover_fail"])
    backend_fail = as_int(data["backend_fail"])
    t1 = as_float(data["stage_one_time"])
    t2 = as_float(data["stage_two_time"])
    c1 = as_float(data["delta_cost_one"])
    c2 = as_float(data["delta_cost_two"])

    ddd_cnt = int(np.sum(ddd_alarm[mask]))
    trig_cnt = int(np.sum(trigger[mask]))
    step_cnt = int(np.sum(mask))
    return {
        "steps": float(step_cnt),
        "DDD_alarm": float(ddd_cnt),
        "trigger_after_gate": float(trig_cnt),
        "skip_by_gate": float(np.sum(skip[mask])),
        "recover_fail": float(np.sum(recover_fail[mask])),
        "backend_fail": float(np.sum(backend_fail[mask])),
        "deployment_rate_per_step": float(trig_cnt / step_cnt) if step_cnt else 0.0,
        "deployment_rate_among_alarms": float(trig_cnt / ddd_cnt) if ddd_cnt else 0.0,
        "backend_fail_rate_per_alarm": float(np.sum(backend_fail[mask]) / ddd_cnt) if ddd_cnt else 0.0,
        "mean_stage_one_time_per_alarm": float(np.sum(t1[mask]) / ddd_cnt) if ddd_cnt else 0.0,
        "mean_stage_two_time_per_alarm": float(np.sum(t2[mask]) / ddd_cnt) if ddd_cnt else 0.0,
        "mean_delta_cost_one_per_alarm": float(np.sum(c1[mask]) / ddd_cnt) if ddd_cnt else 0.0,
        "mean_delta_cost_two_per_alarm": float(np.sum(c2[mask]) / ddd_cnt) if ddd_cnt else 0.0,
        "final_cumulative_stage_time": float(np.sum(t1[mask] + t2[mask])),
        "final_cumulative_delta_cost": float(np.sum(c1[mask] + c2[mask])),
    }


def ratio_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return float("nan")
    return (new - old) / old


def write_summary_txt(out_path: str, checks: Dict[str, bool], base_all: Dict[str, float], gate_all: Dict[str, float],
                      base_clean: Dict[str, float], gate_clean: Dict[str, float],
                      base_attack: Dict[str, float], gate_attack: Dict[str, float],
                      baseline_path: str, gated_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("==== Paired mixed-timeline comparison ====\n")
        f.write(f"baseline_file: {baseline_path}\n")
        f.write(f"gated_file: {gated_path}\n")
        f.write("\n==== Sanity checks ====\n")
        for k, v in checks.items():
            f.write(f"{k}: {v}\n")
        f.write("\n==== Whole timeline ====\n")
        _write_pair_block(f, base_all, gate_all)
        f.write("\n==== Clean segments only ====\n")
        _write_pair_block(f, base_clean, gate_clean)
        f.write("\n==== Attack segments only ====\n")
        _write_pair_block(f, base_attack, gate_attack)


def _write_pair_block(f, base: Dict[str, float], gate: Dict[str, float]) -> None:
    keys = [
        "steps",
        "DDD_alarm",
        "trigger_after_gate",
        "skip_by_gate",
        "backend_fail",
        "deployment_rate_per_step",
        "deployment_rate_among_alarms",
        "backend_fail_rate_per_alarm",
        "mean_stage_one_time_per_alarm",
        "mean_stage_two_time_per_alarm",
        "mean_delta_cost_one_per_alarm",
        "mean_delta_cost_two_per_alarm",
        "final_cumulative_stage_time",
        "final_cumulative_delta_cost",
    ]
    for k in keys:
        b = float(base.get(k, float("nan")))
        g = float(gate.get(k, float("nan")))
        rc = ratio_change(g, b)
        if np.isnan(rc):
            f.write(f"{k}: baseline={b:.6f} | gated={g:.6f} | rel_change=nan\n")
        else:
            f.write(f"{k}: baseline={b:.6f} | gated={g:.6f} | rel_change={rc:.2%}\n")


def plot_cumulative_compare(out_dir: str, base: Dict[str, Any], gate: Dict[str, Any], base_name: str, gate_name: str) -> None:
    t = as_int(base["timeline_step"])
    base_cum_time = as_float(base["cumulative_stage_time"])
    gate_cum_time = as_float(gate["cumulative_stage_time"])
    base_cum_cost = as_float(base["cumulative_delta_cost"])
    gate_cum_cost = as_float(gate["cumulative_delta_cost"])

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(t, base_cum_time, label=f"{base_name} cumulative time")
    ax.plot(t, gate_cum_time, label=f"{gate_name} cumulative time")
    ax.set_xlabel("Timeline step")
    ax.set_ylabel("cumulative defense time")
    ax2 = ax.twinx()
    ax2.plot(t, base_cum_cost, linestyle="--", label=f"{base_name} cumulative cost")
    ax2.plot(t, gate_cum_cost, linestyle=":", label=f"{gate_name} cumulative cost")
    ax2.set_ylabel("cumulative extra cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.savefig(os.path.join(out_dir, "mixed_timeline_cumulative_compare.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "mixed_timeline_cumulative_compare.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_binary_compare(out_dir: str, base: Dict[str, Any], gate: Dict[str, Any], base_name: str, gate_name: str) -> None:
    t = as_int(base["timeline_step"])
    scenario_code = as_int(base["scenario_code"])
    ddd_alarm = as_int(base["ddd_alarm"])
    base_trigger = as_int(base["trigger_after_gate"])
    gate_trigger = as_int(gate["trigger_after_gate"])
    schedule_segments = list(base.get("schedule_segments", []))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for seg in schedule_segments:
        if seg["kind"] != "clean":
            for ax in axes:
                ax.axvspan(seg["start_step"] - 0.5, seg["end_step"] + 0.5, alpha=0.08)

    axes[0].step(t, scenario_code, where="mid")
    axes[0].set_ylabel("Scenario")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["clean", "weak", "medium", "strong"])

    axes[1].step(t, ddd_alarm, where="mid", label="DDD alarm")
    axes[1].step(t, base_trigger, where="mid", label=f"{base_name} trigger")
    axes[1].set_ylabel("Baseline")
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc="upper right")

    axes[2].step(t, ddd_alarm, where="mid", label="DDD alarm")
    axes[2].step(t, gate_trigger, where="mid", label=f"{gate_name} trigger")
    axes[2].set_ylabel("Gated")
    axes[2].set_xlabel("Timeline step")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="upper right")

    fig.savefig(os.path.join(out_dir, "mixed_timeline_binary_compare.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "mixed_timeline_binary_compare.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    base = load_metric(args.baseline)
    gate = load_metric(args.gated)

    checks = check_same_timeline(base, gate)
    if not all(checks.values()):
        print("==== Warning: paired timeline checks ==== ")
        for k, v in checks.items():
            print(k, v)
    else:
        print("==== Paired timeline checks all passed ====")

    is_attack = as_int(base["is_attack_step"]).astype(bool)
    is_clean = ~is_attack
    all_mask = np.ones_like(is_attack, dtype=bool)

    base_all = summarize_one(base, all_mask, "all")
    gate_all = summarize_one(gate, all_mask, "all")
    base_clean = summarize_one(base, is_clean, "clean")
    gate_clean = summarize_one(gate, is_clean, "clean")
    base_attack = summarize_one(base, is_attack, "attack")
    gate_attack = summarize_one(gate, is_attack, "attack")

    out_txt = os.path.join(args.output_dir, "mixed_timeline_compare_summary.txt")
    write_summary_txt(out_txt, checks, base_all, gate_all, base_clean, gate_clean, base_attack, gate_attack, args.baseline, args.gated)

    plot_cumulative_compare(args.output_dir, base, gate, "baseline", "gated")
    plot_binary_compare(args.output_dir, base, gate, "baseline", "gated")

    print("Saved figures to:", args.output_dir)
    print("Saved summary to:", out_txt)
    print("\n==== Whole timeline (baseline -> gated) ====")
    print("trigger_after_gate:", int(base_all["trigger_after_gate"]), "->", int(gate_all["trigger_after_gate"]))
    print("final_cumulative_stage_time:", f"{base_all['final_cumulative_stage_time']:.6f}", "->", f"{gate_all['final_cumulative_stage_time']:.6f}")
    print("final_cumulative_delta_cost:", f"{base_all['final_cumulative_delta_cost']:.6f}", "->", f"{gate_all['final_cumulative_delta_cost']:.6f}")
    print("\n==== Clean segments only (baseline -> gated) ====")
    print("trigger_after_gate:", int(base_clean["trigger_after_gate"]), "->", int(gate_clean["trigger_after_gate"]))
    print("deployment_rate_among_alarms:", f"{base_clean['deployment_rate_among_alarms']:.6f}", "->", f"{gate_clean['deployment_rate_among_alarms']:.6f}")
    print("backend_fail_rate_per_alarm:", f"{base_clean['backend_fail_rate_per_alarm']:.6f}", "->", f"{gate_clean['backend_fail_rate_per_alarm']:.6f}")


if __name__ == "__main__":
    main()

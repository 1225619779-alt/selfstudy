from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from configs.config import mtd_config, save_metric, sys_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize attack baseline/main/strict metric from a cached attack run.")
    p.add_argument("--cache", required=True)
    p.add_argument("--tau_verify", type=float, required=True)
    p.add_argument("--next_load_mode", choices=["sample_length", "offset"], default="sample_length")
    p.add_argument("--output", default="")
    return p.parse_args()


def make_output_path(tau_verify: float) -> str:
    return (
        f"metric/{sys_config['case_name']}/"
        f"metric_event_trigger_tau_{tau_verify}_"
        f"mode_{mtd_config['mode']}_"
        f"{round(np.sqrt(mtd_config['varrho_square']), 5)}_{mtd_config['upper_scale']}.npy"
    )


def safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def main() -> None:
    args = parse_args()
    tau_verify = float(args.tau_verify)
    next_load_mode = str(args.next_load_mode)
    output = str(args.output).strip() or make_output_path(tau_verify)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.cache, allow_pickle=True).item()
    metadata = dict(cache["metadata"])
    groups = cache["groups"]

    TP_DDD: Dict[str, List[bool]] = {}
    att_strength: Dict[str, List[float]] = {}
    varrho_summary: Dict[str, List[float]] = {}
    verify_score: Dict[str, List[float]] = {}
    trigger_after_verification: Dict[str, List[bool]] = {}
    skip_by_verification: Dict[str, List[bool]] = {}
    recover_deviation: Dict[str, List[float]] = {}
    pre_deviation: Dict[str, List[float]] = {}
    recovery_ite_no: Dict[str, List[int]] = {}
    recovery_error: Dict[str, List[bool]] = {}
    obj_one: Dict[str, List[float]] = {}
    obj_two: Dict[str, List[float]] = {}
    worst_primal: Dict[str, List[float]] = {}
    worst_dual: Dict[str, List[float]] = {}
    fail: Dict[str, List[int]] = {}
    x_ratio_stage_one: Dict[str, List[np.ndarray]] = {}
    x_ratio_stage_two: Dict[str, List[np.ndarray]] = {}
    mtd_stage_one_eff: Dict[str, List[float]] = {}
    mtd_stage_two_eff: Dict[str, List[float]] = {}
    mtd_stage_one_hidden: Dict[str, List[float]] = {}
    mtd_stage_two_hidden: Dict[str, List[float]] = {}
    cost_no_mtd: Dict[str, List[float]] = {}
    cost_with_mtd_one: Dict[str, List[float]] = {}
    cost_with_mtd_two: Dict[str, List[float]] = {}
    backend_metric_fail: Dict[str, List[bool]] = {}
    group_summary: Dict[str, Dict[str, float]] = {}

    recovery_time_global: List[float] = []
    residual_no_att_global: List[float] = []
    post_mtd_opf_converge_global: List[bool] = []
    mtd_stage_one_time_global: List[float] = []
    mtd_stage_two_time_global: List[float] = []

    for key, data in groups.items():
        variant = data["variants"][next_load_mode]

        TP_DDD[key] = [bool(x) for x in data["front_end_alarm"]]
        att_strength[key] = []
        varrho_summary[key] = []
        verify_score[key] = []
        trigger_after_verification[key] = []
        skip_by_verification[key] = []
        recover_deviation[key] = []
        pre_deviation[key] = []
        recovery_ite_no[key] = []
        recovery_error[key] = []
        obj_one[key] = []
        obj_two[key] = []
        worst_primal[key] = []
        worst_dual[key] = []
        fail[key] = []
        x_ratio_stage_one[key] = []
        x_ratio_stage_two[key] = []
        mtd_stage_one_eff[key] = []
        mtd_stage_two_eff[key] = []
        mtd_stage_one_hidden[key] = []
        mtd_stage_two_hidden[key] = []
        cost_no_mtd[key] = []
        cost_with_mtd_one[key] = []
        cost_with_mtd_two[key] = []
        backend_metric_fail[key] = []

        front_end_alarm_count = 0
        backend_trigger_count = 0

        for i, score in enumerate(data["verify_score"]):
            verify_score[key].append(float(score))
            att_strength[key].append(float(score))
            recover_deviation[key].append(float(data["recover_deviation"][i]))
            pre_deviation[key].append(float(data["pre_deviation"][i]))
            recovery_ite_no[key].append(int(data["recovery_ite_no"][i]))
            recovery_error[key].append(bool(data["recovery_error"][i]))
            recovery_time_global.append(float(data["recovery_time"][i]))

            front_end_alarm_count += 1

            if bool(data["recovery_error"][i]):
                trigger_after_verification[key].append(False)
                skip_by_verification[key].append(False)
                continue

            should_trigger = bool((tau_verify < 0.0) or (np.isfinite(score) and score >= tau_verify))
            trigger_after_verification[key].append(should_trigger)
            skip_by_verification[key].append(not should_trigger)
            if not should_trigger:
                continue

            backend_trigger_count += 1

            varrho_summary[key].append(float(data["varrho"][i]))
            obj_one[key].append(float(data["obj_one"][i]))
            obj_two[key].append(float(data["obj_two"][i]))
            worst_primal[key].append(float(data["worst_primal"][i]))
            worst_dual[key].append(float(data["worst_dual"][i]))
            fail[key].append(int(data["backend_mtd_fail"][i]))
            x_ratio_stage_one[key].append(np.asarray(variant["x_ratio_stage_one"][i], dtype=float))
            x_ratio_stage_two[key].append(np.asarray(variant["x_ratio_stage_two"][i], dtype=float))
            mtd_stage_one_eff[key].append(float(variant["stage_one_eff"][i]))
            mtd_stage_two_eff[key].append(float(variant["stage_two_eff"][i]))
            mtd_stage_one_hidden[key].append(float(variant["stage_one_hidden"][i]))
            mtd_stage_two_hidden[key].append(float(variant["stage_two_hidden"][i]))
            cost_no_mtd[key].append(float(variant["cost_no_mtd"][i]))
            cost_with_mtd_one[key].append(float(variant["cost_with_mtd_one"][i]))
            cost_with_mtd_two[key].append(float(variant["cost_with_mtd_two"][i]))
            backend_metric_fail[key].append(bool(variant["backend_metric_fail"][i]))

            residual_no_att_global.append(float(variant["residual_no_att"][i]))
            post_mtd_opf_converge_global.append(bool(variant["post_mtd_opf_converge"][i]))
            mtd_stage_one_time_global.append(float(data["stage_one_time"][i]))
            mtd_stage_two_time_global.append(float(data["stage_two_time"][i]))

        group_summary[key] = {
            "front_end_alarms": int(front_end_alarm_count),
            "recovery_error_count": int(np.asarray(recovery_error[key], dtype=bool).sum()),
            "backend_triggers": int(backend_trigger_count),
            "arr": safe_rate(backend_trigger_count, front_end_alarm_count),
            "mean_verify_score": float(np.nanmean(np.asarray(verify_score[key], dtype=float))) if verify_score[key] else float("nan"),
            "median_verify_score": float(np.nanmedian(np.asarray(verify_score[key], dtype=float))) if verify_score[key] else float("nan"),
        }

    payload = {
        "metadata": {
            **metadata,
            "tau_verify": tau_verify,
            "next_load_mode": next_load_mode,
        },
        "TP_DDD": TP_DDD,
        "att_strength": att_strength,
        "varrho_summary": varrho_summary,
        "verify_score": verify_score,
        "trigger_after_verification": trigger_after_verification,
        "skip_by_verification": skip_by_verification,
        "recover_deviation": recover_deviation,
        "pre_deviation": pre_deviation,
        "recovery_ite_no": recovery_ite_no,
        "recovery_time": recovery_time_global,
        "recovery_error": recovery_error,
        "obj_one": obj_one,
        "obj_two": obj_two,
        "worst_primal": worst_primal,
        "worst_dual": worst_dual,
        "fail": fail,
        "x_ratio_stage_one": x_ratio_stage_one,
        "x_ratio_stage_two": x_ratio_stage_two,
        "residual_no_att": residual_no_att_global,
        "post_mtd_opf_converge": post_mtd_opf_converge_global,
        "mtd_stage_one_time": mtd_stage_one_time_global,
        "mtd_stage_two_time": mtd_stage_two_time_global,
        "mtd_stage_one_eff": mtd_stage_one_eff,
        "mtd_stage_two_eff": mtd_stage_two_eff,
        "mtd_stage_one_hidden": mtd_stage_one_hidden,
        "mtd_stage_two_hidden": mtd_stage_two_hidden,
        "cost_no_mtd": cost_no_mtd,
        "cost_with_mtd_one": cost_with_mtd_one,
        "cost_with_mtd_two": cost_with_mtd_two,
        "backend_metric_fail": backend_metric_fail,
    }

    save_metric(output, **payload)

    summary_path = str(Path(output).with_suffix(".summary.txt"))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Attack evaluation summary ====\n")
        for k, v in payload["metadata"].items():
            if k == "group_summary":
                continue
            f.write(f"{k} = {v}\n")
        f.write("\n[group_summary]\n")
        for key, vals in group_summary.items():
            f.write(
                f"{key}: front_end_alarms={vals['front_end_alarms']} "
                f"recovery_error_count={vals['recovery_error_count']} "
                f"backend_triggers={vals['backend_triggers']} arr={vals['arr']:.6f} "
                f"mean_verify_score={vals['mean_verify_score']:.6f} "
                f"median_verify_score={vals['median_verify_score']:.6f}\n"
            )

    print(f"Saved attack metric: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

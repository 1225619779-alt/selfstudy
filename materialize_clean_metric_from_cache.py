from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from configs.config import mtd_config, save_metric, sys_config


GROUP_KEY = "(0,0.0)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize clean baseline/main/strict metric from a cached clean alarm run.")
    p.add_argument("--cache", required=True)
    p.add_argument("--tau_verify", type=float, required=True)
    p.add_argument("--output", default="")
    return p.parse_args()


def make_output_path(tau_verify: float) -> str:
    return (
        f"metric/{sys_config['case_name']}/"
        f"metric_event_trigger_clean_tau_{tau_verify}_"
        f"mode_{mtd_config['mode']}_{round(np.sqrt(mtd_config['varrho_square']), 5)}_{mtd_config['upper_scale']}.npy"
    )


def arr_float(values: List[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def main() -> None:
    args = parse_args()
    tau_verify = float(args.tau_verify)
    output = str(args.output).strip() or make_output_path(tau_verify)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.cache, allow_pickle=True).item()
    records = cache["records"]
    total_clean_sample = int(cache["total_clean_sample"])
    total_DDD_alarm = int(cache["total_DDD_alarm"])
    metadata = cache["metadata"]

    verify_score_all = list(records.get("verify_score", []))
    recovery_error_all = list(records.get("recovery_error", []))

    trigger_after_verification: List[bool] = []
    skip_by_verification: List[bool] = []
    verify_score_triggered: List[float] = []
    verify_score_skipped: List[float] = []
    clean_triggered_idx: List[int] = []
    clean_skipped_idx: List[int] = []
    fail_alarm: List[int] = []
    stage_one_time_alarm: List[float] = []
    stage_two_time_alarm: List[float] = []
    delta_cost_one_alarm: List[float] = []
    delta_cost_two_alarm: List[float] = []
    backend_mtd_fail_alarm: List[int] = []
    backend_metric_fail_alarm: List[int] = []
    fail_triggered: List[int] = []

    obj_one_triggered: List[float] = []
    obj_two_triggered: List[float] = []
    worst_primal_triggered: List[float] = []
    worst_dual_triggered: List[float] = []
    x_ratio_stage_one_triggered: List[np.ndarray] = []
    x_ratio_stage_two_triggered: List[np.ndarray] = []
    residual_no_att_triggered: List[float] = []
    post_mtd_opf_converge_triggered: List[bool] = []
    mtd_stage_one_time_triggered: List[float] = []
    mtd_stage_two_time_triggered: List[float] = []
    mtd_stage_one_eff_triggered: List[float] = []
    mtd_stage_two_eff_triggered: List[float] = []
    mtd_stage_one_hidden_triggered: List[float] = []
    mtd_stage_two_hidden_triggered: List[float] = []
    cost_no_triggered: List[float] = []
    cost_with_one_triggered: List[float] = []
    cost_with_two_triggered: List[float] = []
    varrho_triggered: List[float] = []

    total_trigger_after_verification = 0
    total_skip_by_verification = 0

    for i, score in enumerate(verify_score_all):
        idx_val = int(records["alarm_idx"][i])
        recovery_error = bool(recovery_error_all[i])

        if recovery_error:
            trigger_after_verification.append(False)
            skip_by_verification.append(True)
            total_skip_by_verification += 1
            verify_score_skipped.append(float("nan"))
            clean_skipped_idx.append(idx_val)

            fail_alarm.append(1)
            stage_one_time_alarm.append(0.0)
            stage_two_time_alarm.append(0.0)
            delta_cost_one_alarm.append(0.0)
            delta_cost_two_alarm.append(0.0)
            backend_mtd_fail_alarm.append(0)
            backend_metric_fail_alarm.append(0)
            continue

        should_trigger = bool((tau_verify < 0.0) or (np.isfinite(score) and score >= tau_verify))
        trigger_after_verification.append(should_trigger)
        skip_by_verification.append(not should_trigger)

        if not should_trigger:
            total_skip_by_verification += 1
            verify_score_skipped.append(float(score))
            clean_skipped_idx.append(idx_val)
            fail_alarm.append(0)
            stage_one_time_alarm.append(0.0)
            stage_two_time_alarm.append(0.0)
            delta_cost_one_alarm.append(0.0)
            delta_cost_two_alarm.append(0.0)
            backend_mtd_fail_alarm.append(0)
            backend_metric_fail_alarm.append(0)
            continue

        total_trigger_after_verification += 1
        verify_score_triggered.append(float(score))
        clean_triggered_idx.append(idx_val)

        backend_mtd_fail = int(records["backend_mtd_fail"][i])
        backend_metric_fail = int(records["backend_metric_fail"][i])
        stage_one_time = float(records["stage_one_time"][i])
        stage_two_time = float(records["stage_two_time"][i])
        cost_no = float(records["cost_no_mtd"][i])
        cost_one = float(records["cost_with_mtd_one"][i])
        cost_two = float(records["cost_with_mtd_two"][i])

        fail_alarm.append(int(backend_mtd_fail))
        stage_one_time_alarm.append(0.0 if not np.isfinite(stage_one_time) else stage_one_time)
        stage_two_time_alarm.append(0.0 if not np.isfinite(stage_two_time) else stage_two_time)
        delta_cost_one_alarm.append(
            0.0 if not np.isfinite(cost_no) or not np.isfinite(cost_one) else float(cost_one - cost_no)
        )
        delta_cost_two_alarm.append(
            0.0 if not np.isfinite(cost_no) or not np.isfinite(cost_two) else float(cost_two - cost_no)
        )
        backend_mtd_fail_alarm.append(backend_mtd_fail)
        backend_metric_fail_alarm.append(backend_metric_fail)
        fail_triggered.append(backend_mtd_fail)

        obj_one_triggered.append(float(records["obj_one"][i]))
        obj_two_triggered.append(float(records["obj_two"][i]))
        worst_primal_triggered.append(float(records["worst_primal"][i]))
        worst_dual_triggered.append(float(records["worst_dual"][i]))
        x_ratio_stage_one_triggered.append(np.asarray(records["x_ratio_stage_one"][i], dtype=float))
        x_ratio_stage_two_triggered.append(np.asarray(records["x_ratio_stage_two"][i], dtype=float))
        residual_no_att_triggered.append(float(records["stage_two_residual"][i]))
        post_mtd_opf_converge_triggered.append(bool(records["post_mtd_opf_converge"][i]))
        mtd_stage_one_time_triggered.append(stage_one_time)
        mtd_stage_two_time_triggered.append(stage_two_time)
        mtd_stage_one_eff_triggered.append(float(records["stage_one_residual"][i]))
        mtd_stage_two_eff_triggered.append(float(records["stage_two_residual"][i]))
        mtd_stage_one_hidden_triggered.append(float(records["stage_one_residual"][i]))
        mtd_stage_two_hidden_triggered.append(float(records["stage_two_residual"][i]))
        cost_no_triggered.append(cost_no)
        cost_with_one_triggered.append(cost_one)
        cost_with_two_triggered.append(cost_two)
        varrho_triggered.append(float(records["varrho"][i]))

    if total_clean_sample > 0:
        false_alarm_rate = float(total_DDD_alarm / total_clean_sample)
        useless_mtd_rate = float(total_trigger_after_verification / total_clean_sample)
    else:
        false_alarm_rate = float("nan")
        useless_mtd_rate = float("nan")

    if total_DDD_alarm > 0:
        trigger_rate = float(total_trigger_after_verification / total_DDD_alarm)
        skip_rate = float(total_skip_by_verification / total_DDD_alarm)
        fail_per_alarm = float(np.mean(arr_float(fail_alarm)))
        stage_one_time_per_alarm = float(np.mean(arr_float(stage_one_time_alarm)))
        stage_two_time_per_alarm = float(np.mean(arr_float(stage_two_time_alarm)))
        delta_cost_one_per_alarm = float(np.mean(arr_float(delta_cost_one_alarm)))
        delta_cost_two_per_alarm = float(np.mean(arr_float(delta_cost_two_alarm)))
    else:
        trigger_rate = float("nan")
        skip_rate = float("nan")
        fail_per_alarm = float("nan")
        stage_one_time_per_alarm = float("nan")
        stage_two_time_per_alarm = float("nan")
        delta_cost_one_per_alarm = float("nan")
        delta_cost_two_per_alarm = float("nan")

    payload: Dict[str, Any] = {
        "group_key": GROUP_KEY,
        "tau_verify": tau_verify,
        "total_clean_sample": {GROUP_KEY: total_clean_sample},
        "total_DDD_alarm": {GROUP_KEY: total_DDD_alarm},
        "total_trigger_after_verification": {GROUP_KEY: total_trigger_after_verification},
        "total_skip_by_verification": {GROUP_KEY: total_skip_by_verification},
        "false_alarm_rate": {GROUP_KEY: false_alarm_rate},
        "trigger_rate": {GROUP_KEY: trigger_rate},
        "skip_rate": {GROUP_KEY: skip_rate},
        "useless_mtd_rate": {GROUP_KEY: useless_mtd_rate},
        "TP_DDD": {GROUP_KEY: [bool(x) for x in records.get("front_end_alarm", [True] * total_DDD_alarm)]},
        "att_strength": {GROUP_KEY: [float(x) for x in verify_score_all]},
        "varrho_summary": {GROUP_KEY: varrho_triggered},
        "verify_score": {GROUP_KEY: [float(x) for x in verify_score_all]},
        "verify_score_triggered": {GROUP_KEY: verify_score_triggered},
        "verify_score_skipped": {GROUP_KEY: verify_score_skipped},
        "trigger_after_verification": {GROUP_KEY: trigger_after_verification},
        "skip_by_verification": {GROUP_KEY: skip_by_verification},
        "clean_alarm_idx": {GROUP_KEY: [int(x) for x in records["alarm_idx"]]},
        "clean_triggered_idx": {GROUP_KEY: clean_triggered_idx},
        "clean_skipped_idx": {GROUP_KEY: clean_skipped_idx},
        "recover_deviation": {GROUP_KEY: [float(x) for x in records["recover_deviation"]]},
        "pre_deviation": {GROUP_KEY: [float(x) for x in records["pre_deviation"]]},
        "recovery_ite_no": {GROUP_KEY: [int(x) for x in records["recovery_ite_no"]]},
        "recovery_time": [float(x) for x in records["recovery_time"]],
        "recovery_error_alarm": {GROUP_KEY: [int(bool(x)) for x in records["recovery_error"]]},
        "recovery_error_policy": metadata.get("recovery_error_policy", "skip"),
        "obj_one": {GROUP_KEY: obj_one_triggered},
        "obj_two": {GROUP_KEY: obj_two_triggered},
        "worst_primal": {GROUP_KEY: worst_primal_triggered},
        "worst_dual": {GROUP_KEY: worst_dual_triggered},
        "fail": {GROUP_KEY: fail_triggered},
        "x_ratio_stage_one": {GROUP_KEY: x_ratio_stage_one_triggered},
        "x_ratio_stage_two": {GROUP_KEY: x_ratio_stage_two_triggered},
        "residual_no_att": residual_no_att_triggered,
        "post_mtd_opf_converge": post_mtd_opf_converge_triggered,
        "mtd_stage_one_time": mtd_stage_one_time_triggered,
        "mtd_stage_two_time": mtd_stage_two_time_triggered,
        "mtd_stage_one_eff": {GROUP_KEY: mtd_stage_one_eff_triggered},
        "mtd_stage_two_eff": {GROUP_KEY: mtd_stage_two_eff_triggered},
        "mtd_stage_one_hidden": {GROUP_KEY: mtd_stage_one_hidden_triggered},
        "mtd_stage_two_hidden": {GROUP_KEY: mtd_stage_two_hidden_triggered},
        "cost_no_mtd": {GROUP_KEY: cost_no_triggered},
        "cost_with_mtd_one": {GROUP_KEY: cost_with_one_triggered},
        "cost_with_mtd_two": {GROUP_KEY: cost_with_two_triggered},
        "fail_alarm": {GROUP_KEY: fail_alarm},
        "stage_one_time_alarm": {GROUP_KEY: stage_one_time_alarm},
        "stage_two_time_alarm": {GROUP_KEY: stage_two_time_alarm},
        "delta_cost_one_alarm": {GROUP_KEY: delta_cost_one_alarm},
        "delta_cost_two_alarm": {GROUP_KEY: delta_cost_two_alarm},
        "backend_mtd_fail_alarm": {GROUP_KEY: backend_mtd_fail_alarm},
        "backend_metric_fail_alarm": {GROUP_KEY: backend_metric_fail_alarm},
        "fail_per_alarm": {GROUP_KEY: fail_per_alarm},
        "stage_one_time_per_alarm": {GROUP_KEY: stage_one_time_per_alarm},
        "stage_two_time_per_alarm": {GROUP_KEY: stage_two_time_per_alarm},
        "delta_cost_one_per_alarm": {GROUP_KEY: delta_cost_one_per_alarm},
        "delta_cost_two_per_alarm": {GROUP_KEY: delta_cost_two_per_alarm},
        "metadata": {
            "source_cache": args.cache,
            "next_load_mode": metadata.get("next_load_mode", "sample_length"),
            "next_load_extra": metadata.get("next_load_extra", 7),
            "detector_quantile": metadata.get("detector_quantile"),
            "detector_threshold": metadata.get("detector_threshold"),
        },
    }

    save_metric(output, **payload)
    print(f"Saved clean metric: {output}")


if __name__ == "__main__":
    main()

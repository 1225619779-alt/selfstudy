from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import random

from configs.config import mtd_config, save_metric, sys_config
from configs.nn_setting import nn_setting
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_load_pv
from utils.run_metadata import attach_runtime_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute clean backend results from a front-end clean alarm cache.")
    p.add_argument("--input-cache", required=True)
    p.add_argument("--output-cache", required=True)
    p.add_argument("--next_load_mode", choices=["sample_length", "offset"], default="sample_length")
    p.add_argument("--next_load_extra", type=int, default=7)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def next_load_idx_for(split_start: int, idx_val: int, next_load_mode: str, next_load_extra: int) -> int:
    if next_load_mode == "sample_length":
        return int(split_start + nn_setting["sample_length"] + idx_val)
    return int(split_start + next_load_extra + idx_val)


def unpack_multi_run_result(result: Any) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, float, int]:
    if len(result) == 10:
        (
            b_mtd_one_final,
            b_mtd_two_final,
            obj_one,
            obj_two,
            worst_primal,
            worst_dual,
            _c_worst,
            stage_one_time,
            stage_two_time,
            backend_mtd_fail,
        ) = result
        return (
            b_mtd_one_final,
            b_mtd_two_final,
            float(obj_one),
            float(obj_two),
            float(worst_primal),
            float(worst_dual),
            float(stage_one_time),
            float(stage_two_time),
            int(backend_mtd_fail),
        )
    if len(result) == 7:
        (
            b_mtd_one_final,
            b_mtd_two_final,
            obj_one,
            obj_two,
            stage_one_time,
            stage_two_time,
            backend_mtd_fail,
        ) = result
        return (
            b_mtd_one_final,
            b_mtd_two_final,
            float(obj_one),
            float(obj_two),
            float("nan"),
            float("nan"),
            float(stage_one_time),
            float(stage_two_time),
            int(backend_mtd_fail),
        )
    raise ValueError(f"Unexpected multi_run() result length: {len(result)}")


def main() -> None:
    args = parse_args()
    cache = np.load(args.input_cache, allow_pickle=True).item()
    out = deepcopy(cache)
    records: Dict[str, List[Any]] = out["records"]
    metadata = out["metadata"]

    case_class = load_case()
    load_active, load_reactive, pv_active_, _ = load_load_pv()
    split_start = int(metadata["split_start_idx"])
    seed_base = int(metadata.get("seed_base", 20260324))

    n = len(records.get("alarm_idx", []))
    for i in range(n):
        if bool(records["recovery_error"][i]):
            records["varrho"][i] = float("nan")
            records["obj_one"][i] = float("nan")
            records["obj_two"][i] = float("nan")
            records["worst_primal"][i] = float("nan")
            records["worst_dual"][i] = float("nan")
            records["backend_run_error"][i] = False
            records["backend_run_error_message"][i] = ""
            records["backend_mtd_fail"][i] = 0
            records["backend_metric_fail"][i] = 0
            records["stage_one_time"][i] = float("nan")
            records["stage_two_time"][i] = float("nan")
            records["stage_one_residual"][i] = float("nan")
            records["stage_two_residual"][i] = float("nan")
            records["cost_no_mtd"][i] = float("nan")
            records["cost_with_mtd_one"][i] = float("nan")
            records["cost_with_mtd_two"][i] = float("nan")
            records["x_ratio_stage_one"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
            records["x_ratio_stage_two"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
            records["post_mtd_opf_converge"][i] = False
            continue

        idx_val = int(records["alarm_idx"][i])
        seed_ = seed_base + idx_val
        random.seed(seed_)
        np.random.seed(seed_ % (2**32 - 1))
        next_load_idx = next_load_idx_for(
            split_start=split_start,
            idx_val=idx_val,
            next_load_mode=args.next_load_mode,
            next_load_extra=int(args.next_load_extra),
        )
        v_last = np.asarray(records["v_last"][i], dtype=np.complex128)
        c_recover = np.asarray(records["c_recover_no_ref"][i], dtype=float).reshape(-1, 1)

        backend_run_error = False
        backend_run_error_message = ""
        backend_mtd_fail = 1
        backend_metric_fail = True
        stage_one_time = float("nan")
        stage_two_time = float("nan")
        stage_one_residual = float("nan")
        stage_two_residual = float("nan")
        cost_no_mtd = float("nan")
        cost_with_mtd_one = float("nan")
        cost_with_mtd_two = float("nan")
        x_ratio_stage_one = np.full(case_class.no_brh, np.nan, dtype=float)
        x_ratio_stage_two = np.full(case_class.no_brh, np.nan, dtype=float)
        post_mtd_opf_converge = False
        varrho = float("nan")
        obj_one = float("nan")
        obj_two = float("nan")
        worst_primal = float("nan")
        worst_dual = float("nan")

        try:
            mtd_optim_ = mtd_optim(case_class, v_last, c_recover, mtd_config["varrho_square"])
            varrho = float(np.sqrt(mtd_optim_.varrho_square))
            (
                b_mtd_one_final,
                b_mtd_two_final,
                obj_one,
                obj_two,
                worst_primal,
                worst_dual,
                stage_one_time,
                stage_two_time,
                backend_mtd_fail,
            ) = unpack_multi_run_result(mtd_optim_.multi_run())

            try:
                (
                    _ok1,
                    x_ratio_stage_one,
                    stage_one_residual,
                    cost_no_mtd,
                    cost_with_mtd_one,
                ) = mtd_optim_.mtd_metric_no_attack(
                    b_mtd=b_mtd_one_final,
                    load_active=load_active[next_load_idx],
                    load_reactive=load_reactive[next_load_idx],
                    pv_active=pv_active_[next_load_idx],
                )
                backend_metric_fail = False
            except Exception:
                backend_metric_fail = True

            try:
                (
                    post_mtd_opf_converge,
                    x_ratio_stage_two,
                    stage_two_residual,
                    _cost_no_stage_two,
                    cost_with_mtd_two,
                ) = mtd_optim_.mtd_metric_no_attack(
                    b_mtd=b_mtd_two_final,
                    load_active=load_active[next_load_idx],
                    load_reactive=load_reactive[next_load_idx],
                    pv_active=pv_active_[next_load_idx],
                )
                backend_metric_fail = backend_metric_fail or False
            except Exception:
                backend_metric_fail = True
        except Exception as e:
            backend_run_error = True
            backend_run_error_message = repr(e)

        records["varrho"][i] = float(varrho)
        records["obj_one"][i] = float(obj_one)
        records["obj_two"][i] = float(obj_two)
        records["worst_primal"][i] = float(worst_primal)
        records["worst_dual"][i] = float(worst_dual)
        records["backend_run_error"][i] = bool(backend_run_error)
        records["backend_run_error_message"][i] = backend_run_error_message
        records["backend_mtd_fail"][i] = int(backend_mtd_fail)
        records["backend_metric_fail"][i] = int(backend_metric_fail)
        records["stage_one_time"][i] = float(stage_one_time)
        records["stage_two_time"][i] = float(stage_two_time)
        records["stage_one_residual"][i] = float(stage_one_residual)
        records["stage_two_residual"][i] = float(stage_two_residual)
        records["cost_no_mtd"][i] = float(cost_no_mtd)
        records["cost_with_mtd_one"][i] = float(cost_with_mtd_one)
        records["cost_with_mtd_two"][i] = float(cost_with_mtd_two)
        records["x_ratio_stage_one"][i] = np.asarray(x_ratio_stage_one, dtype=float)
        records["x_ratio_stage_two"][i] = np.asarray(x_ratio_stage_two, dtype=float)
        records["post_mtd_opf_converge"][i] = bool(post_mtd_opf_converge)

        if (i + 1) % max(1, int(args.log_every)) == 0:
            print(f"[progress] recomputed clean backend {i + 1}/{n}")

    metadata["next_load_mode"] = str(args.next_load_mode)
    metadata["next_load_extra"] = int(args.next_load_extra)
    metadata["mtd_mode"] = mtd_config["mode"]
    metadata["varrho"] = float(np.sqrt(mtd_config["varrho_square"]))
    metadata["upper_scale"] = float(mtd_config["upper_scale"])
    metadata["multi_run_no"] = int(mtd_config["multi_run_no"])
    metadata["x_facts_ratio"] = float(mtd_config["x_facts_ratio"])
    metadata["skip_backend"] = False
    metadata["recover_input_mode"] = metadata.get("recover_input_mode", "")
    attach_runtime_metadata(
        metadata,
        repo_root=str(Path(__file__).resolve().parent),
        input_cache_path=args.input_cache,
        output_cache_path=args.output_cache,
        runner_name=Path(__file__).name,
    )

    Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
    save_metric(args.output_cache, **out)
    print(f"Saved recomputed clean cache: {args.output_cache}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import random

from configs.config import mtd_config, save_metric
from configs.nn_setting import nn_setting
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_load_pv
from utils.run_metadata import attach_runtime_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute attack backend results from a front-end attack alarm cache.")
    p.add_argument("--input-cache", required=True)
    p.add_argument("--output-cache", required=True)
    p.add_argument("--next_load_modes", nargs="*", default=["sample_length", "offset"], choices=["sample_length", "offset"])
    p.add_argument("--next_load_extra", type=int, default=7)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def next_load_idx_for(split_start: int, idx_val: int, mode: str, next_load_extra: int) -> int:
    if mode == "sample_length":
        return int(split_start + nn_setting["sample_length"] + idx_val)
    return int(split_start + next_load_extra + idx_val)


def ensure_variant_slots(variant: Dict[str, List[Any]], n: int, case_class: Any) -> None:
    defaults: Dict[str, Any] = {
        "backend_metric_fail": False,
        "backend_metric_error_message": "",
        "x_ratio_stage_one": lambda: np.full(case_class.no_brh, np.nan, dtype=float),
        "x_ratio_stage_two": lambda: np.full(case_class.no_brh, np.nan, dtype=float),
        "cost_no_mtd": float("nan"),
        "cost_with_mtd_one": float("nan"),
        "cost_with_mtd_two": float("nan"),
        "stage_one_hidden": float("nan"),
        "stage_one_eff": float("nan"),
        "stage_two_hidden": float("nan"),
        "stage_two_eff": float("nan"),
        "residual_no_att": float("nan"),
        "post_mtd_opf_converge": False,
    }
    for key, default in defaults.items():
        if key not in variant:
            variant[key] = []
        while len(variant[key]) < n:
            variant[key].append(default() if callable(default) else default)


def full_attack_vector_from_group(group: Dict[str, Any], i: int) -> np.ndarray:
    if "c_true" in group and i < len(group["c_true"]):
        c_true = np.asarray(group["c_true"][i], dtype=float).reshape(-1)
        if np.isfinite(c_true).all():
            return c_true
    v_att_last = np.asarray(group["v_att_last"][i], dtype=np.complex128)
    v_last_clean = np.asarray(group["v_last_clean"][i], dtype=np.complex128)
    return (np.angle(v_att_last) - np.angle(v_last_clean)).reshape(-1)


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
    metadata = out["metadata"]
    groups: Dict[str, Dict[str, Any]] = out["groups"]
    seed_base = int(metadata.get("seed_base", 20260324))

    case_class = load_case()
    load_active, load_reactive, pv_active_, _ = load_load_pv()
    split_start = int(metadata["split_start_idx"])
    next_load_modes = list(dict.fromkeys(args.next_load_modes))

    for key, group in groups.items():
        n = len(group.get("alarm_idx", []))
        for mode in next_load_modes:
            if mode not in group["variants"]:
                group["variants"][mode] = {
                    "backend_metric_fail": [],
                    "backend_metric_error_message": [],
                    "x_ratio_stage_one": [],
                    "x_ratio_stage_two": [],
                    "cost_no_mtd": [],
                    "cost_with_mtd_one": [],
                    "cost_with_mtd_two": [],
                    "stage_one_hidden": [],
                    "stage_one_eff": [],
                    "stage_two_hidden": [],
                    "stage_two_eff": [],
                    "residual_no_att": [],
                    "post_mtd_opf_converge": [],
                }
            ensure_variant_slots(group["variants"][mode], n, case_class)

        for i in range(n):
            if bool(group["recovery_error"][i]):
                group["varrho"][i] = float("nan")
                group["obj_one"][i] = float("nan")
                group["obj_two"][i] = float("nan")
                group["worst_primal"][i] = float("nan")
                group["worst_dual"][i] = float("nan")
                group["backend_run_error"][i] = False
                group["backend_run_error_message"][i] = ""
                group["backend_mtd_fail"][i] = 0
                group["stage_one_time"][i] = float("nan")
                group["stage_two_time"][i] = float("nan")
                for mode in next_load_modes:
                    variant = group["variants"][mode]
                    variant["backend_metric_fail"][i] = False
                    variant["backend_metric_error_message"][i] = ""
                    variant["x_ratio_stage_one"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
                    variant["x_ratio_stage_two"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
                    variant["cost_no_mtd"][i] = float("nan")
                    variant["cost_with_mtd_one"][i] = float("nan")
                    variant["cost_with_mtd_two"][i] = float("nan")
                    variant["stage_one_hidden"][i] = float("nan")
                    variant["stage_one_eff"][i] = float("nan")
                    variant["stage_two_hidden"][i] = float("nan")
                    variant["stage_two_eff"][i] = float("nan")
                    variant["residual_no_att"][i] = float("nan")
                    variant["post_mtd_opf_converge"][i] = False
                continue

            idx_val = int(group["alarm_idx"][i])
            ang_no = int(float(key.strip("()").split(",")[0]))
            ang_str = float(key.strip("()").split(",")[1])
            seed_ = seed_base + 100000 * ang_no + int(round(1000 * ang_str)) + idx_val
            random.seed(seed_)
            np.random.seed(seed_ % (2**32 - 1))
            v_last_clean = np.asarray(group["v_last_clean"][i], dtype=np.complex128)
            c_recover = np.asarray(group["c_recover_no_ref"][i], dtype=float).reshape(-1, 1)
            c_true = full_attack_vector_from_group(group, i)

            backend_run_error = False
            backend_run_error_message = ""
            backend_mtd_fail = 1
            stage_one_time = float("nan")
            stage_two_time = float("nan")
            varrho = float("nan")
            obj_one = float("nan")
            obj_two = float("nan")
            worst_primal = float("nan")
            worst_dual = float("nan")

            try:
                mtd_optim_ = mtd_optim(case_class, v_last_clean, c_recover, mtd_config["varrho_square"])
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

                for mode in next_load_modes:
                    next_load_idx = next_load_idx_for(
                        split_start=split_start,
                        idx_val=idx_val,
                        mode=mode,
                        next_load_extra=int(args.next_load_extra),
                    )
                    variant = group["variants"][mode]
                    backend_metric_fail = False
                    x_ratio_stage_one = np.full(case_class.no_brh, np.nan, dtype=float)
                    x_ratio_stage_two = np.full(case_class.no_brh, np.nan, dtype=float)
                    cost_no_mtd = float("nan")
                    cost_with_mtd_one = float("nan")
                    cost_with_mtd_two = float("nan")
                    stage_one_hidden = float("nan")
                    stage_one_eff = float("nan")
                    stage_two_hidden = float("nan")
                    stage_two_eff = float("nan")
                    residual_no_att = float("nan")
                    post_mtd_opf_converge = False
                    metric_error_messages: List[str] = []

                    try:
                        (
                            _ok1,
                            x_ratio_stage_one,
                            cost_no_mtd,
                            cost_with_mtd_one,
                            _residual_normal,
                            stage_one_hidden,
                            stage_one_eff,
                        ) = mtd_optim_.mtd_metric_with_attack(
                            b_mtd=b_mtd_one_final,
                            c_actual=c_true,
                            load_active=load_active[next_load_idx],
                            load_reactive=load_reactive[next_load_idx],
                            pv_active=pv_active_[next_load_idx],
                            mode=mtd_config["mode"],
                        )
                    except Exception as e:
                        backend_metric_fail = True
                        metric_error_messages.append(f"stage_one:{repr(e)}")

                    try:
                        (
                            post_mtd_opf_converge,
                            x_ratio_stage_two,
                            _cost_no_stage_two,
                            cost_with_mtd_two,
                            residual_no_att,
                            stage_two_hidden,
                            stage_two_eff,
                        ) = mtd_optim_.mtd_metric_with_attack(
                            b_mtd=b_mtd_two_final,
                            c_actual=c_true,
                            load_active=load_active[next_load_idx],
                            load_reactive=load_reactive[next_load_idx],
                            pv_active=pv_active_[next_load_idx],
                            mode=mtd_config["mode"],
                        )
                    except Exception as e:
                        backend_metric_fail = True
                        metric_error_messages.append(f"stage_two:{repr(e)}")

                    variant["backend_metric_fail"][i] = bool(backend_metric_fail)
                    variant["backend_metric_error_message"][i] = " | ".join(metric_error_messages)
                    variant["x_ratio_stage_one"][i] = np.asarray(x_ratio_stage_one, dtype=float)
                    variant["x_ratio_stage_two"][i] = np.asarray(x_ratio_stage_two, dtype=float)
                    variant["cost_no_mtd"][i] = float(cost_no_mtd)
                    variant["cost_with_mtd_one"][i] = float(cost_with_mtd_one)
                    variant["cost_with_mtd_two"][i] = float(cost_with_mtd_two)
                    variant["stage_one_hidden"][i] = float(stage_one_hidden)
                    variant["stage_one_eff"][i] = float(stage_one_eff)
                    variant["stage_two_hidden"][i] = float(stage_two_hidden)
                    variant["stage_two_eff"][i] = float(stage_two_eff)
                    variant["residual_no_att"][i] = float(residual_no_att)
                    variant["post_mtd_opf_converge"][i] = bool(post_mtd_opf_converge)

            except Exception as e:
                backend_run_error = True
                backend_run_error_message = repr(e)
                for mode in next_load_modes:
                    variant = group["variants"][mode]
                    variant["backend_metric_fail"][i] = True
                    variant["backend_metric_error_message"][i] = f"backend_run:{repr(e)}"
                    variant["x_ratio_stage_one"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
                    variant["x_ratio_stage_two"][i] = np.full(case_class.no_brh, np.nan, dtype=float)
                    variant["cost_no_mtd"][i] = float("nan")
                    variant["cost_with_mtd_one"][i] = float("nan")
                    variant["cost_with_mtd_two"][i] = float("nan")
                    variant["stage_one_hidden"][i] = float("nan")
                    variant["stage_one_eff"][i] = float("nan")
                    variant["stage_two_hidden"][i] = float("nan")
                    variant["stage_two_eff"][i] = float("nan")
                    variant["residual_no_att"][i] = float("nan")
                    variant["post_mtd_opf_converge"][i] = False

            group["varrho"][i] = float(varrho)
            group["obj_one"][i] = float(obj_one)
            group["obj_two"][i] = float(obj_two)
            group["worst_primal"][i] = float(worst_primal)
            group["worst_dual"][i] = float(worst_dual)
            group["backend_run_error"][i] = bool(backend_run_error)
            group["backend_run_error_message"][i] = backend_run_error_message
            group["backend_mtd_fail"][i] = int(backend_mtd_fail)
            group["stage_one_time"][i] = float(stage_one_time)
            group["stage_two_time"][i] = float(stage_two_time)

            if (i + 1) % max(1, int(args.log_every)) == 0:
                print(f"[progress] group={key} recomputed attack backend {i + 1}/{n}")

    metadata["next_load_modes"] = next_load_modes
    metadata["next_load_extra"] = int(args.next_load_extra)
    metadata["mtd_mode"] = mtd_config["mode"]
    metadata["varrho"] = float(np.sqrt(mtd_config["varrho_square"]))
    metadata["upper_scale"] = float(mtd_config["upper_scale"])
    metadata["multi_run_no"] = int(mtd_config["multi_run_no"])
    metadata["x_facts_ratio"] = float(mtd_config["x_facts_ratio"])
    metadata["skip_backend"] = False
    attach_runtime_metadata(
        metadata,
        repo_root=str(Path(__file__).resolve().parent),
        input_cache_path=args.input_cache,
        output_cache_path=args.output_cache,
        runner_name=Path(__file__).name,
    )

    Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
    save_metric(args.output_cache, **out)
    print(f"Saved recomputed attack cache: {args.output_cache}")


if __name__ == "__main__":
    main()

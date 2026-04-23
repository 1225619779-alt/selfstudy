from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from configs.config import mtd_config, save_metric, sys_config
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_dataset, load_load_pv, load_measurement
from utils.run_metadata import attach_runtime_metadata


GROUP_KEY = "(0,0.0)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect clean false-alarm cache once, then materialize multiple tau regimes offline."
        )
    )
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--max_total_run", type=int, default=-1)
    p.add_argument("--stop_ddd_alarm_at", type=int, default=-1)
    p.add_argument("--seed_base", type=int, default=20260324)
    p.add_argument("--is_shuffle", action="store_true")
    p.add_argument(
        "--next_load_mode",
        choices=["sample_length", "offset"],
        default="sample_length",
    )
    p.add_argument("--next_load_extra", type=int, default=7)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--skip_backend", action="store_true")
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_output_path(split: str) -> str:
    return (
        f"metric/{sys_config['case_name']}/"
        f"clean_alarm_cache_{split}_mode_{mtd_config['mode']}_"
        f"{round(np.sqrt(mtd_config['varrho_square']), 5)}_{mtd_config['upper_scale']}.npy"
    )


def split_start_idx(feature_size: int, split: str) -> int:
    if split == "valid":
        return int(feature_size * nn_setting["train_prop"])
    if split == "test":
        return int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))
    raise ValueError(f"Unsupported split={split!r}")


def next_load_idx_for(split_start: int, idx_val: int, next_load_mode: str, next_load_extra: int) -> int:
    if next_load_mode == "sample_length":
        return int(split_start + nn_setting["sample_length"] + idx_val)
    return int(split_start + next_load_extra + idx_val)


def append_array(records: Dict[str, List[Any]], key: str, value: Any) -> None:
    records.setdefault(key, []).append(value)


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

    split = str(args.split)
    max_total_run: Optional[int] = None if args.max_total_run < 0 else int(args.max_total_run)
    stop_ddd_alarm_at: Optional[int] = None if args.stop_ddd_alarm_at < 0 else int(args.stop_ddd_alarm_at)
    seed_base = int(args.seed_base)
    is_shuffle = bool(args.is_shuffle)
    next_load_mode = str(args.next_load_mode)
    next_load_extra = int(args.next_load_extra)
    log_every = max(1, int(args.log_every))
    skip_backend = bool(args.skip_backend)
    output = str(args.output).strip() or make_output_path(split)
    ensure_parent(output)

    print("==== Clean alarm cache collector ====")
    print("case_name:", sys_config["case_name"])
    print("split:", split)
    print("max_total_run:", "FULL" if max_total_run is None else max_total_run)
    print("stop_ddd_alarm_at:", "DISABLED" if stop_ddd_alarm_at is None else stop_ddd_alarm_at)
    print("next_load_mode:", next_load_mode)
    print("next_load_extra:", next_load_extra)
    print("skip_backend:", skip_backend)
    print("output:", output)

    case_class = load_case()
    z_noise_summary, _ = load_measurement()
    load_active, load_reactive, pv_active_, _ = load_load_pv()

    (
        test_dataloader_scaled,
        test_dataloader_unscaled,
        valid_dataloader_scaled,
        valid_dataloader_unscaled,
    ) = load_dataset(is_shuffle=is_shuffle)
    _ = test_dataloader_scaled
    _ = valid_dataloader_scaled

    if split == "test":
        dataloader_unscaled = test_dataloader_unscaled
    else:
        dataloader_unscaled = valid_dataloader_unscaled

    feature_size = len(z_noise_summary)
    start_idx = split_start_idx(feature_size=feature_size, split=split)

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))
    _ = lstm_ae

    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    detector_quantile = float(dd_detector.quantile[dd_detector.quantile_idx])
    detector_threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])
    print(f"Detector quantile: {detector_quantile}")
    print(f"Detector threshold: {detector_threshold}")

    total_clean_sample = 0
    total_DDD_alarm = 0
    records: Dict[str, List[Any]] = {
        "sample_idx": [],
        "front_end_alarm": [],
        "detector_loss_all": [],
    }

    for idx_, (idx, input_batch, v_est_pre, v_est_last) in tqdm(
        enumerate(dataloader_unscaled),
        desc=f"clean-cache-{split}",
    ):
        if max_total_run is not None and idx_ >= max_total_run:
            break

        idx_val = safe_int_idx(idx)
        total_clean_sample += 1
        append_array(records, "sample_idx", idx_val)
        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        input_scale = scaler_(input_batch)
        _, _, _, loss_recons = dd_detector.evaluate(input_scale)
        detector_loss = float(loss_recons)
        is_alarm = bool(detector_loss > detector_threshold)
        append_array(records, "detector_loss_all", detector_loss)
        append_array(records, "front_end_alarm", is_alarm)
        if not is_alarm:
            if total_clean_sample % log_every == 0:
                print(
                    f"[progress] samples={total_clean_sample} alarms={total_DDD_alarm} "
                    f"alarm_rate={(total_DDD_alarm / total_clean_sample):.6f}"
                )
            continue

        total_DDD_alarm += 1
        append_array(records, "alarm_idx", idx_val)
        append_array(records, "detector_loss", detector_loss)
        append_array(records, "recovery_error", False)
        append_array(records, "recovery_error_message", "")

        try:
            (
                _z_recover,
                v_recover,
                loss_recover_summary,
                _loss_sparse_real_summary,
                _loss_sparse_imag_summary,
                _loss_v_mag_summary,
                _loss_summary,
                recover_time_single,
            ) = dd_detector.recover(
                attack_batch=input_batch,
                v_pre=v_est_pre,
                v_last=v_est_last,
            )
        except Exception as e:
            records["recovery_error"][-1] = True
            records["recovery_error_message"][-1] = repr(e)
            append_array(records, "verify_score", float("nan"))
            append_array(records, "recover_deviation", float("nan"))
            append_array(records, "pre_deviation", float("nan"))
            append_array(records, "recovery_ite_no", 0)
            append_array(records, "recovery_time", float("nan"))
            append_array(records, "c_recover_no_ref", np.full(case_class.no_bus - 1, np.nan, dtype=float))
            append_array(records, "v_last", np.asarray(v_est_last.numpy(), dtype=np.complex128))
            append_array(records, "v_recover", np.full(case_class.no_bus, np.nan + 1j * np.nan, dtype=np.complex128))
            append_array(records, "varrho", float("nan"))
            append_array(records, "obj_one", float("nan"))
            append_array(records, "obj_two", float("nan"))
            append_array(records, "worst_primal", float("nan"))
            append_array(records, "worst_dual", float("nan"))
            append_array(records, "backend_run_error", True)
            append_array(records, "backend_run_error_message", "")
            append_array(records, "backend_mtd_fail", 0)
            append_array(records, "backend_metric_fail", 0)
            append_array(records, "stage_one_time", float("nan"))
            append_array(records, "stage_two_time", float("nan"))
            append_array(records, "stage_one_residual", float("nan"))
            append_array(records, "stage_two_residual", float("nan"))
            append_array(records, "cost_no_mtd", float("nan"))
            append_array(records, "cost_with_mtd_one", float("nan"))
            append_array(records, "cost_with_mtd_two", float("nan"))
            append_array(records, "x_ratio_stage_one", np.full(case_class.no_brh, np.nan, dtype=float))
            append_array(records, "x_ratio_stage_two", np.full(case_class.no_brh, np.nan, dtype=float))
            append_array(records, "post_mtd_opf_converge", False)
            if stop_ddd_alarm_at is not None and total_DDD_alarm >= stop_ddd_alarm_at:
                break
            continue

        vang_recover = np.angle(v_recover.numpy())
        vang_pre = np.angle(v_est_pre.numpy())
        vang_last = np.angle(v_est_last.numpy())
        c_recover = vang_last - vang_recover
        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
        score_ = float(np.linalg.norm(c_recover_no_ref, 2))

        append_array(records, "verify_score", score_)
        append_array(records, "recover_deviation", float(np.linalg.norm(vang_last - vang_recover, 2)))
        append_array(records, "pre_deviation", float(np.linalg.norm(vang_last - vang_pre, 2)))
        append_array(records, "recovery_ite_no", int(len(loss_recover_summary)))
        append_array(records, "recovery_time", float(recover_time_single))
        append_array(records, "c_recover_no_ref", np.asarray(c_recover_no_ref[:, 0], dtype=float))
        append_array(records, "v_last", np.asarray(v_est_last.numpy(), dtype=np.complex128))
        append_array(records, "v_recover", np.asarray(v_recover.numpy(), dtype=np.complex128))

        if skip_backend:
            append_array(records, "varrho", float("nan"))
            append_array(records, "obj_one", float("nan"))
            append_array(records, "obj_two", float("nan"))
            append_array(records, "worst_primal", float("nan"))
            append_array(records, "worst_dual", float("nan"))
            append_array(records, "backend_run_error", False)
            append_array(records, "backend_run_error_message", "")
            append_array(records, "backend_mtd_fail", 0)
            append_array(records, "backend_metric_fail", 0)
            append_array(records, "stage_one_time", float("nan"))
            append_array(records, "stage_two_time", float("nan"))
            append_array(records, "stage_one_residual", float("nan"))
            append_array(records, "stage_two_residual", float("nan"))
            append_array(records, "cost_no_mtd", float("nan"))
            append_array(records, "cost_with_mtd_one", float("nan"))
            append_array(records, "cost_with_mtd_two", float("nan"))
            append_array(records, "x_ratio_stage_one", np.full(case_class.no_brh, np.nan, dtype=float))
            append_array(records, "x_ratio_stage_two", np.full(case_class.no_brh, np.nan, dtype=float))
            append_array(records, "post_mtd_opf_converge", False)
            if total_DDD_alarm % log_every == 0:
                print(
                    f"[progress] samples={total_clean_sample} alarms={total_DDD_alarm} "
                    f"alarm_rate={(total_DDD_alarm / total_clean_sample):.6f}"
                )
            if stop_ddd_alarm_at is not None and total_DDD_alarm >= stop_ddd_alarm_at:
                break
            continue

        seed_ = seed_base + idx_val
        random.seed(seed_)
        np.random.seed(seed_ % (2**32 - 1))
        next_load_idx = next_load_idx_for(
            split_start=start_idx,
            idx_val=idx_val,
            next_load_mode=next_load_mode,
            next_load_extra=next_load_extra,
        )

        backend_run_error = False
        backend_run_error_message = ""
        backend_metric_fail = False
        backend_mtd_fail = 0
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
            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, mtd_config["varrho_square"])
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
            except Exception:
                backend_metric_fail = True
        except Exception as e:
            backend_run_error = True
            backend_run_error_message = repr(e)
            backend_mtd_fail = 1
            backend_metric_fail = True

        append_array(records, "varrho", varrho)
        append_array(records, "obj_one", float(obj_one))
        append_array(records, "obj_two", float(obj_two))
        append_array(records, "worst_primal", float(worst_primal))
        append_array(records, "worst_dual", float(worst_dual))
        append_array(records, "backend_run_error", bool(backend_run_error))
        append_array(records, "backend_run_error_message", backend_run_error_message)
        append_array(records, "backend_mtd_fail", int(backend_mtd_fail))
        append_array(records, "backend_metric_fail", int(backend_metric_fail))
        append_array(records, "stage_one_time", float(stage_one_time))
        append_array(records, "stage_two_time", float(stage_two_time))
        append_array(records, "stage_one_residual", float(stage_one_residual))
        append_array(records, "stage_two_residual", float(stage_two_residual))
        append_array(records, "cost_no_mtd", float(cost_no_mtd))
        append_array(records, "cost_with_mtd_one", float(cost_with_mtd_one))
        append_array(records, "cost_with_mtd_two", float(cost_with_mtd_two))
        append_array(records, "x_ratio_stage_one", np.asarray(x_ratio_stage_one, dtype=float))
        append_array(records, "x_ratio_stage_two", np.asarray(x_ratio_stage_two, dtype=float))
        append_array(records, "post_mtd_opf_converge", bool(post_mtd_opf_converge))

        if total_DDD_alarm % log_every == 0:
            print(
                f"[progress] samples={total_clean_sample} alarms={total_DDD_alarm} "
                f"alarm_rate={(total_DDD_alarm / total_clean_sample):.6f}"
            )

        if stop_ddd_alarm_at is not None and total_DDD_alarm >= stop_ddd_alarm_at:
            break

    payload = {
        "metadata": {
            "case_name": sys_config["case_name"],
            "split": split,
            "seed_base": seed_base,
            "is_shuffle": is_shuffle,
            "next_load_mode": next_load_mode,
            "next_load_extra": next_load_extra,
            "detector_quantile": detector_quantile,
            "detector_threshold": detector_threshold,
            "recovery_error_policy": "skip",
            "mtd_mode": mtd_config["mode"],
            "varrho": float(np.sqrt(mtd_config["varrho_square"])),
            "upper_scale": float(mtd_config["upper_scale"]),
            "multi_run_no": int(mtd_config["multi_run_no"]),
            "x_facts_ratio": float(mtd_config["x_facts_ratio"]),
            "split_start_idx": int(start_idx),
            "sample_length": int(nn_setting["sample_length"]),
            "skip_backend": skip_backend,
        },
        "group_key": GROUP_KEY,
        "total_clean_sample": int(total_clean_sample),
        "total_DDD_alarm": int(total_DDD_alarm),
        "records": records,
    }
    attach_runtime_metadata(
        payload["metadata"],
        repo_root=str(Path(__file__).resolve().parent),
        output_cache_path=output,
        runner_name=Path(__file__).name,
    )

    save_metric(output, **payload)
    print(f"Saved cache: {output}")


if __name__ == "__main__":
    main()

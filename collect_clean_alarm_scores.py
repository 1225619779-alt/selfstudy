from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from configs.config import sys_config
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from utils.load_data import load_case, load_measurement, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean false-alarm scores for gate ablation."
    )
    parser.add_argument(
        "--max_total_run",
        type=int,
        default=-1,
        help="Max number of clean samples to scan. -1 means full test set.",
    )
    parser.add_argument(
        "--stop_ddd_alarm_at",
        type=int,
        default=-1,
        help="Stop when the number of clean DDD alarms reaches this value. -1 disables early stop.",
    )
    parser.add_argument(
        "--is_shuffle",
        action="store_true",
        help="Shuffle dataloader. Keep this OFF for strict comparability.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"metric/{sys_config['case_name']}/metric_clean_alarm_scores_full.npy",
        help="Output .npy path.",
    )
    return parser.parse_args()


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()

    max_total_run: Optional[int] = None if args.max_total_run < 0 else int(args.max_total_run)
    stop_ddd_alarm_at: Optional[int] = None if args.stop_ddd_alarm_at < 0 else int(args.stop_ddd_alarm_at)
    is_shuffle: bool = bool(args.is_shuffle)
    output_path: str = args.output

    print("==== collect_clean_alarm_scores ====")
    print("case_name:", sys_config["case_name"])
    print("max_total_run:", "FULL" if max_total_run is None else max_total_run)
    print("stop_ddd_alarm_at:", "DISABLED" if stop_ddd_alarm_at is None else stop_ddd_alarm_at)
    print("dataloader shuffle:", is_shuffle)
    print("output:", output_path)

    case_class = load_case()
    z_noise_summary, _ = load_measurement()
    _, test_dataloader_unscaled, _, _ = load_dataset(is_shuffle=is_shuffle)

    feature_size = len(z_noise_summary)
    print("feature_size:", feature_size)

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(
        torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"]))
    )
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])
    print(f"Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}")
    print(f"Threshold: {threshold}")

    group_key = "(0,0.0)"

    total_clean_sample: Dict[str, int] = {group_key: 0}
    total_DDD_alarm: Dict[str, int] = {group_key: 0}
    clean_alarm_idx: Dict[str, List[int]] = {group_key: []}
    ddd_loss_alarm: Dict[str, List[float]] = {group_key: []}
    score_phys_l2: Dict[str, List[float]] = {group_key: []}
    recover_fail: Dict[str, List[bool]] = {group_key: []}
    recover_time_alarm: Dict[str, List[float]] = {group_key: []}

    for idx_, (idx, input_batch, v_est_pre, v_est_last) in tqdm(
        enumerate(test_dataloader_unscaled)
    ):
        if max_total_run is not None and idx_ >= max_total_run:
            break

        idx_val = safe_int_idx(idx)
        total_clean_sample[group_key] += 1

        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        input_scale = scaler_(input_batch)
        _, _, _, loss_recons = dd_detector.evaluate(input_scale)
        loss_recons = float(loss_recons)
        alarm = bool(loss_recons > threshold)

        if not alarm:
            continue

        total_DDD_alarm[group_key] += 1
        clean_alarm_idx[group_key].append(idx_val)
        ddd_loss_alarm[group_key].append(loss_recons)

        try:
            (
                z_recover,
                v_recover,
                loss_recover_summary,
                loss_sparse_real_summary,
                loss_sparse_imag_summary,
                loss_v_mag_summary,
                loss_summary,
                recover_time_single,
            ) = dd_detector.recover(
                attack_batch=input_batch,
                v_pre=v_est_pre,
                v_last=v_est_last,
            )
            _ = z_recover
            _ = loss_recover_summary
            _ = loss_sparse_real_summary
            _ = loss_sparse_imag_summary
            _ = loss_v_mag_summary
            _ = loss_summary

            vang_last = np.angle(v_est_last.numpy())
            vang_recover = np.angle(v_recover.numpy())
            c_recover = vang_last - vang_recover
            c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
            score_ = float(np.linalg.norm(c_recover_no_ref, 2))

            score_phys_l2[group_key].append(score_)
            recover_fail[group_key].append(False)
            recover_time_alarm[group_key].append(float(recover_time_single))
        except Exception as e:
            print(f"[WARN] recovery failed at idx={idx_val}: {repr(e)}")
            score_phys_l2[group_key].append(float("nan"))
            recover_fail[group_key].append(True)
            recover_time_alarm[group_key].append(float("nan"))

        if stop_ddd_alarm_at is not None and total_DDD_alarm[group_key] >= stop_ddd_alarm_at:
            break

    result = {
        "group_key": group_key,
        "max_total_run": max_total_run,
        "stop_ddd_alarm_at": stop_ddd_alarm_at,
        "is_shuffle": is_shuffle,
        "ddd_quantile": dd_detector.quantile[dd_detector.quantile_idx],
        "ddd_threshold": threshold,
        "total_clean_sample": total_clean_sample,
        "total_DDD_alarm": total_DDD_alarm,
        "clean_alarm_idx": clean_alarm_idx,
        "ddd_loss_alarm": ddd_loss_alarm,
        "score_phys_l2": score_phys_l2,
        "recover_fail": recover_fail,
        "recover_time_alarm": recover_time_alarm,
    }

    ensure_parent(output_path)
    np.save(output_path, result, allow_pickle=True)

    n_clean = total_clean_sample[group_key]
    n_alarm = total_DDD_alarm[group_key]
    n_fail = int(np.sum(np.asarray(recover_fail[group_key], dtype=bool)))
    n_finite = int(np.sum(np.isfinite(np.asarray(score_phys_l2[group_key], dtype=float))))

    print("\n==== Summary ====")
    print("total_clean_sample:", n_clean)
    print("total_DDD_alarm:", n_alarm)
    print("recover_fail_count:", n_fail)
    print("finite_phys_score_count:", n_finite)
    if n_alarm > 0:
        print("mean_ddd_loss_alarm:", float(np.nanmean(np.asarray(ddd_loss_alarm[group_key], dtype=float))))
        print("mean_score_phys_l2:", float(np.nanmean(np.asarray(score_phys_l2[group_key], dtype=float))))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

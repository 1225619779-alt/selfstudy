from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from configs.config import mtd_config, sys_config
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from utils.load_data import load_case, load_measurement, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect attack alarm scores for gate ablation."
    )
    parser.add_argument(
        "--total_run",
        type=int,
        default=200,
        help="Number of attacked samples to test per attack group.",
    )
    parser.add_argument(
        "--ang_no_list",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Attack angle-number list.",
    )
    parser.add_argument(
        "--ang_str_list",
        type=float,
        nargs="*",
        default=[0.2, 0.3],
        help="Attack angle-strength list.",
    )
    parser.add_argument(
        "--mag_no",
        type=int,
        default=0,
        help="Attack magnitude-number, keep 0 to match current repo setup.",
    )
    parser.add_argument(
        "--mag_str",
        type=float,
        default=0.0,
        help="Attack magnitude-strength, keep 0.0 to match current repo setup.",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=20260324,
        help="Base seed. Effective seed matches evaluation_event_trigger.py style.",
    )
    parser.add_argument(
        "--is_shuffle",
        action="store_true",
        help="Shuffle dataloader. Keep this OFF for strict comparability.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"metric/{sys_config['case_name']}/metric_attack_alarm_scores_200.npy",
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

    total_run = int(args.total_run)
    ang_no_list = [int(x) for x in args.ang_no_list]
    ang_str_list = [float(x) for x in args.ang_str_list]
    mag_no = int(args.mag_no)
    mag_str = float(args.mag_str)
    seed_base = int(args.seed_base)
    is_shuffle = bool(args.is_shuffle)
    output_path = args.output

    print("==== collect_attack_alarm_scores ====")
    print("case_name:", sys_config["case_name"])
    print("total_run per group:", total_run)
    print("varrho:", np.sqrt(mtd_config["varrho_square"]))
    print("mode:", mtd_config["mode"])
    print("upper_scaling:", mtd_config["upper_scale"])
    print("ang_no_list:", ang_no_list)
    print("ang_str_list:", ang_str_list)
    print("mag_no:", mag_no)
    print("mag_str:", mag_str)
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

    total_tested_sample: Dict[str, int] = {}
    total_DDD_alarm: Dict[str, int] = {}
    attack_alarm_idx: Dict[str, List[int]] = {}
    ddd_loss_alarm: Dict[str, List[float]] = {}
    score_phys_l2: Dict[str, List[float]] = {}
    recover_fail: Dict[str, List[bool]] = {}
    recover_time_alarm: Dict[str, List[float]] = {}

    for ang_no in ang_no_list:
        for ang_str in ang_str_list:
            group_key = f"({ang_no},{ang_str})"
            print("\n----", group_key, "----")

            total_tested_sample[group_key] = 0
            total_DDD_alarm[group_key] = 0
            attack_alarm_idx[group_key] = []
            ddd_loss_alarm[group_key] = []
            score_phys_l2[group_key] = []
            recover_fail[group_key] = []
            recover_time_alarm[group_key] = []

            for idx_, (idx, input_batch, v_est_pre, v_est_last) in tqdm(
                enumerate(test_dataloader_unscaled)
            ):
                if idx_ >= total_run:
                    break

                idx_val = safe_int_idx(idx)
                total_tested_sample[group_key] += 1

                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()

                seed_ = seed_base + 100000 * ang_no + int(round(1000 * ang_str)) + idx_val
                random.seed(seed_)
                np.random.seed(seed_ % (2**32 - 1))

                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                    z_noise=input_batch,
                    v_est_last=v_est_last,
                    ang_no=ang_no,
                    mag_no=mag_no,
                    ang_str=ang_str,
                    mag_str=mag_str,
                )
                v_att_est_last = torch.from_numpy(v_att_est_last)

                z_att_noise_scale = scaler_(z_att_noise)
                _, _, _, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                loss_recons = float(loss_recons)
                alarm = bool(loss_recons > threshold)
                if not alarm:
                    continue

                total_DDD_alarm[group_key] += 1
                attack_alarm_idx[group_key].append(idx_val)
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
                        v_last=v_att_est_last,
                    )
                    _ = z_recover
                    _ = loss_recover_summary
                    _ = loss_sparse_real_summary
                    _ = loss_sparse_imag_summary
                    _ = loss_v_mag_summary
                    _ = loss_summary

                    vang_att = np.angle(v_att_est_last.numpy())
                    vang_recover = np.angle(v_recover.numpy())
                    c_recover = vang_att - vang_recover
                    c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                    score_ = float(np.linalg.norm(c_recover_no_ref, 2))

                    score_phys_l2[group_key].append(score_)
                    recover_fail[group_key].append(False)
                    recover_time_alarm[group_key].append(float(recover_time_single))
                except Exception as e:
                    print(f"[WARN] recovery failed at group={group_key}, idx={idx_val}: {repr(e)}")
                    score_phys_l2[group_key].append(float("nan"))
                    recover_fail[group_key].append(True)
                    recover_time_alarm[group_key].append(float("nan"))

    result = {
        "total_run": total_run,
        "ang_no_list": ang_no_list,
        "ang_str_list": ang_str_list,
        "mag_no": mag_no,
        "mag_str": mag_str,
        "seed_base": seed_base,
        "is_shuffle": is_shuffle,
        "ddd_quantile": dd_detector.quantile[dd_detector.quantile_idx],
        "ddd_threshold": threshold,
        "total_tested_sample": total_tested_sample,
        "total_DDD_alarm": total_DDD_alarm,
        "attack_alarm_idx": attack_alarm_idx,
        "ddd_loss_alarm": ddd_loss_alarm,
        "score_phys_l2": score_phys_l2,
        "recover_fail": recover_fail,
        "recover_time_alarm": recover_time_alarm,
    }

    ensure_parent(output_path)
    np.save(output_path, result, allow_pickle=True)

    print("\n==== Summary ====")
    total_alarm_all = 0
    total_fail_all = 0
    for group_key in sorted(total_DDD_alarm.keys()):
        n_alarm = int(total_DDD_alarm[group_key])
        n_fail = int(np.sum(np.asarray(recover_fail[group_key], dtype=bool)))
        total_alarm_all += n_alarm
        total_fail_all += n_fail
        print(
            group_key,
            "tested=", total_tested_sample[group_key],
            "DDD_alarm=", n_alarm,
            "recover_fail=", n_fail,
        )
    print("total_attack_DDD_alarm:", total_alarm_all)
    print("total_recover_fail:", total_fail_all)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

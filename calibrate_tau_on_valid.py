
"""
Validation-only calibration for verification-gate thresholds.

Purpose
-------
1. Collect recovery-aware verification scores on CLEAN FALSE ALARMS from the
   validation split only.
2. Choose tau_main and tau_strict from validation-only clean-score percentiles.
3. Save the score array and calibration summary for later paper use.

Recommended use
---------------
python calibrate_tau_on_valid.py --p_main 0.90 --p_strict 0.95 --max_total_run -1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from utils.load_data import load_case, load_measurement, load_load_pv, load_dataset
from models.model import LSTM_AE
from models.evaluation import Evaluation
from configs.nn_setting import nn_setting
from configs.config import sys_config, save_metric
from models.dataset import scaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only calibration for recovery-aware verification thresholds."
    )
    parser.add_argument("--p_main", type=float, default=0.90,
                        help="Percentile for main operating point, e.g. 0.90.")
    parser.add_argument("--p_strict", type=float, default=0.95,
                        help="Percentile for stricter operating point, e.g. 0.95.")
    parser.add_argument("--max_total_run", type=int, default=-1,
                        help="-1 means full validation loader. Otherwise stop after this many validation samples.")
    parser.add_argument("--is_shuffle", action="store_true",
                        help="Shuffle dataloader. Default False for reproducibility.")
    return parser.parse_args()


def _safe_int_idx(idx) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def main() -> None:
    args = parse_args()

    if not (0.0 < args.p_main < 1.0 and 0.0 < args.p_strict < 1.0):
        raise ValueError("p_main and p_strict must both be in (0,1).")
    if args.p_main >= args.p_strict:
        raise ValueError("p_main must be smaller than p_strict.")

    max_total_run = None if args.max_total_run < 0 else int(args.max_total_run)

    print("==== Validation-only tau calibration ====")
    print("case_name:", sys_config["case_name"])
    print("p_main:", args.p_main)
    print("p_strict:", args.p_strict)
    print("max_total_run:", "FULL" if max_total_run is None else max_total_run)
    print("dataloader shuffle:", bool(args.is_shuffle))

    case_class = load_case()
    z_noise_summary, v_est_summary = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
    test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(
        is_shuffle=bool(args.is_shuffle)
    )

    # keep detector setup consistent with the repo
    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(
        torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"]))
    )
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    print(f"Detector quantile: {dd_detector.quantile[dd_detector.quantile_idx]}")
    print(f"Detector threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}")

    verify_score_clean_false_alarm = []
    clean_false_alarm_idx = []
    clean_recovery_time = []

    print("Collect validation-split clean false-alarm verify_score ...")
    for local_i, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(valid_dataloader_unscaled)):
        if max_total_run is not None and local_i >= max_total_run:
            break

        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        z_clean_scale = scaler_(input)
        encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_clean_scale)

        # keep only clean false alarms on validation split
        if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
            continue

        z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, \
        loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_single = dd_detector.recover(
            attack_batch=input,
            v_pre=v_est_pre,
            v_last=v_est_last,
        )

        vang_last = np.angle(v_est_last.numpy())
        vang_recover = np.angle(v_recover.numpy())
        c_recover = (vang_last - vang_recover)
        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)

        score_ = float(np.linalg.norm(c_recover_no_ref, 2))
        verify_score_clean_false_alarm.append(score_)
        clean_false_alarm_idx.append(_safe_int_idx(idx))
        clean_recovery_time.append(recover_time_single)

    if len(verify_score_clean_false_alarm) == 0:
        raise RuntimeError("No clean false alarms found on validation split. Cannot calibrate tau.")

    arr = np.asarray(verify_score_clean_false_alarm, dtype=float)

    tau_main = float(np.quantile(arr, args.p_main))
    tau_strict = float(np.quantile(arr, args.p_strict))

    print("\n=== validation-only clean false-alarm summary ===")
    print("count  =", len(arr))
    print("mean   =", float(arr.mean()))
    print("median =", float(np.median(arr)))
    print("p90    =", float(np.quantile(arr, 0.90)))
    print("p95    =", float(np.quantile(arr, 0.95)))
    print("p99    =", float(np.quantile(arr, 0.99)))
    print("\n=== recommended taus from validation split ===")
    print(f"tau_main   (p={args.p_main:.2f}) = {tau_main:.6f}")
    print(f"tau_strict (p={args.p_strict:.2f}) = {tau_strict:.6f}")
    print(f"rounded_main_3dp   = {tau_main:.3f}")
    print(f"rounded_strict_3dp = {tau_strict:.3f}")

    out_dir = Path(f'metric/{sys_config["case_name"]}/tau_calibration_valid')
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "valid_clean_false_alarm_scores.npy", arr)
    np.save(out_dir / "valid_clean_false_alarm_idx.npy", np.asarray(clean_false_alarm_idx, dtype=int))
    np.save(out_dir / "valid_clean_recovery_time.npy", np.asarray(clean_recovery_time, dtype=float))

    summary_txt = out_dir / "tau_calibration_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Validation-only tau calibration summary\n")
        f.write(f"count={len(arr)}\n")
        f.write(f"mean={float(arr.mean()):.12f}\n")
        f.write(f"median={float(np.median(arr)):.12f}\n")
        f.write(f"p90={float(np.quantile(arr, 0.90)):.12f}\n")
        f.write(f"p95={float(np.quantile(arr, 0.95)):.12f}\n")
        f.write(f"p99={float(np.quantile(arr, 0.99)):.12f}\n")
        f.write(f"p_main={args.p_main:.4f}\n")
        f.write(f"p_strict={args.p_strict:.4f}\n")
        f.write(f"tau_main={tau_main:.12f}\n")
        f.write(f"tau_strict={tau_strict:.12f}\n")
        f.write(f"tau_main_rounded_3dp={tau_main:.3f}\n")
        f.write(f"tau_strict_rounded_3dp={tau_strict:.3f}\n")

    # optional compatibility save via repo helper
    save_metric(
        address=str(out_dir / "tau_calibration_metric.npy"),
        verify_score_clean_false_alarm=verify_score_clean_false_alarm,
        clean_false_alarm_idx=clean_false_alarm_idx,
        clean_recovery_time=clean_recovery_time,
        p_main=args.p_main,
        p_strict=args.p_strict,
        tau_main=tau_main,
        tau_strict=tau_strict,
    )

    print("\nSaved:")
    print(out_dir / "valid_clean_false_alarm_scores.npy")
    print(out_dir / "tau_calibration_summary.txt")
    print(out_dir / "tau_calibration_metric.npy")


if __name__ == "__main__":
    main()

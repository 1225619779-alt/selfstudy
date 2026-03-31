"""
Calibrate verification score for recovery-gated trigger.
Goal:
1. collect verify_score on clean false alarms
2. collect verify_score on attack true alarms
"""

from utils.load_data import load_case, load_measurement, load_load_pv, load_dataset
from models.model import LSTM_AE
from models.evaluation import Evaluation
import torch
from configs.nn_setting import nn_setting
from configs.config import sys_config, save_metric
import numpy as np
from tqdm import tqdm
from models.dataset import scaler

# -------------------------
# Preparation
# -------------------------
total_run = 200   # first small run for calibration
print("total_run", total_run)

case_class = load_case()
z_noise_summary, v_est_summary = load_measurement()
load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(is_shuffle=False)

lstm_ae = LSTM_AE()
lstm_ae.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
dd_detector = Evaluation(case_class=case_class)
scaler_ = scaler()

print(f'Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}')
print(f'Threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}')

# attack list
ang_no_list = [1, 2, 3]
mag_no = 0
ang_str_list = [0.2, 0.3]
mag_str = 0

# -------------------------
# Metrics
# -------------------------
verify_score_clean_false_alarm = []
clean_false_alarm_idx = []

verify_score_attack_true_alarm = {}
attack_true_alarm_idx = {}

clean_recovery_time = []
attack_recovery_time = {}

# -------------------------
# Part 1: clean false alarms
# -------------------------
print("Collect clean false-alarm verify_score ...")

for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):
    if idx_ >= total_run:
        break

    # format
    v_est_pre = v_est_pre.flatten()
    v_est_last = v_est_last.flatten()

    # detector on clean data
    z_clean_scale = scaler_(input)
    encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_clean_scale)

    # only keep clean false alarms
    if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
        continue

    # recovery on clean-but-alarmed sample
    z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_single = dd_detector.recover(
        attack_batch=input,   # clean measurement, not scaled
        v_pre=v_est_pre,
        v_last=v_est_last,
    )

    vang_last = np.angle(v_est_last.numpy())
    vang_recover = np.angle(v_recover.numpy())
    c_recover = (vang_last - vang_recover)
    c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)

    score_ = float(np.linalg.norm(c_recover_no_ref, 2))

    verify_score_clean_false_alarm.append(score_)
    clean_false_alarm_idx.append(int(idx.numpy()))
    clean_recovery_time.append(recover_time_single)

# -------------------------
# Part 2: attack true alarms
# -------------------------
print("Collect attack true-alarm verify_score ...")

for ang_no in ang_no_list:
    for ang_str in ang_str_list:
        key = f'({ang_no},{ang_str})'
        print(key)

        verify_score_attack_true_alarm[key] = []
        attack_true_alarm_idx[key] = []
        attack_recovery_time[key] = []

        for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):
            if idx_ >= total_run:
                break

            # format
            v_est_pre = v_est_pre.flatten()
            v_est_last = v_est_last.flatten()

            # generate attack
            z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                z_noise=input,
                v_est_last=v_est_last,
                ang_no=ang_no,
                mag_no=mag_no,
                ang_str=ang_str,
                mag_str=mag_str
            )
            v_att_est_last = torch.from_numpy(v_att_est_last)

            # detect attack
            z_att_noise_scale = scaler_(z_att_noise)
            encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)

            # only keep attack true alarms
            if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                continue

            # recover attacked sample
            z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_single = dd_detector.recover(
                attack_batch=input,   # keep same style as current repo
                v_pre=v_est_pre,
                v_last=v_att_est_last,
            )

            vang_att = np.angle(v_att_est_last.numpy())
            vang_recover = np.angle(v_recover.numpy())
            c_recover = (vang_att - vang_recover)
            c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)

            score_ = float(np.linalg.norm(c_recover_no_ref, 2))

            verify_score_attack_true_alarm[key].append(score_)
            attack_true_alarm_idx[key].append(int(idx.numpy()))
            attack_recovery_time[key].append(recover_time_single)

# -------------------------
# quick summary
# -------------------------
print("\n=== clean false alarm summary ===")
if len(verify_score_clean_false_alarm) == 0:
    print("No clean false alarms found in current total_run.")
else:
    arr = np.array(verify_score_clean_false_alarm, dtype=float)
    print("count =", len(arr))
    print("mean  =", arr.mean())
    print("median=", np.median(arr))
    print("p90   =", np.quantile(arr, 0.90))
    print("p95   =", np.quantile(arr, 0.95))
    print("p99   =", np.quantile(arr, 0.99))

print("\n=== attack true alarm summary ===")
for key, vals in verify_score_attack_true_alarm.items():
    if len(vals) == 0:
        print(key, "-> no attack true alarms found in current total_run.")
        continue
    arr = np.array(vals, dtype=float)
    print(key)
    print("  count =", len(arr))
    print("  mean  =", arr.mean())
    print("  median=", np.median(arr))
    print("  p10   =", np.quantile(arr, 0.10))
    print("  p50   =", np.quantile(arr, 0.50))
    print("  p90   =", np.quantile(arr, 0.90))

# -------------------------
# save
# -------------------------
save_metric(
    address=f'metric/{sys_config["case_name"]}/metric_verify_score_{total_run}.npy',

    verify_score_clean_false_alarm=verify_score_clean_false_alarm,
    clean_false_alarm_idx=clean_false_alarm_idx,
    clean_recovery_time=clean_recovery_time,

    verify_score_attack_true_alarm=verify_score_attack_true_alarm,
    attack_true_alarm_idx=attack_true_alarm_idx,
    attack_recovery_time=attack_recovery_time,
)
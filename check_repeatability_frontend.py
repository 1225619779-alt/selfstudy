from utils.load_data import load_case, load_measurement, load_load_pv, load_dataset
from models.model import LSTM_AE
from models.evaluation import Evaluation
import torch
from configs.nn_setting import nn_setting
import numpy as np
from models.dataset import scaler
import random

TOTAL_RUN = 50
TARGETS = [(1, 0.2), (1, 0.3), (2, 0.2)]


def to_scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else arr


def run_once(ang_no, ang_str):
    case_class = load_case()
    z_noise_summary, v_est_summary = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
    test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(is_shuffle=False)

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting['model_path'], map_location=torch.device(nn_setting['device'])))
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    idx_list = []
    attack_sig_list = []
    loss_list = []
    tp_list = []

    for idx_, (idx, input, v_est_pre, v_est_last) in enumerate(test_dataloader_unscaled):
        if idx_ >= TOTAL_RUN:
            break

        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        # 记录样本顺序
        idx_val = int(idx.item())
        idx_list.append(idx_val)

        seed_ = 20260324 + 100000 * ang_no + int(round(1000 * ang_str)) + idx_val
        random.seed(seed_)
        np.random.seed(seed_ % (2**32 - 1))
        # 生成攻击
        z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
            z_noise=input,
            v_est_last=v_est_last,
            ang_no=ang_no,
            mag_no=0,
            ang_str=ang_str,
            mag_str=0
        )

        # 记录攻击签名（不是为了做研究，只是为了排查两次运行是否生成了同一个攻击）
        z_np = np.asarray(z_att_noise)
        attack_sig = (float(np.sum(z_np)), float(np.linalg.norm(z_np)))
        attack_sig_list.append(attack_sig)

        # detector
        z_att_noise_scale = scaler_(z_att_noise)
        encoded, decoded, loss_lattent, loss_recons = dd_detector.evaluate(z_att_noise_scale)

        loss_val = float(to_scalar(loss_recons))
        loss_list.append(loss_val)

        tp = bool(loss_val > dd_detector.ae_threshold[dd_detector.quantile_idx])
        tp_list.append(tp)

    return {
        "idx_list": idx_list,
        "attack_sig_list": attack_sig_list,
        "loss_list": loss_list,
        "tp_list": tp_list,
    }


def compare_two_runs(name, run_a, run_b):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    same_idx = (run_a["idx_list"] == run_b["idx_list"])
    same_attack = (run_a["attack_sig_list"] == run_b["attack_sig_list"])
    same_tp = (run_a["tp_list"] == run_b["tp_list"])

    print("same idx order?      ", same_idx)
    print("same attack signature?", same_attack)
    print("same TP_DDD list?    ", same_tp)

    if not same_idx:
        diff_idx_pos = [i for i in range(min(len(run_a["idx_list"]), len(run_b["idx_list"]))) if run_a["idx_list"][i] != run_b["idx_list"][i]]
        print("idx different positions:", diff_idx_pos)

    if not same_attack:
        diff_attack_pos = [i for i in range(min(len(run_a["attack_sig_list"]), len(run_b["attack_sig_list"]))) if run_a["attack_sig_list"][i] != run_b["attack_sig_list"][i]]
        print("attack different positions:", diff_attack_pos[:20])

    if not same_tp:
        diff_tp_pos = [i for i in range(min(len(run_a["tp_list"]), len(run_b["tp_list"]))) if run_a["tp_list"][i] != run_b["tp_list"][i]]
        print("TP_DDD different positions:", diff_tp_pos)

        for i in diff_tp_pos[:10]:
            print(f"  pos={i}, idx={run_a['idx_list'][i]}")
            print(f"    runA loss={run_a['loss_list'][i]:.12f}, TP={run_a['tp_list'][i]}")
            print(f"    runB loss={run_b['loss_list'][i]:.12f}, TP={run_b['tp_list'][i]}")


def main():
    print("TOTAL_RUN =", TOTAL_RUN)

    for ang_no, ang_str in TARGETS:
        tag = f"({ang_no},{ang_str})"
        print(f"\nRunning A for {tag} ...")
        run_a = run_once(ang_no, ang_str)

        print(f"Running B for {tag} ...")
        run_b = run_once(ang_no, ang_str)

        compare_two_runs(f"COMPARE {tag}", run_a, run_b)


if __name__ == "__main__":
    main()
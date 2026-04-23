from __future__ import annotations

import argparse
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attack-side DDET-MTD evaluation with a verification gate. "
            "This is a safer CLI replacement for the old hard-coded script."
        )
    )
    parser.add_argument("--tau_verify", type=float, required=True, help="Verification threshold.")
    parser.add_argument("--total_run", type=int, default=50, help="Max number of test samples per attack group.")
    parser.add_argument(
        "--ang_no_list",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Attack bus-count list. Default: 1 2 3",
    )
    parser.add_argument(
        "--ang_str_list",
        type=float,
        nargs="*",
        default=[0.2, 0.3],
        help="Attack-strength list. Default: 0.2 0.3",
    )
    parser.add_argument("--mag_no", type=int, default=0)
    parser.add_argument("--mag_str", type=float, default=0.0)
    parser.add_argument(
        "--seed_base",
        type=int,
        default=20260324,
        help="Base seed. The old script used a deterministic per-group and per-sample seed style.",
    )
    parser.add_argument("--is_shuffle", action="store_true", help="Shuffle dataloader. Not recommended.")
    parser.add_argument(
        "--recover_input_mode",
        choices=["repo_compatible", "attacked_measurement"],
        default="repo_compatible",
        help=(
            "repo_compatible preserves the current repo behavior and passes the original input to recover(); "
            "attacked_measurement passes z_att_noise instead. Use the latter only if you intentionally want to change semantics."
        ),
    )
    parser.add_argument(
        "--next_load_extra",
        type=int,
        default=7,
        help="Compatibility offset used only when --next_load_mode=offset.",
    )
    parser.add_argument(
        "--next_load_mode",
        choices=["sample_length", "offset"],
        default="sample_length",
        help=(
            "sample_length uses test_start_idx + sample_length + idx; "
            "offset preserves the historical next_load_extra behavior."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output .npy path. If omitted, a tau-specific path will be auto-generated.",
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



def make_output_path(tau_verify: float) -> str:
    return (
        f"metric/{sys_config['case_name']}/"
        f"metric_event_trigger_tau_{tau_verify}_"
        f"mode_{mtd_config['mode']}_"
        f"{round(np.sqrt(mtd_config['varrho_square']), 5)}_"
        f"{mtd_config['upper_scale']}.npy"
    )



def main() -> None:
    args = parse_args()

    tau_verify = float(args.tau_verify)
    total_run = int(args.total_run)
    ang_no_list = list(args.ang_no_list)
    ang_str_list = [float(x) for x in args.ang_str_list]
    mag_no = int(args.mag_no)
    mag_str = float(args.mag_str)
    seed_base = int(args.seed_base)
    is_shuffle = bool(args.is_shuffle)
    recover_input_mode = str(args.recover_input_mode)
    next_load_extra = int(args.next_load_extra)
    next_load_mode = str(args.next_load_mode)
    output = args.output.strip() or make_output_path(tau_verify)
    ensure_parent(output)

    print("==== Attack event-trigger evaluation (CLI) ====")
    print("case_name:", sys_config["case_name"])
    print("tau_verify:", tau_verify)
    print("total_run:", total_run)
    print("ang_no_list:", ang_no_list)
    print("ang_str_list:", ang_str_list)
    print("recover_input_mode:", recover_input_mode)
    print("next_load_mode:", next_load_mode)
    print("next_load_extra:", next_load_extra)
    print("output:", output)
    print("varrho:", np.sqrt(mtd_config["varrho_square"]))
    print("mode:", mtd_config["mode"])
    print("upper_scaling:", mtd_config["upper_scale"])

    case_class = load_case()
    z_noise_summary, v_est_summary = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
    test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(
        is_shuffle=is_shuffle
    )
    feature_size = len(z_noise_summary)
    test_start_idx = int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    detector_quantile = float(dd_detector.quantile[dd_detector.quantile_idx])
    detector_threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])
    print(f"Detector quantile: {detector_quantile}")
    print(f"Detector threshold: {detector_threshold}")

    TP_DDD: Dict[str, List[bool]] = {}
    att_strength: Dict[str, List[float]] = {}
    varrho_summary: Dict[str, List[float]] = {}
    verify_score: Dict[str, List[float]] = {}
    trigger_after_verification: Dict[str, List[bool]] = {}
    skip_by_verification: Dict[str, List[bool]] = {}
    recover_deviation: Dict[str, List[float]] = {}
    pre_deviation: Dict[str, List[float]] = {}
    recovery_ite_no: Dict[str, List[int]] = {}
    recovery_time: List[float] = []
    recovery_error: Dict[str, List[bool]] = {}
    obj_one: Dict[str, List[float]] = {}
    obj_two: Dict[str, List[float]] = {}
    worst_primal: Dict[str, List[float]] = {}
    worst_dual: Dict[str, List[float]] = {}
    fail: Dict[str, List[int]] = {}
    x_ratio_stage_one: Dict[str, List[np.ndarray]] = {}
    x_ratio_stage_two: Dict[str, List[np.ndarray]] = {}
    residual_no_att: List[float] = []
    post_mtd_opf_converge: List[bool] = []
    mtd_stage_one_time: List[float] = []
    mtd_stage_two_time: List[float] = []
    mtd_stage_one_eff: Dict[str, List[float]] = {}
    mtd_stage_two_eff: Dict[str, List[float]] = {}
    mtd_stage_one_hidden: Dict[str, List[float]] = {}
    mtd_stage_two_hidden: Dict[str, List[float]] = {}
    cost_no_mtd: Dict[str, List[float]] = {}
    cost_with_mtd_one: Dict[str, List[float]] = {}
    cost_with_mtd_two: Dict[str, List[float]] = {}
    backend_metric_fail: Dict[str, List[bool]] = {}

    group_summary: Dict[str, Dict[str, float]] = {}

    for ang_no in ang_no_list:
        for ang_str in ang_str_list:
            key = f"({ang_no},{ang_str})"
            print("Running group", key)
            TP_DDD[key] = []
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

            for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled), desc=key):
                if idx_ >= total_run:
                    break

                idx_val = safe_int_idx(idx)
                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()

                seed_ = seed_base + 100000 * int(ang_no) + int(round(1000 * float(ang_str))) + idx_val
                random.seed(seed_)
                np.random.seed(seed_ % (2**32 - 1))

                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                    z_noise=input,
                    v_est_last=v_est_last,
                    ang_no=ang_no,
                    mag_no=mag_no,
                    ang_str=ang_str,
                    mag_str=mag_str,
                )
                v_att_est_last = torch.from_numpy(v_att_est_last)

                z_att_noise_scale = scaler_(z_att_noise)
                _, _, _, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                    TP_DDD[key].append(False)
                    continue
                TP_DDD[key].append(True)

                recover_batch = input if recover_input_mode == "repo_compatible" else z_att_noise
                try:
                    z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_single = dd_detector.recover(
                        attack_batch=recover_batch,
                        v_pre=v_est_pre,
                        v_last=v_att_est_last,
                    )
                    recovery_error[key].append(False)
                except Exception as e:
                    print(f"[WARN] recovery failed at group={key}, idx={idx_val}: {repr(e)}")
                    recovery_error[key].append(True)
                    verify_score[key].append(float("nan"))
                    trigger_after_verification[key].append(False)
                    skip_by_verification[key].append(False)
                    recover_deviation[key].append(float("nan"))
                    pre_deviation[key].append(float("nan"))
                    recovery_ite_no[key].append(0)
                    recovery_time.append(float("nan"))
                    continue

                vang_recover = np.angle(v_recover.numpy())
                vang_pre = np.angle(v_est_pre.numpy())
                vang_true = np.angle(v_est_last.numpy())
                recover_deviation[key].append(float(np.linalg.norm(vang_true - vang_recover, 2)))
                pre_deviation[key].append(float(np.linalg.norm(vang_true - vang_pre, 2)))
                recovery_ite_no[key].append(int(len(loss_recover_summary)))
                recovery_time.append(float(recover_time_single))

                vang_att = np.angle(v_att_est_last.numpy())
                c_true = vang_att - vang_true
                c_recover = vang_att - vang_recover
                c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                score_ = float(np.linalg.norm(c_recover_no_ref, 2))
                att_strength[key].append(score_)
                verify_score[key].append(score_)

                should_trigger_mtd = bool(score_ >= tau_verify)
                trigger_after_verification[key].append(should_trigger_mtd)
                skip_by_verification[key].append(not should_trigger_mtd)
                if not should_trigger_mtd:
                    continue

                try:
                    mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, mtd_config["varrho_square"])
                    varrho_summary[key].append(float(np.sqrt(mtd_optim_.varrho_square)))
                    (
                        b_mtd_one_final,
                        b_mtd_two_final,
                        obj_one_final,
                        obj_two_final,
                        obj_worst_primal,
                        obj_worst_dual,
                        c_worst,
                        stage_one_time_,
                        stage_two_time_,
                        is_fail,
                    ) = mtd_optim_.multi_run()

                    obj_one[key].append(obj_one_final)
                    obj_two[key].append(obj_two_final)
                    worst_primal[key].append(obj_worst_primal)
                    worst_dual[key].append(obj_worst_dual)
                    fail[key].append(int(is_fail))
                    mtd_stage_one_time.append(float(stage_one_time_))
                    mtd_stage_two_time.append(float(stage_two_time_))
                except Exception as e:
                    print(f"[WARN] backend MTD failed at group={key}, idx={idx_val}: {repr(e)}")
                    obj_one[key].append(float("nan"))
                    obj_two[key].append(float("nan"))
                    worst_primal[key].append(float("nan"))
                    worst_dual[key].append(float("nan"))
                    fail[key].append(1)
                    mtd_stage_one_time.append(float("nan"))
                    mtd_stage_two_time.append(float("nan"))
                    mtd_stage_one_hidden[key].append(float("nan"))
                    mtd_stage_one_eff[key].append(float("nan"))
                    x_ratio_stage_one[key].append(np.full(case_class.no_brh, np.nan, dtype=float))
                    cost_no_mtd[key].append(float("nan"))
                    cost_with_mtd_one[key].append(float("nan"))
                    mtd_stage_two_hidden[key].append(float("nan"))
                    mtd_stage_two_eff[key].append(float("nan"))
                    post_mtd_opf_converge.append(False)
                    x_ratio_stage_two[key].append(np.full(case_class.no_brh, np.nan, dtype=float))
                    cost_with_mtd_two[key].append(float("nan"))
                    residual_no_att.append(float("nan"))
                    backend_metric_fail[key].append(True)
                    continue

                if next_load_mode == "sample_length":
                    next_load_idx = int(test_start_idx + nn_setting["sample_length"] + idx_val)
                else:
                    next_load_idx = int(test_start_idx + next_load_extra + idx_val)

                metric_failed = False
                nan_ratio = np.full(case_class.no_brh, np.nan, dtype=float)

                try:
                    (
                        is_converged,
                        x_mtd_change_ratio,
                        cost_no_mtd_,
                        cost_with_mtd_,
                        residual_normal,
                        residual_hid,
                        residual_eff,
                    ) = mtd_optim_.mtd_metric_with_attack(
                        b_mtd=b_mtd_one_final,
                        c_actual=c_true,
                        load_active=load_active[next_load_idx],
                        load_reactive=load_reactive[next_load_idx],
                        pv_active=pv_active_[next_load_idx],
                        mode=mtd_config["mode"],
                    )
                    mtd_stage_one_hidden[key].append(float(residual_hid))
                    mtd_stage_one_eff[key].append(float(residual_eff))
                    x_ratio_stage_one[key].append(x_mtd_change_ratio)
                    cost_no_mtd[key].append(float(cost_no_mtd_))
                    cost_with_mtd_one[key].append(float(cost_with_mtd_))
                except Exception as e:
                    print(f"[WARN] stage-one backend metric failed at group={key}, idx={idx_val}: {repr(e)}")
                    metric_failed = True
                    mtd_stage_one_hidden[key].append(float("nan"))
                    mtd_stage_one_eff[key].append(float("nan"))
                    x_ratio_stage_one[key].append(nan_ratio.copy())
                    cost_no_mtd[key].append(float("nan"))
                    cost_with_mtd_one[key].append(float("nan"))

                try:
                    (
                        is_converged,
                        x_mtd_change_ratio,
                        cost_no_mtd_,
                        cost_with_mtd_,
                        residual_normal_,
                        residual_hid,
                        residual_eff,
                    ) = mtd_optim_.mtd_metric_with_attack(
                        b_mtd=b_mtd_two_final,
                        c_actual=c_true,
                        load_active=load_active[next_load_idx],
                        load_reactive=load_reactive[next_load_idx],
                        pv_active=pv_active_[next_load_idx],
                        mode=mtd_config["mode"],
                    )
                    mtd_stage_two_hidden[key].append(float(residual_hid))
                    mtd_stage_two_eff[key].append(float(residual_eff))
                    post_mtd_opf_converge.append(bool(is_converged))
                    x_ratio_stage_two[key].append(x_mtd_change_ratio)
                    cost_with_mtd_two[key].append(float(cost_with_mtd_))
                    residual_no_att.append(float(residual_normal_))
                except Exception as e:
                    print(f"[WARN] stage-two backend metric failed at group={key}, idx={idx_val}: {repr(e)}")
                    metric_failed = True
                    mtd_stage_two_hidden[key].append(float("nan"))
                    mtd_stage_two_eff[key].append(float("nan"))
                    post_mtd_opf_converge.append(False)
                    x_ratio_stage_two[key].append(nan_ratio.copy())
                    cost_with_mtd_two[key].append(float("nan"))
                    residual_no_att.append(float("nan"))

                backend_metric_fail[key].append(metric_failed)

            front_end_alarms = int(sum(TP_DDD[key]))
            backend_triggers = int(sum(trigger_after_verification[key]))
            arr = float(backend_triggers / front_end_alarms) if front_end_alarms > 0 else float("nan")
            group_summary[key] = {
                "front_end_alarms": front_end_alarms,
                "recovery_error_count": int(np.asarray(recovery_error[key], dtype=bool).sum()),
                "backend_triggers": backend_triggers,
                "arr": arr,
                "mean_verify_score": float(np.nanmean(verify_score[key])) if verify_score[key] else float("nan"),
                "median_verify_score": float(np.nanmedian(verify_score[key])) if verify_score[key] else float("nan"),
            }
            print(key, group_summary[key])

    metadata = {
        "case_name": sys_config["case_name"],
        "tau_verify": tau_verify,
        "total_run": total_run,
        "ang_no_list": ang_no_list,
        "ang_str_list": ang_str_list,
        "mag_no": mag_no,
        "mag_str": mag_str,
        "seed_base": seed_base,
        "is_shuffle": is_shuffle,
        "recover_input_mode": recover_input_mode,
        "next_load_mode": next_load_mode,
        "next_load_extra": next_load_extra,
        "detector_quantile": detector_quantile,
        "detector_threshold": detector_threshold,
        "recovery_error_policy": "count_as_non_trigger_non_skip",
        "group_summary": group_summary,
    }

    save_metric(
        address=output,
        metadata=metadata,
        TP_DDD=TP_DDD,
        att_strength=att_strength,
        varrho_summary=varrho_summary,
        verify_score=verify_score,
        trigger_after_verification=trigger_after_verification,
        skip_by_verification=skip_by_verification,
        recover_deviation=recover_deviation,
        pre_deviation=pre_deviation,
        recovery_ite_no=recovery_ite_no,
        recovery_time=recovery_time,
        recovery_error=recovery_error,
        obj_one=obj_one,
        obj_two=obj_two,
        worst_primal=worst_primal,
        worst_dual=worst_dual,
        fail=fail,
        x_ratio_stage_one=x_ratio_stage_one,
        x_ratio_stage_two=x_ratio_stage_two,
        residual_no_att=residual_no_att,
        post_mtd_opf_converge=post_mtd_opf_converge,
        mtd_stage_one_time=mtd_stage_one_time,
        mtd_stage_two_time=mtd_stage_two_time,
        mtd_stage_one_eff=mtd_stage_one_eff,
        mtd_stage_two_eff=mtd_stage_two_eff,
        mtd_stage_one_hidden=mtd_stage_one_hidden,
        mtd_stage_two_hidden=mtd_stage_two_hidden,
        cost_no_mtd=cost_no_mtd,
        cost_with_mtd_one=cost_with_mtd_one,
        cost_with_mtd_two=cost_with_mtd_two,
        backend_metric_fail=backend_metric_fail,
    )

    summary_path = str(Path(output).with_suffix(".summary.txt"))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Attack evaluation summary ====\n")
        for k, v in metadata.items():
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

    print(f"Saved metric: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

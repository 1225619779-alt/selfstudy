from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect attack-side alarm cache once, then materialize multiple tau regimes offline."
        )
    )
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--total_run", type=int, default=50)
    p.add_argument("--ang_no_list", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--ang_str_list", type=float, nargs="*", default=[0.2, 0.3])
    p.add_argument("--mag_no", type=int, default=0)
    p.add_argument("--mag_str", type=float, default=0.0)
    p.add_argument("--seed_base", type=int, default=20260324)
    p.add_argument("--is_shuffle", action="store_true")
    p.add_argument(
        "--recover_input_mode",
        choices=["repo_compatible", "attacked_measurement"],
        default="repo_compatible",
    )
    p.add_argument(
        "--next_load_modes",
        nargs="*",
        default=["sample_length", "offset"],
        choices=["sample_length", "offset"],
    )
    p.add_argument("--next_load_extra", type=int, default=7)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--skip_backend", action="store_true")
    p.add_argument("--output", default="")
    return p.parse_args()


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_output_path(split: str, total_run: int, recover_input_mode: str) -> str:
    return (
        f"metric/{sys_config['case_name']}/"
        f"attack_alarm_cache_{split}_{total_run}_{recover_input_mode}_"
        f"mode_{mtd_config['mode']}_{round(np.sqrt(mtd_config['varrho_square']), 5)}_{mtd_config['upper_scale']}.npy"
    )


def split_start_idx(feature_size: int, split: str) -> int:
    if split == "valid":
        return int(feature_size * nn_setting["train_prop"])
    if split == "test":
        return int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))
    raise ValueError(f"Unsupported split={split!r}")


def next_load_idx_for(split_start: int, idx_val: int, mode: str, next_load_extra: int) -> int:
    if mode == "sample_length":
        return int(split_start + nn_setting["sample_length"] + idx_val)
    return int(split_start + next_load_extra + idx_val)


def append_array(store: Dict[str, List[Any]], key: str, value: Any) -> None:
    store.setdefault(key, []).append(value)


def group_key(ang_no: int, ang_str: float) -> str:
    return f"({int(ang_no)},{float(ang_str):.1f})"


def empty_variant_store() -> Dict[str, List[Any]]:
    return {
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


def main() -> None:
    args = parse_args()

    split = str(args.split)
    total_run = int(args.total_run)
    ang_no_list = list(args.ang_no_list)
    ang_str_list = [float(x) for x in args.ang_str_list]
    mag_no = int(args.mag_no)
    mag_str = float(args.mag_str)
    seed_base = int(args.seed_base)
    is_shuffle = bool(args.is_shuffle)
    recover_input_mode = str(args.recover_input_mode)
    next_load_modes = list(dict.fromkeys(args.next_load_modes))
    next_load_extra = int(args.next_load_extra)
    log_every = max(1, int(args.log_every))
    skip_backend = bool(args.skip_backend)
    output = str(args.output).strip() or make_output_path(split, total_run, recover_input_mode)
    ensure_parent(output)

    print("==== Attack alarm cache collector ====")
    print("case_name:", sys_config["case_name"])
    print("split:", split)
    print("total_run:", total_run)
    print("recover_input_mode:", recover_input_mode)
    print("next_load_modes:", next_load_modes)
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

    groups: Dict[str, Dict[str, Any]] = {}

    for ang_no in ang_no_list:
        for ang_str in ang_str_list:
            key = group_key(ang_no=ang_no, ang_str=ang_str)
            print("Running group", key)
            group_records: Dict[str, Any] = {
                "sample_idx": [],
                "front_end_alarm": [],
                "detector_loss_all": [],
                "alarm_idx": [],
                "detector_loss_alarm": [],
                "verify_score": [],
                "recovery_error": [],
                "recovery_error_message": [],
                "recover_deviation": [],
                "pre_deviation": [],
                "recovery_ite_no": [],
                "recovery_time": [],
                "c_true": [],
                "c_true_no_ref": [],
                "c_recover_no_ref": [],
                "v_last_clean": [],
                "v_att_last": [],
                "v_recover": [],
                "varrho": [],
                "obj_one": [],
                "obj_two": [],
                "worst_primal": [],
                "worst_dual": [],
                "backend_run_error": [],
                "backend_run_error_message": [],
                "backend_mtd_fail": [],
                "stage_one_time": [],
                "stage_two_time": [],
                "variants": {mode: empty_variant_store() for mode in next_load_modes},
            }

            for idx_, (idx, input_batch, v_est_pre, v_est_last) in tqdm(
                enumerate(dataloader_unscaled),
                desc=f"attack-cache-{key}",
            ):
                if idx_ >= total_run:
                    break

                idx_val = safe_int_idx(idx)
                group_records["sample_idx"].append(idx_val)

                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()

                seed_ = seed_base + 100000 * int(ang_no) + int(round(1000 * float(ang_str))) + idx_val
                random.seed(seed_)
                np.random.seed(seed_ % (2**32 - 1))

                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                    z_noise=input_batch,
                    v_est_last=v_est_last,
                    ang_no=int(ang_no),
                    mag_no=mag_no,
                    ang_str=float(ang_str),
                    mag_str=mag_str,
                )
                v_att_est_last = torch.from_numpy(v_att_est_last)

                z_att_noise_scale = scaler_(z_att_noise)
                _, _, _, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                detector_loss = float(loss_recons)
                is_alarm = bool(detector_loss > detector_threshold)
                group_records["detector_loss_all"].append(detector_loss)
                group_records["front_end_alarm"].append(is_alarm)
                if not is_alarm:
                    if (idx_ + 1) % log_every == 0:
                        alarm_count = int(np.asarray(group_records["front_end_alarm"], dtype=bool).sum())
                        print(f"[progress] group={key} samples={idx_ + 1}/{total_run} front_end_alarms={alarm_count}")
                    continue

                group_records["alarm_idx"].append(idx_val)
                group_records["detector_loss_alarm"].append(detector_loss)
                group_records["recovery_error"].append(False)
                group_records["recovery_error_message"].append("")

                recover_batch = input_batch if recover_input_mode == "repo_compatible" else z_att_noise

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
                        attack_batch=recover_batch,
                        v_pre=v_est_pre,
                        v_last=v_att_est_last,
                    )
                except Exception as e:
                    group_records["recovery_error"][-1] = True
                    group_records["recovery_error_message"][-1] = repr(e)
                    group_records["verify_score"].append(float("nan"))
                    group_records["recover_deviation"].append(float("nan"))
                    group_records["pre_deviation"].append(float("nan"))
                    group_records["recovery_ite_no"].append(0)
                    group_records["recovery_time"].append(float("nan"))
                    group_records["c_true"].append(np.full(case_class.no_bus, np.nan, dtype=float))
                    group_records["c_true_no_ref"].append(np.full(case_class.no_bus - 1, np.nan, dtype=float))
                    group_records["c_recover_no_ref"].append(np.full(case_class.no_bus - 1, np.nan, dtype=float))
                    group_records["v_last_clean"].append(np.asarray(v_est_last.numpy(), dtype=np.complex128))
                    group_records["v_att_last"].append(np.asarray(v_att_est_last.numpy(), dtype=np.complex128))
                    group_records["v_recover"].append(np.full(case_class.no_bus, np.nan + 1j * np.nan, dtype=np.complex128))
                    group_records["varrho"].append(float("nan"))
                    group_records["obj_one"].append(float("nan"))
                    group_records["obj_two"].append(float("nan"))
                    group_records["worst_primal"].append(float("nan"))
                    group_records["worst_dual"].append(float("nan"))
                    group_records["backend_run_error"].append(False)
                    group_records["backend_run_error_message"].append("")
                    group_records["backend_mtd_fail"].append(0)
                    group_records["stage_one_time"].append(float("nan"))
                    group_records["stage_two_time"].append(float("nan"))
                    for mode in next_load_modes:
                        variant = group_records["variants"][mode]
                        variant["backend_metric_fail"].append(False)
                        variant["backend_metric_error_message"].append("")
                        variant["x_ratio_stage_one"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["x_ratio_stage_two"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["cost_no_mtd"].append(float("nan"))
                        variant["cost_with_mtd_one"].append(float("nan"))
                        variant["cost_with_mtd_two"].append(float("nan"))
                        variant["stage_one_hidden"].append(float("nan"))
                        variant["stage_one_eff"].append(float("nan"))
                        variant["stage_two_hidden"].append(float("nan"))
                        variant["stage_two_eff"].append(float("nan"))
                        variant["residual_no_att"].append(float("nan"))
                        variant["post_mtd_opf_converge"].append(False)
                    continue

                vang_recover = np.angle(v_recover.numpy())
                vang_att = np.angle(v_att_est_last.numpy())
                vang_true = np.angle(v_est_last.numpy())
                c_true = vang_att - vang_true
                c_true_no_ref = np.expand_dims(c_true[case_class.non_ref_index], 1)
                c_recover = vang_att - vang_recover
                c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                score_ = float(np.linalg.norm(c_recover_no_ref, 2))

                group_records["verify_score"].append(score_)
                group_records["recover_deviation"].append(float(np.linalg.norm(vang_true - vang_recover, 2)))
                group_records["pre_deviation"].append(float(np.linalg.norm(vang_true - np.angle(v_est_pre.numpy()), 2)))
                group_records["recovery_ite_no"].append(int(len(loss_recover_summary)))
                group_records["recovery_time"].append(float(recover_time_single))
                group_records["c_true"].append(np.asarray(c_true, dtype=float))
                group_records["c_true_no_ref"].append(np.asarray(c_true_no_ref[:, 0], dtype=float))
                group_records["c_recover_no_ref"].append(np.asarray(c_recover_no_ref[:, 0], dtype=float))
                group_records["v_last_clean"].append(np.asarray(v_est_last.numpy(), dtype=np.complex128))
                group_records["v_att_last"].append(np.asarray(v_att_est_last.numpy(), dtype=np.complex128))
                group_records["v_recover"].append(np.asarray(v_recover.numpy(), dtype=np.complex128))

                if skip_backend:
                    group_records["varrho"].append(float("nan"))
                    group_records["obj_one"].append(float("nan"))
                    group_records["obj_two"].append(float("nan"))
                    group_records["worst_primal"].append(float("nan"))
                    group_records["worst_dual"].append(float("nan"))
                    group_records["backend_run_error"].append(False)
                    group_records["backend_run_error_message"].append("")
                    group_records["backend_mtd_fail"].append(0)
                    group_records["stage_one_time"].append(float("nan"))
                    group_records["stage_two_time"].append(float("nan"))
                    for mode in next_load_modes:
                        variant = group_records["variants"][mode]
                        variant["backend_metric_fail"].append(False)
                        variant["backend_metric_error_message"].append("")
                        variant["x_ratio_stage_one"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["x_ratio_stage_two"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["cost_no_mtd"].append(float("nan"))
                        variant["cost_with_mtd_one"].append(float("nan"))
                        variant["cost_with_mtd_two"].append(float("nan"))
                        variant["stage_one_hidden"].append(float("nan"))
                        variant["stage_one_eff"].append(float("nan"))
                        variant["stage_two_hidden"].append(float("nan"))
                        variant["stage_two_eff"].append(float("nan"))
                        variant["residual_no_att"].append(float("nan"))
                        variant["post_mtd_opf_converge"].append(False)
                    if (idx_ + 1) % log_every == 0:
                        alarm_count = int(np.asarray(group_records["front_end_alarm"], dtype=bool).sum())
                        print(f"[progress] group={key} samples={idx_ + 1}/{total_run} front_end_alarms={alarm_count}")
                    continue

                backend_run_error = False
                backend_run_error_message = ""
                backend_mtd_fail = 0
                stage_one_time = float("nan")
                stage_two_time = float("nan")
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
                        _c_worst,
                        stage_one_time,
                        stage_two_time,
                        backend_mtd_fail,
                    ) = mtd_optim_.multi_run()

                    for mode in next_load_modes:
                        next_load_idx = next_load_idx_for(
                            split_start=start_idx,
                            idx_val=idx_val,
                            mode=mode,
                            next_load_extra=next_load_extra,
                        )
                        variant = group_records["variants"][mode]
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

                        variant["backend_metric_fail"].append(bool(backend_metric_fail))
                        variant["backend_metric_error_message"].append(" | ".join(metric_error_messages))
                        variant["x_ratio_stage_one"].append(np.asarray(x_ratio_stage_one, dtype=float))
                        variant["x_ratio_stage_two"].append(np.asarray(x_ratio_stage_two, dtype=float))
                        variant["cost_no_mtd"].append(float(cost_no_mtd))
                        variant["cost_with_mtd_one"].append(float(cost_with_mtd_one))
                        variant["cost_with_mtd_two"].append(float(cost_with_mtd_two))
                        variant["stage_one_hidden"].append(float(stage_one_hidden))
                        variant["stage_one_eff"].append(float(stage_one_eff))
                        variant["stage_two_hidden"].append(float(stage_two_hidden))
                        variant["stage_two_eff"].append(float(stage_two_eff))
                        variant["residual_no_att"].append(float(residual_no_att))
                        variant["post_mtd_opf_converge"].append(bool(post_mtd_opf_converge))

                except Exception as e:
                    backend_run_error = True
                    backend_run_error_message = repr(e)
                    backend_mtd_fail = 1
                    for mode in next_load_modes:
                        variant = group_records["variants"][mode]
                        variant["backend_metric_fail"].append(True)
                        variant["backend_metric_error_message"].append(f"backend_run:{repr(e)}")
                        variant["x_ratio_stage_one"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["x_ratio_stage_two"].append(np.full(case_class.no_brh, np.nan, dtype=float))
                        variant["cost_no_mtd"].append(float("nan"))
                        variant["cost_with_mtd_one"].append(float("nan"))
                        variant["cost_with_mtd_two"].append(float("nan"))
                        variant["stage_one_hidden"].append(float("nan"))
                        variant["stage_one_eff"].append(float("nan"))
                        variant["stage_two_hidden"].append(float("nan"))
                        variant["stage_two_eff"].append(float("nan"))
                        variant["residual_no_att"].append(float("nan"))
                        variant["post_mtd_opf_converge"].append(False)

                group_records["varrho"].append(float(varrho))
                group_records["obj_one"].append(float(obj_one))
                group_records["obj_two"].append(float(obj_two))
                group_records["worst_primal"].append(float(worst_primal))
                group_records["worst_dual"].append(float(worst_dual))
                group_records["backend_run_error"].append(bool(backend_run_error))
                group_records["backend_run_error_message"].append(backend_run_error_message)
                group_records["backend_mtd_fail"].append(int(backend_mtd_fail))
                group_records["stage_one_time"].append(float(stage_one_time))
                group_records["stage_two_time"].append(float(stage_two_time))

                if (idx_ + 1) % log_every == 0:
                    alarm_count = int(np.asarray(group_records["front_end_alarm"], dtype=bool).sum())
                    print(f"[progress] group={key} samples={idx_ + 1}/{total_run} front_end_alarms={alarm_count}")

            groups[key] = group_records

    payload = {
        "metadata": {
            "case_name": sys_config["case_name"],
            "split": split,
            "total_run": total_run,
            "ang_no_list": ang_no_list,
            "ang_str_list": ang_str_list,
            "mag_no": mag_no,
            "mag_str": mag_str,
            "seed_base": seed_base,
            "is_shuffle": is_shuffle,
            "recover_input_mode": recover_input_mode,
            "next_load_modes": next_load_modes,
            "next_load_extra": next_load_extra,
            "detector_quantile": detector_quantile,
            "detector_threshold": detector_threshold,
            "recovery_error_policy": "count_as_non_trigger_non_skip",
            "mtd_mode": mtd_config["mode"],
            "varrho": float(np.sqrt(mtd_config["varrho_square"])),
            "upper_scale": float(mtd_config["upper_scale"]),
            "multi_run_no": int(mtd_config["multi_run_no"]),
            "x_facts_ratio": float(mtd_config["x_facts_ratio"]),
            "split_start_idx": int(start_idx),
            "sample_length": int(nn_setting["sample_length"]),
            "skip_backend": skip_backend,
        },
        "groups": groups,
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

"""
Clean false-alarm experiment for DDET-MTD (case14 first version)
Goal:
- Run DDD on CLEAN data (no injected attack)
- For clean false alarms (DDD alarms), run recovery to compute verify_score = ||c_recover_no_ref||_2
- Insert a verification gate between recovery and backend MTD trigger
- Compare baseline tau_verify=-1.0 vs gated tau_verify=0.021

This script is a minimal runnable adaptation of:
- evaluation_event_trigger.py  (event-trigger pipeline + metric saving style)
- evaluation_verify_score.py   (clean false-alarm verify_score construction)

Author: generated for dev3025 user (庞文博)
Date: 2026-03-24 (America/Los_Angeles)
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from configs.config import sys_config, mtd_config, save_metric
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_load_pv, load_measurement, load_dataset


def parse_args() -> argparse.Namespace:
    """
    MOD: add argparse for easy baseline/gated runs without editing code.
    Defaults are chosen for quick reproducibility; for paper-grade results, run full test loader.
    """
    parser = argparse.ArgumentParser(description="DDET-MTD clean false-alarm verification-gated event-trigger experiment")

    parser.add_argument("--tau_verify", type=float, default=0.021,
                        help="Verification gate threshold. Baseline: -1.0 (always trigger). Gated: 0.021.")
    parser.add_argument("--max_total_run", type=int, default=-1,
                        help="Max number of clean samples to scan. -1 means scan the full test loader.")
    parser.add_argument("--stop_ddd_alarm_at", type=int, default=50,
                        help="Stop once total_DDD_alarm reaches this number. -1 means no stop (scan full).")
    parser.add_argument("--seed_base", type=int, default=20260324,
                        help="Base seed for deterministic MTD multi-run init per sample (seed_base + idx).")
    parser.add_argument("--is_shuffle", action="store_true",
                        help="If set, shuffle dataloader (NOT recommended for reproducible comparison). Default: False.")
    return parser.parse_args()


def safe_int_idx(idx: Any) -> int:
    """Convert dataloader idx to int robustly."""
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def main() -> None:
    args = parse_args()

    tau_verify: float = float(args.tau_verify)
    max_total_run: Optional[int] = None if args.max_total_run < 0 else int(args.max_total_run)
    stop_ddd_alarm_at: Optional[int] = None if args.stop_ddd_alarm_at < 0 else int(args.stop_ddd_alarm_at)
    seed_base: int = int(args.seed_base)
    is_shuffle: bool = bool(args.is_shuffle)

    # -------------------------
    # Preparation
    # -------------------------
    print("==== Clean false-alarm event-trigger (verification-gated) ====")
    print("case_name:", sys_config["case_name"])
    print("tau_verify:", tau_verify)
    print("varrho:", float(np.sqrt(mtd_config["varrho_square"])))
    print("mode:", mtd_config["mode"])
    print("upper_scaling:", mtd_config["upper_scale"])
    print("max_total_run:", "FULL" if max_total_run is None else max_total_run)
    print("stop_ddd_alarm_at:", "DISABLED" if stop_ddd_alarm_at is None else stop_ddd_alarm_at)
    print("dataloader shuffle:", is_shuffle)

    # Load cases, measurement, and load (same as evaluation_event_trigger.py)
    case_class = load_case()
    z_noise_summary, v_est_summary = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()

    # MOD: force deterministic order by default (recommended). Set --is_shuffle if needed.
    test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(
        is_shuffle=is_shuffle
    )

    feature_size = len(z_noise_summary)
    test_start_idx = int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))  # same as repo scripts

    # Detector
    # NOTE: LSTM_AE instance is not directly used later, but kept to be consistent with repo scripts.
    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))

    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()

    print(f"Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}")
    print(f"Threshold: {dd_detector.ae_threshold[dd_detector.quantile_idx]}")

    # -------------------------
    # Metrics (preserve existing style + add clean-specific aggregates)
    # -------------------------
    # MOD: use a single group_key for clean scenario
    group_key = "(0,0.0)"
    print("group_key:", group_key)

    # Counts/rates (required for paper table)
    total_clean_sample: Dict[str, int] = {group_key: 0}
    total_DDD_alarm: Dict[str, int] = {group_key: 0}
    total_trigger_after_verification: Dict[str, int] = {group_key: 0}
    total_skip_by_verification: Dict[str, int] = {group_key: 0}

    false_alarm_rate: Dict[str, float] = {group_key: float("nan")}
    trigger_rate: Dict[str, float] = {group_key: float("nan")}
    skip_rate: Dict[str, float] = {group_key: float("nan")}
    useless_mtd_rate: Dict[str, float] = {group_key: float("nan")}

    # Keep the same names as evaluation_event_trigger.py for compatibility,
    # but note: on clean data, "TP_DDD" is effectively "DDD_alarm" (false alarms).
    TP_DDD: Dict[str, List[bool]] = {group_key: []}
    att_strength: Dict[str, List[float]] = {group_key: []}
    varrho_summary: Dict[str, List[float]] = {group_key: []}

    # Verification gate logs (required)
    verify_score: Dict[str, List[float]] = {group_key: []}
    verify_score_triggered: Dict[str, List[float]] = {group_key: []}
    verify_score_skipped: Dict[str, List[float]] = {group_key: []}

    trigger_after_verification: Dict[str, List[bool]] = {group_key: []}
    skip_by_verification: Dict[str, List[bool]] = {group_key: []}

    # Optional indices for debugging/traceability
    clean_alarm_idx: Dict[str, List[int]] = {group_key: []}
    clean_triggered_idx: Dict[str, List[int]] = {group_key: []}
    clean_skipped_idx: Dict[str, List[int]] = {group_key: []}

    # Recovery logs (mostly preserved)
    recover_deviation: Dict[str, List[float]] = {group_key: []}
    pre_deviation: Dict[str, List[float]] = {group_key: []}
    recovery_ite_no: Dict[str, List[int]] = {group_key: []}
    recovery_time: List[float] = []

    # MTD triggered-only logs (preserve names)
    obj_one: Dict[str, List[float]] = {group_key: []}
    obj_two: Dict[str, List[float]] = {group_key: []}
    worst_primal: Dict[str, List[float]] = {group_key: []}
    worst_dual: Dict[str, List[float]] = {group_key: []}
    fail: Dict[str, List[int]] = {group_key: []}

    x_ratio_stage_one: Dict[str, List[np.ndarray]] = {group_key: []}
    x_ratio_stage_two: Dict[str, List[np.ndarray]] = {group_key: []}

    residual_no_att: List[float] = []
    post_mtd_opf_converge: List[bool] = []
    mtd_stage_one_time: List[float] = []
    mtd_stage_two_time: List[float] = []

    # For clean scenario, effectiveness/hiddenness are not central; we map them to residual_after_MTD (no-attack) for compatibility.
    mtd_stage_one_eff: Dict[str, List[float]] = {group_key: []}
    mtd_stage_two_eff: Dict[str, List[float]] = {group_key: []}
    mtd_stage_one_hidden: Dict[str, List[float]] = {group_key: []}
    mtd_stage_two_hidden: Dict[str, List[float]] = {group_key: []}

    cost_no_mtd: Dict[str, List[float]] = {group_key: []}
    cost_with_mtd_one: Dict[str, List[float]] = {group_key: []}
    cost_with_mtd_two: Dict[str, List[float]] = {group_key: []}

    # MOD: per-alarm burden logs (include zeros for skipped alarms)
    fail_alarm: Dict[str, List[int]] = {group_key: []}
    stage_one_time_alarm: Dict[str, List[float]] = {group_key: []}
    stage_two_time_alarm: Dict[str, List[float]] = {group_key: []}
    delta_cost_one_alarm: Dict[str, List[float]] = {group_key: []}
    delta_cost_two_alarm: Dict[str, List[float]] = {group_key: []}

    # Aggregated per-alarm values (required for paper table)
    fail_per_alarm: Dict[str, float] = {group_key: float("nan")}
    stage_one_time_per_alarm: Dict[str, float] = {group_key: float("nan")}
    stage_two_time_per_alarm: Dict[str, float] = {group_key: float("nan")}
    delta_cost_one_per_alarm: Dict[str, float] = {group_key: float("nan")}
    delta_cost_two_per_alarm: Dict[str, float] = {group_key: float("nan")}

    # -------------------------
    # Iteration (clean data)
    # -------------------------
    for idx_, (idx, input, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):
        if max_total_run is not None and idx_ >= max_total_run:
            break

        idx_val = safe_int_idx(idx)
        total_clean_sample[group_key] += 1

        # format (same as repo scripts)
        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        # -------------------------
        # DDD detection on CLEAN measurement
        # MOD: no attack generation block here
        # -------------------------
        input_scale = scaler_(input)
        _, _, _, loss_recons = dd_detector.evaluate(input_scale)

        alarm = bool(loss_recons > dd_detector.ae_threshold[dd_detector.quantile_idx])
        TP_DDD[group_key].append(alarm)

        if not alarm:
            continue

        # This is a CLEAN false alarm
        total_DDD_alarm[group_key] += 1
        clean_alarm_idx[group_key].append(idx_val)

        # -------------------------
        # Recovery (reuse clean false-alarm logic from evaluation_verify_score.py)
        # -------------------------
        try:
            z_recover, v_recover, loss_recover_summary, loss_sparse_real_summary, loss_sparse_imag_summary, loss_v_mag_summary, loss_summary, recover_time_single = dd_detector.recover(
                attack_batch=input,  # clean measurement window, NOT scaled
                v_pre=v_est_pre,
                v_last=v_est_last,
            )
        except Exception as e:
            # If recovery fails, we treat it as an alarm that cannot be verified => force trigger (baseline-safe) but record zeros
            print(f"[WARN] Recovery error at idx={idx_val}: {repr(e)}")
            # mark as skipped from perspective of backend (no MTD)
            trigger_after_verification[group_key].append(False)
            skip_by_verification[group_key].append(True)
            total_skip_by_verification[group_key] += 1
            verify_score[group_key].append(float("nan"))
            verify_score_skipped[group_key].append(float("nan"))
            clean_skipped_idx[group_key].append(idx_val)

            fail_alarm[group_key].append(1)
            stage_one_time_alarm[group_key].append(0.0)
            stage_two_time_alarm[group_key].append(0.0)
            delta_cost_one_alarm[group_key].append(0.0)
            delta_cost_two_alarm[group_key].append(0.0)

            if stop_ddd_alarm_at is not None and total_DDD_alarm[group_key] >= stop_ddd_alarm_at:
                break
            continue

        recovery_time.append(float(recover_time_single))

        vang_recover = np.angle(v_recover.numpy())
        vang_pre = np.angle(v_est_pre.numpy())
        vang_true = np.angle(v_est_last.numpy())

        recover_deviation[group_key].append(float(np.linalg.norm(vang_true - vang_recover, 2)))
        pre_deviation[group_key].append(float(np.linalg.norm(vang_true - vang_pre, 2)))
        recovery_ite_no[group_key].append(int(len(loss_recover_summary)))

        # -------------------------
        # Verification score (||c_recover_no_ref||_2)
        # MOD: follow evaluation_verify_score.py clean branch
        # -------------------------
        vang_last = np.angle(v_est_last.numpy())
        c_recover = (vang_last - vang_recover)
        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
        score_ = float(np.linalg.norm(c_recover_no_ref, 2))

        verify_score[group_key].append(score_)
        att_strength[group_key].append(score_)  # keep same name as event_trigger script

        should_trigger_mtd = bool((tau_verify < 0.0) or (score_ >= tau_verify))
        trigger_after_verification[group_key].append(should_trigger_mtd)
        skip_by_verification[group_key].append(not should_trigger_mtd)

        if should_trigger_mtd:
            total_trigger_after_verification[group_key] += 1
            verify_score_triggered[group_key].append(score_)
            clean_triggered_idx[group_key].append(idx_val)
        else:
            total_skip_by_verification[group_key] += 1
            verify_score_skipped[group_key].append(score_)
            clean_skipped_idx[group_key].append(idx_val)

            # per-alarm zeros (key design point)
            fail_alarm[group_key].append(0)
            stage_one_time_alarm[group_key].append(0.0)
            stage_two_time_alarm[group_key].append(0.0)
            delta_cost_one_alarm[group_key].append(0.0)
            delta_cost_two_alarm[group_key].append(0.0)

            if stop_ddd_alarm_at is not None and total_DDD_alarm[group_key] >= stop_ddd_alarm_at:
                break
            continue

        # -------------------------
        # MTD backend (only when gate passes)
        # -------------------------
        # MOD: deterministic seed per alarm sample to keep baseline vs gated comparable even if trigger counts differ
        seed_ = seed_base + idx_val
        random.seed(seed_)
        np.random.seed(seed_ % (2**32 - 1))

        varrho_square = mtd_config["varrho_square"]

        # Use next load index for cost evaluation (consistent with repo scripts).
        # MOD: generalize the "6+1" magic number by sample_length
        next_load_idx = test_start_idx + int(nn_setting["sample_length"]) + idx_val

        try:
            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, varrho_square)
            varrho_summary[group_key].append(float(np.sqrt(mtd_optim_.varrho_square)))

            ret = mtd_optim_.multi_run()
            if isinstance(ret, tuple) and len(ret) == 10:
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
                ) = ret
            elif isinstance(ret, tuple) and len(ret) == 7:
                # compatibility if mtd_config["is_worst"] changes
                (b_mtd_one_final, b_mtd_two_final, obj_one_final, obj_two_final, stage_one_time_, stage_two_time_, is_fail) = ret
                obj_worst_primal = float("nan")
                obj_worst_dual = float("nan")
                c_worst = None
            else:
                raise RuntimeError(f"Unexpected return from multi_run(): type={type(ret)}, len={len(ret) if hasattr(ret,'__len__') else 'NA'}")

            # triggered-only logs (preserve)
            obj_one[group_key].append(float(obj_one_final))
            obj_two[group_key].append(float(obj_two_final))
            worst_primal[group_key].append(float(obj_worst_primal))
            worst_dual[group_key].append(float(obj_worst_dual))
            fail[group_key].append(int(is_fail))

            mtd_stage_one_time.append(float(stage_one_time_))
            mtd_stage_two_time.append(float(stage_two_time_))

            # Stage-one evaluation: no attack
            ok1, x_ratio1, residual1, cost_base1, cost_mtd1 = mtd_optim_.mtd_metric_no_attack(
                b_mtd=b_mtd_one_final,
                load_active=load_active[next_load_idx],
                load_reactive=load_reactive[next_load_idx],
                pv_active=pv_active_[next_load_idx],
            )

            x_ratio_stage_one[group_key].append(x_ratio1)
            cost_no_mtd[group_key].append(float(cost_base1))
            cost_with_mtd_one[group_key].append(float(cost_mtd1))

            # Map "eff/hidden" to residual_after_MTD for compatibility (clean case has no attacker)
            mtd_stage_one_eff[group_key].append(float(residual1))
            mtd_stage_one_hidden[group_key].append(float(residual1))

            # Stage-two evaluation: no attack
            ok2, x_ratio2, residual2, cost_base2, cost_mtd2 = mtd_optim_.mtd_metric_no_attack(
                b_mtd=b_mtd_two_final,
                load_active=load_active[next_load_idx],
                load_reactive=load_reactive[next_load_idx],
                pv_active=pv_active_[next_load_idx],
            )

            x_ratio_stage_two[group_key].append(x_ratio2)
            cost_with_mtd_two[group_key].append(float(cost_mtd2))

            mtd_stage_two_eff[group_key].append(float(residual2))
            mtd_stage_two_hidden[group_key].append(float(residual2))

            post_mtd_opf_converge.append(bool(ok2))
            residual_no_att.append(float(residual2))

            # per-alarm burden (key outputs)
            fail_alarm[group_key].append(int(is_fail))
            stage_one_time_alarm[group_key].append(float(stage_one_time_))
            stage_two_time_alarm[group_key].append(float(stage_two_time_))
            delta_cost_one_alarm[group_key].append(float(cost_mtd1 - cost_base1))
            delta_cost_two_alarm[group_key].append(float(cost_mtd2 - cost_base2))

        except Exception as e:
            # Make the script robust: do not crash the whole run on solver/OPF failure.
            print(f"[WARN] MTD backend error at idx={idx_val}: {repr(e)}")

            fail_alarm[group_key].append(1)
            stage_one_time_alarm[group_key].append(0.0)
            stage_two_time_alarm[group_key].append(0.0)
            delta_cost_one_alarm[group_key].append(0.0)
            delta_cost_two_alarm[group_key].append(0.0)

        if stop_ddd_alarm_at is not None and total_DDD_alarm[group_key] >= stop_ddd_alarm_at:
            break

    # -------------------------
    # Aggregate metrics for paper table
    # -------------------------
    n_clean = total_clean_sample[group_key]
    n_alarm = total_DDD_alarm[group_key]
    n_trig = total_trigger_after_verification[group_key]
    n_skip = total_skip_by_verification[group_key]

    if n_clean > 0:
        false_alarm_rate[group_key] = float(n_alarm / n_clean)
        useless_mtd_rate[group_key] = float(n_trig / n_clean)
    else:
        false_alarm_rate[group_key] = float("nan")
        useless_mtd_rate[group_key] = float("nan")

    if n_alarm > 0:
        trigger_rate[group_key] = float(n_trig / n_alarm)
        skip_rate[group_key] = float(n_skip / n_alarm)

        fail_per_alarm[group_key] = float(np.mean(np.array(fail_alarm[group_key], dtype=float)))
        stage_one_time_per_alarm[group_key] = float(np.mean(np.array(stage_one_time_alarm[group_key], dtype=float)))
        stage_two_time_per_alarm[group_key] = float(np.mean(np.array(stage_two_time_alarm[group_key], dtype=float)))
        delta_cost_one_per_alarm[group_key] = float(np.mean(np.array(delta_cost_one_alarm[group_key], dtype=float)))
        delta_cost_two_per_alarm[group_key] = float(np.mean(np.array(delta_cost_two_alarm[group_key], dtype=float)))
    else:
        trigger_rate[group_key] = float("nan")
        skip_rate[group_key] = float("nan")
        fail_per_alarm[group_key] = float("nan")
        stage_one_time_per_alarm[group_key] = float("nan")
        stage_two_time_per_alarm[group_key] = float("nan")
        delta_cost_one_per_alarm[group_key] = float("nan")
        delta_cost_two_per_alarm[group_key] = float("nan")

    print("\n==== Summary (clean false-alarm) ====")
    print("total_clean_sample:", n_clean)
    print("total_DDD_alarm:", n_alarm, "false_alarm_rate:", false_alarm_rate[group_key])
    print("total_trigger_after_verification:", n_trig, "trigger_rate:", trigger_rate[group_key])
    print("total_skip_by_verification:", n_skip, "skip_rate:", skip_rate[group_key])
    print("useless_mtd_rate (per clean sample):", useless_mtd_rate[group_key])
    print("fail_per_alarm:", fail_per_alarm[group_key])
    print("stage_one_time_per_alarm:", stage_one_time_per_alarm[group_key])
    print("stage_two_time_per_alarm:", stage_two_time_per_alarm[group_key])
    print("delta_cost_one_per_alarm:", delta_cost_one_per_alarm[group_key])
    print("delta_cost_two_per_alarm:", delta_cost_two_per_alarm[group_key])

    # -------------------------
    # Save (preserve metric style + add new fields)
    # -------------------------
    address = (
        f"metric/{sys_config['case_name']}/"
        f"metric_event_trigger_clean_tau_{tau_verify}_"
        f"mode_{mtd_config['mode']}_{round(np.sqrt(mtd_config['varrho_square']), 5)}_{mtd_config['upper_scale']}.npy"
    )

    save_metric(
        address=address,

        # Meta / settings
        group_key=group_key,
        tau_verify=tau_verify,
        max_total_run=max_total_run,
        stop_ddd_alarm_at=stop_ddd_alarm_at,
        seed_base=seed_base,
        is_shuffle=is_shuffle,

        # Required counts/rates
        total_clean_sample=total_clean_sample,
        total_DDD_alarm=total_DDD_alarm,
        total_trigger_after_verification=total_trigger_after_verification,
        total_skip_by_verification=total_skip_by_verification,
        false_alarm_rate=false_alarm_rate,
        trigger_rate=trigger_rate,
        skip_rate=skip_rate,
        useless_mtd_rate=useless_mtd_rate,

        # Required per-alarm aggregated burdens
        fail_per_alarm=fail_per_alarm,
        stage_one_time_per_alarm=stage_one_time_per_alarm,
        stage_two_time_per_alarm=stage_two_time_per_alarm,
        delta_cost_one_per_alarm=delta_cost_one_per_alarm,
        delta_cost_two_per_alarm=delta_cost_two_per_alarm,

        # Per-alarm raw logs (auditable)
        fail_alarm=fail_alarm,
        stage_one_time_alarm=stage_one_time_alarm,
        stage_two_time_alarm=stage_two_time_alarm,
        delta_cost_one_alarm=delta_cost_one_alarm,
        delta_cost_two_alarm=delta_cost_two_alarm,

        # DDD / gate logs
        TP_DDD=TP_DDD,
        verify_score=verify_score,
        verify_score_triggered=verify_score_triggered,
        verify_score_skipped=verify_score_skipped,
        trigger_after_verification=trigger_after_verification,
        skip_by_verification=skip_by_verification,
        att_strength=att_strength,
        varrho_summary=varrho_summary,

        # Recovery logs
        recover_deviation=recover_deviation,
        pre_deviation=pre_deviation,
        recovery_ite_no=recovery_ite_no,
        recovery_time=recovery_time,

        # Optional idx logs
        clean_alarm_idx=clean_alarm_idx,
        clean_triggered_idx=clean_triggered_idx,
        clean_skipped_idx=clean_skipped_idx,

        # MTD logs (triggered-only, preserved names)
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
    )

    print("\nSaved:", address)


if __name__ == "__main__":
    main()

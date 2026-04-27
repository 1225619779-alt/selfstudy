from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import mtd_config, save_metric, sys_config
from configs.nn_setting import nn_setting
from evaluation_mixed_timeline import (
    iter_with_offset,
    parse_schedule,
    safe_int_idx,
    unpack_multi_run,
)
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_dataset, load_load_pv, load_measurement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpointed isolated copy of evaluation_mixed_timeline.py for Gate 6c profiling."
    )
    parser.add_argument("--tau_verify", type=float, default=-1.0)
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--seed_base", type=int, default=20260711)
    parser.add_argument("--is_shuffle", action="store_true")
    parser.add_argument("--next_load_extra", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--partial_output", type=str, required=True)
    parser.add_argument("--runtime_jsonl", type=str, required=True)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--max_wall_seconds", type=float, default=43200.0)
    return parser.parse_args()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_save_metric(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_metric(address=str(tmp), **payload)
    tmp_npy = Path(str(tmp) + ".npy") if not str(tmp).endswith(".npy") else tmp
    if tmp_npy != tmp and tmp_npy.exists():
        os.replace(tmp_npy, path)
    else:
        os.replace(tmp, path)


def result_payload(
    *,
    args: argparse.Namespace,
    schedule: List[Any],
    threshold: float,
    dd_detector: Evaluation,
    total_steps: int,
    completed_steps: int,
    status: str,
    partial_reason: str,
    test_start_idx: int,
    timeline_step: List[int],
    segment_id: List[int],
    idx_summary: List[int],
    scenario_label: List[str],
    scenario_code: List[int],
    is_attack_step: List[int],
    ang_no_summary: List[int],
    ang_str_summary: List[float],
    ddd_alarm: List[int],
    verify_score: List[float],
    trigger_after_gate: List[int],
    skip_by_gate: List[int],
    recover_fail: List[int],
    backend_fail: List[int],
    stage_one_time: List[float],
    stage_two_time: List[float],
    delta_cost_one: List[float],
    delta_cost_two: List[float],
    cumulative_stage_time: List[float],
    cumulative_delta_cost: List[float],
    ddd_loss_recons: List[float],
    recovery_time: List[float],
    note: List[str],
    segment_meta: List[Dict[str, Any]],
    cum_time_total: float,
    cum_cost_total: float,
) -> Dict[str, Any]:
    scenario_breakdown: Dict[str, Dict[str, int]] = {}
    for label in sorted(set(scenario_label), key=scenario_label.index):
        mask = np.array([lab == label for lab in scenario_label], dtype=bool)
        scenario_breakdown[label] = {
            "count": int(mask.sum()),
            "DDD_alarm": int(np.sum(np.asarray(ddd_alarm, dtype=int)[mask])),
            "trigger_after_gate": int(np.sum(np.asarray(trigger_after_gate, dtype=int)[mask])),
            "skip_by_gate": int(np.sum(np.asarray(skip_by_gate, dtype=int)[mask])),
            "recover_fail": int(np.sum(np.asarray(recover_fail, dtype=int)[mask])),
        }
    return {
        "status": status,
        "partial_reason": partial_reason,
        "completed_steps": int(completed_steps),
        "total_steps_requested": int(total_steps),
        "tau_verify": float(args.tau_verify),
        "schedule_spec": args.schedule,
        "schedule_segments": segment_meta,
        "start_offset": int(args.start_offset),
        "seed_base": int(args.seed_base),
        "is_shuffle": bool(args.is_shuffle),
        "next_load_extra": int(args.next_load_extra),
        "case_name": sys_config["case_name"],
        "model_path": nn_setting["model_path"],
        "test_start_idx": int(test_start_idx),
        "ddd_quantile": dd_detector.quantile[dd_detector.quantile_idx],
        "ddd_threshold": float(threshold),
        "timeline_step": timeline_step,
        "segment_id": segment_id,
        "idx_summary": idx_summary,
        "scenario_label": scenario_label,
        "scenario_code": scenario_code,
        "is_attack_step": is_attack_step,
        "ang_no_summary": ang_no_summary,
        "ang_str_summary": ang_str_summary,
        "ddd_alarm": ddd_alarm,
        "verify_score": verify_score,
        "trigger_after_gate": trigger_after_gate,
        "skip_by_gate": skip_by_gate,
        "recover_fail": recover_fail,
        "backend_fail": backend_fail,
        "stage_one_time": stage_one_time,
        "stage_two_time": stage_two_time,
        "delta_cost_one": delta_cost_one,
        "delta_cost_two": delta_cost_two,
        "cumulative_stage_time": cumulative_stage_time,
        "cumulative_delta_cost": cumulative_delta_cost,
        "ddd_loss_recons": ddd_loss_recons,
        "recovery_time": recovery_time,
        "note": note,
        "summary": {
            "total_steps": int(completed_steps),
            "total_clean_steps": int(np.sum(1 - np.asarray(is_attack_step, dtype=int))) if is_attack_step else 0,
            "total_attack_steps": int(np.sum(np.asarray(is_attack_step, dtype=int))) if is_attack_step else 0,
            "total_DDD_alarm": int(np.sum(np.asarray(ddd_alarm, dtype=int))) if ddd_alarm else 0,
            "total_trigger_after_gate": int(np.sum(np.asarray(trigger_after_gate, dtype=int))) if trigger_after_gate else 0,
            "total_skip_by_gate": int(np.sum(np.asarray(skip_by_gate, dtype=int))) if skip_by_gate else 0,
            "total_recover_fail": int(np.sum(np.asarray(recover_fail, dtype=int))) if recover_fail else 0,
            "total_backend_fail": int(np.sum(np.asarray(backend_fail, dtype=int))) if backend_fail else 0,
            "final_cumulative_stage_time": float(cum_time_total),
            "final_cumulative_delta_cost": float(cum_cost_total),
        },
        "scenario_breakdown": scenario_breakdown,
    }


def append_runtime(path: str | Path, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if sys_config["case_name"] != "case39":
        raise RuntimeError(f"Gate 6c requires DDET_CASE_NAME=case39, got {sys_config['case_name']}")

    start_wall = time.time()
    schedule = parse_schedule(args.schedule)
    total_steps = int(sum(seg.length for seg in schedule))

    print("==== checkpointed_evaluation_mixed_timeline ====")
    print("case_name:", sys_config["case_name"])
    print("tau_verify:", args.tau_verify)
    print("schedule:", args.schedule)
    print("start_offset:", args.start_offset)
    print("seed_base:", args.seed_base)
    print("max_wall_seconds:", args.max_wall_seconds)
    print("output:", args.output)
    print("partial_output:", args.partial_output)
    print("runtime_jsonl:", args.runtime_jsonl)
    print("total_steps:", total_steps)

    case_class = load_case()
    z_noise_summary, _ = load_measurement()
    load_active, load_reactive, pv_active_, _pv_reactive = load_load_pv()
    _, test_dataloader_unscaled, _, _ = load_dataset(is_shuffle=bool(args.is_shuffle))

    feature_size = len(z_noise_summary)
    test_start_idx = int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))
    _ = lstm_ae
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()
    threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])

    print(f"Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}")
    print(f"Threshold: {threshold}")

    iterator = iter_with_offset(test_dataloader_unscaled, int(args.start_offset))

    timeline_step: List[int] = []
    segment_id: List[int] = []
    idx_summary: List[int] = []
    scenario_label: List[str] = []
    scenario_code: List[int] = []
    is_attack_step: List[int] = []
    ang_no_summary: List[int] = []
    ang_str_summary: List[float] = []
    ddd_alarm: List[int] = []
    verify_score: List[float] = []
    trigger_after_gate: List[int] = []
    skip_by_gate: List[int] = []
    recover_fail: List[int] = []
    backend_fail: List[int] = []
    stage_one_time: List[float] = []
    stage_two_time: List[float] = []
    delta_cost_one: List[float] = []
    delta_cost_two: List[float] = []
    cumulative_stage_time: List[float] = []
    cumulative_delta_cost: List[float] = []
    ddd_loss_recons: List[float] = []
    recovery_time: List[float] = []
    note: List[str] = []
    segment_meta: List[Dict[str, Any]] = []
    cum_time_total = 0.0
    cum_cost_total = 0.0
    current_step = 0

    def save_checkpoint(status: str, reason: str) -> None:
        payload = result_payload(
            args=args,
            schedule=schedule,
            threshold=threshold,
            dd_detector=dd_detector,
            total_steps=total_steps,
            completed_steps=current_step,
            status=status,
            partial_reason=reason,
            test_start_idx=test_start_idx,
            timeline_step=timeline_step,
            segment_id=segment_id,
            idx_summary=idx_summary,
            scenario_label=scenario_label,
            scenario_code=scenario_code,
            is_attack_step=is_attack_step,
            ang_no_summary=ang_no_summary,
            ang_str_summary=ang_str_summary,
            ddd_alarm=ddd_alarm,
            verify_score=verify_score,
            trigger_after_gate=trigger_after_gate,
            skip_by_gate=skip_by_gate,
            recover_fail=recover_fail,
            backend_fail=backend_fail,
            stage_one_time=stage_one_time,
            stage_two_time=stage_two_time,
            delta_cost_one=delta_cost_one,
            delta_cost_two=delta_cost_two,
            cumulative_stage_time=cumulative_stage_time,
            cumulative_delta_cost=cumulative_delta_cost,
            ddd_loss_recons=ddd_loss_recons,
            recovery_time=recovery_time,
            note=note,
            segment_meta=segment_meta,
            cum_time_total=cum_time_total,
            cum_cost_total=cum_cost_total,
        )
        atomic_save_metric(args.partial_output, **payload)

    progress = tqdm(total=total_steps)
    for seg_id, seg in enumerate(schedule):
        seg_start = current_step
        for _ in range(seg.length):
            step_wall = time.time()
            try:
                idx, input_batch, v_est_pre, v_est_last = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Test dataloader exhausted before timeline finished. Requested total_steps={total_steps}, start_offset={args.start_offset}."
                ) from exc

            idx_val = safe_int_idx(idx)
            v_est_pre = v_est_pre.flatten()
            v_est_last = v_est_last.flatten()
            this_ddd_alarm = 0
            this_verify = float("nan")
            this_trigger = 0
            this_skip = 0
            this_recover_fail = 0
            this_backend_fail = 0
            this_t1 = 0.0
            this_t2 = 0.0
            this_c1 = 0.0
            this_c2 = 0.0
            this_ddd_loss = float("nan")
            this_recovery_time = 0.0
            this_note = ""

            if seg.kind == "clean":
                input_scale = scaler_(input_batch)
                _, _, _, loss_recons = dd_detector.evaluate(input_scale)
                this_ddd_loss = float(loss_recons)
                if this_ddd_loss > threshold:
                    this_ddd_alarm = 1
                    try:
                        (
                            _z_recover,
                            v_recover,
                            _loss_recover_summary,
                            _loss_sparse_real_summary,
                            _loss_sparse_imag_summary,
                            _loss_v_mag_summary,
                            _loss_summary,
                            recover_time_single,
                        ) = dd_detector.recover(attack_batch=input_batch, v_pre=v_est_pre, v_last=v_est_last)
                        this_recovery_time = float(recover_time_single)
                        vang_last = np.angle(v_est_last.numpy())
                        vang_recover = np.angle(v_recover.numpy())
                        c_recover = vang_last - vang_recover
                        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                        this_verify = float(np.linalg.norm(c_recover_no_ref, 2))
                        should_trigger_mtd = bool((float(args.tau_verify) < 0.0) or (this_verify >= float(args.tau_verify)))
                        if should_trigger_mtd:
                            this_trigger = 1
                            next_load_idx = test_start_idx + int(nn_setting["sample_length"]) + idx_val + int(args.next_load_extra)
                            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, mtd_config["varrho_square"])
                            ret = unpack_multi_run(mtd_optim_.multi_run())
                            (
                                b_mtd_one_final,
                                b_mtd_two_final,
                                _obj_one_final,
                                _obj_two_final,
                                _obj_worst_primal,
                                _obj_worst_dual,
                                _c_worst,
                                stage_one_time_,
                                stage_two_time_,
                                is_fail,
                            ) = ret
                            this_t1 = float(stage_one_time_)
                            this_t2 = float(stage_two_time_)
                            this_backend_fail = int(is_fail)
                            _ok1, _x_ratio1, _residual1, cost_base1, cost_mtd1 = mtd_optim_.mtd_metric_no_attack(
                                b_mtd=b_mtd_one_final,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                            )
                            _ok2, _x_ratio2, _residual2, cost_base2, cost_mtd2 = mtd_optim_.mtd_metric_no_attack(
                                b_mtd=b_mtd_two_final,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                            )
                            this_c1 = float(cost_mtd1 - cost_base1)
                            this_c2 = float(cost_mtd2 - cost_base2)
                        else:
                            this_skip = 1
                    except Exception as exc:
                        this_recover_fail = 1
                        this_backend_fail = 1
                        this_skip = 1
                        this_note = f"clean_recover_or_backend_error: {repr(exc)}"
            else:
                seed_ = int(args.seed_base) + 100000 * seg.ang_no + int(round(1000 * seg.ang_str)) + idx_val
                random.seed(seed_)
                np.random.seed(seed_ % (2**32 - 1))
                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                    z_noise=input_batch,
                    v_est_last=v_est_last,
                    ang_no=seg.ang_no,
                    mag_no=0,
                    ang_str=seg.ang_str,
                    mag_str=0.0,
                )
                v_att_est_last = torch.from_numpy(v_att_est_last)
                z_att_noise_scale = scaler_(z_att_noise)
                _, _, _, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                this_ddd_loss = float(loss_recons)
                if this_ddd_loss > threshold:
                    this_ddd_alarm = 1
                    try:
                        (
                            _z_recover,
                            v_recover,
                            _loss_recover_summary,
                            _loss_sparse_real_summary,
                            _loss_sparse_imag_summary,
                            _loss_v_mag_summary,
                            _loss_summary,
                            recover_time_single,
                        ) = dd_detector.recover(attack_batch=input_batch, v_pre=v_est_pre, v_last=v_att_est_last)
                        this_recovery_time = float(recover_time_single)
                        vang_att = np.angle(v_att_est_last.numpy())
                        vang_true = np.angle(v_est_last.numpy())
                        vang_recover = np.angle(v_recover.numpy())
                        c_true = vang_att - vang_true
                        c_recover = vang_att - vang_recover
                        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                        this_verify = float(np.linalg.norm(c_recover_no_ref, 2))
                        should_trigger_mtd = bool((float(args.tau_verify) < 0.0) or (this_verify >= float(args.tau_verify)))
                        if should_trigger_mtd:
                            this_trigger = 1
                            next_load_idx = test_start_idx + int(nn_setting["sample_length"]) + idx_val + int(args.next_load_extra)
                            mtd_optim_ = mtd_optim(case_class, v_est_last.numpy(), c_recover_no_ref, mtd_config["varrho_square"])
                            ret = unpack_multi_run(mtd_optim_.multi_run())
                            (
                                b_mtd_one_final,
                                b_mtd_two_final,
                                _obj_one_final,
                                _obj_two_final,
                                _obj_worst_primal,
                                _obj_worst_dual,
                                _c_worst,
                                stage_one_time_,
                                stage_two_time_,
                                is_fail,
                            ) = ret
                            this_t1 = float(stage_one_time_)
                            this_t2 = float(stage_two_time_)
                            this_backend_fail = int(is_fail)
                            (
                                _is_converged1,
                                _x_ratio1,
                                cost_base1,
                                cost_mtd1,
                                _residual_normal1,
                                _residual_hid1,
                                _residual_eff1,
                            ) = mtd_optim_.mtd_metric_with_attack(
                                b_mtd=b_mtd_one_final,
                                c_actual=c_true,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                                mode=mtd_config["mode"],
                            )
                            (
                                _is_converged2,
                                _x_ratio2,
                                cost_base2,
                                cost_mtd2,
                                _residual_normal2,
                                _residual_hid2,
                                _residual_eff2,
                            ) = mtd_optim_.mtd_metric_with_attack(
                                b_mtd=b_mtd_two_final,
                                c_actual=c_true,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                                mode=mtd_config["mode"],
                            )
                            this_c1 = float(cost_mtd1 - cost_base1)
                            this_c2 = float(cost_mtd2 - cost_base2)
                        else:
                            this_skip = 1
                    except Exception as exc:
                        this_recover_fail = 1
                        this_backend_fail = 1
                        this_skip = 1
                        this_note = f"attack_recover_or_backend_error: {repr(exc)}"

            cum_time_total += this_t1 + this_t2
            cum_cost_total += this_c1 + this_c2
            timeline_step.append(current_step)
            segment_id.append(seg_id)
            idx_summary.append(idx_val)
            scenario_label.append(seg.label)
            scenario_code.append(seg.scenario_code)
            is_attack_step.append(1 if seg.kind == "attack" else 0)
            ang_no_summary.append(seg.ang_no)
            ang_str_summary.append(float(seg.ang_str))
            ddd_alarm.append(this_ddd_alarm)
            verify_score.append(this_verify)
            trigger_after_gate.append(this_trigger)
            skip_by_gate.append(this_skip)
            recover_fail.append(this_recover_fail)
            backend_fail.append(this_backend_fail)
            stage_one_time.append(this_t1)
            stage_two_time.append(this_t2)
            delta_cost_one.append(this_c1)
            delta_cost_two.append(this_c2)
            cumulative_stage_time.append(cum_time_total)
            cumulative_delta_cost.append(cum_cost_total)
            ddd_loss_recons.append(this_ddd_loss)
            recovery_time.append(this_recovery_time)
            note.append(this_note)

            current_step += 1
            elapsed_step = time.time() - step_wall
            append_runtime(
                args.runtime_jsonl,
                {
                    "step": current_step,
                    "segment_id": seg_id,
                    "label": seg.label,
                    "is_attack": 1 if seg.kind == "attack" else 0,
                    "ddd_alarm": this_ddd_alarm,
                    "trigger_after_gate": this_trigger,
                    "backend_fail": this_backend_fail,
                    "recover_fail": this_recover_fail,
                    "elapsed_step_seconds": elapsed_step,
                    "elapsed_total_seconds": time.time() - start_wall,
                },
            )
            if current_step % max(int(args.checkpoint_every), 1) == 0:
                save_checkpoint("partial", "periodic_checkpoint")
            progress.update(1)
            if time.time() - start_wall >= float(args.max_wall_seconds):
                save_checkpoint("partial_timeout", "max_wall_seconds_reached")
                progress.close()
                print(f"Stopped after max_wall_seconds at completed_steps={current_step}")
                return 124

        seg_end = current_step - 1
        segment_meta.append(
            {
                "segment_id": seg_id,
                "kind": seg.kind,
                "label": seg.label,
                "length": seg.length,
                "start_step": seg_start,
                "end_step": seg_end,
                "ang_no": seg.ang_no,
                "ang_str": seg.ang_str,
                "scenario_code": seg.scenario_code,
            }
        )

    progress.close()
    payload = result_payload(
        args=args,
        schedule=schedule,
        threshold=threshold,
        dd_detector=dd_detector,
        total_steps=total_steps,
        completed_steps=current_step,
        status="complete",
        partial_reason="complete",
        test_start_idx=test_start_idx,
        timeline_step=timeline_step,
        segment_id=segment_id,
        idx_summary=idx_summary,
        scenario_label=scenario_label,
        scenario_code=scenario_code,
        is_attack_step=is_attack_step,
        ang_no_summary=ang_no_summary,
        ang_str_summary=ang_str_summary,
        ddd_alarm=ddd_alarm,
        verify_score=verify_score,
        trigger_after_gate=trigger_after_gate,
        skip_by_gate=skip_by_gate,
        recover_fail=recover_fail,
        backend_fail=backend_fail,
        stage_one_time=stage_one_time,
        stage_two_time=stage_two_time,
        delta_cost_one=delta_cost_one,
        delta_cost_two=delta_cost_two,
        cumulative_stage_time=cumulative_stage_time,
        cumulative_delta_cost=cumulative_delta_cost,
        ddd_loss_recons=ddd_loss_recons,
        recovery_time=recovery_time,
        note=note,
        segment_meta=segment_meta,
        cum_time_total=cum_time_total,
        cum_cost_total=cum_cost_total,
    )
    atomic_save_metric(args.output, **payload)
    atomic_save_metric(args.partial_output, **payload)
    print("Saved complete output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

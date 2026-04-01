from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from configs.config import mtd_config, save_metric, sys_config
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from optim.optimization import mtd_optim
from utils.load_data import load_case, load_load_pv, load_measurement, load_dataset


@dataclass
class Segment:
    kind: str  # 'clean' or 'attack'
    length: int
    label: str
    scenario_code: int  # 0 clean, 1 weak, 2 medium, 3 strong
    ang_no: int = 0
    ang_str: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mixed clean/attack timeline case study for the verification-gated DDET-MTD pipeline."
    )
    parser.add_argument(
        "--tau_verify",
        type=float,
        default=0.021,
        help="Verification threshold. Default keeps the current main operating point.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="clean:80;att-1-0.2:30;clean:40;att-2-0.2:30;clean:40;att-3-0.3:30;clean:80",
        help=(
            "Semicolon-separated timeline specification. "
            "Use clean:N or att-angNo-angStr:N, e.g. clean:80;att-1-0.2:30;clean:40."
        ),
    )
    parser.add_argument(
        "--start_offset",
        type=int,
        default=0,
        help="How many test-loader samples to skip before starting the timeline.",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=20260331,
        help="Base seed. Attack steps follow the repo's per-sample deterministic seeding style.",
    )
    parser.add_argument(
        "--is_shuffle",
        action="store_true",
        help="Shuffle the dataloader. Keep this OFF for reproducible timelines.",
    )
    parser.add_argument(
        "--next_load_extra",
        type=int,
        default=0,
        help=(
            "Extra offset added to next_load_idx. Default 0 follows the cleaned-up indexing "
            "used in evaluation_event_trigger_clean.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"metric/{sys_config['case_name']}/metric_mixed_timeline_tau_0.021.npy",
        help="Output .npy path.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def parse_schedule(spec: str) -> List[Segment]:
    segments: List[Segment] = []
    tokens = [tok.strip() for tok in spec.split(";") if tok.strip()]
    if not tokens:
        raise ValueError("Empty schedule.")

    for token in tokens:
        try:
            head, length_s = token.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid segment '{token}'. Expected form clean:N or att-angNo-angStr:N") from exc

        length = int(length_s)
        if length <= 0:
            raise ValueError(f"Segment length must be positive: {token}")

        if head == "clean":
            segments.append(Segment(kind="clean", length=length, label="clean", scenario_code=0))
            continue

        if head.startswith("att-"):
            parts = head.split("-")
            if len(parts) != 3:
                raise ValueError(f"Invalid attack segment '{token}'. Use att-angNo-angStr:N")
            ang_no = int(parts[1])
            ang_str = float(parts[2])
            if ang_no == 1 and ang_str <= 0.2:
                code = 1
                label = f"weak({ang_no},{ang_str})"
            elif ang_no <= 2 and ang_str <= 0.2:
                code = 2
                label = f"medium({ang_no},{ang_str})"
            else:
                code = 3
                label = f"strong({ang_no},{ang_str})"
            segments.append(
                Segment(
                    kind="attack",
                    length=length,
                    label=label,
                    scenario_code=code,
                    ang_no=ang_no,
                    ang_str=ang_str,
                )
            )
            continue

        raise ValueError(f"Unknown segment head '{head}'.")

    return segments


def iter_with_offset(iterable: Iterable[Any], offset: int) -> Iterator[Any]:
    iterator = iter(iterable)
    for _ in range(offset):
        next(iterator)
    return iterator


def unpack_multi_run(ret: Any) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, Any, float, float, int]:
    if isinstance(ret, tuple) and len(ret) == 10:
        return ret  # type: ignore[return-value]
    raise RuntimeError(
        f"Unexpected return from multi_run(): type={type(ret)}, len={len(ret) if hasattr(ret, '__len__') else 'NA'}"
    )


def main() -> None:
    args = parse_args()
    tau_verify = float(args.tau_verify)
    schedule = parse_schedule(args.schedule)
    start_offset = int(args.start_offset)
    seed_base = int(args.seed_base)
    is_shuffle = bool(args.is_shuffle)
    next_load_extra = int(args.next_load_extra)
    output_path = str(args.output)

    print("==== evaluation_mixed_timeline ====")
    print("case_name:", sys_config["case_name"])
    print("tau_verify:", tau_verify)
    print("schedule:", args.schedule)
    print("start_offset:", start_offset)
    print("is_shuffle:", is_shuffle)
    print("next_load_extra:", next_load_extra)
    print("output:", output_path)

    total_steps = int(sum(seg.length for seg in schedule))
    print("total_steps:", total_steps)

    case_class = load_case()
    z_noise_summary, _ = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
    _, test_dataloader_unscaled, _, _ = load_dataset(is_shuffle=is_shuffle)

    feature_size = len(z_noise_summary)
    test_start_idx = int(feature_size * (nn_setting["train_prop"] + nn_setting["valid_prop"]))

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(
        torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"]))
    )
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()
    threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])

    print(f"Quantile: {dd_detector.quantile[dd_detector.quantile_idx]}")
    print(f"Threshold: {threshold}")

    iterator = iter_with_offset(test_dataloader_unscaled, start_offset)

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
    progress = tqdm(total=total_steps)

    for seg_id, seg in enumerate(schedule):
        seg_start = current_step

        for _ in range(seg.length):
            try:
                idx, input_batch, v_est_pre, v_est_last = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Test dataloader exhausted before timeline finished. Requested total_steps={total_steps}, start_offset={start_offset}."
                ) from exc

            idx_val = safe_int_idx(idx)
            v_est_pre = v_est_pre.flatten()
            v_est_last = v_est_last.flatten()

            # defaults
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
                        ) = dd_detector.recover(
                            attack_batch=input_batch,
                            v_pre=v_est_pre,
                            v_last=v_est_last,
                        )
                        this_recovery_time = float(recover_time_single)

                        vang_last = np.angle(v_est_last.numpy())
                        vang_recover = np.angle(v_recover.numpy())
                        c_recover = vang_last - vang_recover
                        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                        this_verify = float(np.linalg.norm(c_recover_no_ref, 2))

                        should_trigger_mtd = bool((tau_verify < 0.0) or (this_verify >= tau_verify))
                        if should_trigger_mtd:
                            this_trigger = 1
                            next_load_idx = test_start_idx + int(nn_setting["sample_length"]) + idx_val + next_load_extra
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

                            ok1, _x_ratio1, _residual1, cost_base1, cost_mtd1 = mtd_optim_.mtd_metric_no_attack(
                                b_mtd=b_mtd_one_final,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                            )
                            ok2, _x_ratio2, _residual2, cost_base2, cost_mtd2 = mtd_optim_.mtd_metric_no_attack(
                                b_mtd=b_mtd_two_final,
                                load_active=load_active[next_load_idx],
                                load_reactive=load_reactive[next_load_idx],
                                pv_active=pv_active_[next_load_idx],
                            )
                            _ = ok1
                            _ = ok2
                            this_c1 = float(cost_mtd1 - cost_base1)
                            this_c2 = float(cost_mtd2 - cost_base2)
                        else:
                            this_skip = 1
                    except Exception as exc:
                        this_recover_fail = 1
                        this_backend_fail = 1
                        this_skip = 1
                        this_note = f"clean_recover_or_backend_error: {repr(exc)}"
                # else: no DDD alarm, keep all-zero backend actions

            else:  # attack step
                seed_ = seed_base + 100000 * seg.ang_no + int(round(1000 * seg.ang_str)) + idx_val
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
                        ) = dd_detector.recover(
                            attack_batch=input_batch,
                            v_pre=v_est_pre,
                            v_last=v_att_est_last,
                        )
                        this_recovery_time = float(recover_time_single)

                        vang_att = np.angle(v_att_est_last.numpy())
                        vang_true = np.angle(v_est_last.numpy())
                        vang_recover = np.angle(v_recover.numpy())
                        c_true = vang_att - vang_true
                        c_recover = vang_att - vang_recover
                        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                        this_verify = float(np.linalg.norm(c_recover_no_ref, 2))

                        should_trigger_mtd = bool((tau_verify < 0.0) or (this_verify >= tau_verify))
                        if should_trigger_mtd:
                            this_trigger = 1
                            next_load_idx = test_start_idx + int(nn_setting["sample_length"]) + idx_val + next_load_extra
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
                # else: no DDD alarm

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
            progress.update(1)

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

    result = {
        "tau_verify": tau_verify,
        "schedule_spec": args.schedule,
        "schedule_segments": segment_meta,
        "start_offset": start_offset,
        "seed_base": seed_base,
        "is_shuffle": is_shuffle,
        "next_load_extra": next_load_extra,
        "ddd_quantile": dd_detector.quantile[dd_detector.quantile_idx],
        "ddd_threshold": threshold,
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
            "total_steps": total_steps,
            "total_clean_steps": int(np.sum(1 - np.asarray(is_attack_step, dtype=int))),
            "total_attack_steps": int(np.sum(np.asarray(is_attack_step, dtype=int))),
            "total_DDD_alarm": int(np.sum(np.asarray(ddd_alarm, dtype=int))),
            "total_trigger_after_gate": int(np.sum(np.asarray(trigger_after_gate, dtype=int))),
            "total_skip_by_gate": int(np.sum(np.asarray(skip_by_gate, dtype=int))),
            "total_recover_fail": int(np.sum(np.asarray(recover_fail, dtype=int))),
            "total_backend_fail": int(np.sum(np.asarray(backend_fail, dtype=int))),
            "final_cumulative_stage_time": float(cum_time_total),
            "final_cumulative_delta_cost": float(cum_cost_total),
        },
        "scenario_breakdown": scenario_breakdown,
    }

    ensure_parent(output_path)
    save_metric(address=output_path, **result)

    print("\n==== Summary ====")
    print("total_steps:", result["summary"]["total_steps"])
    print("total_clean_steps:", result["summary"]["total_clean_steps"])
    print("total_attack_steps:", result["summary"]["total_attack_steps"])
    print("total_DDD_alarm:", result["summary"]["total_DDD_alarm"])
    print("total_trigger_after_gate:", result["summary"]["total_trigger_after_gate"])
    print("total_skip_by_gate:", result["summary"]["total_skip_by_gate"])
    print("total_recover_fail:", result["summary"]["total_recover_fail"])
    print("total_backend_fail:", result["summary"]["total_backend_fail"])
    print("final_cumulative_stage_time:", result["summary"]["final_cumulative_stage_time"])
    print("final_cumulative_delta_cost:", result["summary"]["final_cumulative_delta_cost"])
    print("\n---- Scenario breakdown ----")
    for key, value in scenario_breakdown.items():
        print(key, value)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

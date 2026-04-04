from __future__ import annotations

"""
Validation-only joint operating-point selection for the recovery-aware gate.

What this script does
---------------------
1. Collect CLEAN false-alarm verification scores on the VALIDATION split.
2. Collect ATTACK true-alarm verification scores on the VALIDATION split.
3. Sweep candidate tau values (validation-only).
4. Select tau_main and tau_strict under joint ARR constraints.

Default rule (editable via CLI)
-------------------------------
Main operating point:
  maximize tau subject to
    - overall ARR >= 0.90
    - ARR >= 0.95 for protected groups: (2,0.3), (3,0.2), (3,0.3)

Stricter operating point:
  maximize tau subject to
    - overall ARR >= 0.85
    - ARR >= 0.90 for protected groups: (2,0.3), (3,0.2), (3,0.3)

This is intentionally validation-only. No test split is used here.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from configs.config import mtd_config, save_metric, sys_config
from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from utils.load_data import load_case, load_dataset, load_load_pv, load_measurement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-only joint tau selection under ARR constraints.")
    p.add_argument("--total_run_attack", type=int, default=50,
                   help="Validation samples per attack group. Default 50 to match the current attack-supporting protocol.")
    p.add_argument("--is_shuffle", action="store_true", help="Shuffle dataloader. Default False.")
    p.add_argument("--seed_base", type=int, default=20260403)
    p.add_argument("--ang_no_list", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--ang_str_list", type=float, nargs="*", default=[0.2, 0.3])
    p.add_argument("--mag_no", type=int, default=0)
    p.add_argument("--mag_str", type=float, default=0.0)
    p.add_argument("--recover_input_mode", choices=["repo_compatible", "attacked_measurement"], default="repo_compatible")
    p.add_argument("--next_load_extra", type=int, default=7)
    p.add_argument("--main_overall_arr_min", type=float, default=0.90)
    p.add_argument("--main_protected_arr_min", type=float, default=0.95)
    p.add_argument("--strict_overall_arr_min", type=float, default=0.85)
    p.add_argument("--strict_protected_arr_min", type=float, default=0.90)
    p.add_argument("--protected_groups", nargs="*", default=["(2,0.3)", "(3,0.2)", "(3,0.3)"],
                   help="Groups that should retain high ARR. Use keys like '(2,0.3)'.")
    p.add_argument("--candidate_quantiles", nargs="*", type=float, default=[],
                   help="Optional clean-score quantiles to report in summary (e.g. 0.8 0.85 0.9 0.95).")
    p.add_argument("--output_dir", type=str, default="",
                   help="Output directory. Default: metric/<case>/tau_selection_joint_valid")
    return p.parse_args()


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def fmt_group(k: int, eps: float) -> str:
    return f"({int(k)},{float(eps):.1f})"


def load_objects(is_shuffle: bool):
    case_class = load_case()
    z_noise_summary, v_est_summary = load_measurement()
    load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
    test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(
        is_shuffle=is_shuffle
    )
    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()
    return case_class, valid_dataloader_unscaled, dd_detector, scaler_


def collect_clean_validation_scores(valid_loader, dd_detector, scaler_) -> Dict[str, Any]:
    total_clean_sample = 0
    front_end_alarms = 0
    scores: List[float] = []
    idxs: List[int] = []

    for idx, input_batch, v_est_pre, v_est_last in tqdm(valid_loader, desc="valid-clean"):
        total_clean_sample += 1
        v_est_pre = v_est_pre.flatten()
        v_est_last = v_est_last.flatten()

        z_clean_scale = scaler_(input_batch)
        _, _, _, loss_recons = dd_detector.evaluate(z_clean_scale)
        if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
            continue

        front_end_alarms += 1
        z_recover, v_recover, *_ = dd_detector.recover(
            attack_batch=input_batch,
            v_pre=v_est_pre,
            v_last=v_est_last,
        )

        vang_last = np.angle(v_est_last.numpy())
        vang_recover = np.angle(v_recover.numpy())
        c_recover = vang_last - vang_recover
        c_recover_no_ref = np.expand_dims(c_recover[dd_detector.case_class.non_ref_index], 1)
        score_ = float(np.linalg.norm(c_recover_no_ref, 2))
        scores.append(score_)
        idxs.append(safe_int_idx(idx))

    if front_end_alarms == 0:
        raise RuntimeError("No clean false alarms found on validation split.")

    return {
        "total_clean_sample": total_clean_sample,
        "front_end_alarms": front_end_alarms,
        "scores": np.asarray(scores, dtype=float),
        "idx": np.asarray(idxs, dtype=int),
    }


def collect_attack_validation_scores(
    case_class,
    valid_loader,
    dd_detector,
    scaler_,
    total_run: int,
    seed_base: int,
    ang_no_list: List[int],
    ang_str_list: List[float],
    mag_no: int,
    mag_str: float,
    recover_input_mode: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for ang_no in ang_no_list:
        for ang_str in ang_str_list:
            key = fmt_group(ang_no, ang_str)
            front_end_alarms = 0
            scores: List[float] = []
            idxs: List[int] = []
            used = 0

            for idx_local, (idx, input_batch, v_est_pre, v_est_last) in tqdm(enumerate(valid_loader), desc=f"valid-attack-{key}"):
                if used >= total_run:
                    break

                idx_val = safe_int_idx(idx)
                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()

                seed_ = seed_base + 100000 * int(ang_no) + int(round(1000 * float(ang_str))) + idx_val
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
                if loss_recons <= dd_detector.ae_threshold[dd_detector.quantile_idx]:
                    continue

                front_end_alarms += 1
                used += 1

                recover_input = input_batch if recover_input_mode == "repo_compatible" else z_att_noise
                z_recover, v_recover, *_ = dd_detector.recover(
                    attack_batch=recover_input,
                    v_pre=v_est_pre,
                    v_last=v_att_est_last,
                )

                vang_att = np.angle(v_att_est_last.numpy())
                vang_recover = np.angle(v_recover.numpy())
                c_recover = vang_att - vang_recover
                c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
                score_ = float(np.linalg.norm(c_recover_no_ref, 2))
                scores.append(score_)
                idxs.append(idx_val)

            if front_end_alarms == 0:
                raise RuntimeError(f"No attack alarms collected for group {key} on validation split.")

            out[key] = {
                "front_end_alarms": front_end_alarms,
                "scores": np.asarray(scores, dtype=float),
                "idx": np.asarray(idxs, dtype=int),
            }
    return out


def evaluate_tau(
    tau: float,
    clean_pack: Dict[str, Any],
    attack_pack: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    clean_scores = clean_pack["scores"]
    total_clean_sample = int(clean_pack["total_clean_sample"])
    clean_front_end_alarms = int(clean_pack["front_end_alarms"])
    clean_backend_triggers = int(np.sum(clean_scores >= tau))

    group_metrics = {}
    total_attack_alarms = 0
    total_attack_triggers = 0
    for key, pack in attack_pack.items():
        n_alarm = int(pack["front_end_alarms"])
        n_trig = int(np.sum(pack["scores"] >= tau))
        arr = float(n_trig / n_alarm) if n_alarm > 0 else float("nan")
        total_attack_alarms += n_alarm
        total_attack_triggers += n_trig
        group_metrics[key] = {
            "front_end_alarms": n_alarm,
            "backend_triggers": n_trig,
            "arr": arr,
            "mean_verify_score": float(np.mean(pack["scores"])),
            "median_verify_score": float(np.median(pack["scores"])),
        }

    overall_arr = float(total_attack_triggers / total_attack_alarms) if total_attack_alarms > 0 else float("nan")

    return {
        "tau": float(tau),
        "clean_total_sample": total_clean_sample,
        "clean_front_end_alarms": clean_front_end_alarms,
        "clean_backend_triggers": clean_backend_triggers,
        "clean_deployment_among_alarms": float(clean_backend_triggers / clean_front_end_alarms),
        "clean_udr": float(clean_backend_triggers / total_clean_sample),
        "overall_attack_alarms": total_attack_alarms,
        "overall_attack_triggers": total_attack_triggers,
        "overall_arr": overall_arr,
        "group_metrics": group_metrics,
    }


def feasible(metrics: Dict[str, Any], overall_min: float, protected_min: float, protected_groups: List[str]) -> bool:
    if metrics["overall_arr"] < overall_min:
        return False
    gm = metrics["group_metrics"]
    for key in protected_groups:
        if key not in gm:
            raise KeyError(f"Protected group {key} missing from group metrics.")
        if gm[key]["arr"] < protected_min:
            return False
    return True


def pick_best(results: List[Dict[str, Any]], overall_min: float, protected_min: float, protected_groups: List[str]) -> Dict[str, Any] | None:
    feas = [r for r in results if feasible(r, overall_min, protected_min, protected_groups)]
    if not feas:
        return None
    # UDR monotonically decreases with tau, but keep an explicit tie-break rule.
    feas.sort(key=lambda r: (r["clean_udr"], -r["tau"]))
    best_udr = feas[0]["clean_udr"]
    tied = [r for r in feas if abs(r["clean_udr"] - best_udr) < 1e-15]
    tied.sort(key=lambda r: r["tau"], reverse=True)
    return tied[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir.strip() or f"metric/{sys_config['case_name']}/tau_selection_joint_valid")
    out_dir.mkdir(parents=True, exist_ok=True)

    case_class, valid_loader, dd_detector, scaler_ = load_objects(args.is_shuffle)
    detector_quantile = float(dd_detector.quantile[dd_detector.quantile_idx])
    detector_threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])

    print("==== Validation-only JOINT tau selection ====")
    print("case_name:", sys_config["case_name"])
    print("detector_quantile:", detector_quantile)
    print("detector_threshold:", detector_threshold)
    print("total_run_attack:", args.total_run_attack)
    print("recover_input_mode:", args.recover_input_mode)
    print("protected_groups:", args.protected_groups)
    print("main constraints: overall >=", args.main_overall_arr_min, ", protected >=", args.main_protected_arr_min)
    print("strict constraints: overall >=", args.strict_overall_arr_min, ", protected >=", args.strict_protected_arr_min)

    clean_pack = collect_clean_validation_scores(valid_loader, dd_detector, scaler_)
    attack_pack = collect_attack_validation_scores(
        case_class=case_class,
        valid_loader=valid_loader,
        dd_detector=dd_detector,
        scaler_=scaler_,
        total_run=int(args.total_run_attack),
        seed_base=int(args.seed_base),
        ang_no_list=list(args.ang_no_list),
        ang_str_list=[float(x) for x in args.ang_str_list],
        mag_no=int(args.mag_no),
        mag_str=float(args.mag_str),
        recover_input_mode=str(args.recover_input_mode),
    )

    clean_scores = clean_pack["scores"]
    attack_scores_all = np.concatenate([pack["scores"] for pack in attack_pack.values()])
    candidates = np.unique(np.concatenate(([0.0], clean_scores, attack_scores_all)))
    candidates.sort()

    results = [evaluate_tau(float(tau), clean_pack, attack_pack) for tau in candidates]
    best_main = pick_best(results, args.main_overall_arr_min, args.main_protected_arr_min, args.protected_groups)
    best_strict = pick_best(results, args.strict_overall_arr_min, args.strict_protected_arr_min, args.protected_groups)

    if best_main is None:
        raise RuntimeError("No feasible tau_main found under the requested validation constraints.")
    if best_strict is None:
        raise RuntimeError("No feasible tau_strict found under the requested validation constraints.")
    if best_strict["tau"] < best_main["tau"]:
        # stricter point should not be looser than main point; if it happens, keep the stricter choice as the max feasible under stricter constraints.
        pass

    candidate_rows_path = out_dir / "candidate_metrics.csv"
    with open(candidate_rows_path, "w", encoding="utf-8") as f:
        header = [
            "tau", "clean_udr", "clean_deployment_among_alarms", "overall_arr"
        ] + [f"arr_{g}" for g in attack_pack.keys()]
        f.write(",".join(header) + "\n")
        for r in results:
            row = [
                f"{r['tau']:.12f}",
                f"{r['clean_udr']:.12f}",
                f"{r['clean_deployment_among_alarms']:.12f}",
                f"{r['overall_arr']:.12f}",
            ] + [f"{r['group_metrics'][g]['arr']:.12f}" for g in attack_pack.keys()]
            f.write(",".join(row) + "\n")

    summary_txt = out_dir / "tau_selection_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Validation-only joint tau selection summary\n")
        f.write(f"detector_quantile={detector_quantile:.12f}\n")
        f.write(f"detector_threshold={detector_threshold:.12f}\n")
        f.write(f"clean_total_sample={int(clean_pack['total_clean_sample'])}\n")
        f.write(f"clean_front_end_alarms={int(clean_pack['front_end_alarms'])}\n")
        f.write(f"attack_total_run_per_group={int(args.total_run_attack)}\n")
        f.write(f"protected_groups={json.dumps(args.protected_groups)}\n")
        f.write(f"main_overall_arr_min={float(args.main_overall_arr_min):.6f}\n")
        f.write(f"main_protected_arr_min={float(args.main_protected_arr_min):.6f}\n")
        f.write(f"strict_overall_arr_min={float(args.strict_overall_arr_min):.6f}\n")
        f.write(f"strict_protected_arr_min={float(args.strict_protected_arr_min):.6f}\n")
        f.write(f"tau_main={best_main['tau']:.12f}\n")
        f.write(f"tau_main_rounded_3dp={best_main['tau']:.3f}\n")
        f.write(f"tau_main_clean_udr={best_main['clean_udr']:.12f}\n")
        f.write(f"tau_main_overall_arr={best_main['overall_arr']:.12f}\n")
        for g, gm in best_main["group_metrics"].items():
            f.write(f"tau_main_arr_{g}={gm['arr']:.12f}\n")
        f.write(f"tau_strict={best_strict['tau']:.12f}\n")
        f.write(f"tau_strict_rounded_3dp={best_strict['tau']:.3f}\n")
        f.write(f"tau_strict_clean_udr={best_strict['clean_udr']:.12f}\n")
        f.write(f"tau_strict_overall_arr={best_strict['overall_arr']:.12f}\n")
        for g, gm in best_strict["group_metrics"].items():
            f.write(f"tau_strict_arr_{g}={gm['arr']:.12f}\n")

    payload = {
        "clean_pack": {
            "total_clean_sample": int(clean_pack["total_clean_sample"]),
            "front_end_alarms": int(clean_pack["front_end_alarms"]),
            "scores": clean_pack["scores"].tolist(),
            "idx": clean_pack["idx"].tolist(),
        },
        "attack_pack": {
            k: {
                "front_end_alarms": int(v["front_end_alarms"]),
                "scores": v["scores"].tolist(),
                "idx": v["idx"].tolist(),
            }
            for k, v in attack_pack.items()
        },
        "tau_main": float(best_main["tau"]),
        "tau_strict": float(best_strict["tau"]),
        "best_main": best_main,
        "best_strict": best_strict,
    }
    with open(out_dir / "tau_selection_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_metric(
        address=str(out_dir / "tau_selection_metric.npy"),
        tau_main=float(best_main["tau"]),
        tau_strict=float(best_strict["tau"]),
        best_main=best_main,
        best_strict=best_strict,
        detector_quantile=detector_quantile,
        detector_threshold=detector_threshold,
        protected_groups=args.protected_groups,
    )

    print("\n==== Selected operating points ====")
    print(f"tau_main   = {best_main['tau']:.12f}  (3dp: {best_main['tau']:.3f})")
    print(f"  clean_udr   = {best_main['clean_udr']:.6f}")
    print(f"  overall_arr = {best_main['overall_arr']:.6f}")
    for g in args.protected_groups:
        print(f"  {g} arr = {best_main['group_metrics'][g]['arr']:.6f}")

    print(f"tau_strict = {best_strict['tau']:.12f}  (3dp: {best_strict['tau']:.3f})")
    print(f"  clean_udr   = {best_strict['clean_udr']:.6f}")
    print(f"  overall_arr = {best_strict['overall_arr']:.6f}")
    for g in args.protected_groups:
        print(f"  {g} arr = {best_strict['group_metrics'][g]['arr']:.6f}")

    if args.candidate_quantiles:
        arr = np.asarray(clean_pack["scores"], dtype=float)
        print("\n==== Selected clean quantile references ====")
        for q in args.candidate_quantiles:
            print(f"q={q:.3f} -> tau={float(np.quantile(arr, q)):.12f}")

    print("\nSaved:")
    print(summary_txt)
    print(candidate_rows_path)
    print(out_dir / "tau_selection_payload.json")
    print(out_dir / "tau_selection_metric.npy")


if __name__ == "__main__":
    main()

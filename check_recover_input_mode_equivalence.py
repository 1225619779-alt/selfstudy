from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from configs.nn_setting import nn_setting
from models.dataset import scaler
from models.evaluation import Evaluation
from models.model import LSTM_AE
from utils.load_data import load_case, load_dataset, load_measurement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check equivalence of recover_input_mode on validation attack alarms.")
    p.add_argument("--total_run", type=int, default=50, help="Validation attack alarms per group.")
    p.add_argument("--seed_base", type=int, default=20260403)
    p.add_argument("--ang_no_list", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--ang_str_list", type=float, nargs="*", default=[0.2, 0.3])
    p.add_argument("--mag_no", type=int, default=0)
    p.add_argument("--mag_str", type=float, default=0.0)
    p.add_argument("--tau_main", type=float, default=0.013319196253)
    p.add_argument("--tau_strict", type=float, default=0.016153267226)
    p.add_argument("--output_csv", default="metric/case39/recover_input_mode_equivalence.csv")
    p.add_argument("--output_json", default="metric/case39/recover_input_mode_equivalence.json")
    p.add_argument("--output_md", default="reports/recover_input_mode_equivalence.md")
    return p.parse_args()


def safe_int_idx(idx: Any) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


def compute_score(
    dd_detector: Evaluation,
    case_class,
    attack_batch,
    v_pre,
    v_last,
) -> tuple[float, bool]:
    try:
        _, v_recover, *_ , recover_time = dd_detector.recover(
            attack_batch=attack_batch,
            v_pre=v_pre,
            v_last=v_last,
        )
        _ = recover_time
        vang_att = np.angle(v_last.numpy())
        vang_recover = np.angle(v_recover.numpy())
        c_recover = vang_att - vang_recover
        c_recover_no_ref = np.expand_dims(c_recover[case_class.non_ref_index], 1)
        return float(np.linalg.norm(c_recover_no_ref, 2)), False
    except Exception:
        return float("nan"), True


def finite_trigger(score: float, tau: float) -> bool:
    return bool(np.isfinite(score) and score >= tau)


def fmt(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.12f}"


def main() -> None:
    args = parse_args()

    case_class = load_case()
    z_noise_summary, _ = load_measurement()
    _, _, _, valid_dataloader_unscaled = load_dataset(is_shuffle=False)
    _ = len(z_noise_summary)

    lstm_ae = LSTM_AE()
    lstm_ae.load_state_dict(torch.load(nn_setting["model_path"], map_location=torch.device(nn_setting["device"])))
    dd_detector = Evaluation(case_class=case_class)
    scaler_ = scaler()
    threshold = float(dd_detector.ae_threshold[dd_detector.quantile_idx])

    rows: List[Dict[str, Any]] = []
    overall_repo_scores: List[float] = []
    overall_att_scores: List[float] = []
    overall_diff: List[float] = []
    overall_changed_main = 0
    overall_changed_strict = 0
    overall_pairs = 0
    overall_repo_fail = 0
    overall_att_fail = 0

    for ang_no in args.ang_no_list:
        for ang_str in args.ang_str_list:
            group = f"({int(ang_no)},{float(ang_str):.1f})"
            repo_scores: List[float] = []
            att_scores: List[float] = []
            diffs: List[float] = []
            changed_main = 0
            changed_strict = 0
            repo_fail = 0
            att_fail = 0
            used = 0

            for _, (idx, input_batch, v_est_pre, v_est_last) in tqdm(enumerate(valid_dataloader_unscaled), desc=f"equiv-{group}"):
                if used >= args.total_run:
                    break

                idx_val = safe_int_idx(idx)
                v_est_pre = v_est_pre.flatten()
                v_est_last = v_est_last.flatten()

                seed_ = args.seed_base + 100000 * int(ang_no) + int(round(1000 * float(ang_str))) + idx_val
                random.seed(seed_)
                np.random.seed(seed_ % (2**32 - 1))

                z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                    z_noise=input_batch,
                    v_est_last=v_est_last,
                    ang_no=int(ang_no),
                    mag_no=int(args.mag_no),
                    ang_str=float(ang_str),
                    mag_str=float(args.mag_str),
                )
                v_att_est_last = torch.from_numpy(v_att_est_last)

                z_att_noise_scale = scaler_(z_att_noise)
                _, _, _, loss_recons = dd_detector.evaluate(z_att_noise_scale)
                if float(loss_recons) <= threshold:
                    continue

                used += 1

                score_repo, repo_err = compute_score(
                    dd_detector=dd_detector,
                    case_class=case_class,
                    attack_batch=input_batch,
                    v_pre=v_est_pre,
                    v_last=v_att_est_last,
                )
                score_att, att_err = compute_score(
                    dd_detector=dd_detector,
                    case_class=case_class,
                    attack_batch=z_att_noise,
                    v_pre=v_est_pre,
                    v_last=v_att_est_last,
                )

                repo_scores.append(score_repo)
                att_scores.append(score_att)
                if repo_err:
                    repo_fail += 1
                if att_err:
                    att_fail += 1

                diff = float(abs(score_repo - score_att)) if np.isfinite(score_repo) and np.isfinite(score_att) else float("nan")
                diffs.append(diff)
                trig_repo_main = finite_trigger(score_repo, args.tau_main)
                trig_att_main = finite_trigger(score_att, args.tau_main)
                trig_repo_strict = finite_trigger(score_repo, args.tau_strict)
                trig_att_strict = finite_trigger(score_att, args.tau_strict)
                changed_main += int(trig_repo_main != trig_att_main)
                changed_strict += int(trig_repo_strict != trig_att_strict)

            repo_arr = np.asarray(repo_scores, dtype=float)
            att_arr = np.asarray(att_scores, dtype=float)
            diff_arr = np.asarray(diffs, dtype=float)
            finite_diff = diff_arr[np.isfinite(diff_arr)]

            row = {
                "group": group,
                "sample_count": int(used),
                "repo_recover_fail_count": int(repo_fail),
                "attacked_recover_fail_count": int(att_fail),
                "finite_pair_count": int(finite_diff.size),
                "max_abs_diff": float(np.max(finite_diff)) if finite_diff.size else float("nan"),
                "mean_abs_diff": float(np.mean(finite_diff)) if finite_diff.size else float("nan"),
                "median_abs_diff": float(np.median(finite_diff)) if finite_diff.size else float("nan"),
                "changed_trigger_count_tau_main_exact": int(changed_main),
                "changed_trigger_count_tau_strict_exact": int(changed_strict),
            }
            rows.append(row)

            overall_repo_scores.extend(repo_arr.tolist())
            overall_att_scores.extend(att_arr.tolist())
            overall_diff.extend(diff_arr.tolist())
            overall_changed_main += changed_main
            overall_changed_strict += changed_strict
            overall_pairs += used
            overall_repo_fail += repo_fail
            overall_att_fail += att_fail

    overall_diff_arr = np.asarray(overall_diff, dtype=float)
    finite_overall = overall_diff_arr[np.isfinite(overall_diff_arr)]
    overall = {
        "group": "overall",
        "sample_count": int(overall_pairs),
        "repo_recover_fail_count": int(overall_repo_fail),
        "attacked_recover_fail_count": int(overall_att_fail),
        "finite_pair_count": int(finite_overall.size),
        "max_abs_diff": float(np.max(finite_overall)) if finite_overall.size else float("nan"),
        "mean_abs_diff": float(np.mean(finite_overall)) if finite_overall.size else float("nan"),
        "median_abs_diff": float(np.median(finite_overall)) if finite_overall.size else float("nan"),
        "changed_trigger_count_tau_main_exact": int(overall_changed_main),
        "changed_trigger_count_tau_strict_exact": int(overall_changed_strict),
    }
    rows.append(overall)

    payload = {
        "tau_main_exact": float(args.tau_main),
        "tau_strict_exact": float(args.tau_strict),
        "threshold": threshold,
        "rows": rows,
    }

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        "# Recover Input Mode Equivalence",
        "",
        f"- tau_main_exact: `{fmt(float(args.tau_main))}`",
        f"- tau_strict_exact: `{fmt(float(args.tau_strict))}`",
        f"- detector_threshold: `{fmt(threshold)}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['group']}",
                "",
                f"- sample_count: `{row['sample_count']}`",
                f"- repo_recover_fail_count: `{row['repo_recover_fail_count']}`",
                f"- attacked_recover_fail_count: `{row['attacked_recover_fail_count']}`",
                f"- finite_pair_count: `{row['finite_pair_count']}`",
                f"- max_abs_diff: `{fmt(float(row['max_abs_diff']))}`",
                f"- mean_abs_diff: `{fmt(float(row['mean_abs_diff']))}`",
                f"- median_abs_diff: `{fmt(float(row['median_abs_diff']))}`",
                f"- changed_trigger_count@tau_main_exact: `{row['changed_trigger_count_tau_main_exact']}`",
                f"- changed_trigger_count@tau_strict_exact: `{row['changed_trigger_count_tau_strict_exact']}`",
                "",
            ]
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved MD: {out_md}")


if __name__ == "__main__":
    main()

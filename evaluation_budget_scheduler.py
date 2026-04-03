from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

from scheduler.calibration import (
    BinnedStatisticModel,
    fit_attack_posterior_from_banks,
    fit_service_models_from_mixed_bank,
    mixed_bank_to_alarm_arrays,
)
from scheduler.policies import AlarmJob, SimulationConfig, build_jobs_from_arrays, simulate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-1 offline evaluation for budget-aware backend MTD scheduling."
    )
    parser.add_argument("--clean_bank", type=str, required=True, help="Path to metric_clean_alarm_scores*.npy")
    parser.add_argument("--attack_bank", type=str, required=True, help="Path to metric_attack_alarm_scores*.npy")
    parser.add_argument("--fit_bank", type=str, required=True, help="Path to mixed timeline bank for fitting service models")
    parser.add_argument("--eval_bank", type=str, required=True, help="Path to mixed timeline bank for evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="metric/case14/budget_scheduler_phase1_result.npy",
        help="Where to save the result dictionary (.npy)",
    )
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--slot_budget_list", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--max_wait_steps", type=int, default=10)
    parser.add_argument("--time_budget_quantile", type=float, default=0.50)
    parser.add_argument("--cost_budget_quantile", type=float, default=0.50)
    parser.add_argument("--threshold_quantiles", type=float, nargs="*", default=[0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--vq_v_grid", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--vq_age_grid", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--vq_fail_grid", type=float, nargs="*", default=[0.0, 0.05, 0.10])
    parser.add_argument("--clean_penalty", type=float, default=0.50)
    parser.add_argument("--delay_penalty", type=float, default=0.15)
    parser.add_argument("--time_overuse_penalty", type=float, default=0.25)
    parser.add_argument("--cost_overuse_penalty", type=float, default=0.25)
    parser.add_argument("--rng_seed", type=int, default=20260402)
    return parser.parse_args()



def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)



def _normalize_value_proxy(verify_score: np.ndarray, fit_verify_score: np.ndarray) -> np.ndarray:
    fit_verify = np.asarray(fit_verify_score, dtype=float)
    fit_verify = fit_verify[np.isfinite(fit_verify)]
    scale = float(np.quantile(fit_verify, 0.90)) if fit_verify.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(np.asarray(verify_score, dtype=float) / scale, 0.0, 3.0)



def _predict_jobs(
    arrays: Dict[str, np.ndarray],
    *,
    posterior_model: BinnedStatisticModel,
    posterior_signal_key: str,
    service_models: Dict[str, BinnedStatisticModel],
    service_signal_key: str,
    fit_verify_score: np.ndarray,
) -> Tuple[List[AlarmJob], int]:
    if posterior_signal_key not in arrays:
        raise KeyError(f"posterior_signal_key={posterior_signal_key!r} not found in arrays")
    if service_signal_key not in arrays:
        raise KeyError(f"service_signal_key={service_signal_key!r} not found in arrays")

    x_post = np.asarray(arrays[posterior_signal_key], dtype=float)
    x_srv = np.asarray(arrays[service_signal_key], dtype=float)
    p_hat = posterior_model.predict(x_post)
    tau_hat = service_models["service_time"].predict(x_srv)
    cost_hat = service_models["service_cost"].predict(x_srv)
    fail_hat = service_models["backend_fail"].predict(x_srv)
    value_proxy = _normalize_value_proxy(arrays["verify_score"], fit_verify_score)
    return build_jobs_from_arrays(
        arrays,
        p_hat=p_hat,
        tau_hat=tau_hat,
        cost_hat=cost_hat,
        fail_hat=fail_hat,
        value_proxy=value_proxy,
    )



def _threshold_candidates(x: np.ndarray, quantiles: List[float]) -> List[float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0]
    candidates = sorted({float(np.quantile(arr, q)) for q in quantiles})
    return candidates



def _objective(
    summary: Dict[str, float],
    *,
    max_wait_steps: int,
    clean_penalty: float,
    delay_penalty: float,
    time_overuse_penalty: float,
    cost_overuse_penalty: float,
    average_time_budget: float | None,
    average_cost_budget: float | None,
) -> float:
    score = float(summary["weighted_attack_recall_no_backend_fail"])
    score -= float(clean_penalty) * float(summary["clean_service_ratio"])
    score -= float(delay_penalty) * float(summary["attack_delay_mean"] / max(max_wait_steps, 1))
    if average_time_budget is not None and average_time_budget > 0:
        over_t = max(float(summary["average_service_time_per_step"]) / average_time_budget - 1.0, 0.0)
        score -= float(time_overuse_penalty) * over_t
    if average_cost_budget is not None and average_cost_budget > 0:
        over_c = max(float(summary["average_service_cost_per_step"]) / average_cost_budget - 1.0, 0.0)
        score -= float(cost_overuse_penalty) * over_c
    return float(score)



def _run_one_policy(jobs: List[AlarmJob], total_steps: int, cfg: SimulationConfig) -> Dict[str, object]:
    return simulate_policy(jobs, total_steps=total_steps, cfg=cfg)



def _tune_threshold_policy(
    jobs_fit: List[AlarmJob],
    total_steps_fit: int,
    *,
    threshold_candidates: List[float],
    policy_name: str,
    slot_budget: int,
    max_wait_steps: int,
    rng_seed: int,
    average_time_budget: float | None,
    average_cost_budget: float | None,
    score_kwargs: Dict[str, float],
) -> Tuple[float, Dict[str, object]]:
    best_thr = threshold_candidates[0]
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for thr in threshold_candidates:
        cfg = SimulationConfig(
            policy_name=policy_name,
            slot_budget=slot_budget,
            max_wait_steps=max_wait_steps,
            threshold=float(thr),
            rng_seed=rng_seed,
            average_time_budget=average_time_budget,
            average_cost_budget=average_cost_budget,
        )
        res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
        obj = _objective(res["summary"], average_time_budget=average_time_budget, average_cost_budget=average_cost_budget, **score_kwargs)
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
            best_res = res
    assert best_res is not None
    return best_thr, best_res



def _tune_proposed_policy(
    jobs_fit: List[AlarmJob],
    total_steps_fit: int,
    *,
    slot_budget: int,
    max_wait_steps: int,
    average_time_budget: float | None,
    average_cost_budget: float | None,
    v_grid: List[float],
    age_grid: List[float],
    fail_grid: List[float],
    rng_seed: int,
    score_kwargs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    best_params = {"v_weight": v_grid[0], "age_bonus": age_grid[0], "fail_penalty": fail_grid[0]}
    best_res: Dict[str, object] | None = None
    best_obj = -1e18
    for v_weight in v_grid:
        for age_bonus in age_grid:
            for fail_penalty in fail_grid:
                cfg = SimulationConfig(
                    policy_name="proposed_vq",
                    slot_budget=slot_budget,
                    max_wait_steps=max_wait_steps,
                    average_time_budget=average_time_budget,
                    average_cost_budget=average_cost_budget,
                    v_weight=float(v_weight),
                    age_bonus=float(age_bonus),
                    fail_penalty=float(fail_penalty),
                    rng_seed=rng_seed,
                )
                res = _run_one_policy(jobs_fit, total_steps_fit, cfg)
                obj = _objective(res["summary"], average_time_budget=average_time_budget, average_cost_budget=average_cost_budget, **score_kwargs)
                if obj > best_obj:
                    best_obj = obj
                    best_params = {
                        "v_weight": float(v_weight),
                        "age_bonus": float(age_bonus),
                        "fail_penalty": float(fail_penalty),
                    }
                    best_res = res
    assert best_res is not None
    return best_params, best_res



def _budget_targets(arrays_fit: Dict[str, np.ndarray], slot_budget: int, q_time: float, q_cost: float) -> Tuple[float, float]:
    time_arr = np.asarray(arrays_fit["service_time"], dtype=float)
    cost_arr = np.asarray(arrays_fit["service_cost"], dtype=float)
    time_arr = time_arr[np.isfinite(time_arr)]
    cost_arr = cost_arr[np.isfinite(cost_arr)]
    base_time = float(np.quantile(time_arr, q_time)) if time_arr.size else 0.0
    base_cost = float(np.quantile(cost_arr, q_cost)) if cost_arr.size else 0.0
    return float(base_time * slot_budget), float(base_cost * slot_budget)



def _summary_row(policy_label: str, policy_res: Dict[str, object], tune_payload: Dict[str, object] | None = None) -> Dict[str, object]:
    row = {"policy": policy_label}
    row.update(policy_res["summary"])
    if tune_payload is not None:
        row["tuned"] = tune_payload
    return row



def main() -> None:
    args = parse_args()
    ensure_parent(args.output)

    arrays_fit = mixed_bank_to_alarm_arrays(args.fit_bank)
    arrays_eval = mixed_bank_to_alarm_arrays(args.eval_bank)

    posterior_verify = fit_attack_posterior_from_banks(
        args.clean_bank,
        args.attack_bank,
        signal_key="score_phys_l2",
        n_bins=args.n_bins,
    )
    posterior_ddd = fit_attack_posterior_from_banks(
        args.clean_bank,
        args.attack_bank,
        signal_key="ddd_loss_alarm",
        n_bins=args.n_bins,
    )
    service_models = fit_service_models_from_mixed_bank(args.fit_bank, signal_key="verify_score", n_bins=args.n_bins)

    jobs_fit, total_steps_fit = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
    )
    jobs_eval, total_steps_eval = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_verify,
        posterior_signal_key="verify_score",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
    )

    # Alternative ddd-only posterior for one baseline family.
    jobs_fit_ddd, _ = _predict_jobs(
        arrays_fit,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
    )
    jobs_eval_ddd, _ = _predict_jobs(
        arrays_eval,
        posterior_model=posterior_ddd,
        posterior_signal_key="ddd_loss_recons",
        service_models=service_models,
        service_signal_key="verify_score",
        fit_verify_score=np.asarray(arrays_fit["verify_score"], dtype=float),
    )

    verify_threshold_candidates = _threshold_candidates(arrays_fit["verify_score"], list(args.threshold_quantiles))
    ddd_threshold_candidates = _threshold_candidates(arrays_fit["ddd_loss_recons"], list(args.threshold_quantiles))

    score_kwargs = {
        "max_wait_steps": int(args.max_wait_steps),
        "clean_penalty": float(args.clean_penalty),
        "delay_penalty": float(args.delay_penalty),
        "time_overuse_penalty": float(args.time_overuse_penalty),
        "cost_overuse_penalty": float(args.cost_overuse_penalty),
    }

    all_results: Dict[str, object] = {
        "config": vars(args),
        "slot_budget_results": {},
        "notes": {
            "phase": "phase1_offline_scheduler_validation",
            "fit_bank": args.fit_bank,
            "eval_bank": args.eval_bank,
        },
    }

    for slot_budget in [int(x) for x in args.slot_budget_list]:
        avg_time_budget, avg_cost_budget = _budget_targets(
            arrays_fit,
            slot_budget=slot_budget,
            q_time=float(args.time_budget_quantile),
            q_cost=float(args.cost_budget_quantile),
        )

        per_budget: Dict[str, object] = {
            "average_time_budget": avg_time_budget,
            "average_cost_budget": avg_cost_budget,
            "policies": {},
            "tuning": {},
            "summary_rows": [],
        }

        # Tune threshold baselines on fit bank.
        thr_verify, thr_verify_fit_res = _tune_threshold_policy(
            jobs_fit,
            total_steps_fit,
            threshold_candidates=verify_threshold_candidates,
            policy_name="threshold_verify_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            average_time_budget=avg_time_budget,
            average_cost_budget=avg_cost_budget,
            score_kwargs=score_kwargs,
        )
        thr_ddd, thr_ddd_fit_res = _tune_threshold_policy(
            jobs_fit_ddd,
            total_steps_fit,
            threshold_candidates=ddd_threshold_candidates,
            policy_name="threshold_ddd_fifo",
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            rng_seed=int(args.rng_seed),
            average_time_budget=avg_time_budget,
            average_cost_budget=avg_cost_budget,
            score_kwargs=score_kwargs,
        )
        proposed_params, proposed_fit_res = _tune_proposed_policy(
            jobs_fit,
            total_steps_fit,
            slot_budget=slot_budget,
            max_wait_steps=int(args.max_wait_steps),
            average_time_budget=avg_time_budget,
            average_cost_budget=avg_cost_budget,
            v_grid=list(args.vq_v_grid),
            age_grid=list(args.vq_age_grid),
            fail_grid=list(args.vq_fail_grid),
            rng_seed=int(args.rng_seed),
            score_kwargs=score_kwargs,
        )

        per_budget["tuning"] = {
            "threshold_verify": {
                "best_threshold": thr_verify,
                "fit_summary": thr_verify_fit_res["summary"],
            },
            "threshold_ddd": {
                "best_threshold": thr_ddd,
                "fit_summary": thr_ddd_fit_res["summary"],
            },
            "proposed_vq": {
                "best_params": proposed_params,
                "fit_summary": proposed_fit_res["summary"],
            },
        }

        # Evaluate policies on eval bank.
        eval_policy_payloads = {
            "always_fifo": SimulationConfig(
                policy_name="fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "random": SimulationConfig(
                policy_name="random",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "topk_verify": SimulationConfig(
                policy_name="topk_verify",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "static_value_cost": SimulationConfig(
                policy_name="static_value_cost",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "threshold_verify_fifo": SimulationConfig(
                policy_name="threshold_verify_fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                threshold=thr_verify,
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "threshold_ddd_fifo": SimulationConfig(
                policy_name="threshold_ddd_fifo",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                threshold=thr_ddd,
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                rng_seed=int(args.rng_seed),
            ),
            "proposed_vq": SimulationConfig(
                policy_name="proposed_vq",
                slot_budget=slot_budget,
                max_wait_steps=int(args.max_wait_steps),
                average_time_budget=avg_time_budget,
                average_cost_budget=avg_cost_budget,
                v_weight=proposed_params["v_weight"],
                age_bonus=proposed_params["age_bonus"],
                fail_penalty=proposed_params["fail_penalty"],
                rng_seed=int(args.rng_seed),
            ),
        }

        policy_jobs_lookup = {
            "always_fifo": jobs_eval,
            "random": jobs_eval,
            "topk_verify": jobs_eval,
            "static_value_cost": jobs_eval,
            "threshold_verify_fifo": jobs_eval,
            "threshold_ddd_fifo": jobs_eval_ddd,
            "proposed_vq": jobs_eval,
        }

        for label, cfg in eval_policy_payloads.items():
            jobs_this = policy_jobs_lookup[label]
            res = _run_one_policy(jobs_this, total_steps_eval, cfg)
            per_budget["policies"][label] = res
            tune_payload = None
            if label == "threshold_verify_fifo":
                tune_payload = {"best_threshold": thr_verify}
            elif label == "threshold_ddd_fifo":
                tune_payload = {"best_threshold": thr_ddd}
            elif label == "proposed_vq":
                tune_payload = proposed_params
            per_budget["summary_rows"].append(_summary_row(label, res, tune_payload=tune_payload))

        all_results["slot_budget_results"][str(slot_budget)] = per_budget

    np.save(args.output, all_results, allow_pickle=True)

    summary_json_path = os.path.splitext(args.output)[0] + ".summary.json"
    compact = {
        "config": all_results["config"],
        "slot_budget_results": {
            key: {
                "average_time_budget": value["average_time_budget"],
                "average_cost_budget": value["average_cost_budget"],
                "tuning": value["tuning"],
                "summary_rows": value["summary_rows"],
            }
            for key, value in all_results["slot_budget_results"].items()
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print("Saved result npy:", args.output)
    print("Saved compact summary json:", summary_json_path)
    print("\n==== Compact summaries ====")
    for slot_budget, payload in all_results["slot_budget_results"].items():
        print(f"\n-- slot_budget={slot_budget} --")
        rows = payload["summary_rows"]
        rows_sorted = sorted(rows, key=lambda r: r["weighted_attack_recall_no_backend_fail"], reverse=True)
        for row in rows_sorted:
            print(
                row["policy"],
                {
                    "weighted_attack_recall_no_backend_fail": round(float(row["weighted_attack_recall_no_backend_fail"]), 4),
                    "unnecessary_mtd_count": int(row["unnecessary_mtd_count"]),
                    "queue_delay_p95": round(float(row["queue_delay_p95"]), 4),
                    "average_service_cost_per_step": round(float(row["average_service_cost_per_step"]), 6),
                },
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_CLEAN_SCORES = "metric/case14/metric_clean_alarm_scores_full.npy"
DEFAULT_ATTACK_SCORES = "metric/case14/metric_attack_alarm_scores_200.npy"
DEFAULT_BASELINE_CLEAN = "metric/case14/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy"
DEFAULT_MAIN_CLEAN = "metric/case14/metric_event_trigger_clean_tau_0.021_mode_0_0.03_1.1.npy"
DEFAULT_STRICT_CANDIDATES = [
    "metric/case14/metric_event_trigger_clean_tau_0.03_mode_0_0.03_1.1.npy",
    "metric/case14/metric_event_trigger_clean_tau_0.030_mode_0_0.03_1.1.npy",
]
DEFAULT_OUT_NPY = "metric/case14/metric_gate_ablation_summary.npy"
DEFAULT_OUT_CSV = "metric/case14/metric_gate_ablation_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze matched-budget gate ablation: detector loss vs proposed physical score."
    )
    parser.add_argument("--clean_scores", type=str, default=DEFAULT_CLEAN_SCORES)
    parser.add_argument("--attack_scores", type=str, default=DEFAULT_ATTACK_SCORES)
    parser.add_argument("--baseline_clean_metric", type=str, default=DEFAULT_BASELINE_CLEAN)
    parser.add_argument("--main_clean_metric", type=str, default=DEFAULT_MAIN_CLEAN)
    parser.add_argument(
        "--strict_clean_metric",
        type=str,
        default="",
        help="Optional. If empty, the script tries common 0.03/0.030 candidates.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit budgets. If omitted, infer from main/strict clean metrics, otherwise fallback to 130 90.",
    )
    parser.add_argument(
        "--weak_keys",
        nargs="*",
        default=["(1,0.2)", "(1,0.3)", "(2,0.2)"],
    )
    parser.add_argument(
        "--strong_keys",
        nargs="*",
        default=["(2,0.3)", "(3,0.2)", "(3,0.3)"],
    )
    parser.add_argument("--out_npy", type=str, default=DEFAULT_OUT_NPY)
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_npy_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def maybe_load_npy_dict(path: str) -> Dict[str, Any] | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return load_npy_dict(path)


def resolve_strict_metric(explicit: str) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    for p in DEFAULT_STRICT_CANDIDATES:
        if os.path.exists(p):
            return p
    for p in glob.glob("metric/case14/metric_event_trigger_clean_tau_0.03*_mode_0_0.03_1.1.npy"):
        return p
    return ""


def detect_group_key(data: Dict[str, Any], preferred: str = "(0,0.0)") -> str:
    candidate_fields = [
        "clean_alarm_idx",
        "total_clean_sample",
        "total_DDD_alarm",
        "TP_DDD",
        "verify_score",
    ]
    for field in candidate_fields:
        value = data.get(field, None)
        if isinstance(value, dict) and len(value) > 0:
            keys = list(value.keys())
            if preferred in keys:
                return preferred
            return keys[0]
    raise KeyError("Cannot detect clean group key.")


def get_group_list(data: Dict[str, Any], field: str, group_key: str) -> List[Any]:
    value = data.get(field, None)
    if isinstance(value, dict):
        return list(value.get(group_key, []))
    return []


def get_group_scalar(data: Dict[str, Any], field: str, group_key: str, default: float = float("nan")) -> float:
    value = data.get(field, None)
    if isinstance(value, dict):
        if group_key in value:
            return value[group_key]
    return default


def as_float_array(x: Sequence[Any]) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(x, dtype=float)


def as_int_array(x: Sequence[Any]) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=int)
    return np.asarray(x, dtype=int)


def topk_mask(scores: Sequence[Any], k: int) -> Tuple[np.ndarray, float, np.ndarray]:
    scores_arr = as_float_array(scores)
    valid = np.where(np.isfinite(scores_arr))[0]
    if k < 0:
        raise ValueError(f"k must be nonnegative, got {k}")
    if k > len(valid):
        raise ValueError(f"Requested top-{k} but only {len(valid)} finite scores are available.")
    mask = np.zeros(len(scores_arr), dtype=bool)
    if k == 0:
        return mask, float("inf"), np.asarray([], dtype=int)
    order = valid[np.argsort(-scores_arr[valid], kind="mergesort")]
    sel = order[:k]
    mask[sel] = True
    tau = float(scores_arr[sel[-1]])
    return mask, tau, sel


def pooled_retention(score_dict: Dict[str, List[Any]], tau: float, keys: Iterable[str]) -> float:
    pooled: List[np.ndarray] = []
    for key in keys:
        if key not in score_dict:
            continue
        pooled.append(as_float_array(score_dict[key]))
    if not pooled:
        return float("nan")
    arr = np.concatenate(pooled)
    if arr.size == 0:
        return float("nan")
    keep = np.isfinite(arr) & (arr >= tau)
    return float(np.mean(keep.astype(float)))


def compute_clean_row(
    total_clean_sample: int,
    total_clean_alarm: int,
    mask: np.ndarray,
    fail_alarm: np.ndarray,
    stage_one_time_alarm: np.ndarray,
    stage_two_time_alarm: np.ndarray,
    delta_cost_one_alarm: np.ndarray,
    delta_cost_two_alarm: np.ndarray,
) -> Dict[str, float]:
    if len(mask) != total_clean_alarm:
        raise ValueError(f"mask length {len(mask)} != total_clean_alarm {total_clean_alarm}")

    mask_f = mask.astype(float)
    selected = int(mask.sum())

    fail_sel = fail_alarm * mask_f
    s1_sel = stage_one_time_alarm * mask_f
    s2_sel = stage_two_time_alarm * mask_f
    c1_sel = delta_cost_one_alarm * mask_f
    c2_sel = delta_cost_two_alarm * mask_f

    return {
        "clean_selected_alarms": float(selected),
        "uMTD_rate": float(selected / total_clean_sample) if total_clean_sample > 0 else float("nan"),
        "trigger_rate": float(selected / total_clean_alarm) if total_clean_alarm > 0 else float("nan"),
        "fail_per_alarm": float(np.mean(fail_sel)) if total_clean_alarm > 0 else float("nan"),
        "stage_one_time_per_alarm": float(np.mean(s1_sel)) if total_clean_alarm > 0 else float("nan"),
        "stage_two_time_per_alarm": float(np.mean(s2_sel)) if total_clean_alarm > 0 else float("nan"),
        "delta_cost_one_per_alarm": float(np.mean(c1_sel)) if total_clean_alarm > 0 else float("nan"),
        "delta_cost_two_per_alarm": float(np.mean(c2_sel)) if total_clean_alarm > 0 else float("nan"),
    }


def fmt(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if np.isnan(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.3f}"
    if abs(x) >= 1:
        return f"{x:.6f}"
    return f"{x:.6f}"


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_budget_items(
    args_budgets: List[int] | None,
    main_metric: Dict[str, Any] | None,
    strict_metric: Dict[str, Any] | None,
    clean_group_key: str,
) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str, int, str]] = []
    seen = set()

    if main_metric is not None:
        k = int(get_group_scalar(main_metric, "total_trigger_after_verification", clean_group_key, default=-1))
        if k >= 0 and k not in seen:
            items.append(("Main OP", k, "main_metric"))
            seen.add(k)

    if strict_metric is not None:
        k = int(get_group_scalar(strict_metric, "total_trigger_after_verification", clean_group_key, default=-1))
        if k >= 0 and k not in seen:
            items.append(("Strict OP", k, "strict_metric"))
            seen.add(k)

    if args_budgets:
        for k in args_budgets:
            if k not in seen:
                items.append((f"Budget-{k}", int(k), "cli"))
                seen.add(k)

    if not items:
        for k, label in [(130, "Main OP"), (90, "Strict OP")]:
            items.append((label, k, "fallback"))

    return items


def main() -> None:
    args = parse_args()

    strict_metric_path = resolve_strict_metric(args.strict_clean_metric)

    print("==== analyze_gate_ablation ====")
    print("clean_scores:", args.clean_scores)
    print("attack_scores:", args.attack_scores)
    print("baseline_clean_metric:", args.baseline_clean_metric)
    print("main_clean_metric:", args.main_clean_metric)
    print("strict_clean_metric:", strict_metric_path if strict_metric_path else "NOT FOUND")

    clean_scores = load_npy_dict(args.clean_scores)
    attack_scores = load_npy_dict(args.attack_scores)
    baseline_clean = load_npy_dict(args.baseline_clean_metric)
    main_clean = maybe_load_npy_dict(args.main_clean_metric)
    strict_clean = maybe_load_npy_dict(strict_metric_path)

    clean_group_key = detect_group_key(clean_scores)
    baseline_group_key = detect_group_key(baseline_clean)
    if clean_group_key != baseline_group_key:
        raise RuntimeError(
            f"clean group key mismatch: collector={clean_group_key}, baseline={baseline_group_key}"
        )

    total_clean_sample = int(get_group_scalar(baseline_clean, "total_clean_sample", clean_group_key, default=-1))
    total_clean_alarm = int(get_group_scalar(baseline_clean, "total_DDD_alarm", clean_group_key, default=-1))

    clean_alarm_idx_from_collector = as_int_array(get_group_list(clean_scores, "clean_alarm_idx", clean_group_key))
    clean_alarm_idx_from_baseline = as_int_array(get_group_list(baseline_clean, "clean_alarm_idx", clean_group_key))

    aligned = np.array_equal(clean_alarm_idx_from_collector, clean_alarm_idx_from_baseline)
    if not aligned:
        raise RuntimeError("clean_alarm_idx mismatch between collector and baseline clean metric")

    ddd_loss_clean = as_float_array(get_group_list(clean_scores, "ddd_loss_alarm", clean_group_key))
    phys_score_clean = as_float_array(get_group_list(clean_scores, "score_phys_l2", clean_group_key))
    recover_fail_clean = np.asarray(get_group_list(clean_scores, "recover_fail", clean_group_key), dtype=bool)

    if len(ddd_loss_clean) != total_clean_alarm or len(phys_score_clean) != total_clean_alarm:
        raise RuntimeError(
            f"collector length mismatch: loss={len(ddd_loss_clean)}, phys={len(phys_score_clean)}, total_clean_alarm={total_clean_alarm}"
        )

    fail_alarm = as_float_array(get_group_list(baseline_clean, "fail_alarm", clean_group_key))
    stage_one_time_alarm = as_float_array(get_group_list(baseline_clean, "stage_one_time_alarm", clean_group_key))
    stage_two_time_alarm = as_float_array(get_group_list(baseline_clean, "stage_two_time_alarm", clean_group_key))
    delta_cost_one_alarm = as_float_array(get_group_list(baseline_clean, "delta_cost_one_alarm", clean_group_key))
    delta_cost_two_alarm = as_float_array(get_group_list(baseline_clean, "delta_cost_two_alarm", clean_group_key))

    budget_items = build_budget_items(args.budgets, main_clean, strict_clean, clean_group_key)
    print("budget_items:", budget_items)

    attack_phys = attack_scores.get("score_phys_l2", {})
    attack_det = attack_scores.get("ddd_loss_alarm", {})
    attack_keys = sorted(attack_phys.keys())
    weak_keys = [k for k in args.weak_keys if k in attack_keys]
    strong_keys = [k for k in args.strong_keys if k in attack_keys]

    rows: List[Dict[str, Any]] = []
    sanity: Dict[str, Any] = {
        "clean_alarm_idx_aligned": bool(aligned),
        "clean_recover_fail_count": int(recover_fail_clean.sum()),
        "attack_recover_fail_count": {
            k: int(np.sum(np.asarray(attack_scores.get("recover_fail", {}).get(k, []), dtype=bool)))
            for k in attack_keys
        },
        "budget_items": budget_items,
    }

    baseline_row = {
        "budget_label": "Baseline",
        "score_name": "always_trigger",
        "budget_k": total_clean_alarm,
        "tau_on_clean": float("-inf"),
        "clean_selected_alarms": total_clean_alarm,
        "total_clean_false_alarms": total_clean_alarm,
        "uMTD_rate": float(get_group_scalar(baseline_clean, "useless_mtd_rate", clean_group_key)),
        "trigger_rate": float(get_group_scalar(baseline_clean, "trigger_rate", clean_group_key)),
        "fail_per_alarm": float(get_group_scalar(baseline_clean, "fail_per_alarm", clean_group_key)),
        "stage_one_time_per_alarm": float(get_group_scalar(baseline_clean, "stage_one_time_per_alarm", clean_group_key)),
        "stage_two_time_per_alarm": float(get_group_scalar(baseline_clean, "stage_two_time_per_alarm", clean_group_key)),
        "delta_cost_one_per_alarm": float(get_group_scalar(baseline_clean, "delta_cost_one_per_alarm", clean_group_key)),
        "delta_cost_two_per_alarm": float(get_group_scalar(baseline_clean, "delta_cost_two_per_alarm", clean_group_key)),
        "attack_retention_overall": 1.0,
        "weak_retention": 1.0,
        "strong_retention": 1.0,
    }
    rows.append(baseline_row)

    main_metric_trigger_match = None
    strict_metric_trigger_match = None

    for budget_label, k, source in budget_items:
        det_mask, det_tau, det_sel = topk_mask(ddd_loss_clean, k)
        phys_mask, phys_tau, phys_sel = topk_mask(phys_score_clean, k)

        det_clean_stats = compute_clean_row(
            total_clean_sample=total_clean_sample,
            total_clean_alarm=total_clean_alarm,
            mask=det_mask,
            fail_alarm=fail_alarm,
            stage_one_time_alarm=stage_one_time_alarm,
            stage_two_time_alarm=stage_two_time_alarm,
            delta_cost_one_alarm=delta_cost_one_alarm,
            delta_cost_two_alarm=delta_cost_two_alarm,
        )
        phys_clean_stats = compute_clean_row(
            total_clean_sample=total_clean_sample,
            total_clean_alarm=total_clean_alarm,
            mask=phys_mask,
            fail_alarm=fail_alarm,
            stage_one_time_alarm=stage_one_time_alarm,
            stage_two_time_alarm=stage_two_time_alarm,
            delta_cost_one_alarm=delta_cost_one_alarm,
            delta_cost_two_alarm=delta_cost_two_alarm,
        )

        det_row = {
            "budget_label": budget_label,
            "score_name": "detector_loss",
            "budget_k": k,
            "tau_on_clean": det_tau,
            "total_clean_false_alarms": total_clean_alarm,
            **det_clean_stats,
            "attack_retention_overall": pooled_retention(attack_det, det_tau, attack_keys),
            "weak_retention": pooled_retention(attack_det, det_tau, weak_keys),
            "strong_retention": pooled_retention(attack_det, det_tau, strong_keys),
        }
        phys_row = {
            "budget_label": budget_label,
            "score_name": "proposed_phys_score",
            "budget_k": k,
            "tau_on_clean": phys_tau,
            "total_clean_false_alarms": total_clean_alarm,
            **phys_clean_stats,
            "attack_retention_overall": pooled_retention(attack_phys, phys_tau, attack_keys),
            "weak_retention": pooled_retention(attack_phys, phys_tau, weak_keys),
            "strong_retention": pooled_retention(attack_phys, phys_tau, strong_keys),
        }
        rows.append(det_row)
        rows.append(phys_row)

        selected_idx_phys = clean_alarm_idx_from_collector[phys_mask]
        sanity[f"{budget_label}_det_tau"] = det_tau
        sanity[f"{budget_label}_phys_tau"] = phys_tau
        sanity[f"{budget_label}_phys_selected_idx"] = selected_idx_phys.tolist()

        if main_clean is not None and budget_label == "Main OP":
            metric_idx = as_int_array(get_group_list(main_clean, "clean_triggered_idx", clean_group_key))
            main_metric_trigger_match = np.array_equal(metric_idx, selected_idx_phys)
            sanity["main_metric_trigger_set_match_topk_phys"] = bool(main_metric_trigger_match)
        if strict_clean is not None and budget_label == "Strict OP":
            metric_idx = as_int_array(get_group_list(strict_clean, "clean_triggered_idx", clean_group_key))
            strict_metric_trigger_match = np.array_equal(metric_idx, selected_idx_phys)
            sanity["strict_metric_trigger_set_match_topk_phys"] = bool(strict_metric_trigger_match)

    ensure_parent(args.out_npy)
    ensure_parent(args.out_csv)
    np.save(
        args.out_npy,
        {
            "rows": rows,
            "sanity": sanity,
            "clean_scores_path": args.clean_scores,
            "attack_scores_path": args.attack_scores,
            "baseline_clean_metric_path": args.baseline_clean_metric,
            "main_clean_metric_path": args.main_clean_metric,
            "strict_clean_metric_path": strict_metric_path,
        },
        allow_pickle=True,
    )
    write_csv(args.out_csv, rows)

    print("\n==== Sanity checks ====")
    print("clean_alarm_idx_aligned:", sanity["clean_alarm_idx_aligned"])
    print("clean_recover_fail_count:", sanity["clean_recover_fail_count"])
    print("attack_recover_fail_count:", sanity["attack_recover_fail_count"])
    if main_metric_trigger_match is not None:
        print("main_metric_trigger_set_match_topk_phys:", main_metric_trigger_match)
    if strict_metric_trigger_match is not None:
        print("strict_metric_trigger_set_match_topk_phys:", strict_metric_trigger_match)

    print("\n==== Matched-budget ablation table ====")
    header = [
        "budget_label",
        "score_name",
        "budget_k",
        "tau_on_clean",
        "clean_selected_alarms",
        "uMTD_rate",
        "fail_per_alarm",
        "stage_one_time_per_alarm",
        "stage_two_time_per_alarm",
        "delta_cost_one_per_alarm",
        "delta_cost_two_per_alarm",
        "attack_retention_overall",
        "weak_retention",
        "strong_retention",
    ]
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(fmt(row[h]) for h in header))

    print("\nSaved npy:", args.out_npy)
    print("Saved csv:", args.out_csv)


if __name__ == "__main__":
    main()

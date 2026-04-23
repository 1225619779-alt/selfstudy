from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PROTECTED_GROUPS = ["(2,0.3)", "(3,0.2)", "(3,0.3)"]
ALL_ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a validation-only minimal score ablation from existing case39 clean/attack caches. "
            "No detector/recovery/backend rerun is performed."
        )
    )
    p.add_argument(
        "--clean-valid-cache",
        default="metric/case39/clean_alarm_cache_valid_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--attack-valid-cache",
        default="metric/case39/attack_alarm_cache_valid_50_repo_compatible_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--clean-test-cache",
        default="metric/case39/clean_alarm_cache_test_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--attack-test-cache",
        default="metric/case39/attack_alarm_cache_test_50_repo_compatible_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--main-clean-metric",
        default="metric/case39/metric_event_trigger_clean_tau_0.013319196253_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--strict-clean-metric",
        default="metric/case39/metric_event_trigger_clean_tau_0.016153267226_mode_0_0.03_1.1.npy",
    )
    p.add_argument(
        "--out-csv",
        default="metric/case39/minimal_score_ablation.csv",
    )
    p.add_argument(
        "--out-json",
        default="metric/case39/minimal_score_ablation.json",
    )
    p.add_argument(
        "--out-md",
        default="reports/case39_minimal_score_ablation.md",
    )
    return p.parse_args()


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def load_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def detect_group_key(metric: Dict[str, Any]) -> str:
    for field in ["total_clean_sample", "total_DDD_alarm", "total_trigger_after_verification"]:
        if isinstance(metric.get(field), dict) and metric[field]:
            keys = list(metric[field].keys())
            if "(0,0.0)" in keys:
                return "(0,0.0)"
            return keys[0]
    raise KeyError("Cannot detect clean group key from clean metric.")


def score_detector_loss_clean(records: Dict[str, List[Any]]) -> np.ndarray:
    return np.asarray(records["detector_loss"], dtype=float)


def score_angle_l2_clean(records: Dict[str, List[Any]]) -> np.ndarray:
    return np.asarray(records["verify_score"], dtype=float)


def score_angle_linf_clean(records: Dict[str, List[Any]]) -> np.ndarray:
    vals = [float(np.max(np.abs(np.asarray(x, dtype=float)))) for x in records["c_recover_no_ref"]]
    return np.asarray(vals, dtype=float)


def score_joint_angle_vmag_l2_clean(records: Dict[str, List[Any]]) -> np.ndarray:
    out: List[float] = []
    for c_vec, v_last, v_recover in zip(
        records["c_recover_no_ref"],
        records["v_last"],
        records["v_recover"],
    ):
        angle_part = np.asarray(c_vec, dtype=float)
        v_last_arr = np.asarray(v_last, dtype=np.complex128)
        v_recover_arr = np.asarray(v_recover, dtype=np.complex128)
        mag_part = np.abs(v_last_arr) - np.abs(v_recover_arr)
        out.append(float(np.sqrt(np.sum(np.square(angle_part)) + np.sum(np.square(mag_part)))))
    return np.asarray(out, dtype=float)


def score_detector_loss_attack(group: Dict[str, List[Any]]) -> np.ndarray:
    return np.asarray(group["detector_loss_alarm"], dtype=float)


def score_angle_l2_attack(group: Dict[str, List[Any]]) -> np.ndarray:
    return np.asarray(group["verify_score"], dtype=float)


def score_angle_linf_attack(group: Dict[str, List[Any]]) -> np.ndarray:
    vals = [float(np.max(np.abs(np.asarray(x, dtype=float)))) for x in group["c_recover_no_ref"]]
    return np.asarray(vals, dtype=float)


def score_joint_angle_vmag_l2_attack(group: Dict[str, List[Any]]) -> np.ndarray:
    out: List[float] = []
    for c_vec, v_att_last, v_recover in zip(
        group["c_recover_no_ref"],
        group["v_att_last"],
        group["v_recover"],
    ):
        angle_part = np.asarray(c_vec, dtype=float)
        v_att_arr = np.asarray(v_att_last, dtype=np.complex128)
        v_recover_arr = np.asarray(v_recover, dtype=np.complex128)
        mag_part = np.abs(v_att_arr) - np.abs(v_recover_arr)
        out.append(float(np.sqrt(np.sum(np.square(angle_part)) + np.sum(np.square(mag_part)))))
    return np.asarray(out, dtype=float)


def build_clean_scores(records: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    return {
        "detector_loss": score_detector_loss_clean(records),
        "angle_l2": score_angle_l2_clean(records),
        "angle_linf": score_angle_linf_clean(records),
        "joint_angle_vmag_l2": score_joint_angle_vmag_l2_clean(records),
    }


def build_attack_scores(groups: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for g in ALL_ATTACK_GROUPS:
        group = groups[g]
        out[g] = {
            "detector_loss": score_detector_loss_attack(group),
            "angle_l2": score_angle_l2_attack(group),
            "angle_linf": score_angle_linf_attack(group),
            "joint_angle_vmag_l2": score_joint_angle_vmag_l2_attack(group),
        }
    return out


def valid_mask(values: Iterable[Any]) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=float))


def overall_arr_from_masks(mask_by_group: Dict[str, np.ndarray], group_sizes: Dict[str, int]) -> float:
    triggers = 0
    alarms = 0
    for g in ALL_ATTACK_GROUPS:
        alarms += int(group_sizes[g])
        triggers += int(mask_by_group[g].sum())
    return float(triggers / alarms) if alarms else float("nan")


def protected_min_arr_from_masks(mask_by_group: Dict[str, np.ndarray], group_sizes: Dict[str, int]) -> float:
    vals = []
    for g in PROTECTED_GROUPS:
        alarms = int(group_sizes[g])
        vals.append(float(mask_by_group[g].sum() / alarms) if alarms else float("nan"))
    return float(np.nanmin(np.asarray(vals, dtype=float)))


def select_tau_validation(
    clean_scores: np.ndarray,
    attack_scores: Dict[str, np.ndarray],
    clean_recovery_error: np.ndarray,
    attack_recovery_error: Dict[str, np.ndarray],
    *,
    overall_target: float,
    protected_target: float,
) -> Dict[str, Any]:
    clean_valid = (~clean_recovery_error) & valid_mask(clean_scores)
    attack_valid = {g: (~attack_recovery_error[g]) & valid_mask(attack_scores[g]) for g in ALL_ATTACK_GROUPS}
    group_sizes = {g: int(attack_valid[g].sum()) for g in ALL_ATTACK_GROUPS}

    candidates = [float(x) for x in clean_scores[clean_valid]]
    for g in ALL_ATTACK_GROUPS:
        candidates.extend(float(x) for x in attack_scores[g][attack_valid[g]])
    if not candidates:
        raise ValueError("No finite validation scores available for tau selection.")
    tau_candidates = np.asarray(sorted(set(candidates), reverse=True), dtype=float)

    best: Dict[str, Any] | None = None
    for tau in tau_candidates:
        clean_mask = clean_valid & (clean_scores >= tau)
        attack_masks = {
            g: attack_valid[g] & (attack_scores[g] >= tau)
            for g in ALL_ATTACK_GROUPS
        }
        overall_arr = overall_arr_from_masks(attack_masks, group_sizes)
        protected_min_arr = protected_min_arr_from_masks(attack_masks, group_sizes)
        if overall_arr + 1e-12 < overall_target:
            continue
        if protected_min_arr + 1e-12 < protected_target:
            continue
        candidate = {
            "tau": float(tau),
            "clean_trigger_count": int(clean_mask.sum()),
            "overall_arr": float(overall_arr),
            "protected_min_arr": float(protected_min_arr),
        }
        if best is None:
            best = candidate
            continue
        if candidate["clean_trigger_count"] < best["clean_trigger_count"]:
            best = candidate
            continue
        if (
            candidate["clean_trigger_count"] == best["clean_trigger_count"]
            and candidate["tau"] > best["tau"]
        ):
            best = candidate

    if best is None:
        raise ValueError(
            f"No validation tau satisfies overall>={overall_target} and protected>={protected_target}."
        )
    return best


def clean_stats_from_mask(records: Dict[str, List[Any]], mask: np.ndarray) -> Dict[str, Any]:
    total_alarms = len(records["alarm_idx"])
    total_triggers = int(mask.sum())
    backend_mtd_fail = np.asarray(records["backend_mtd_fail"], dtype=float)
    backend_metric_fail = np.asarray(records["backend_metric_fail"], dtype=float)
    stage_two_time = np.asarray(records["stage_two_time"], dtype=float)
    cost_no = np.asarray(records["cost_no_mtd"], dtype=float)
    cost_two = np.asarray(records["cost_with_mtd_two"], dtype=float)

    fail_alarm = np.where(mask, backend_mtd_fail, 0.0)
    stage_two_alarm = np.where(mask & np.isfinite(stage_two_time), stage_two_time, 0.0)
    delta_two = np.where(mask & np.isfinite(cost_no) & np.isfinite(cost_two), cost_two - cost_no, 0.0)

    return {
        "test_clean_alarm_count": int(total_alarms),
        "test_clean_trigger_count": int(total_triggers),
        "test_clean_trigger_rate": float(total_triggers / total_alarms) if total_alarms else float("nan"),
        "test_clean_backend_mtd_fail_count": int(np.asarray(backend_mtd_fail[mask], dtype=float).sum()) if total_triggers else 0,
        "test_clean_backend_metric_fail_count": int(np.asarray(backend_metric_fail[mask], dtype=float).sum()) if total_triggers else 0,
        "test_clean_fail_per_alarm": float(np.mean(fail_alarm)) if total_alarms else float("nan"),
        "test_clean_fail_per_trigger": float(np.mean(backend_mtd_fail[mask])) if total_triggers else float("nan"),
        "test_clean_backend_metric_fail_rate_among_triggers": float(np.mean(backend_metric_fail[mask])) if total_triggers else float("nan"),
        "test_clean_stage_two_time_per_alarm": float(np.mean(stage_two_alarm)) if total_alarms else float("nan"),
        "test_clean_delta_cost_two_per_alarm": float(np.mean(delta_two)) if total_alarms else float("nan"),
    }


def attack_stats_from_mask(
    groups: Dict[str, Dict[str, List[Any]]],
    masks: Dict[str, np.ndarray],
    *,
    next_load_mode: str = "sample_length",
) -> Dict[str, Any]:
    total_alarms = 0
    total_triggers = 0
    total_backend_metric_fail = 0
    total_backend_mtd_fail = 0
    protected_arr: Dict[str, float] = {}

    for g in ALL_ATTACK_GROUPS:
        group = groups[g]
        mask = masks[g]
        alarms = len(group["alarm_idx"])
        triggers = int(mask.sum())
        total_alarms += alarms
        total_triggers += triggers
        total_backend_metric_fail += int(
            np.asarray(group["variants"][next_load_mode]["backend_metric_fail"], dtype=bool)[mask].sum()
        )
        total_backend_mtd_fail += int(np.asarray(group["backend_mtd_fail"], dtype=float)[mask].sum())
        arr = float(triggers / alarms) if alarms else float("nan")
        if g in PROTECTED_GROUPS:
            protected_arr[g] = arr

    protected_min = float(np.nanmin(np.asarray(list(protected_arr.values()), dtype=float)))
    return {
        "test_attack_alarm_count": int(total_alarms),
        "test_attack_trigger_count": int(total_triggers),
        "test_attack_overall_arr": float(total_triggers / total_alarms) if total_alarms else float("nan"),
        "test_attack_protected_min_arr": protected_min,
        "test_attack_backend_metric_fail_count": int(total_backend_metric_fail),
        "test_attack_backend_mtd_fail_count": int(total_backend_mtd_fail),
        "test_attack_backend_metric_fail_rate_among_triggers": float(total_backend_metric_fail / total_triggers) if total_triggers else float("nan"),
        "test_attack_backend_mtd_fail_rate_among_triggers": float(total_backend_mtd_fail / total_triggers) if total_triggers else float("nan"),
    }


def group_arrs_from_mask(groups: Dict[str, Dict[str, List[Any]]], masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for g in ALL_ATTACK_GROUPS:
        alarms = len(groups[g]["alarm_idx"])
        out[g] = float(masks[g].sum() / alarms) if alarms else float("nan")
    return out


def load_budget_k(metric_path: str) -> int:
    d = load_dict(metric_path)
    group_key = detect_group_key(d)
    return int(d["total_trigger_after_verification"][group_key])


def make_markdown(rows: List[Dict[str, Any]]) -> str:
    header = [
        "regime",
        "score_family",
        "tau_valid",
        "valid_clean_trigger_count",
        "test_clean_trigger_count",
        "test_attack_trigger_count",
        "test_attack_overall_arr",
        "test_attack_protected_min_arr",
        "test_clean_backend_mtd_fail_count",
        "test_clean_stage_two_time_per_alarm",
        "test_clean_delta_cost_two_per_alarm",
        "test_clean_fail_per_alarm",
        "test_clean_fail_per_trigger",
    ]
    lines = [
        "# Case39 Minimal Score Ablation",
        "",
        "Validation-only selection policy:",
        "- Main: overall ARR >= 0.90 and protected-group ARR >= 0.95",
        "- Strict: overall ARR >= 0.85 and protected-group ARR >= 0.90",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        vals = []
        for key in header:
            value = row[key]
            if isinstance(value, float):
                if np.isnan(value):
                    vals.append("nan")
                else:
                    vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `angle_l2` is the current main score.")
    lines.append("- `angle_linf` uses max abs angle recovery deviation on non-reference buses.")
    lines.append("- `joint_angle_vmag_l2` uses sqrt(||dtheta||_2^2 + ||d|V|||_2^2).")
    lines.append("- Burden metrics are reported per clean alarm, not per triggered alarm.")
    lines.append("- Count columns are included to make failure denominators explicit.")
    return "\n".join(lines) + "\n"


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    clean_valid = load_dict(args.clean_valid_cache)
    attack_valid = load_dict(args.attack_valid_cache)
    clean_test = load_dict(args.clean_test_cache)
    attack_test = load_dict(args.attack_test_cache)

    valid_clean_records = clean_valid["records"]
    test_clean_records = clean_test["records"]
    valid_attack_groups = attack_valid["groups"]
    test_attack_groups = attack_test["groups"]

    valid_clean_scores = build_clean_scores(valid_clean_records)
    valid_attack_scores = build_attack_scores(valid_attack_groups)
    test_clean_scores = build_clean_scores(test_clean_records)
    test_attack_scores = build_attack_scores(test_attack_groups)

    valid_clean_recovery_error = np.asarray(valid_clean_records["recovery_error"], dtype=bool)
    test_clean_recovery_error = np.asarray(test_clean_records["recovery_error"], dtype=bool)
    valid_attack_recovery_error = {
        g: np.asarray(valid_attack_groups[g]["recovery_error"], dtype=bool)
        for g in ALL_ATTACK_GROUPS
    }
    test_attack_recovery_error = {
        g: np.asarray(test_attack_groups[g]["recovery_error"], dtype=bool)
        for g in ALL_ATTACK_GROUPS
    }

    regime_specs = {
        "main": {
            "overall_target": 0.90,
            "protected_target": 0.95,
            "paper_budget_k": load_budget_k(args.main_clean_metric),
        },
        "strict": {
            "overall_target": 0.85,
            "protected_target": 0.90,
            "paper_budget_k": load_budget_k(args.strict_clean_metric),
        },
    }

    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "clean_valid_cache": args.clean_valid_cache,
        "attack_valid_cache": args.attack_valid_cache,
        "clean_test_cache": args.clean_test_cache,
        "attack_test_cache": args.attack_test_cache,
        "regimes": {},
    }

    for regime, spec in regime_specs.items():
        payload["regimes"][regime] = {}
        for score_family in ["angle_l2", "angle_linf", "joint_angle_vmag_l2", "detector_loss"]:
            selected = select_tau_validation(
                clean_scores=valid_clean_scores[score_family],
                attack_scores={g: valid_attack_scores[g][score_family] for g in ALL_ATTACK_GROUPS},
                clean_recovery_error=valid_clean_recovery_error,
                attack_recovery_error=valid_attack_recovery_error,
                overall_target=float(spec["overall_target"]),
                protected_target=float(spec["protected_target"]),
            )

            test_clean_valid = (~test_clean_recovery_error) & valid_mask(test_clean_scores[score_family])
            test_clean_mask = test_clean_valid & (test_clean_scores[score_family] >= selected["tau"])
            test_attack_masks = {
                g: ((~test_attack_recovery_error[g]) & valid_mask(test_attack_scores[g][score_family]) & (test_attack_scores[g][score_family] >= selected["tau"]))
                for g in ALL_ATTACK_GROUPS
            }

            row = {
                "regime": regime,
                "score_family": score_family,
                "tau_valid": float(selected["tau"]),
                "valid_clean_trigger_count": int(selected["clean_trigger_count"]),
                "paper_clean_budget_k": int(spec["paper_budget_k"]),
                "valid_overall_arr": float(selected["overall_arr"]),
                "valid_protected_min_arr": float(selected["protected_min_arr"]),
            }
            row.update(clean_stats_from_mask(test_clean_records, test_clean_mask))
            row.update(attack_stats_from_mask(test_attack_groups, test_attack_masks))
            row["groupwise_arr_json"] = json.dumps(group_arrs_from_mask(test_attack_groups, test_attack_masks), ensure_ascii=True, sort_keys=True)
            rows.append(row)

            payload["regimes"][regime][score_family] = dict(row)

    ensure_parent(args.out_csv)
    ensure_parent(args.out_json)
    ensure_parent(args.out_md)
    write_csv(args.out_csv, rows)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(make_markdown(rows), encoding="utf-8")

    print(f"Saved csv: {args.out_csv}")
    print(f"Saved json: {args.out_json}")
    print(f"Saved md: {args.out_md}")


if __name__ == "__main__":
    main()

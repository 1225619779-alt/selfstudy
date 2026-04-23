from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.run_metadata import file_fingerprint, git_head


ATTACK_GROUPS = ["(1,0.2)", "(1,0.3)", "(2,0.2)", "(2,0.3)", "(3,0.2)", "(3,0.3)"]
PROTECTED_GROUPS = ["(2,0.3)", "(3,0.2)", "(3,0.3)"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize one backend candidate from recomputed clean/attack caches.")
    p.add_argument("--clean-cache", required=True)
    p.add_argument("--attack-cache", required=True)
    p.add_argument("--tau-main", type=float, required=True)
    p.add_argument("--tau-strict", type=float, required=True)
    p.add_argument("--x-facts-ratio", type=float, required=True)
    p.add_argument("--varrho", type=float, required=True)
    p.add_argument("--upper-scale", type=float, required=True)
    p.add_argument("--multi-run-no", type=int, required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def matched_threshold(detector_scores: np.ndarray, target_triggers: int) -> float:
    if target_triggers <= 0:
        return float("inf")
    scores = np.asarray(detector_scores, dtype=float)
    if target_triggers >= scores.size:
        return float(np.nanmin(scores)) - 1e-12
    idx = np.argpartition(scores, scores.size - target_triggers)[scores.size - target_triggers]
    return float(scores[idx])


def clean_regime(clean_cache: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    rec = clean_cache["records"]
    total_alarms = len(rec["alarm_idx"])
    total_triggers = int(mask.sum())
    backend_mtd_fail = np.asarray(rec["backend_mtd_fail"], dtype=float)
    backend_metric_fail = np.asarray(rec["backend_metric_fail"], dtype=float)
    stage_one = np.asarray(rec["stage_one_time"], dtype=float)
    stage_two = np.asarray(rec["stage_two_time"], dtype=float)
    cost_no = np.asarray(rec["cost_no_mtd"], dtype=float)
    cost_one = np.asarray(rec["cost_with_mtd_one"], dtype=float)
    cost_two = np.asarray(rec["cost_with_mtd_two"], dtype=float)
    fail_alarm = np.where(mask, backend_mtd_fail, 0.0)
    stage_one_alarm = np.where(mask & np.isfinite(stage_one), stage_one, 0.0)
    stage_two_alarm = np.where(mask & np.isfinite(stage_two), stage_two, 0.0)
    delta_one = np.where(mask & np.isfinite(cost_no) & np.isfinite(cost_one), cost_one - cost_no, 0.0)
    delta_two = np.where(mask & np.isfinite(cost_no) & np.isfinite(cost_two), cost_two - cost_no, 0.0)
    return {
        "total_alarms": total_alarms,
        "clean_alarm_count": total_alarms,
        "total_triggers": total_triggers,
        "clean_trigger_count": total_triggers,
        "clean_trigger_rate": float(total_triggers / total_alarms) if total_alarms else float("nan"),
        "fail_count": int(np.asarray(backend_mtd_fail[mask], dtype=float).sum()) if total_triggers else 0,
        "backend_metric_fail_count": int(np.asarray(backend_metric_fail[mask], dtype=float).sum()) if total_triggers else 0,
        "fail_per_alarm": float(np.mean(fail_alarm)) if total_alarms else float("nan"),
        "fail_per_trigger": float(np.mean(backend_mtd_fail[mask])) if total_triggers else float("nan"),
        "backend_metric_fail_rate_among_triggers": float(np.mean(backend_metric_fail[mask])) if total_triggers else float("nan"),
        "stage_one_time_per_alarm": float(np.mean(stage_one_alarm)) if total_alarms else float("nan"),
        "stage_one_time_per_trigger": float(np.nanmean(stage_one[mask])) if total_triggers else float("nan"),
        "stage_two_time_per_alarm": float(np.mean(stage_two_alarm)) if total_alarms else float("nan"),
        "stage_two_time_per_trigger": float(np.nanmean(stage_two[mask])) if total_triggers else float("nan"),
        "delta_cost_one_per_alarm": float(np.mean(delta_one)) if total_alarms else float("nan"),
        "delta_cost_two_per_alarm": float(np.mean(delta_two)) if total_alarms else float("nan"),
        "delta_cost_two_per_trigger_success_only": float(np.nanmean((cost_two - cost_no)[mask & np.isfinite(cost_no) & np.isfinite(cost_two)]))
        if np.any(mask & np.isfinite(cost_no) & np.isfinite(cost_two))
        else float("nan"),
    }


def attack_regime(attack_cache: Dict[str, Any], gate_masks: Dict[str, np.ndarray], next_load_mode: str = "sample_length") -> Dict[str, Any]:
    groups = attack_cache["groups"]
    total_alarms = 0
    total_triggers = 0
    total_backend_metric_fail = 0
    total_backend_mtd_fail = 0
    protected_arr = {}
    for key in ATTACK_GROUPS:
        g = groups[key]
        alarms = len(g["alarm_idx"])
        mask = gate_masks[key]
        triggers = int(mask.sum())
        total_alarms += alarms
        total_triggers += triggers
        total_backend_metric_fail += int(np.asarray(g["variants"][next_load_mode]["backend_metric_fail"], dtype=bool)[mask].sum())
        total_backend_mtd_fail += int(np.asarray(g["backend_mtd_fail"], dtype=float)[mask].sum())
        arr = float(triggers / alarms) if alarms else float("nan")
        if key in PROTECTED_GROUPS:
            protected_arr[key] = arr
    return {
        "overall_arr": float(total_triggers / total_alarms) if total_alarms else float("nan"),
        "total_alarms": total_alarms,
        "attack_alarm_count": total_alarms,
        "total_triggers": total_triggers,
        "attack_trigger_count": total_triggers,
        "backend_metric_fail_count": int(total_backend_metric_fail),
        "backend_mtd_fail_count": int(total_backend_mtd_fail),
        "backend_metric_fail_rate_among_triggers": float(total_backend_metric_fail / total_triggers) if total_triggers else float("nan"),
        "backend_mtd_fail_rate_among_triggers": float(total_backend_mtd_fail / total_triggers) if total_triggers else float("nan"),
        "protected_group_arr": protected_arr,
    }


def main() -> None:
    args = parse_args()
    clean_cache = np.load(args.clean_cache, allow_pickle=True).item()
    attack_cache = np.load(args.attack_cache, allow_pickle=True).item()

    clean_scores = np.asarray(clean_cache["records"]["verify_score"], dtype=float)
    clean_detector = np.asarray(clean_cache["records"]["detector_loss"], dtype=float)
    clean_recovery_error = np.asarray(clean_cache["records"]["recovery_error"], dtype=bool)
    clean_valid = ~clean_recovery_error

    clean_mask_baseline = clean_valid.copy()
    clean_mask_main = clean_valid & (clean_scores >= float(args.tau_main))
    clean_mask_strict = clean_valid & (clean_scores >= float(args.tau_strict))
    det_th_main = matched_threshold(clean_detector[clean_valid], int(clean_mask_main.sum()))
    det_th_strict = matched_threshold(clean_detector[clean_valid], int(clean_mask_strict.sum()))
    clean_mask_det_main = clean_valid & (clean_detector >= det_th_main)
    clean_mask_det_strict = clean_valid & (clean_detector >= det_th_strict)

    attack_masks_phys_main: Dict[str, np.ndarray] = {}
    attack_masks_phys_strict: Dict[str, np.ndarray] = {}
    attack_masks_det_main: Dict[str, np.ndarray] = {}
    attack_masks_det_strict: Dict[str, np.ndarray] = {}
    attack_masks_baseline: Dict[str, np.ndarray] = {}

    for key in ATTACK_GROUPS:
        g = attack_cache["groups"][key]
        scores = np.asarray(g["verify_score"], dtype=float)
        det = np.asarray(g["detector_loss_alarm"], dtype=float)
        rec_err = np.asarray(g["recovery_error"], dtype=bool)
        valid = ~rec_err
        attack_masks_baseline[key] = valid.copy()
        attack_masks_phys_main[key] = valid & (scores >= float(args.tau_main))
        attack_masks_phys_strict[key] = valid & (scores >= float(args.tau_strict))
        attack_masks_det_main[key] = valid & (det >= det_th_main)
        attack_masks_det_strict[key] = valid & (det >= det_th_strict)

    clean_meta = dict(clean_cache.get("metadata", {}))
    attack_meta = dict(attack_cache.get("metadata", {}))
    expected = {
        "varrho": float(args.varrho),
        "upper_scale": float(args.upper_scale),
        "multi_run_no": int(args.multi_run_no),
        "x_facts_ratio": float(args.x_facts_ratio),
        "tau_main_exact": float(args.tau_main),
        "tau_strict_exact": float(args.tau_strict),
    }
    consistency = {
        "clean_matches_candidate": bool(
            np.isclose(float(clean_meta.get("varrho", np.nan)), expected["varrho"])
            and np.isclose(float(clean_meta.get("upper_scale", np.nan)), expected["upper_scale"])
            and int(clean_meta.get("multi_run_no", -1)) == expected["multi_run_no"]
            and np.isclose(float(clean_meta.get("x_facts_ratio", np.nan)), expected["x_facts_ratio"])
        ),
        "attack_matches_candidate": bool(
            np.isclose(float(attack_meta.get("varrho", np.nan)), expected["varrho"])
            and np.isclose(float(attack_meta.get("upper_scale", np.nan)), expected["upper_scale"])
            and int(attack_meta.get("multi_run_no", -1)) == expected["multi_run_no"]
            and np.isclose(float(attack_meta.get("x_facts_ratio", np.nan)), expected["x_facts_ratio"])
        ),
        "split_matches": clean_meta.get("split") == attack_meta.get("split"),
        "case_matches": clean_meta.get("case_name") == attack_meta.get("case_name"),
        "seed_matches": clean_meta.get("seed_base") == attack_meta.get("seed_base"),
    }

    payload = {
        "candidate": {
            "x_facts_ratio": float(args.x_facts_ratio),
            "varrho": float(args.varrho),
            "upper_scale": float(args.upper_scale),
            "multi_run_no": int(args.multi_run_no),
        },
        "tau_main_exact": float(args.tau_main),
        "tau_strict_exact": float(args.tau_strict),
        "runtime": {
            "git_head": git_head(cwd=str(Path(__file__).resolve().parent)),
            "clean_cache_fingerprint": file_fingerprint(args.clean_cache),
            "attack_cache_fingerprint": file_fingerprint(args.attack_cache),
        },
        "clean_metadata": clean_meta,
        "attack_metadata": attack_meta,
        "consistency": consistency,
        "detector_loss_threshold_main_budget": float(det_th_main),
        "detector_loss_threshold_strict_budget": float(det_th_strict),
        "clean": {
            "baseline": clean_regime(clean_cache, clean_mask_baseline),
            "proposed_main": clean_regime(clean_cache, clean_mask_main),
            "proposed_strict": clean_regime(clean_cache, clean_mask_strict),
            "detector_main_budget": clean_regime(clean_cache, clean_mask_det_main),
            "detector_strict_budget": clean_regime(clean_cache, clean_mask_det_strict),
        },
        "attack": {
            "baseline": attack_regime(attack_cache, attack_masks_baseline),
            "proposed_main": attack_regime(attack_cache, attack_masks_phys_main),
            "proposed_strict": attack_regime(attack_cache, attack_masks_phys_strict),
            "detector_main_budget": attack_regime(attack_cache, attack_masks_det_main),
            "detector_strict_budget": attack_regime(attack_cache, attack_masks_det_strict),
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved summary: {args.output_json}")


if __name__ == "__main__":
    main()

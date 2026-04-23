from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.run_metadata import git_head


ROOT = Path(__file__).resolve().parent
CAND_DIR = ROOT / "metric" / "case39" / "backend_calibration_valid_grid" / "candidates"
CONSISTENCY_JSON = ROOT / "metric" / "case39" / "backend_calibration_consistency.json"
SPOT_CACHE = ROOT / "metric" / "case39" / "spot_rerun" / "xf_0.2_var_0.015_up_1.05_mr_15_clean_from_scratch.npy"
SPOT_REF_JSON = ROOT / "metric" / "case39" / "backend_calibration_valid_grid" / "candidates" / "xf_0.2_var_0.015_up_1.05_mr_15.json"
REPORT_MD = ROOT / "reports" / "case39_backend_calibration_early_stop_handoff.md"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_candidate(candidates: List[Path]) -> Path:
    def key(path: Path) -> tuple[float, float]:
        payload = load_json(path)
        clean = payload["clean"]["proposed_main"]
        return (float(clean["fail_per_alarm"]), float(clean["stage_two_time_per_alarm"]))

    return min(candidates, key=key)


def maybe_load_spot() -> Dict[str, Any] | None:
    if not SPOT_CACHE.exists() or not SPOT_REF_JSON.exists():
        return None
    payload = np.load(SPOT_CACHE, allow_pickle=True).item()
    ref = load_json(SPOT_REF_JSON)
    tau_main = float(ref["tau_main_exact"])
    records = payload["records"]
    verify_score = np.asarray(records["verify_score"], dtype=float)
    recovery_error = np.asarray(records["recovery_error"], dtype=bool)
    backend_mtd_fail = np.asarray(records["backend_mtd_fail"], dtype=float)
    stage_one = np.asarray(records["stage_one_time"], dtype=float)
    stage_two = np.asarray(records["stage_two_time"], dtype=float)
    valid = ~recovery_error
    mask = valid & (verify_score >= tau_main)
    total_alarms = int(len(records["alarm_idx"]))
    total_triggers = int(mask.sum())
    return {
        "candidate_name": SPOT_REF_JSON.name,
        "total_alarms": total_alarms,
        "trigger_count": total_triggers,
        "fail_count": int(np.asarray(backend_mtd_fail[mask], dtype=float).sum()) if total_triggers else 0,
        "fail_per_alarm": float(np.mean(np.where(mask, backend_mtd_fail, 0.0))) if total_alarms else float("nan"),
        "fail_per_trigger": float(np.mean(backend_mtd_fail[mask])) if total_triggers else float("nan"),
        "stage_one_time_per_alarm": float(np.mean(np.where(mask & np.isfinite(stage_one), stage_one, 0.0))) if total_alarms else float("nan"),
        "stage_two_time_per_alarm": float(np.mean(np.where(mask & np.isfinite(stage_two), stage_two, 0.0))) if total_alarms else float("nan"),
    }


def main() -> None:
    candidates = sorted(CAND_DIR.glob("*.json"))
    targeted = [p for p in candidates if p.name.startswith("xf_0.5_var_0.015_")]
    consistency = load_json(CONSISTENCY_JSON) if CONSISTENCY_JSON.exists() else {}
    best = best_candidate(candidates) if candidates else None
    spot = maybe_load_spot()

    lines = [
        "# Case39 Backend Calibration Early-Stop Handoff",
        "",
        f"- git_head: `{git_head(cwd=str(ROOT))}`",
        f"- completed_candidate_count: `{len(candidates)}`",
        f"- targeted_candidate_count: `{len(targeted)}`",
        "",
        "## Consistency",
    ]
    if consistency:
        inv = consistency.get("invariance", {})
        lines.extend(
            [
                f"- metadata_mismatch_candidates: `{len(consistency.get('metadata_mismatch_candidates', []))}`",
                f"- clean_trigger_count_main_unique: `{inv.get('clean_trigger_count_main_unique', [])}`",
                f"- clean_trigger_count_strict_unique: `{inv.get('clean_trigger_count_strict_unique', [])}`",
                f"- attack_trigger_count_main_unique: `{inv.get('attack_trigger_count_main_unique', [])}`",
                f"- attack_overall_arr_main_unique: `{inv.get('attack_overall_arr_main_unique', [])}`",
                f"- attack_trigger_count_strict_unique: `{inv.get('attack_trigger_count_strict_unique', [])}`",
                f"- attack_overall_arr_strict_unique: `{inv.get('attack_overall_arr_strict_unique', [])}`",
            ]
        )
    else:
        lines.append("- consistency summary not available")

    if best is not None:
        payload = load_json(best)
        clean = payload["clean"]["proposed_main"]
        attack = payload["attack"]["proposed_main"]
        lines.extend(
            [
                "",
                "## Best Current Candidate",
                f"- candidate: `{best.name}`",
                f"- clean main fail/alarm: `{clean['fail_per_alarm']:.12f}`",
                f"- clean main fail/trigger: `{clean['fail_per_trigger']:.12f}`",
                f"- clean main trigger_count: `{clean['clean_trigger_count']}`",
                f"- clean main stage_two_time/alarm: `{clean['stage_two_time_per_alarm']:.12f}`",
                f"- attack main overall_arr: `{attack['overall_arr']:.12f}`",
                f"- attack main trigger_count: `{attack['attack_trigger_count']}`",
                f"- attack main backend_mtd_fail_rate_among_triggers: `{attack['backend_mtd_fail_rate_among_triggers']:.12f}`",
            ]
        )

    lines.extend(["", "## Targeted x=0.5, var=0.015"])
    if targeted:
        for path in targeted:
            payload = load_json(path)
            clean = payload["clean"]["proposed_main"]
            attack = payload["attack"]["proposed_main"]
            lines.extend(
                [
                    f"- `{path.name}`: clean fail/alarm=`{clean['fail_per_alarm']:.12f}`, "
                    f"clean fail/trigger=`{clean['fail_per_trigger']:.12f}`, "
                    f"stage_two_time/alarm=`{clean['stage_two_time_per_alarm']:.12f}`, "
                    f"attack overall_arr=`{attack['overall_arr']:.12f}`, "
                    f"attack backend_mtd_fail_rate_among_triggers=`{attack['backend_mtd_fail_rate_among_triggers']:.12f}`",
                ]
            )
    else:
        lines.append("- targeted candidates not finished yet")

    lines.extend(["", "## From-Scratch Spot Rerun"])
    if spot is not None:
        payload = load_json(SPOT_REF_JSON)
        clean = payload["clean"]["proposed_main"]
        lines.extend(
            [
                f"- reference candidate: `{SPOT_REF_JSON.name}`",
                f"- cached candidate trigger_count: `{clean['clean_trigger_count']}`",
                f"- spot rerun trigger_count: `{spot['trigger_count']}`",
                f"- cached candidate fail_count: `{clean['fail_count']}`",
                f"- spot rerun fail_count: `{spot['fail_count']}`",
                f"- cached candidate fail/alarm: `{clean['fail_per_alarm']:.12f}`",
                f"- spot rerun fail/alarm: `{spot['fail_per_alarm']:.12f}`",
                f"- cached candidate fail/trigger: `{clean['fail_per_trigger']:.12f}`",
                f"- spot rerun fail/trigger: `{spot['fail_per_trigger']:.12f}`",
                f"- cached candidate stage_one_time/alarm: `{clean['stage_one_time_per_alarm']:.12f}`",
                f"- spot rerun stage_one_time/alarm: `{spot['stage_one_time_per_alarm']:.12f}`",
                f"- cached candidate stage_two_time/alarm: `{clean['stage_two_time_per_alarm']:.12f}`",
                f"- spot rerun stage_two_time/alarm: `{spot['stage_two_time_per_alarm']:.12f}`",
            ]
        )
    else:
        lines.append("- spot rerun not finished yet")

    recommendation = "case39 = stress-test / limitation evidence"
    if targeted:
        best_fail = min(
            load_json(path)["clean"]["proposed_main"]["fail_per_alarm"] for path in targeted
        )
        if best_fail <= 0.60:
            recommendation = "case39 maybe weak larger benchmark; needs explicit manual review"

    lines.extend(
        [
            "",
            "## Recommendation",
            f"- {recommendation}",
            "- case14 remains the main detailed evidence",
            "- do not claim robust end-to-end case39 backend success",
        ]
    )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()

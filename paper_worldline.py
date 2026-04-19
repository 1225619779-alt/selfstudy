from __future__ import annotations

from typing import Iterable


# Current paper worldline for the case14 study.
CASE_NAME = "case14"

TAU_BASELINE_LABEL = "-1.0"
TAU_MAIN_LABEL = "0.03318"
TAU_STRICT_LABEL = "0.03675"

TAU_BASELINE = float(TAU_BASELINE_LABEL)
TAU_MAIN = float(TAU_MAIN_LABEL)
TAU_STRICT = float(TAU_STRICT_LABEL)

VAR_RHO_LABEL = "0.03"
UPPER_SCALE_LABEL = "1.1"
MODE_LABEL = "0"

ATTACK_SCORE_TOTAL_RUN = 50
DEFAULT_MIXED_SCHEDULE = "clean:80;att-1-0.2:30;clean:40;att-2-0.2:30;clean:40;att-3-0.3:30;clean:80"


def metric_dir(case_name: str = CASE_NAME) -> str:
    return f"metric/{case_name}"


def clean_metric_path(
    tau_label: str,
    *,
    case_name: str = CASE_NAME,
    mode_label: str = MODE_LABEL,
    var_rho_label: str = VAR_RHO_LABEL,
    upper_scale_label: str = UPPER_SCALE_LABEL,
) -> str:
    return (
        f"{metric_dir(case_name)}/metric_event_trigger_clean_tau_{tau_label}"
        f"_mode_{mode_label}_{var_rho_label}_{upper_scale_label}.npy"
    )


def mixed_metric_path(tau_label: str, *, case_name: str = CASE_NAME) -> str:
    return f"{metric_dir(case_name)}/metric_mixed_timeline_tau_{tau_label}.npy"


BASELINE_CLEAN_METRIC = clean_metric_path(TAU_BASELINE_LABEL)
MAIN_CLEAN_METRIC = clean_metric_path(TAU_MAIN_LABEL)
STRICT_CLEAN_METRIC = clean_metric_path(TAU_STRICT_LABEL)

LEGACY_MAIN_CLEAN_METRIC = clean_metric_path("0.021")
LEGACY_STRICT_CLEAN_METRICS = [
    clean_metric_path("0.03"),
    clean_metric_path("0.030"),
]
STRICT_CLEAN_METRIC_CANDIDATES = [STRICT_CLEAN_METRIC, *LEGACY_STRICT_CLEAN_METRICS]

BASELINE_MIXED_METRIC = mixed_metric_path(TAU_BASELINE_LABEL)
MAIN_MIXED_METRIC = mixed_metric_path(TAU_MAIN_LABEL)
STRICT_MIXED_METRIC = mixed_metric_path(TAU_STRICT_LABEL)

CLEAN_SCORE_METRIC = f"{metric_dir()}/metric_clean_alarm_scores_full.npy"
ATTACK_SCORE_METRIC = f"{metric_dir()}/metric_attack_alarm_scores_{ATTACK_SCORE_TOTAL_RUN}.npy"
GATE_ABLATION_NPY = f"{metric_dir()}/metric_gate_ablation_summary_{TAU_MAIN_LABEL}_{TAU_STRICT_LABEL}.npy"
GATE_ABLATION_CSV = f"{metric_dir()}/metric_gate_ablation_summary_{TAU_MAIN_LABEL}_{TAU_STRICT_LABEL}.csv"


def first_existing(paths: Iterable[str]) -> str | None:
    import os

    for path in paths:
        if path and os.path.exists(path):
            return path
    return None

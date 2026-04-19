"""
Contains the descriptions on the configuration of
1. state estimation, and
2. optimal power flow
"""

from __future__ import annotations

import os

import numpy as np
from pypower.api import ppoption

# SE settings: output
se_config = {
    "tol": 1e-3,       # the tolerance on the minimum jacobian matrix norm changes before considered as converged
    "max_it": 100,     # maximum iteration
    "verbose": 0,      # description settings on the output
}

# OPF settings: no output
opt = ppoption()
opt["VERBOSE"] = 0
opt["OUT_ALL"] = 0
opt["OPF_FLOW_LIM"] = 1
opt["OPF_VIOLATION"] = 5e-4


SUPPORTED_CASES = ("case14", "case39")


CASE_PROFILES = {
    "case14": {
        "case_name": "case14",
        "load_resolution": "5min",
        "fpr": 0.02,
        "noise_ratio_power": 0.02,
        "noise_ratio_voltage": 0.001,
        "pv_bus": np.array([4, 5, 11, 13]) - 1,
        "measure_type": "HALF_RTU",
    },
    "case39": {
        "case_name": "case39",
        "load_resolution": "5min",
        "fpr": 0.02,
        "noise_ratio_power": 0.02,
        "noise_ratio_voltage": 0.001,
        # Use nonzero-load buses first to avoid an initial negative-load-heavy PV placement.
        "pv_bus": np.array([1, 3, 4, 7, 8, 9, 12, 15, 16, 18]) - 1,
        "measure_type": "HALF_RTU",
    },
}


case_name = os.getenv("DDET_CASE_NAME", "case14").strip().lower()
if case_name not in CASE_PROFILES:
    raise ValueError(
        f"Unsupported DDET_CASE_NAME={case_name!r}. Supported cases: {', '.join(SUPPORTED_CASES)}"
    )

sys_config = dict(CASE_PROFILES[case_name])

# MTD setting
mtd_config = {
    "max_ite": 100,
    "multi_run_no": 15,   # The number of multi-runs in stage one
    "upper_scale": 1.1,   # Improve the detection threshold
    "tol_one": 0.1,
    "tol_two": 1,
    "verbose": True,
    "is_worst": True,
    "x_facts_ratio": 0.5,
    "varrho_square": 0.03**2,
    "total_run": 200,
    "mode": 0,            # The attacker either uses 0: true state and 1: estimated state
    "comment": "reduce_scaling",
}


def save_metric(address, **kwargs):
    metric = {}
    for key in kwargs.keys():
        metric[f"{key}"] = kwargs[key]

    np.save(address, metric, allow_pickle=True)

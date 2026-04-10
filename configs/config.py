"""
Contains the descriptions on the configuration of 
1. state estimation, and 
2. optimal power flow
"""

from pypower.api import ppoption
import numpy as np
import os

# SE settings: output
se_config = {
    'tol' : 1e-3,       # the tolerance on the minimum jacobian matrix norm changes before considered as converged
    'max_it' : 100,     # maximum iteration
    'verbose' : 0       # description settings on the output, 0 for nothing to show; 1 for showing the norm loss in the Newton-Raphson method
}

# OPF settings: no output
opt = ppoption()              # OPF options
opt['VERBOSE'] = 0            # Dont show anything
opt['OUT_ALL'] = 0
opt['OPF_FLOW_LIM'] = 1       # Constraint on the active power flow
opt['OPF_VIOLATION'] = 5e-4   # 


case_name = os.environ.get("DDET_CASE_NAME", "case14")

if case_name == "case14":
    # System setting
    sys_config = {
        'case_name': 'case14',
        'load_resolution': '5min',
        'fpr': 0.02,
        'noise_ratio_power': 0.02,
        'noise_ratio_voltage': 0.001,
        'pv_bus': np.array([4,5,11,13])-1,
        'measure_type': 'HALF_RTU', 
    }

    # MTD setting
    mtd_config = {
        'max_ite': 100, 
        'multi_run_no': 15,   # The number of multi-runs in stage one, default 15
        'upper_scale': 1.1,   # Improve the detection threshold
        'tol_one': 0.1,   #default 0.1
        'tol_two': 1,     # default 1
        'verbose': True,
        'is_worst': True,
        'x_facts_ratio': 0.5,
        'varrho_square': 0.03**2,    # The identification uncertainty
        'total_run': 200,
        'mode': 0,      # The attacker either uses 0: true state and 1: estimated state
        'comment': 'reduce_scaling' 
    }

elif case_name == "case39":
    # Native case39 support for second-stage system expansion.
    # Keep global defaults aligned with case14 unless a case39-specific reason exists.
    sys_config = {
        'case_name': 'case39',
        'load_resolution': '5min',
        'fpr': 0.02,
        'noise_ratio_power': 0.02,
        'noise_ratio_voltage': 0.001,
        # Derived from pypower case39 + HALF_RTU structural probe.
        'pv_bus': np.array([29,31,32,33,34,35,36,37,38]),
        'measure_type': 'HALF_RTU',
    }

    # Keep MTD defaults identical to the frozen case14 stack for transfer comparability.
    mtd_config = {
        'max_ite': 100,
        'multi_run_no': 15,
        'upper_scale': 1.1,
        'tol_one': 0.1,
        'tol_two': 1,
        'verbose': True,
        'is_worst': True,
        'x_facts_ratio': 0.5,
        'varrho_square': 0.03**2,
        'total_run': 200,
        'mode': 0,
        'comment': 'reduce_scaling'
    }
else:
    raise ValueError(f"Unsupported case_name: {case_name}")


def save_metric(address, **kwargs):
    metric = {}
    for key in kwargs.keys():
        metric[f'{key}'] = kwargs[key]
    
    np.save(address, metric, allow_pickle=True)
"""
Settings for Neural Network
"""

from __future__ import annotations

import torch

from configs.config import sys_config
from configs.config_mea_idx import define_mea_idx_noise
from gen_data.gen_data import gen_case


CASE_MODEL_PROFILES = {
    "case14": {
        "lattent_dim": 10,
        "no_layer": 3,
    },
    "case39": {
        "lattent_dim": 24,
        "no_layer": 3,
    },
}


def infer_feature_size() -> int:
    case = gen_case(sys_config["case_name"])
    _, no_mea, _ = define_mea_idx_noise(case, choice=sys_config["measure_type"])
    return int(no_mea)


if sys_config["case_name"] not in CASE_MODEL_PROFILES:
    raise ValueError(f"Unsupported case_name={sys_config['case_name']!r} for nn_setting")

model_profile = CASE_MODEL_PROFILES[sys_config["case_name"]]

nn_setting = {
    # Network Structure
    "sample_length": 6,
    "lattent_dim": model_profile["lattent_dim"],
    "no_layer": model_profile["no_layer"],
    "feature_size": infer_feature_size(),

    # Training
    "epochs": 1000,
    "lr": 1e-3,
    "patience": 10,
    "delta": 0,
    "model_path": f"saved_model/{sys_config['case_name']}/checkpoint_rnn.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "lattent_weight": 0.0,

    "train_prop": 0.6,
    "valid_prop": 0.2,

    # Recover Setting
    "recover_lr": 5 * 1e-3,
    "beta_real": 0.1,
    "beta_imag": 0.1,
    "beta_mag": 100,
    "mode": "pre",
    "max_step_size": 1000,
    "min_step_size": 50,
}

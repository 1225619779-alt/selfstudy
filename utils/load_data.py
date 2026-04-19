"""
Load the case class based on the specifications in sys_config
"""

from __future__ import annotations

import numpy as np
from pypower.api import case14, case39
from torch.utils.data import DataLoader

from configs.config import sys_config
from configs.config_mea_idx import define_mea_idx_noise
from gen_data.gen_data import gen_case
from models.dataset import rnn_dataset, rnn_dataset_evl
from utils.fdi_att import FDI


BASE_CASE_FACTORY = {
    "case14": case14,
    "case39": case39,
}


def load_case():
    """
    Return the instance case class
    """
    case_name = sys_config["case_name"]
    if case_name not in BASE_CASE_FACTORY:
        raise ValueError(f"Unsupported case_name={case_name!r}")

    # Validate that the requested PyPower case exists before applying local modifications.
    _ = BASE_CASE_FACTORY[case_name]()
    case = gen_case(case_name)

    noise_sigma_dir = f"gen_data/{case_name}/noise_sigma.npy"
    idx, no_mea, _ = define_mea_idx_noise(case, sys_config["measure_type"])
    noise_sigma = np.load(noise_sigma_dir)

    case_class = FDI(case, noise_sigma, idx, sys_config["fpr"])
    return case_class


def load_load_pv():
    case_name = sys_config["case_name"]

    load_active_dir = f"gen_data/{case_name}/load_active.npy"
    load_reactive_dir = f"gen_data/{case_name}/load_reactive.npy"
    pv_active_dir = f"gen_data/{case_name}/pv_active.npy"
    pv_reactive_dir = f"gen_data/{case_name}/pv_reactive.npy"

    load_active = np.load(load_active_dir)
    load_reactive = np.load(load_reactive_dir)
    pv_active = np.load(pv_active_dir)
    pv_reactive = np.load(pv_reactive_dir)

    # set the length equal to the bus number
    pv_active_ = np.zeros((load_active.shape[0], load_reactive.shape[1]))
    pv_reactive_ = np.zeros((load_reactive.shape[0], load_reactive.shape[1]))
    pv_active_[:, sys_config["pv_bus"]] = pv_active
    pv_reactive_[:, sys_config["pv_bus"]] = pv_reactive

    return load_active, load_reactive, pv_active_, pv_reactive_


def load_measurement():
    case_name = sys_config["case_name"]

    z_noise_summary = np.load(f"gen_data/{case_name}/z_noise_summary.npy")
    v_est_summary = np.load(f"gen_data/{case_name}/v_est_summary.npy")
    success_summary = np.load(f"gen_data/{case_name}/success_summary.npy")
    print(f"z noise size: {z_noise_summary.shape}")
    print(f"v est size: {v_est_summary.shape}")
    print(f"success size: {success_summary.shape}")

    return z_noise_summary, v_est_summary


def load_dataset(is_shuffle=True):
    # Test dataset dataloader: SCALED
    test_dataset_scaled = rnn_dataset(mode="test", istransform=True)
    test_dataloader_scaled = DataLoader(
        dataset=test_dataset_scaled,
        shuffle=is_shuffle,
        batch_size=1,
    )
    print(f"test dataset size scaled: {len(test_dataloader_scaled.dataset)}")

    # Test dataset dataloader: UNSCALED
    test_dataset_unscaled = rnn_dataset_evl(mode="test", istransform=False)
    test_dataloader_unscaled = DataLoader(
        dataset=test_dataset_unscaled,
        shuffle=is_shuffle,
        batch_size=1,
    )
    print(f"test dataset size unscaled: {len(test_dataloader_unscaled.dataset)}")

    # Valid dataset: SCALED
    valid_dataset_scaled = rnn_dataset(mode="valid", istransform=True)
    valid_dataloader_scaled = DataLoader(
        dataset=valid_dataset_scaled,
        shuffle=is_shuffle,
        batch_size=1,
    )
    print(f"valid dataset size scaled: {len(valid_dataloader_scaled.dataset)}")

    # Validation dataset dataloader: UNSCALED
    valid_dataset_unscaled = rnn_dataset_evl(mode="valid", istransform=False)
    valid_dataloader_unscaled = DataLoader(
        dataset=valid_dataset_unscaled,
        shuffle=is_shuffle,
        batch_size=1,
    )
    print(f"valid dataset size unscaled: {len(valid_dataloader_unscaled.dataset)}")

    return (
        test_dataloader_scaled,
        test_dataloader_unscaled,
        valid_dataloader_scaled,
        valid_dataloader_unscaled,
    )

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gen_data.gen_data import improve_resolution, add_cloud, gen_case, gen_load, gen_pv
from configs.config import sys_config
from configs.config_mea_idx import define_mea_idx_noise
from utils.class_se import SE


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    raw_dir = repo_root / 'gen_data' / 'raw_data'
    case_dir = repo_root / 'gen_data' / sys_config['case_name']
    case_dir.mkdir(parents=True, exist_ok=True)

    load_csv = raw_dir / 'load.csv'
    pv_csv = raw_dir / 'pv.csv'
    if not load_csv.exists():
        raise FileNotFoundError(f'Missing raw load file: {load_csv}')
    if not pv_csv.exists():
        raise FileNotFoundError(f'Missing raw PV file: {pv_csv}')

    print('[1/6] Reading raw CSV files...')
    raw_load = pd.read_csv(load_csv)
    raw_pv = pd.read_csv(pv_csv)
    if 'DateTime' not in raw_load.columns or 'DateTime' not in raw_pv.columns:
        raise ValueError('Both load.csv and pv.csv must contain a DateTime column.')

    # Follow the existing case14 helper logic.
    raw_load['DateTime'] = pd.to_datetime(raw_load['DateTime'])
    raw_pv['DateTime'] = pd.to_datetime(raw_pv['DateTime'])
    raw_load = raw_load[(raw_load['DateTime'] >= '2012-06-01 00:00:00') & (raw_load['DateTime'] <= '2012-09-30 23:45:00')].reset_index(drop=True)
    raw_pv = raw_pv[(raw_pv['DateTime'] >= '2019-06-01 00:00:00') & (raw_pv['DateTime'] <= '2019-09-30 23:45:00')].reset_index(drop=True)
    print(f'    raw_load shape = {raw_load.shape}')
    print(f'    raw_pv shape = {raw_pv.shape}')

    print('[2/6] Interpolating raw data to 5-minute resolution...')
    load_high, pv_high = improve_resolution(load_raw=raw_load.copy(), pv_raw=raw_pv.copy(), res=sys_config['load_resolution'])

    load_high_to_save = load_high.reset_index().rename(columns={load_high.reset_index().columns[0]: 'DateTime'})
    pv_high_to_save = pv_high.reset_index().rename(columns={pv_high.reset_index().columns[0]: 'DateTime'})
    load_high_to_save.to_csv(raw_dir / 'load_high.csv', index=False)
    pv_high_to_save.to_csv(raw_dir / 'pv_high.csv', index=False)
    print('    Saved: gen_data/raw_data/load_high.csv')
    print('    Saved: gen_data/raw_data/pv_high.csv')

    print('[3/6] Adding cloud fluctuations to PV profile...')
    pv_high_cloud = add_cloud(pv_high.copy(), unchange_rate=0.5, max_reduce=0.8)
    pv_high_cloud_to_save = pv_high_cloud.reset_index().rename(columns={pv_high_cloud.reset_index().columns[0]: 'DateTime'})

    print(f'[4/6] Building modified {sys_config["case_name"]} and generating load/PV arrays...')
    case = gen_case(sys_config['case_name'])
    load_active, load_reactive = gen_load(case=case, load_raw=load_high_to_save.copy())
    pv_active, pv_reactive = gen_pv(sys_config['pv_bus'], pv_high_cloud_to_save.copy(), load_active, penetration_ratio=0.3)

    print('[5/6] Computing noise_sigma exactly the way the current helper does it...')
    idx, no_mea, noise_sigma0 = define_mea_idx_noise(case, choice=sys_config['measure_type'])
    case_class = SE(case, noise_sigma0, idx, fpr=sys_config['fpr'])
    result = case_class.run_opf(verbose=False)
    z, z_noise, vang_ref, vmag_ref = case_class.construct_mea(result)
    noise_sigma = np.abs(z * sys_config['noise_ratio_power']).flatten() + 1e-3

    print('[6/6] Saving the five basic .npy files...')
    np.save(case_dir / 'noise_sigma.npy', noise_sigma, allow_pickle=True)
    np.save(case_dir / 'load_active.npy', load_active, allow_pickle=True)
    np.save(case_dir / 'load_reactive.npy', load_reactive, allow_pickle=True)
    np.save(case_dir / 'pv_active.npy', pv_active, allow_pickle=True)
    np.save(case_dir / 'pv_reactive.npy', pv_reactive, allow_pickle=True)

    print(f'\n[OK] Saved files under gen_data/{sys_config["case_name"]}:')
    print('    noise_sigma.npy ', noise_sigma.shape)
    print('    load_active.npy ', load_active.shape)
    print('    load_reactive.npy ', load_reactive.shape)
    print('    pv_active.npy ', pv_active.shape)
    print('    pv_reactive.npy ', pv_reactive.shape)
    print('\nNext step: run measurement generation:')
    print('    DDET_CASE_NAME=%s python gen_data/gen_data.py' % sys_config['case_name'])


if __name__ == '__main__':
    main()

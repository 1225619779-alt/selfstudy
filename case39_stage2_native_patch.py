#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

CONFIG_CASE39_BLOCK = '''elif case_name == "case39":
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
else:'''

NN_CASE39_BLOCK = '''elif sys_config["case_name"] == "case39":

    nn_setting = {
        # Network Structure
        "sample_length": 6,
        "lattent_dim": 10,
        "no_layer": 3,
        # Derived from pypower case39 + HALF_RTU structural probe.
        "feature_size": 170,

        # Training
        "epochs": 1000,
        "lr": 1e-3,
        "patience": 10,
        "delta": 0,
        "model_path": "saved_model/case39/checkpoint_rnn.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "device": "cpu",
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
else:'''

GENERATE_CASE_BASIC_NPY = r'''from __future__ import annotations

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
'''

REBUILD_CONFIRM_FROM_MANIFEST = r'''from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str], cwd: Path) -> None:
    print('$', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate native test banks and baseline summaries from an existing confirm manifest.')
    parser.add_argument('--manifest', required=True, help='Path to confirm manifest json.')
    parser.add_argument('--force', action='store_true', help='Regenerate even if outputs already exist.')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    workdir = manifest_path.parent.parent.parent.resolve()
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding='utf-8'))

    for hold in manifest['holdouts']:
        test_bank = workdir / hold['test_bank']
        result_npy = workdir / hold['result_npy']
        result_summary = workdir / hold['result_summary']

        if test_bank.exists() and test_bank.is_symlink():
            test_bank.unlink()
        if args.force and result_npy.exists():
            result_npy.unlink()
        if args.force and result_summary.exists():
            result_summary.unlink()

        if args.force or not test_bank.exists():
            test_bank.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable,
                'evaluation_mixed_timeline.py',
                '--tau_verify', '-1',
                '--schedule', str(hold['schedule']),
                '--seed_base', str(hold['seed_base']),
                '--start_offset', str(hold['start_offset']),
                '--output', str(test_bank),
            ], cwd=workdir)
        else:
            print(f'[skip] existing test bank: {test_bank}')

        if args.force or (not result_npy.exists()) or (not result_summary.exists()):
            result_npy.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable,
                'evaluation_budget_scheduler_phase3_holdout.py',
                '--clean_bank', str(workdir / manifest['clean_bank']),
                '--attack_bank', str(workdir / manifest['attack_bank']),
                '--train_bank', str(workdir / manifest['train_bank']),
                '--val_bank', str(workdir / manifest['val_bank']),
                '--test_bank', str(test_bank),
                '--slot_budget_list', *[str(x) for x in manifest['frozen_regime']['slot_budget_list']],
                '--decision_step_group', str(manifest['frozen_regime']['decision_step_group']),
                '--busy_time_quantile', str(manifest['frozen_regime']['busy_time_quantile']),
                '--output', str(result_npy),
            ], cwd=workdir)
        else:
            print(f'[skip] existing baseline summary: {result_summary}')

    print(json.dumps({'manifest': str(manifest_path), 'done': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
'''


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding='utf-8')
    if old not in text:
        raise RuntimeError(f'Could not find expected block in {path}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def replace_regex(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text(encoding='utf-8')
    new_text, n = re.subn(pattern, repl, text, flags=re.S)
    if n == 0:
        raise RuntimeError(f'Could not match pattern in {path}: {pattern}')
    path.write_text(new_text, encoding='utf-8')


def main() -> None:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path('.').resolve()

    # 1) config.py
    config_path = repo_root / 'configs' / 'config.py'
    replace_regex(
        config_path,
        r'''elif case_name == "case39":\n\s+raise RuntimeError\(\n\s+"case39 raw generation is disabled in the first-stage bridge\. "\n\s+"Stage precomputed case39 banks under metric/case39 via prepare_case_bank_assets\.py\. "\n\s+"Do not fall back to case14 raw assets or checkpoint\."\n\s+\)\nelse:''',
        CONFIG_CASE39_BLOCK,
    )

    # 2) nn_setting.py
    nn_path = repo_root / 'configs' / 'nn_setting.py'
    replace_regex(
        nn_path,
        r'''elif sys_config\["case_name"\] == "case39":\n\s+model_path = Path\("saved_model/case39/checkpoint_rnn\.pt"\)\n\s+raise RuntimeError\(\n\s+"case39 raw generation is disabled in the first-stage bridge\. "\n\s+f"Do not fall back to case14 checkpoint\. Expected future model path would be: \{model_path\}"\n\s+\)\nelse:''',
        NN_CASE39_BLOCK,
    )

    # 3) gen_data/gen_data.py
    gen_data_path = repo_root / 'gen_data' / 'gen_data.py'
    text = gen_data_path.read_text(encoding='utf-8')
    if 'from pathlib import Path' not in text:
        text = 'from pathlib import Path\n' + text
    text = text.replace('from pypower.api import ext2int, bustypes, case14', 'from pypower.api import ext2int, bustypes, case14, case39')
    if "elif case_name == 'case39':" not in text:
        marker = "    return case\n"
        insert = """
    elif case_name == 'case39':
        # Conservative native case39 support.
        # Do not invent case14-style load surgery without evidence; start from pypower case39 as-is.
        case = case39()
        # Keep generator reactive bounds symmetric, same generic adjustment used in case14 support.
        case['gen'][:,QMIN] = -case['gen'][:,QMAX]

"""
        if marker not in text:
            raise RuntimeError('Could not find return marker in gen_data/gen_data.py')
        text = text.replace(marker, insert + marker, 1)
    # Normalize __main__ path handling for Linux/WSL
    text = re.sub(
        r'''\n\s*# Loading path\n\s*noise_sigma_dir = .*?\n\s*# Saving path\n\s*z_noise_summary_dir = .*?\n\s*success_summary_dir = .*?\n\s*# Modify case''',
        '''\n    # Loading/saving paths\n    case_dir = Path('gen_data') / sys_config["case_name"]\n    case_dir.mkdir(parents=True, exist_ok=True)\n    noise_sigma_dir = case_dir / 'noise_sigma.npy'\n    load_active_dir = case_dir / 'load_active.npy'\n    load_reactive_dir = case_dir / 'load_reactive.npy'\n    pv_active_dir = case_dir / 'pv_active.npy'\n    pv_reactive_dir = case_dir / 'pv_reactive.npy'\n\n    z_noise_summary_dir = case_dir / 'z_noise_summary.npy'\n    v_est_summary_dir = case_dir / 'v_est_summary.npy'\n    success_summary_dir = case_dir / 'success_summary.npy'\n\n    # Modify case''',
        text,
        flags=re.S,
    )
    gen_data_path.write_text(text, encoding='utf-8')

    # 4) utils/load_data.py
    load_data_path = repo_root / 'utils' / 'load_data.py'
    text = load_data_path.read_text(encoding='utf-8')
    text = text.replace('from pypower.api import case14\n', '')
    text = re.sub(
        r'''def load_case\(\):.*?return case_class''',
        '''def load_case():\n    """\n    Return the instance case class\n    """\n    case = gen_case(sys_config['case_name'])\n    noise_sigma_dir = f'gen_data/{sys_config["case_name"]}/noise_sigma.npy'\n    idx, no_mea, _ = define_mea_idx_noise(case, sys_config['measure_type'])\n    noise_sigma = np.load(noise_sigma_dir)\n\n    case_class = FDI(case, noise_sigma, idx, sys_config['fpr'])\n\n    return case_class''',
        text,
        flags=re.S,
    )
    text = text.replace("    z_noise_summary = np.load('gen_data/case14/z_noise_summary.npy')     # Measurement with noise\n    v_est_summary = np.load('gen_data/case14/v_est_summary.npy')          # Estimated voltage state\n    success_summary = np.load('gen_data/case14/success_summary.npy')\n", "    z_noise_summary = np.load(f'gen_data/{sys_config[\"case_name\"]}/z_noise_summary.npy')     # Measurement with noise\n    v_est_summary = np.load(f'gen_data/{sys_config[\"case_name\"]}/v_est_summary.npy')          # Estimated voltage state\n    success_summary = np.load(f'gen_data/{sys_config[\"case_name\"]}/success_summary.npy')\n")
    load_data_path.write_text(text, encoding='utf-8')

    # 5) add generate_case_basic_npy.py
    (repo_root / 'generate_case_basic_npy.py').write_text(GENERATE_CASE_BASIC_NPY, encoding='utf-8')

    # 6) add rebuild_confirm_holdouts_from_manifest.py
    (repo_root / 'rebuild_confirm_holdouts_from_manifest.py').write_text(REBUILD_CONFIRM_FROM_MANIFEST, encoding='utf-8')

    print(f'Applied case39 native stage-2 patch under {repo_root}')
    print('Created: generate_case_basic_npy.py')
    print('Created: rebuild_confirm_holdouts_from_manifest.py')


if __name__ == '__main__':
    main()

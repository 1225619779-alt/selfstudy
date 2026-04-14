#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _set_repo_env(repo_root: str, case_name: str) -> None:
    os.environ["DDET_CASE_NAME"] = case_name
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_case_class(repo_root: str, case_name: str):
    _set_repo_env(repo_root, case_name)
    from configs.config import sys_config  # type: ignore
    from configs.config_mea_idx import define_mea_idx_noise  # type: ignore
    from gen_data.gen_data import gen_case  # type: ignore
    from utils.class_se import SE  # type: ignore

    case = gen_case(case_name)
    idx, _no_mea, _ = define_mea_idx_noise(case, choice=sys_config["measure_type"])
    noise_sigma = np.load(Path("gen_data") / case_name / "noise_sigma.npy")
    case_class = SE(case, noise_sigma, idx, fpr=sys_config["fpr"])
    pv_bus = np.array(sys_config["pv_bus"], dtype=int)
    return case_class, pv_bus, noise_sigma, sys_config


def _compute_slice(repo_root: str, case_name: str, start_idx: int, end_idx: int, noise_seed: int):
    repo_root = str(Path(repo_root).resolve())
    os.chdir(repo_root)
    case_class, pv_bus, noise_sigma, sys_config = _load_case_class(repo_root, case_name)

    root = Path("gen_data") / case_name
    load_active = np.load(root / "load_active.npy")
    load_reactive = np.load(root / "load_reactive.npy")
    pv_active = np.load(root / "pv_active.npy")
    pv_reactive = np.load(root / "pv_reactive.npy")

    z_list = []
    v_list = []
    succ_list = []
    per_iter = []

    first_success = False
    prev_z = None
    prev_v = None

    for global_idx in range(start_idx, end_idx):
        t0 = time.perf_counter()
        pv_active_pad = np.zeros(load_active.shape[1], dtype=float)
        pv_reactive_pad = np.zeros(load_reactive.shape[1], dtype=float)
        pv_active_pad[pv_bus] = pv_active[global_idx]
        pv_reactive_pad[pv_bus] = pv_reactive[global_idx]

        result = case_class.run_opf(
            load_active=load_active[global_idx] - pv_active_pad,
            load_reactive=load_reactive[global_idx] - pv_reactive_pad,
        )
        success = bool(result.get("success", False))
        succ_list.append(success)

        if not success:
            if not first_success:
                raise RuntimeError(
                    f"First failing step encountered at global_idx={global_idx}; no previous measurement exists for forward-fill."
                )
            z_list.append(prev_z.copy())
            v_list.append(prev_v.copy())
            per_iter.append(time.perf_counter() - t0)
            continue

        z, _z_noise_unused, vang_ref, vmag_ref = case_class.construct_mea(result)
        rng = np.random.default_rng(noise_seed + global_idx)
        noise_vec = rng.normal(loc=0.0, scale=noise_sigma, size=noise_sigma.shape[0])
        z_noise = z.flatten() + noise_vec
        z_noise = np.expand_dims(z_noise, axis=1)
        v_est = case_class.ac_se_pypower(z_noise=z_noise, vang_ref=vang_ref, vmag_ref=vmag_ref)
        z_flat = z_noise.flatten()
        v_arr = np.asarray(v_est)

        z_list.append(z_flat)
        v_list.append(v_arr)
        prev_z = z_flat
        prev_v = v_arr
        first_success = True
        per_iter.append(time.perf_counter() - t0)

    z_out = np.stack(z_list, axis=0)
    v_out = np.stack(v_list, axis=0)
    s_out = np.array(succ_list, dtype=bool)
    return z_out, v_out, s_out, per_iter


def _complex_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit case39_measure_parallel_v2 outputs against a direct sequential recomputation on the same slice."
    )
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--case_name", default="case39")
    parser.add_argument("--parallel_out_root", required=True, help="Path like gen_data/case39_smoke_parallel_v2_16")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--noise_seed", type=int, default=20260408)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = str(Path(args.repo_root).resolve())
    os.chdir(repo_root)
    _set_repo_env(repo_root, args.case_name)

    parallel_root = Path(args.parallel_out_root)
    z_par = np.load(parallel_root / "z_noise_summary.npy")
    v_par = np.load(parallel_root / "v_est_summary.npy")
    s_par = np.load(parallel_root / "success_summary.npy")

    if z_par.shape[0] != (args.end_idx - args.start_idx):
        raise ValueError(
            f"Parallel output length mismatch: got {z_par.shape[0]}, expected {args.end_idx - args.start_idx}"
        )

    t0 = time.perf_counter()
    z_seq, v_seq, s_seq, per_iter = _compute_slice(
        repo_root=repo_root,
        case_name=args.case_name,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        noise_seed=args.noise_seed,
    )
    elapsed_seq = time.perf_counter() - t0

    report = {
        "repo_root": repo_root,
        "case_name": args.case_name,
        "slice": {
            "start_idx": args.start_idx,
            "end_idx_exclusive": args.end_idx,
            "n_steps": args.end_idx - args.start_idx,
            "noise_seed": args.noise_seed,
        },
        "parallel_out_root": str(parallel_root),
        "parallel_shapes": {
            "z": list(z_par.shape),
            "v": list(v_par.shape),
            "success": list(s_par.shape),
        },
        "sequential_shapes": {
            "z": list(z_seq.shape),
            "v": list(v_seq.shape),
            "success": list(s_seq.shape),
        },
        "sequential_runtime": {
            "elapsed_sec": float(elapsed_seq),
            "sec_per_iter_mean": float(np.mean(per_iter)),
            "sec_per_iter_min": float(np.min(per_iter)),
            "sec_per_iter_max": float(np.max(per_iter)),
        },
        "agreement": {
            "success_exact_equal": bool(np.array_equal(s_par.astype(bool), s_seq.astype(bool))),
            "z_max_abs_diff": _complex_max_abs(np.asarray(z_par), np.asarray(z_seq)),
            "v_max_abs_diff": _complex_max_abs(np.asarray(v_par), np.asarray(v_seq)),
            "z_allclose_rtol1e-7_atol1e-9": bool(np.allclose(z_par, z_seq, rtol=1e-7, atol=1e-9, equal_nan=True)),
            "v_allclose_rtol1e-7_atol1e-9": bool(np.allclose(v_par, v_seq, rtol=1e-7, atol=1e-9, equal_nan=True)),
        },
        "note": "This audit compares the saved v2 parallel outputs against a direct sequential recomputation using the same deterministic noise rule (noise_seed + global_idx).",
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

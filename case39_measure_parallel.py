#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Standard-library only at module import time. Repo imports happen inside functions
# after DDET_CASE_NAME is set, so this script is robust to case-aware config logic.


def _set_repo_env(repo_root: str, case_name: str) -> None:
    os.environ["DDET_CASE_NAME"] = case_name
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_basic_arrays(case_name: str) -> dict[str, np.ndarray]:
    root = Path("gen_data") / case_name
    return {
        "noise_sigma": np.load(root / "noise_sigma.npy"),
        "load_active": np.load(root / "load_active.npy", mmap_mode="r"),
        "load_reactive": np.load(root / "load_reactive.npy", mmap_mode="r"),
        "pv_active": np.load(root / "pv_active.npy", mmap_mode="r"),
        "pv_reactive": np.load(root / "pv_reactive.npy", mmap_mode="r"),
    }


def _build_case_class(repo_root: str, case_name: str):
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
    return case_class, pv_bus, noise_sigma


def _probe_shapes_and_seed(repo_root: str, case_name: str, start_idx: int, noise_seed: int):
    case_class, pv_bus, noise_sigma = _build_case_class(repo_root, case_name)
    arr = _load_basic_arrays(case_name)

    i = start_idx
    pv_active_pad = np.zeros(arr["load_active"].shape[1], dtype=float)
    pv_reactive_pad = np.zeros(arr["load_reactive"].shape[1], dtype=float)
    pv_active_pad[pv_bus] = arr["pv_active"][i]
    pv_reactive_pad[pv_bus] = arr["pv_reactive"][i]

    result = case_class.run_opf(
        load_active=arr["load_active"][i] - pv_active_pad,
        load_reactive=arr["load_reactive"][i] - pv_reactive_pad,
    )
    if not result["success"]:
        raise RuntimeError(
            "The first probed step did not converge. The original sequential code would also fail "
            "if the first step had no previous measurement to copy from. Try a different start_idx."
        )

    # Use deterministic per-step noise instead of relying on global np.random state.
    # This is not byte-identical to the old implementation, but it is logically cleaner and
    # reproducible, which is preferable for the new native case39 branch.
    z, _z_noise_unused, vang_ref, vmag_ref = case_class.construct_mea(result)
    rng = np.random.default_rng(noise_seed + i)
    noise_vec = rng.normal(loc=0.0, scale=noise_sigma, size=noise_sigma.shape[0])
    z_noise = z.flatten() + noise_vec
    z_noise = np.expand_dims(z_noise, axis=1)
    v_est = case_class.ac_se_pypower(z_noise=z_noise, vang_ref=vang_ref, vmag_ref=vmag_ref)

    return {
        "z_dim": int(z.shape[0]),
        "v_shape": tuple(np.asarray(v_est).shape),
        "v_dtype": np.asarray(v_est).dtype.str,
        "n_bus": int(arr["load_active"].shape[1]),
        "pv_bus": pv_bus.tolist(),
        "total_steps": int(arr["load_active"].shape[0]),
        "noise_sigma_len": int(noise_sigma.shape[0]),
    }


def _chunk_ranges(start: int, stop: int, workers: int, oversubscribe: int = 4) -> list[tuple[int, int]]:
    total = max(0, stop - start)
    if total == 0:
        return []
    n_chunks = min(total, max(workers * oversubscribe, workers))
    chunk_size = math.ceil(total / n_chunks)
    chunks = []
    s = start
    while s < stop:
        e = min(stop, s + chunk_size)
        chunks.append((s, e))
        s = e
    return chunks


def _fill_nan_value(dtype_str: str, shape: Sequence[int]) -> np.ndarray:
    dtype = np.dtype(dtype_str)
    out = np.empty(shape, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        out.fill(np.nan + 1j * np.nan)
    else:
        out.fill(np.nan)
    return out


def _worker_process(
    repo_root: str,
    case_name: str,
    start: int,
    end: int,
    z_dim: int,
    v_shape: Sequence[int],
    v_dtype_str: str,
    noise_seed: int,
    tmp_dir: str,
) -> str:
    _set_repo_env(repo_root, case_name)
    case_class, pv_bus, noise_sigma = _build_case_class(repo_root, case_name)
    arr = _load_basic_arrays(case_name)

    n = end - start
    z_chunk = np.empty((n, z_dim), dtype=float)
    v_chunk = np.empty((n,) + tuple(v_shape), dtype=np.dtype(v_dtype_str))
    success = np.zeros((n,), dtype=bool)

    v_nan = _fill_nan_value(v_dtype_str, v_shape)

    for local_idx, global_idx in enumerate(range(start, end)):
        pv_active_pad = np.zeros(arr["load_active"].shape[1], dtype=float)
        pv_reactive_pad = np.zeros(arr["load_reactive"].shape[1], dtype=float)
        pv_active_pad[pv_bus] = arr["pv_active"][global_idx]
        pv_reactive_pad[pv_bus] = arr["pv_reactive"][global_idx]

        result = case_class.run_opf(
            load_active=arr["load_active"][global_idx] - pv_active_pad,
            load_reactive=arr["load_reactive"][global_idx] - pv_reactive_pad,
        )

        if not result["success"]:
            z_chunk[local_idx].fill(np.nan)
            v_chunk[local_idx] = v_nan
            success[local_idx] = False
            continue

        z, _z_noise_unused, vang_ref, vmag_ref = case_class.construct_mea(result)
        rng = np.random.default_rng(noise_seed + global_idx)
        noise_vec = rng.normal(loc=0.0, scale=noise_sigma, size=noise_sigma.shape[0])
        z_noise = z.flatten() + noise_vec
        z_noise = np.expand_dims(z_noise, axis=1)

        v_est = case_class.ac_se_pypower(z_noise=z_noise, vang_ref=vang_ref, vmag_ref=vmag_ref)
        z_chunk[local_idx] = z_noise.flatten()
        v_chunk[local_idx] = np.asarray(v_est)
        success[local_idx] = True

    out_path = Path(tmp_dir) / f"chunk_{start}_{end}.npz"
    np.savez_compressed(out_path, start=start, end=end, z=z_chunk, v=v_chunk, success=success)
    return str(out_path)


def _forward_fill_failures(z_full: np.ndarray, v_full: np.ndarray, success_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if z_full.shape[0] == 0:
        return z_full, v_full
    if not bool(success_full[0]):
        raise RuntimeError(
            "The first step failed. The original sequential implementation would also have no prior "
            "measurement to copy. Pick a different start_idx or inspect OPF convergence."
        )
    for i in range(1, z_full.shape[0]):
        if not bool(success_full[i]):
            z_full[i] = z_full[i - 1]
            v_full[i] = v_full[i - 1]
    return z_full, v_full


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel measurement generation for gen_data/{case}/z_noise_summary.npy, "
            "v_est_summary.npy, success_summary.npy. Uses deterministic per-step noise seeds "
            "for reproducibility."
        )
    )
    parser.add_argument("--repo_root", default=".", help="Repository root.")
    parser.add_argument("--case_name", default="case39", help="Case name, usually case39.")
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="Exclusive end idx. -1 means full length.")
    parser.add_argument("--noise_seed", type=int, default=20260408)
    parser.add_argument("--out_root", default=None, help="Defaults to gen_data/{case_name}")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    repo_root = str(Path(args.repo_root).resolve())
    os.chdir(repo_root)
    _set_repo_env(repo_root, args.case_name)

    arrays = _load_basic_arrays(args.case_name)
    total_steps = int(arrays["load_active"].shape[0])
    end_idx = total_steps if args.end_idx < 0 else min(args.end_idx, total_steps)
    start_idx = max(0, args.start_idx)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid slice: start_idx={start_idx}, end_idx={end_idx}, total_steps={total_steps}")

    out_root = Path(args.out_root) if args.out_root else Path("gen_data") / args.case_name
    out_root.mkdir(parents=True, exist_ok=True)

    probe = _probe_shapes_and_seed(repo_root, args.case_name, start_idx, args.noise_seed)

    chunks = _chunk_ranges(start_idx, end_idx, args.workers)
    tmp_dir = tempfile.mkdtemp(prefix=f"measure_parallel_{args.case_name}_", dir=str(out_root))
    t0 = time.time()

    chunk_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _worker_process,
                repo_root,
                args.case_name,
                s,
                e,
                probe["z_dim"],
                probe["v_shape"],
                probe["v_dtype"],
                args.noise_seed,
                tmp_dir,
            ): (s, e)
            for s, e in chunks
        }
        for fut in as_completed(futs):
            s, e = futs[fut]
            path = fut.result()
            print(f"finished chunk [{s}, {e}) -> {path}", flush=True)
            chunk_paths.append(path)

    # Reassemble in order.
    loaded = []
    for path in sorted(chunk_paths, key=lambda p: int(Path(p).stem.split("_")[1])):
        with np.load(path, allow_pickle=True) as npz:
            loaded.append((int(npz["start"]), int(npz["end"]), npz["z"], npz["v"], npz["success"]))

    z_full = np.concatenate([x[2] for x in loaded], axis=0)
    v_full = np.concatenate([x[3] for x in loaded], axis=0)
    success_full = np.concatenate([x[4] for x in loaded], axis=0)

    z_full, v_full = _forward_fill_failures(z_full, v_full, success_full)

    z_out = out_root / "z_noise_summary.npy"
    v_out = out_root / "v_est_summary.npy"
    s_out = out_root / "success_summary.npy"
    rep_out = out_root / "parallel_measure_report.json"

    # Save in a layout close to the original sequential implementation.
    np.save(z_out, [row.copy() for row in z_full], allow_pickle=True)
    np.save(v_out, [row.copy() for row in v_full], allow_pickle=True)
    np.save(s_out, success_full.tolist(), allow_pickle=True)

    elapsed = time.time() - t0
    report = {
        "repo_root": repo_root,
        "case_name": args.case_name,
        "workers": args.workers,
        "start_idx": start_idx,
        "end_idx_exclusive": end_idx,
        "n_steps": end_idx - start_idx,
        "noise_seed": args.noise_seed,
        "probe": probe,
        "chunks": [{"start": s, "end": e} for s, e in chunks],
        "success_rate_raw": float(np.mean(success_full.astype(float))),
        "n_forward_filled": int(np.sum(~success_full)),
        "elapsed_sec": elapsed,
        "sec_per_iter_effective": elapsed / max(1, (end_idx - start_idx)),
        "outputs": {
            "z_noise_summary": str(z_out),
            "v_est_summary": str(v_out),
            "success_summary": str(s_out),
        },
        "note": (
            "This parallel generator preserves the sequential fallback semantics for failed OPF steps "
            "via a final forward-fill pass. Noise is made deterministic per step using noise_seed + index."
        ),
    }
    rep_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

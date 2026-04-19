from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from configs.config import CASE_PROFILES, se_config
from configs.config_mea_idx import define_mea_idx_noise
from gen_data.gen_data import gen_case
from utils.class_se import SE


CHUNK_RE = re.compile(r"^z_noise_(\d+)_(\d+)\.npy$")


def build_case(case_name: str) -> tuple[SE, dict]:
    case_cfg = dict(CASE_PROFILES[case_name])
    case = gen_case(case_name)
    idx, _, _ = define_mea_idx_noise(case, choice=case_cfg["measure_type"])
    noise_sigma = np.load(Path("gen_data") / case_name / "noise_sigma.npy")
    return SE(case, noise_sigma, idx, fpr=case_cfg["fpr"]), case_cfg


def process_chunk(
    case_name: str,
    start: int,
    stop: int,
    seed_base: int | None,
    warm_start: bool,
    use_pv_reactive: bool,
    chunk_dir: str,
) -> dict:
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    case_class, case_cfg = build_case(case_name)
    case_dir = Path("gen_data") / case_name

    load_active = np.load(case_dir / "load_active.npy", mmap_mode="r")
    load_reactive = np.load(case_dir / "load_reactive.npy", mmap_mode="r")
    pv_active = np.load(case_dir / "pv_active.npy", mmap_mode="r")
    pv_reactive = np.load(case_dir / "pv_reactive.npy", mmap_mode="r") if use_pv_reactive else None

    rows = stop - start
    z_chunk = np.empty((rows, case_class.no_mea), dtype=np.float64)
    v_chunk = np.empty((rows, case_class.no_bus), dtype=np.complex128)
    success_chunk = np.empty((rows,), dtype=bool)

    pv_bus = np.asarray(case_cfg["pv_bus"], dtype=int)
    prev_z_noise = None
    prev_v_est = None
    total_opf = 0.0
    total_se = 0.0
    total_iter = 0

    for local_idx, global_idx in enumerate(range(start, stop)):
        if seed_base is not None:
            np.random.seed(seed_base + global_idx)

        pv_active_row = np.zeros((load_active.shape[1],), dtype=np.float64)
        pv_reactive_row = np.zeros((load_reactive.shape[1],), dtype=np.float64)
        pv_active_row[pv_bus] = pv_active[global_idx]
        if use_pv_reactive:
            pv_reactive_row[pv_bus] = pv_reactive[global_idx]

        t0 = time.time()
        result = case_class.run_opf(
            load_active=load_active[global_idx] - pv_active_row,
            load_reactive=load_reactive[global_idx] - pv_reactive_row,
        )
        t1 = time.time()
        z, z_noise, vang_ref, vmag_ref = case_class.construct_mea(result)

        if result["success"] is False:
            if prev_z_noise is None or prev_v_est is None:
                raise RuntimeError(
                    f"Chunk {start}:{stop} hit OPF failure at its first sample {global_idx}; "
                    "cannot safely borrow the previous sample inside this chunk."
                )
            z_chunk[local_idx] = prev_z_noise
            v_chunk[local_idx] = prev_v_est
            success_chunk[local_idx] = False
            total_opf += (t1 - t0)
            continue

        se_kwargs = {"return_meta": True}
        se_kwargs["config"] = se_config
        if warm_start and prev_v_est is not None:
            se_kwargs["v_init"] = prev_v_est

        t2 = time.time()
        v_est, meta = case_class.ac_se_pypower(
            z_noise=z_noise,
            vang_ref=vang_ref,
            vmag_ref=vmag_ref,
            **se_kwargs,
        )
        t3 = time.time()

        z_noise_flat = z_noise.flatten()
        z_chunk[local_idx] = z_noise_flat
        v_chunk[local_idx] = v_est
        success_chunk[local_idx] = True
        prev_z_noise = z_noise_flat
        prev_v_est = v_est

        total_opf += (t1 - t0)
        total_se += (t3 - t2)
        total_iter += meta["iterations"]

    chunk_root = Path(chunk_dir)
    chunk_root.mkdir(parents=True, exist_ok=True)
    z_path = chunk_root / f"z_noise_{start}_{stop}.npy"
    v_path = chunk_root / f"v_est_{start}_{stop}.npy"
    s_path = chunk_root / f"success_{start}_{stop}.npy"
    np.save(z_path, z_chunk, allow_pickle=True)
    np.save(v_path, v_chunk, allow_pickle=True)
    np.save(s_path, success_chunk, allow_pickle=True)

    return {
        "start": start,
        "stop": stop,
        "rows": rows,
        "z_path": str(z_path),
        "v_path": str(v_path),
        "s_path": str(s_path),
        "opf_seconds": total_opf,
        "se_seconds": total_se,
        "avg_iter": (total_iter / rows) if rows else 0.0,
    }


def build_ranges(total_rows: int, chunk_size: int, start: int, stop: int | None) -> list[tuple[int, int]]:
    actual_stop = total_rows if stop is None else min(stop, total_rows)
    ranges = []
    cursor = start
    while cursor < actual_stop:
        nxt = min(cursor + chunk_size, actual_stop)
        ranges.append((cursor, nxt))
        cursor = nxt
    return ranges


def scan_existing_chunks(
    chunk_dir: Path,
    expected_ranges: list[tuple[int, int]],
) -> list[dict]:
    expected = set(expected_ranges)
    completed = []

    if not chunk_dir.exists():
        return completed

    for z_path in sorted(chunk_dir.glob("z_noise_*.npy")):
        match = CHUNK_RE.match(z_path.name)
        if match is None:
            continue

        start = int(match.group(1))
        stop = int(match.group(2))
        if (start, stop) not in expected:
            continue

        v_path = chunk_dir / f"v_est_{start}_{stop}.npy"
        s_path = chunk_dir / f"success_{start}_{stop}.npy"
        if not v_path.exists() or not s_path.exists():
            continue

        completed.append(
            {
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "z_path": str(z_path),
                "v_path": str(v_path),
                "s_path": str(s_path),
                "opf_seconds": 0.0,
                "se_seconds": 0.0,
                "avg_iter": 0.0,
                "restored": True,
            }
        )

    completed.sort(key=lambda row: row["start"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel measurement-bank generator for DDET-MTD.")
    parser.add_argument("--case_name", default="case39", choices=sorted(CASE_PROFILES.keys()))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--use_pv_reactive", action="store_true")
    parser.add_argument("--chunk_dir", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    args = parser.parse_args()

    case_dir = Path("gen_data") / args.case_name
    load_active = np.load(case_dir / "load_active.npy", mmap_mode="r")
    total_rows = int(load_active.shape[0])
    ranges = build_ranges(total_rows, args.chunk_size, args.start, args.stop)

    if not ranges:
        raise ValueError("No rows selected for generation.")

    case_class, _ = build_case(args.case_name)
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else case_dir / "measurement_chunks"
    summary_json = Path(args.summary_json) if args.summary_json else case_dir / "measurement_parallel_summary.json"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    existing_completed = scan_existing_chunks(chunk_dir, ranges)
    existing_keys = {(row["start"], row["stop"]) for row in existing_completed}
    pending_ranges = [row for row in ranges if row not in existing_keys]

    started_at_utc = datetime.now(timezone.utc).isoformat()

    print(
        f"[parallel-gen] case={args.case_name} rows={ranges[0][0]}:{ranges[-1][1]} "
        f"workers={args.workers} chunk_size={args.chunk_size} warm_start={args.warm_start}"
    , flush=True)
    print(f"[parallel-gen] started_at_utc={started_at_utc}", flush=True)
    print(
        f"[parallel-gen] restored_chunks={len(existing_completed)} pending_chunks={len(pending_ranges)} "
        f"chunk_dir={chunk_dir}"
    , flush=True)

    start_time = time.time()
    completed = list(existing_completed)

    if pending_ranges:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
            futures = {
                executor.submit(
                    process_chunk,
                    args.case_name,
                    chunk_start,
                    chunk_stop,
                    args.seed,
                    args.warm_start,
                    args.use_pv_reactive,
                    str(chunk_dir),
                ): (chunk_start, chunk_stop)
                for chunk_start, chunk_stop in pending_ranges
            }

            restored_count = len(existing_completed)
            for done_idx, future in enumerate(as_completed(futures), start=1):
                chunk_start, chunk_stop = futures[future]
                result = future.result()
                completed.append(result)

                elapsed = time.time() - start_time
                finished_chunks = restored_count + done_idx
                avg_chunk = elapsed / done_idx
                eta = avg_chunk * (len(pending_ranges) - done_idx)
                print(
                    f"[parallel-gen] chunk {chunk_start}:{chunk_stop} done "
                    f"({finished_chunks}/{len(ranges)}) | "
                    f"opf={result['opf_seconds']:.1f}s se={result['se_seconds']:.1f}s "
                    f"avg_iter={result['avg_iter']:.2f} | "
                    f"elapsed={elapsed/60:.1f} min eta={eta/60:.1f} min"
                , flush=True)
    else:
        print("[parallel-gen] no pending chunks; rebuilding outputs from existing chunk files", flush=True)

    completed.sort(key=lambda row: row["start"])

    if [(row["start"], row["stop"]) for row in completed] != ranges:
        raise RuntimeError("Chunk coverage is incomplete or out of order; refusing to build summary arrays.")

    target_rows = sum(row["rows"] for row in completed)
    z_all = np.empty((target_rows, case_class.no_mea), dtype=np.float64)
    v_all = np.empty((target_rows, case_class.no_bus), dtype=np.complex128)
    s_all = np.empty((target_rows,), dtype=bool)

    cursor = 0
    for row in completed:
        z_chunk = np.load(row["z_path"])
        v_chunk = np.load(row["v_path"])
        s_chunk = np.load(row["s_path"])
        chunk_rows = z_chunk.shape[0]
        z_all[cursor:cursor + chunk_rows] = z_chunk
        v_all[cursor:cursor + chunk_rows] = v_chunk
        s_all[cursor:cursor + chunk_rows] = s_chunk
        cursor += chunk_rows

    stop_label = ranges[-1][1]
    start_label = ranges[0][0]
    if start_label == 0 and stop_label == total_rows:
        z_out = case_dir / "z_noise_summary.npy"
        v_out = case_dir / "v_est_summary.npy"
        s_out = case_dir / "success_summary.npy"
    else:
        z_out = case_dir / f"z_noise_summary_{start_label}_{stop_label}.npy"
        v_out = case_dir / f"v_est_summary_{start_label}_{stop_label}.npy"
        s_out = case_dir / f"success_summary_{start_label}_{stop_label}.npy"

    np.save(z_out, z_all, allow_pickle=True)
    np.save(v_out, v_all, allow_pickle=True)
    np.save(s_out, s_all, allow_pickle=True)

    total_elapsed = time.time() - start_time
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    summary = {
        "case_name": args.case_name,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "start": start_label,
        "stop": stop_label,
        "rows": target_rows,
        "warm_start": args.warm_start,
        "use_pv_reactive": args.use_pv_reactive,
        "seed": args.seed,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "total_elapsed_seconds": total_elapsed,
        "avg_seconds_per_row": total_elapsed / target_rows,
        "z_output": str(z_out),
        "v_output": str(v_out),
        "success_output": str(s_out),
        "chunk_dir": str(chunk_dir),
        "chunks": completed,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "z_output": str(z_out),
                "v_output": str(v_out),
                "success_output": str(s_out),
                "total_elapsed_seconds": total_elapsed,
                "avg_seconds_per_row": total_elapsed / target_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        ,
        flush=True
    )


if __name__ == "__main__":
    main()

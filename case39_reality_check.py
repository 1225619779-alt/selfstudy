
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        return e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reality check for native case39 runtime and actual GPU usage."
    )
    parser.add_argument("--repo_root", default=".", help="Repo root")
    parser.add_argument("--case_name", default="case39", help="Case name")
    parser.add_argument("--n_steps", type=int, default=8, help="Number of timesteps to benchmark")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in load/pv arrays")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    os.chdir(repo_root)
    os.environ["DDET_CASE_NAME"] = args.case_name

    report: dict[str, object] = {
        "repo_root": str(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "case_name": args.case_name,
        "cwd": str(Path.cwd()),
    }

    torch_obj = maybe_import_torch()
    if isinstance(torch_obj, Exception):
        report["torch_import_error"] = repr(torch_obj)
    else:
        torch = torch_obj
        gpu_info: dict[str, object] = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "torch_hip_version": getattr(torch.version, "hip", None),
        }
        if torch.cuda.is_available():
            try:
                gpu_info["device_name_0"] = torch.cuda.get_device_name(0)
            except Exception as e:
                gpu_info["device_name_0_error"] = repr(e)
        report["torch"] = gpu_info

    # Delayed imports after env var is set
    from configs.config import sys_config
    from configs.config_mea_idx import define_mea_idx_noise
    from gen_data.gen_data import gen_case
    from utils.class_se import SE

    report["sys_config"] = {
        "case_name": sys_config.get("case_name"),
        "measure_type": sys_config.get("measure_type"),
        "pv_bus": [int(x) for x in np.array(sys_config.get("pv_bus", [])).tolist()],
        "fpr": float(sys_config.get("fpr")),
    }

    gen_root = Path("gen_data") / args.case_name
    basic_paths = {
        "noise_sigma": gen_root / "noise_sigma.npy",
        "load_active": gen_root / "load_active.npy",
        "load_reactive": gen_root / "load_reactive.npy",
        "pv_active": gen_root / "pv_active.npy",
        "pv_reactive": gen_root / "pv_reactive.npy",
    }
    report["basic_paths"] = {k: str(v) for k, v in basic_paths.items()}
    report["basic_exists"] = {k: v.exists() for k, v in basic_paths.items()}

    missing = [k for k, v in basic_paths.items() if not v.exists()]
    if missing:
        report["status"] = "MISSING_BASIC_ARRAYS"
    else:
        case = gen_case(args.case_name)
        idx, no_mea, _ = define_mea_idx_noise(case, choice=sys_config["measure_type"])
        noise_sigma = np.load(basic_paths["noise_sigma"])
        case_class = SE(case, noise_sigma, idx, fpr=sys_config["fpr"])

        load_active = np.load(basic_paths["load_active"])
        load_reactive = np.load(basic_paths["load_reactive"])
        pv_active = np.load(basic_paths["pv_active"])
        pv_reactive = np.load(basic_paths["pv_reactive"])

        total_steps = int(load_active.shape[0])
        report["array_shapes"] = {
            "load_active": list(load_active.shape),
            "load_reactive": list(load_reactive.shape),
            "pv_active": list(pv_active.shape),
            "pv_reactive": list(pv_reactive.shape),
            "noise_sigma": list(noise_sigma.shape) if hasattr(noise_sigma, "shape") else None,
            "total_steps": total_steps,
        }

        # pad PV to bus dimension
        pv_active_full = np.zeros((load_active.shape[0], load_reactive.shape[1]))
        pv_reactive_full = np.zeros((load_reactive.shape[0], load_reactive.shape[1]))
        pv_bus = np.array(sys_config["pv_bus"], dtype=int)
        pv_active_full[:, pv_bus] = pv_active
        pv_reactive_full[:, pv_bus] = pv_reactive

        start = max(0, min(args.start_idx, total_steps - 1))
        end = min(total_steps, start + args.n_steps)
        if end <= start:
            raise ValueError("Invalid benchmark slice")

        durations = []
        success_flags = []
        z_shapes = []
        for i in range(start, end):
            t0 = time.perf_counter()
            result = case_class.run_opf(
                load_active=load_active[i] - pv_active_full[i],
                load_reactive=load_reactive[i] - pv_reactive_full[i],
            )
            success_flags.append(bool(result.get("success", False)))
            if result.get("success", False):
                z, z_noise, vang_ref, vmag_ref = case_class.construct_mea(result)
                v_est = case_class.ac_se_pypower(z_noise=z_noise, vang_ref=vang_ref, vmag_ref=vmag_ref)
                z_shapes.append([int(x) for x in np.array(z_noise).shape])
                _ = v_est
            t1 = time.perf_counter()
            durations.append(t1 - t0)

        sec_per_iter = float(np.mean(durations))
        estimate_hours = float(sec_per_iter * total_steps / 3600.0)
        report["benchmark"] = {
            "start_idx": start,
            "end_idx_exclusive": end,
            "n_steps": end - start,
            "sec_per_iter_mean": sec_per_iter,
            "sec_per_iter_min": float(np.min(durations)),
            "sec_per_iter_max": float(np.max(durations)),
            "success_rate_in_slice": float(np.mean(success_flags)) if success_flags else None,
            "first_z_noise_shape": z_shapes[0] if z_shapes else None,
            "estimated_hours_for_total_steps_if_constant": estimate_hours,
        }
        report["status"] = "OK"

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

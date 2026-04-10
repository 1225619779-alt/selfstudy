#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

CANONICAL_BANKS = {
    "clean_bank": "metric/case39/metric_clean_alarm_scores_full.npy",
    "attack_bank": "metric/case39/metric_attack_alarm_scores_400.npy",
    "train_bank": "metric/case39/mixed_bank_fit.npy",
    "val_bank": "metric/case39/mixed_bank_eval.npy",
}

RAW_EXTS = {".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle", ".csv", ".xlsx", ".mat", ".json", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", ".venv_q1", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def find_files(root: Path, needle: str) -> List[str]:
    out: List[str] = []
    for p in root.rglob("*"):
        if should_skip(p) or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if needle.lower() in rel.lower():
            out.append(rel)
    return sorted(out)


def inspect_canonical(repo_root: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for key, rel in CANONICAL_BANKS.items():
        p = repo_root / rel
        item: Dict[str, object] = {
            "path": str(p),
            "exists": p.exists(),
            "is_symlink": p.is_symlink(),
        }
        if p.exists() or p.is_symlink():
            try:
                resolved = p.resolve(strict=True)
                item["resolved_path"] = str(resolved)
                item["resolved_exists"] = resolved.exists()
                item["resolved_is_case14"] = "metric/case14" in str(resolved).replace("\\", "/")
                item["size_bytes"] = resolved.stat().st_size
                item["mtime_epoch"] = resolved.stat().st_mtime
                item["sha256"] = sha256_of_file(resolved)
            except Exception as e:
                item["resolve_error"] = repr(e)
        out[key] = item
    return out


def inspect_checkpoint(repo_root: Path) -> Dict[str, object]:
    candidate = repo_root / "saved_model" / "case39" / "checkpoint_rnn.pt"
    out: Dict[str, object] = {
        "expected_path": str(candidate),
        "exists": candidate.exists(),
    }
    if candidate.exists():
        out["size_bytes"] = candidate.stat().st_size
        out["mtime_epoch"] = candidate.stat().st_mtime
        out["sha256"] = sha256_of_file(candidate)
    # broader search
    broader: List[str] = []
    for p in repo_root.rglob("checkpoint_rnn.pt"):
        if should_skip(p) or not p.is_file():
            continue
        broader.append(p.relative_to(repo_root).as_posix())
    out["checkpoint_rnn_hits_under_repo"] = sorted(broader)
    return out


def inspect_code(repo_root: Path) -> Dict[str, object]:
    config_py = repo_root / "configs" / "config.py"
    nn_setting_py = repo_root / "configs" / "nn_setting.py"
    manifest_v1 = repo_root / "make_phase3_confirm_manifest.py"
    manifest_v2 = repo_root / "make_phase3_confirm_manifest_v2.py"

    cfg_txt = read_text_safe(config_py)
    nn_txt = read_text_safe(nn_setting_py)
    m1_txt = read_text_safe(manifest_v1)
    m2_txt = read_text_safe(manifest_v2)

    return {
        "config_py": {
            "path": str(config_py),
            "mentions_case39": "case39" in cfg_txt,
            "mentions_env_case_name": ("DDET_CASE_NAME" in cfg_txt) or ("CASE_NAME" in cfg_txt),
            "hardcodes_case14": "case14" in cfg_txt,
        },
        "nn_setting_py": {
            "path": str(nn_setting_py),
            "mentions_case39": "case39" in nn_txt,
            "mentions_saved_model_case39": "saved_model/case39" in nn_txt,
            "hardcodes_saved_model_case14": "saved_model/case14" in nn_txt,
        },
        "manifest_v1": {
            "path": str(manifest_v1),
            "mentions_case_name_arg": "--case_name" in m1_txt,
            "mentions_metric_case39": "metric/case39" in m1_txt,
            "hardcodes_metric_case14": "metric/case14" in m1_txt,
        },
        "manifest_v2": {
            "path": str(manifest_v2),
            "mentions_case_name_arg": "--case_name" in m2_txt,
            "mentions_metric_case39": "metric/case39" in m2_txt,
            "hardcodes_metric_case14": "metric/case14" in m2_txt,
        },
    }


def inspect_case39_paths(repo_root: Path) -> Dict[str, List[str]]:
    return {
        "metric_case39_hits": find_files(repo_root, "metric/case39"),
        "saved_model_case39_hits": find_files(repo_root, "saved_model/case39"),
        "case39_name_hits": find_files(repo_root, "case39"),
    }


def classify_status(canonical: Dict[str, Dict[str, object]], checkpoint: Dict[str, object], code: Dict[str, object]) -> str:
    all_exist = all(v.get("exists") for v in canonical.values())
    any_case14_backing = any(v.get("resolved_is_case14") for v in canonical.values())
    ckpt_exists = bool(checkpoint.get("exists"))
    code_has_case39_cfg = bool(code["config_py"].get("mentions_case39") or code["config_py"].get("mentions_env_case_name"))
    nn_has_case39 = bool(code["nn_setting_py"].get("mentions_case39") or code["nn_setting_py"].get("mentions_saved_model_case39"))
    manifests_caseaware = bool(code["manifest_v1"].get("mentions_case_name_arg") and code["manifest_v2"].get("mentions_case_name_arg"))

    if all_exist and any_case14_backing:
        return "BRIDGE_ONLY_CASE14_BACKED"
    if all_exist and ckpt_exists and code_has_case39_cfg and nn_has_case39 and manifests_caseaware and not any_case14_backing:
        return "NATIVE_CASE39_READY"
    if manifests_caseaware and (code_has_case39_cfg or nn_has_case39):
        return "PARTIAL_NATIVE_SUPPORT"
    return "NEEDS_NATIVE_CASE39_SUPPORT"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether the repo is bridge-only or ready for native case39 assets.")
    ap.add_argument("--repo_root", default=".", help="Repo root")
    ap.add_argument("--output", required=True, help="JSON output path")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    canonical = inspect_canonical(repo_root)
    checkpoint = inspect_checkpoint(repo_root)
    code = inspect_code(repo_root)
    case39_paths = inspect_case39_paths(repo_root)
    status = classify_status(canonical, checkpoint, code)

    summary = {
        "method": "case39_native_readiness_audit",
        "repo_root": str(repo_root),
        "status": status,
        "canonical_assets": canonical,
        "checkpoint": checkpoint,
        "code_support": code,
        "case39_path_hits": case39_paths,
        "recommendations": [
            "If status is BRIDGE_ONLY_CASE14_BACKED, do not report these runs as native case39 results.",
            "If saved_model/case39/checkpoint_rnn.pt is missing, raw case39 bank generation is not yet runnable.",
            "If canonical metric/case39 banks resolve to metric/case14, current runs are only valid as bridge/regression evidence.",
            "Native case39 requires either true case39 precomputed banks or a valid case39 raw-generation path with matching checkpoint/data.",
        ],
    }

    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "output": str(output),
        "checkpoint_exists": checkpoint.get("exists", False),
        "canonical_exists": {k: v.get("exists", False) for k, v in canonical.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

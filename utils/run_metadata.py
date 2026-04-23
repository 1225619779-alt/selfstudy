from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict


def git_head(cwd: str | None = None) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True)
            .strip()
        )
    except Exception:
        return ""


def file_fingerprint(path: str) -> Dict[str, Any]:
    p = Path(path)
    try:
        stat = p.stat()
        return {
            "path": str(p),
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
    except FileNotFoundError:
        return {
            "path": str(p),
            "size": None,
            "mtime": None,
        }


def attach_runtime_metadata(
    metadata: Dict[str, Any],
    *,
    repo_root: str,
    input_cache_path: str | None = None,
    output_cache_path: str | None = None,
    runner_name: str = "",
) -> None:
    metadata["git_head"] = git_head(cwd=repo_root)
    metadata["repo_root"] = repo_root
    metadata["runner_name"] = runner_name
    metadata["cwd"] = os.getcwd()
    if input_cache_path is not None:
        metadata["input_cache_fingerprint"] = file_fingerprint(input_cache_path)
    if output_cache_path is not None:
        metadata["output_cache_fingerprint"] = file_fingerprint(output_cache_path)

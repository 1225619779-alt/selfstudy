from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

FROZEN_REGIME = {
    "decision_step_group": 1,
    "busy_time_quantile": 0.65,
    "use_cost_budget": False,
    "cost_budget_window_steps": 20,
    "cost_budget_quantile": 0.60,
    "slot_budget_list": [1, 2],
    "max_wait_steps": 10,
}

PURE_MIGRATION_CONTRACT = {
    "stage": "case39_pure_migration_v1",
    "winner": "oracle_protected_ec",
    "allowed_methods": [
        "phase3_proposed",
        "oracle_protected_ec",
        "best-threshold",
        "topk_expected_consequence",
    ],
    "allow_retune": False,
    "frozen_regime": FROZEN_REGIME,
}


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()



def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(f"Destination is a directory: {path}")
        path.unlink()



def _stage(src: Path, dst: Path, mode: str, force: bool) -> Dict[str, object]:
    src = src.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Missing source asset: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(f"Destination already exists: {dst}")
        _remove_existing(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    target = dst.resolve()
    stat = target.stat()
    return {
        "source_path": str(src),
        "destination_path": str(dst),
        "resolved_destination_path": str(target),
        "mode": mode,
        "size_bytes": int(stat.st_size),
        "mtime_epoch": float(stat.st_mtime),
        "sha256": _sha256(target),
    }



def _upsert_asset(protocol: Dict[str, object], key: str, payload: Dict[str, object]) -> None:
    assets = protocol.setdefault("assets", {})
    assets[key] = payload



def _load_protocol(path: Path) -> Dict[str, object]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}



def _rel_to(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))



def _stage_holdouts_from_manifest(
    protocol: Dict[str, object],
    *,
    manifest_path: Path,
    holdout_src_dir: Path,
    workdir_override: Optional[Path],
    mode: str,
    force: bool,
) -> None:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    workdir = workdir_override.resolve() if workdir_override is not None else Path(manifest["workdir"]).resolve()
    holdout_src_dir = holdout_src_dir.expanduser().resolve()
    if not holdout_src_dir.is_dir():
        raise NotADirectoryError(f"Missing holdout source directory: {holdout_src_dir}")

    staged = protocol.setdefault("holdout_test_banks", {})
    for hold in manifest["holdouts"]:
        dst = (workdir / hold["test_bank"]).resolve()
        src = holdout_src_dir / dst.name
        info = _stage(src=src, dst=dst, mode=mode, force=force)
        info["tag"] = str(hold["tag"])
        info["family_tag"] = hold.get("family_tag")
        staged[str(hold["tag"])] = info



def _canonical_out_root(case_name: str, out_root: Optional[str]) -> Path:
    if out_root:
        return Path(out_root).expanduser().resolve()
    return Path("metric") / case_name



def main() -> None:
    p = argparse.ArgumentParser(description="Stage canonical case assets and optional blind confirm holdout banks with provenance.")
    p.add_argument("--case_name", default="case39")
    p.add_argument("--out_root", default=None)
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    p.add_argument("--force", action="store_true")

    p.add_argument("--clean_src", default=None)
    p.add_argument("--attack_src", default=None)
    p.add_argument("--train_src", default=None)
    p.add_argument("--val_src", default=None)
    p.add_argument("--test_src", default=None, help="Optional single smoke test bank staged to mixed_bank_test_smoke.npy")
    p.add_argument("--skip_canonical", action="store_true")

    p.add_argument("--manifest", default=None)
    p.add_argument("--holdout_src_dir", default=None, help="Directory containing precomputed mixed_bank_test_*.npy files matching manifest basenames.")
    p.add_argument("--workdir", default=None, help="Optional override for manifest['workdir'] when staging holdout banks.")
    args = p.parse_args()

    out_root = _canonical_out_root(args.case_name, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    protocol_path = out_root / "asset_protocol.json"
    contract_path = out_root / "bridge_contract.json"
    protocol = _load_protocol(protocol_path)
    protocol.update(
        {
            "case_name": args.case_name,
            "output_root": str(out_root),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    if not args.skip_canonical:
        required = {
            "clean_bank": (args.clean_src, out_root / "metric_clean_alarm_scores_full.npy"),
            "attack_bank": (args.attack_src, out_root / "metric_attack_alarm_scores_400.npy"),
            "train_bank": (args.train_src, out_root / "mixed_bank_fit.npy"),
            "val_bank": (args.val_src, out_root / "mixed_bank_eval.npy"),
        }
        missing = [k for k, (src, _) in required.items() if not src]
        if missing:
            raise ValueError(f"Missing required canonical sources: {missing}")
        for key, (src, dst) in required.items():
            info = _stage(src=Path(src), dst=dst, mode=args.mode, force=bool(args.force))
            _upsert_asset(protocol, key, info)

        if args.test_src:
            info = _stage(src=Path(args.test_src), dst=out_root / "mixed_bank_test_smoke.npy", mode=args.mode, force=bool(args.force))
            _upsert_asset(protocol, "test_bank_smoke", info)

    if args.manifest and args.holdout_src_dir:
        _stage_holdouts_from_manifest(
            protocol,
            manifest_path=Path(args.manifest).expanduser().resolve(),
            holdout_src_dir=Path(args.holdout_src_dir),
            workdir_override=Path(args.workdir).expanduser().resolve() if args.workdir else None,
            mode=args.mode,
            force=bool(args.force),
        )
        protocol["last_manifest"] = str(Path(args.manifest).expanduser().resolve())

    with protocol_path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    contract = dict(PURE_MIGRATION_CONTRACT)
    contract.update(
        {
            "case_name": args.case_name,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "First-stage bridge is pure migration only.",
                "Do not run case39 raw generation through configs/config.py or configs/nn_setting.py.",
                "Do not report select_regime_phase3_val.py outputs as case39 main results.",
            ],
        }
    )
    with contract_path.open("w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "out_root": str(out_root),
        "asset_protocol": str(protocol_path),
        "bridge_contract": str(contract_path),
        "mode": args.mode,
        "skip_canonical": bool(args.skip_canonical),
        "manifest": None if not args.manifest else str(Path(args.manifest).expanduser().resolve()),
        "holdout_src_dir": None if not args.holdout_src_dir else str(Path(args.holdout_src_dir).expanduser().resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

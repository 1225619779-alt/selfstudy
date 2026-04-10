from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PREPARE_CASE_BANK_ASSETS = r'''from __future__ import annotations

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
'''

MANIFEST_V1 = r'''from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")

DEFAULT_FAMILIES = [
    {
        "family_tag": "cfA_frontloaded",
        "schedule": "clean:90;att-1-0.15:90;clean:45;att-2-0.25:90;clean:45;att-3-0.25:60;clean:120",
        "seeds": [20260511, 20260512],
        "offsets": [1020, 1080],
    },
    {
        "family_tag": "cfB_backloaded",
        "schedule": "clean:150;att-2-0.20:60;clean:30;att-3-0.35:90;clean:30;att-1-0.15:60;clean:120",
        "seeds": [20260521, 20260522],
        "offsets": [1140, 1200],
    },
]

FROZEN_REGIME = {
    "decision_step_group": 1,
    "busy_time_quantile": 0.65,
    "use_cost_budget": False,
    "cost_budget_window_steps": 20,
    "cost_budget_quantile": 0.60,
    "slot_budget_list": [1, 2],
    "max_wait_steps": 10,
}


def _run(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def _summary_path_from_npy(path: Path) -> Path:
    return path.with_suffix(".summary.json")


def _default_bank_paths(case_name: str) -> Dict[str, str]:
    root = Path("metric") / case_name
    return {
        "clean_bank": str(root / "metric_clean_alarm_scores_full.npy"),
        "attack_bank": str(root / "metric_attack_alarm_scores_400.npy"),
        "train_bank": str(root / "mixed_bank_fit.npy"),
        "val_bank": str(root / "mixed_bank_eval.npy"),
    }


def build_manifest(
    workdir: Path,
    out_dir: Path,
    *,
    case_name: str,
    clean_bank: str | None,
    attack_bank: str | None,
    train_bank: str | None,
    val_bank: str | None,
) -> Dict[str, object]:
    banks_dir = out_dir / "banks"
    results_dir = out_dir / "results"
    banks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    holdouts = []
    for family in DEFAULT_FAMILIES:
        for idx, (seed, offset) in enumerate(zip(family["seeds"], family["offsets"])):
            tag = f"{family['family_tag']}_{idx}_seed{seed}_off{offset}"
            test_bank = banks_dir / f"mixed_bank_test_{tag}.npy"
            result_npy = results_dir / f"budget_scheduler_phase3_holdout_{tag}.npy"
            result_summary = _summary_path_from_npy(result_npy)
            holdouts.append(
                {
                    "tag": tag,
                    "family_tag": family["family_tag"],
                    "schedule": family["schedule"],
                    "seed_base": int(seed),
                    "start_offset": int(offset),
                    "test_bank": _rel(test_bank, workdir),
                    "result_npy": _rel(result_npy, workdir),
                    "result_summary": _rel(result_summary, workdir),
                }
            )

    banks = _default_bank_paths(case_name)
    return {
        "workdir": str(workdir.resolve()),
        "case_name": case_name,
        "clean_bank": clean_bank or banks["clean_bank"],
        "attack_bank": attack_bank or banks["attack_bank"],
        "train_bank": train_bank or banks["train_bank"],
        "val_bank": val_bank or banks["val_bank"],
        "schedule": "confirm_multi_family_v1",
        "confirm_families": DEFAULT_FAMILIES,
        "frozen_regime": FROZEN_REGIME,
        "holdouts": holdouts,
    }


def generate_assets(manifest: Dict[str, object], workdir: Path, force: bool = False, allow_raw_generation: bool = False) -> None:
    case_name = str(manifest.get("case_name", "case14"))
    for hold in manifest["holdouts"]:
        test_bank = workdir / hold["test_bank"]
        result_npy = workdir / hold["result_npy"]
        result_summary = workdir / hold["result_summary"]

        if force or not test_bank.exists():
            if case_name != "case14" and not allow_raw_generation:
                raise RuntimeError(
                    f"Missing staged test bank for {case_name}: {test_bank}. "
                    "Stage precomputed holdout banks first, or explicitly enable raw generation after wiring case39 raw support."
                )
            cmd = [
                sys.executable,
                "evaluation_mixed_timeline.py",
                "--tau_verify",
                "-1",
                "--schedule",
                str(hold["schedule"]),
                "--seed_base",
                str(hold["seed_base"]),
                "--start_offset",
                str(hold["start_offset"]),
                "--output",
                str(test_bank),
            ]
            _run(cmd, cwd=workdir)
        else:
            print(f"[skip] existing test bank: {test_bank}")

        if force or (not result_npy.exists()) or (not result_summary.exists()):
            cmd = [
                sys.executable,
                "evaluation_budget_scheduler_phase3_holdout.py",
                "--clean_bank",
                str(workdir / manifest["clean_bank"]),
                "--attack_bank",
                str(workdir / manifest["attack_bank"]),
                "--train_bank",
                str(workdir / manifest["train_bank"]),
                "--val_bank",
                str(workdir / manifest["val_bank"]),
                "--test_bank",
                str(test_bank),
                "--slot_budget_list",
                "1",
                "2",
                "--decision_step_group",
                str(manifest["frozen_regime"]["decision_step_group"]),
                "--busy_time_quantile",
                str(manifest["frozen_regime"]["busy_time_quantile"]),
                "--max_wait_steps",
                str(manifest["frozen_regime"]["max_wait_steps"]),
                "--output",
                str(result_npy),
            ]
            _run(cmd, cwd=workdir)
        else:
            print(f"[skip] existing baseline holdout summary: {result_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 4-holdout blind confirm manifest and baseline phase3 holdout summaries.")
    parser.add_argument("--workdir", default=".", help="Repo root.")
    parser.add_argument("--case_name", default=DEFAULT_CASE_NAME, help="Case tag used for default metric root.")
    parser.add_argument("--output_dir", default=None, help="Directory for confirm manifest/banks/results.")
    parser.add_argument("--clean_bank", default=None, help="Optional override for clean bank path (relative to workdir).")
    parser.add_argument("--attack_bank", default=None, help="Optional override for attack bank path (relative to workdir).")
    parser.add_argument("--train_bank", default=None, help="Optional override for train bank path (relative to workdir).")
    parser.add_argument("--val_bank", default=None, help="Optional override for val bank path (relative to workdir).")
    parser.add_argument("--manifest_only", action="store_true", help="Only write manifest; do not generate banks/results.")
    parser.add_argument("--force", action="store_true", help="Regenerate banks/results even if files already exist.")
    parser.add_argument("--allow_raw_generation", action="store_true", help="Allow non-case14 raw generation. Disabled by default for case39 pure migration.")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    output_dir = args.output_dir or f"metric/{args.case_name}/phase3_confirm_blind_v1"
    out_dir = (workdir / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        workdir=workdir,
        out_dir=out_dir,
        case_name=str(args.case_name),
        clean_bank=args.clean_bank,
        attack_bank=args.attack_bank,
        train_bank=args.train_bank,
        val_bank=args.val_bank,
    )
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.manifest_only:
        generate_assets(
            manifest=manifest,
            workdir=workdir,
            force=bool(args.force),
            allow_raw_generation=bool(args.allow_raw_generation),
        )

    print(json.dumps({
        "manifest_path": str(manifest_path),
        "output_dir": str(out_dir),
        "manifest_only": bool(args.manifest_only),
        "case_name": args.case_name,
        "n_holdouts": len(manifest["holdouts"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''

MANIFEST_V2 = MANIFEST_V1.replace('confirm_multi_family_v1', 'confirm_multi_family_v2').replace('phase3_confirm_blind_v1', 'phase3_confirm_blind_v2').replace('Create a 4-holdout blind confirm manifest and baseline phase3 holdout summaries.', 'Create 4 additional blind confirm holdouts from 2 new schedule families and baseline phase3 summaries.').replace('DEFAULT_FAMILIES = [\n    {\n        "family_tag": "cfA_frontloaded",\n        "schedule": "clean:90;att-1-0.15:90;clean:45;att-2-0.25:90;clean:45;att-3-0.25:60;clean:120",\n        "seeds": [20260511, 20260512],\n        "offsets": [1020, 1080],\n    },\n    {\n        "family_tag": "cfB_backloaded",\n        "schedule": "clean:150;att-2-0.20:60;clean:30;att-3-0.35:90;clean:30;att-1-0.15:60;clean:120",\n        "seeds": [20260521, 20260522],\n        "offsets": [1140, 1200],\n    },\n]','DEFAULT_FAMILIES = [\n    {\n        "family_tag": "cfC_interleaved",\n        "schedule": "clean:120;att-1-0.15:45;clean:60;att-2-0.25:45;clean:60;att-3-0.35:45;clean:60;att-2-0.20:45;clean:60",\n        "seeds": [20260611, 20260612],\n        "offsets": [1260, 1320],\n    },\n    {\n        "family_tag": "cfD_tailheavy",\n        "schedule": "clean:180;att-1-0.10:45;clean:45;att-2-0.20:45;clean:30;att-3-0.40:105;clean:90",\n        "seeds": [20260621, 20260622],\n        "offsets": [1380, 1440],\n    },\n]').replace('Create a 4-holdout blind confirm manifest and baseline phase3 holdout summaries.', 'Create 4 additional blind confirm holdouts from 2 new schedule families and baseline phase3 summaries.')

NN_SETTING = r'''"""
Settings for Neural Network
"""
from pathlib import Path

import torch

from configs.config import sys_config

if sys_config["case_name"] == "case14":

    nn_setting = {
        # Network Structure
        "sample_length": 6,
        "lattent_dim": 10,
        "no_layer": 3,
        "feature_size": 68,

        # Training
        "epochs": 1000,
        "lr": 1e-3,
        "patience": 10,
        "delta": 0,
        "model_path": "saved_model/case14/checkpoint_rnn.pt",
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
elif sys_config["case_name"] == "case39":
    model_path = Path("saved_model/case39/checkpoint_rnn.pt")
    raise RuntimeError(
        "case39 raw generation is disabled in the first-stage bridge. "
        f"Do not fall back to case14 checkpoint. Expected future model path would be: {model_path}"
    )
else:
    raise ValueError(f"Unsupported case_name: {sys_config['case_name']}")
'''


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {path}: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_config(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, 'import numpy as np\n', 'import numpy as np\nimport os\n', path)
    text = replace_once(text, "case_name = 'case14'\n\nif case_name == 'case14':\n", 'case_name = os.environ.get("DDET_CASE_NAME", "case14")\n\nif case_name == \"case14\":\n', path)
    text = replace_once(text, "\n\n\ndef save_metric", '\n\nelif case_name == "case39":\n    raise RuntimeError(\n        "case39 raw generation is disabled in the first-stage bridge. "\n        "Stage precomputed case39 banks under metric/case39 via prepare_case_bank_assets.py. "\n        "Do not fall back to case14 raw assets or checkpoint."\n    )\nelse:\n    raise ValueError(f"Unsupported case_name: {case_name}")\n\n\ndef save_metric', path)
    path.write_text(text, encoding='utf-8')


def patch_default_output(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding='utf-8')


def patch_file(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('repo_root')
    args = ap.parse_args()
    root = Path(args.repo_root).expanduser().resolve()

    patch_config(root / 'configs' / 'config.py')
    patch_file(root / 'configs' / 'nn_setting.py', NN_SETTING)
    patch_file(root / 'prepare_case_bank_assets.py', PREPARE_CASE_BANK_ASSETS)
    patch_file(root / 'make_phase3_confirm_manifest.py', MANIFEST_V1)
    patch_file(root / 'make_phase3_confirm_manifest_v2.py', MANIFEST_V2)
    patch_default_output(
        root / 'evaluation_budget_scheduler_phase3_holdout.py',
        'parser.add_argument("--output", type=str, default="metric/case14/budget_scheduler_phase3_holdout.npy")',
        'parser.add_argument("--output", type=str, default=f"metric/{os.environ.get(\"DDET_CASE_NAME\", \"case14\")}/budget_scheduler_phase3_holdout.npy")',
    )
    patch_default_output(
        root / 'evaluation_budget_scheduler_phase3.py',
        'parser.add_argument("--output", type=str, default="metric/case14/budget_scheduler_phase3_ca.npy")',
        'parser.add_argument("--output", type=str, default=f"metric/{os.environ.get(\"DDET_CASE_NAME\", \"case14\")}/budget_scheduler_phase3_ca.npy")',
    )
    patch_default_output(
        root / 'select_regime_phase3_val.py',
        'p.add_argument("--output", type=str, default="metric/case14/phase3_val_regime_ranking.json")',
        'p.add_argument("--output", type=str, default=f"metric/{os.environ.get(\"DDET_CASE_NAME\", \"case14\")}/phase3_val_regime_ranking.json")',
    )
    # optional wrapper safety defaults
    seq_path = root / 'run_phase3_holdout_sequential.py'
    if seq_path.exists():
        text = seq_path.read_text(encoding='utf-8')
        if 'DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")' not in text:
            text = text.replace('import numpy as np\n\n', 'import numpy as np\n\nDEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")\n\n', 1)
        repls = {
            'p.add_argument("--clean_bank", type=str, default="metric/case14/metric_clean_alarm_scores_full.npy")': 'p.add_argument("--clean_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_clean_alarm_scores_full.npy")',
            'p.add_argument("--attack_bank", type=str, default="metric/case14/metric_attack_alarm_scores_400.npy")': 'p.add_argument("--attack_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_attack_alarm_scores_400.npy")',
            'p.add_argument("--train_bank", type=str, default="metric/case14/mixed_bank_fit.npy")': 'p.add_argument("--train_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_fit.npy")',
            'p.add_argument("--val_bank", type=str, default="metric/case14/mixed_bank_eval.npy")': 'p.add_argument("--val_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_eval.npy")',
            'p.add_argument("--test_bank", type=str, default="metric/case14/mixed_bank_test_holdout.npy")': 'p.add_argument("--test_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_test_holdout.npy")',
            'p.add_argument("--ranking_output", type=str, default="metric/case14/phase3_val_regime_ranking_holdout.json")': 'p.add_argument("--ranking_output", type=str, default=f"metric/{DEFAULT_CASE_NAME}/phase3_val_regime_ranking_holdout.json")',
            'p.add_argument("--holdout_output", type=str, default="metric/case14/budget_scheduler_phase3_holdout_auto.npy")': 'p.add_argument("--holdout_output", type=str, default=f"metric/{DEFAULT_CASE_NAME}/budget_scheduler_phase3_holdout_auto.npy")',
        }
        for old, new in repls.items():
            text = replace_once(text, old, new, seq_path)
        seq_path.write_text(text, encoding='utf-8')

    batch_path = root / 'run_phase3_multi_holdout_batch.py'
    if batch_path.exists():
        text = batch_path.read_text(encoding='utf-8')
        if 'DEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")' not in text:
            text = text.replace('from typing import Any, Dict, List, Sequence, Tuple\n\n', 'from typing import Any, Dict, List, Sequence, Tuple\n\nDEFAULT_CASE_NAME = os.environ.get("DDET_CASE_NAME", "case14")\n\n', 1)
        repls = {
            'p.add_argument("--clean_bank", type=str, default="metric/case14/metric_clean_alarm_scores_full.npy")': 'p.add_argument("--clean_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_clean_alarm_scores_full.npy")',
            'p.add_argument("--attack_bank", type=str, default="metric/case14/metric_attack_alarm_scores_400.npy")': 'p.add_argument("--attack_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/metric_attack_alarm_scores_400.npy")',
            'p.add_argument("--train_bank", type=str, default="metric/case14/mixed_bank_fit.npy")': 'p.add_argument("--train_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_fit.npy")',
            'p.add_argument("--val_bank", type=str, default="metric/case14/mixed_bank_eval.npy")': 'p.add_argument("--val_bank", type=str, default=f"metric/{DEFAULT_CASE_NAME}/mixed_bank_eval.npy")',
            'p.add_argument("--out_dir", type=str, default="metric/case14/phase3_multi_holdout")': 'p.add_argument("--out_dir", type=str, default=f"metric/{DEFAULT_CASE_NAME}/phase3_multi_holdout")',
        }
        for old, new in repls.items():
            text = replace_once(text, old, new, batch_path)
        batch_path.write_text(text, encoding='utf-8')

    print(f'Patched case39 bridge files under {root}')


if __name__ == '__main__':
    main()

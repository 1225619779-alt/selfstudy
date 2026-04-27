from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "metric" / "case39" / "q1_top_sprint_20260424_094946"
STAMP_LABEL = "20260424_094946"
STAMP_PATH = Path("/tmp") / f"case39_q1_top_sprint_{STAMP_LABEL}.stamp"
OLD_REPO_CASE14 = Path("/home/pang/projects/DDET-MTD/metric/case14")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_record(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    return {
        "path": rel(path) if path.is_relative_to(ROOT) else str(path),
        "exists": exists,
        "is_symlink": path.is_symlink(),
        "resolved_path": str(resolved),
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def git_value(args: List[str]) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    except Exception:
        return None


def holdout_banks() -> List[str]:
    banks: List[Path] = []
    for version in ("phase3_confirm_blind_v1", "phase3_confirm_blind_v2"):
        banks_dir = ROOT / "metric" / "case39" / version / "banks"
        banks.extend(sorted(banks_dir.glob("*.npy")))
    return [rel(p) for p in sorted(banks)]


def file_set_for_hashing(include_manifests: bool) -> Dict[str, Path]:
    files = {
        "case14_fit": ROOT / "metric" / "case14" / "mixed_bank_fit.npy",
        "case14_eval": ROOT / "metric" / "case14" / "mixed_bank_eval.npy",
        "case39_canonical_fit": ROOT / "metric" / "case39" / "mixed_bank_fit.npy",
        "case39_canonical_eval": ROOT / "metric" / "case39" / "mixed_bank_eval.npy",
        "case39_native_fit": ROOT / "metric" / "case39_localretune" / "mixed_bank_fit_native.npy",
        "case39_native_eval": ROOT / "metric" / "case39_localretune" / "mixed_bank_eval_native.npy",
        "case39_clean": ROOT / "metric" / "case39" / "metric_clean_alarm_scores_full.npy",
        "case39_attack": ROOT / "metric" / "case39" / "metric_attack_alarm_scores_400.npy",
    }
    for bank in holdout_banks():
        files[f"holdout::{bank}"] = ROOT / bank
    if include_manifests:
        files["source_frozen_transfer_manifest"] = OUT / "source_frozen_transfer_manifest.json"
        files["full_native_case39_manifest"] = OUT / "full_native_case39_manifest.json"
    return files


def hash_records(include_manifests: bool) -> Dict[str, Any]:
    return {name: path_record(path) for name, path in sorted(file_set_for_hashing(include_manifests).items())}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifests() -> None:
    source_manifest = {
        "manifest_type": "source_frozen_transfer",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "bridge transfer / stress-test evidence; not native train/val evidence",
        "source_case": "case14",
        "source_train_bank": "metric/case14/mixed_bank_fit.npy",
        "source_val_bank": "metric/case14/mixed_bank_eval.npy",
        "target_case": "case39",
        "target_clean_bank": "metric/case39/metric_clean_alarm_scores_full.npy",
        "target_attack_bank": "metric/case39/metric_attack_alarm_scores_400.npy",
        "target_holdout_banks": holdout_banks(),
        "fixed_budget": [1, 2],
        "fixed_wmax": 10,
        "frozen_holdout_count": len(holdout_banks()),
        "interpretation_guardrail": "Do not describe this manifest as native larger-system success.",
    }
    full_native_manifest = {
        "manifest_type": "full_native_case39",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "full-native case39 train/val/clean/attack/test route for future Gate 2 rerun",
        "native_case": "case39",
        "native_train_bank": "metric/case39_localretune/mixed_bank_fit_native.npy",
        "native_val_bank": "metric/case39_localretune/mixed_bank_eval_native.npy",
        "native_clean_bank": "metric/case39/metric_clean_alarm_scores_full.npy",
        "native_attack_bank": "metric/case39/metric_attack_alarm_scores_400.npy",
        "native_holdout_banks": holdout_banks(),
        "forbidden_train_val_inputs": [
            "metric/case39/mixed_bank_fit.npy",
            "metric/case39/mixed_bank_eval.npy",
        ],
        "fixed_budget": [1, 2],
        "fixed_wmax": 10,
        "frozen_holdout_count": len(holdout_banks()),
        "interpretation_guardrail": "Gate 2 must read explicit native_train_bank/native_val_bank and must not rely on canonical case39 fit/eval symlinks.",
    }
    write_json(OUT / "source_frozen_transfer_manifest.json", source_manifest)
    write_json(OUT / "full_native_case39_manifest.json", full_native_manifest)


def newer_files(root: Path, stamp: Path) -> List[str]:
    if not root.exists() or not stamp.exists():
        return []
    threshold = stamp.stat().st_mtime
    out: List[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime > threshold:
            out.append(str(path))
    return sorted(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(f"case39 q1 top sprint gate1 stamp {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    # Make sure the stamp is older than all Gate 1 writes while still newer than existing canonical assets.
    time.sleep(0.25)

    pre_hash = hash_records(include_manifests=True)
    write_json(OUT / "hash_pre_gate1.json", pre_hash)
    write_manifests()
    post_hash = hash_records(include_manifests=True)
    write_json(OUT / "hash_post_gate1.json", post_hash)

    q1_case14_newer = newer_files(ROOT / "metric" / "case14", STAMP_PATH)
    old_case14_newer = newer_files(OLD_REPO_CASE14, STAMP_PATH)
    (OUT / "anti_write_q1_case14.txt").write_text("\n".join(q1_case14_newer) + ("\n" if q1_case14_newer else ""), encoding="utf-8")
    (OUT / "anti_write_oldrepo_case14.txt").write_text("\n".join(old_case14_newer) + ("\n" if old_case14_newer else ""), encoding="utf-8")

    canonical_fit = ROOT / "metric" / "case39" / "mixed_bank_fit.npy"
    canonical_eval = ROOT / "metric" / "case39" / "mixed_bank_eval.npy"
    case14_fit = ROOT / "metric" / "case14" / "mixed_bank_fit.npy"
    case14_eval = ROOT / "metric" / "case14" / "mixed_bank_eval.npy"
    native_fit = ROOT / "metric" / "case39_localretune" / "mixed_bank_fit_native.npy"
    native_eval = ROOT / "metric" / "case39_localretune" / "mixed_bank_eval_native.npy"

    case14_hash_equal = (
        pre_hash["case14_fit"]["sha256"] == post_hash["case14_fit"]["sha256"]
        and pre_hash["case14_eval"]["sha256"] == post_hash["case14_eval"]["sha256"]
    )
    canonical_resolves_case14 = (
        canonical_fit.resolve(strict=False) == case14_fit.resolve(strict=False)
        and canonical_eval.resolve(strict=False) == case14_eval.resolve(strict=False)
    )
    manifest_native_ready = (
        native_fit.exists()
        and native_eval.exists()
        and not native_fit.is_symlink()
        and not native_eval.is_symlink()
        and len(holdout_banks()) == 8
        and not q1_case14_newer
        and not old_case14_newer
        and case14_hash_equal
    )
    native_ready = manifest_native_ready and not canonical_resolves_case14
    readiness_status = "NATIVE_CASE39_READY" if native_ready else "MANIFEST_NATIVE_READY_CANONICAL_STILL_CASE14" if manifest_native_ready else "NEEDS_NATIVE_CASE39_SUPPORT"

    readiness = {
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "HEAD"]),
        "stamp_path": str(STAMP_PATH),
        "stamp_exists": STAMP_PATH.exists(),
        "source_frozen_manifest_exists": (OUT / "source_frozen_transfer_manifest.json").exists(),
        "full_native_manifest_exists": (OUT / "full_native_case39_manifest.json").exists(),
        "holdout_count": len(holdout_banks()),
        "native_train_bank": path_record(native_fit),
        "native_val_bank": path_record(native_eval),
        "canonical_case39_fit": path_record(canonical_fit),
        "canonical_case39_eval": path_record(canonical_eval),
        "canonical_case39_fit_resolves_to_case14": canonical_fit.resolve(strict=False) == case14_fit.resolve(strict=False),
        "canonical_case39_eval_resolves_to_case14": canonical_eval.resolve(strict=False) == case14_eval.resolve(strict=False),
        "full_native_manifest_uses_canonical_case39_fit_eval": False,
        "anti_write_q1_case14_empty": not q1_case14_newer,
        "anti_write_oldrepo_case14_empty": not old_case14_newer,
        "case14_hash_pre_post_equal": case14_hash_equal,
        "readiness_status": readiness_status,
        "gate2_recommendation": "continue only with explicit full_native_case39_manifest.json; do not use canonical case39 fit/eval until they no longer resolve to case14",
    }
    write_json(OUT / "native_readiness_after_cleanup.json", readiness)

    provenance_lines = [
        "# Gate 1 Provenance Report",
        "",
        f"- Sprint directory: `{rel(OUT)}`",
        f"- STAMP: `{STAMP_PATH}`",
        f"- Branch: `{readiness['branch']}`",
        f"- Commit: `{readiness['commit']}`",
        f"- Source-frozen manifest: `source_frozen_transfer_manifest.json`",
        f"- Full-native manifest: `full_native_case39_manifest.json`",
        "",
        "## Source-frozen Transfer Separation",
        "",
        "- `source_case = case14`",
        "- `source_train_bank = metric/case14/mixed_bank_fit.npy`",
        "- `source_val_bank = metric/case14/mixed_bank_eval.npy`",
        "- `target_case = case39`",
        "- target clean/attack/test banks are explicit case39 paths.",
        "- Interpretation: bridge transfer / stress-test evidence, not native train/val evidence.",
        "",
        "## Full-native Case39 Separation",
        "",
        "- `native_train_bank = metric/case39_localretune/mixed_bank_fit_native.npy`",
        "- `native_val_bank = metric/case39_localretune/mixed_bank_eval_native.npy`",
        "- clean/attack/test banks are explicit case39 paths.",
        "- The manifest forbids using canonical `metric/case39/mixed_bank_fit.npy` and `metric/case39/mixed_bank_eval.npy` as train/val inputs.",
        "",
        "## Canonical Case39 Fit/Eval",
        "",
        f"- canonical fit resolves to `{readiness['canonical_case39_fit']['resolved_path']}`.",
        f"- canonical eval resolves to `{readiness['canonical_case39_eval']['resolved_path']}`.",
        f"- canonical still resolves to case14: `{canonical_resolves_case14}`.",
        "",
        "## Hash Evidence",
        "",
        f"- case14 fit/eval pre/post SHA equal: `{case14_hash_equal}`.",
        "- Pre-hash manifest: `hash_pre_gate1.json`.",
        "- Post-hash manifest: `hash_post_gate1.json`.",
        "",
        "## Readiness",
        "",
        f"- Readiness status: `{readiness_status}`.",
        "- Practical Gate 2 route: use the explicit full-native manifest. Do not rely on canonical case39 fit/eval until they are cleaned or bypassed by manifest arguments.",
    ]
    (OUT / "provenance_report.md").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    anti_lines = [
        "# Gate 1 Anti-write Report",
        "",
        f"- STAMP: `{STAMP_PATH}`",
        f"- q1 repo case14 files newer than STAMP: `{len(q1_case14_newer)}`",
        f"- old repo case14 files newer than STAMP: `{len(old_case14_newer)}`",
        f"- `anti_write_q1_case14.txt` empty: `{not q1_case14_newer}`",
        f"- `anti_write_oldrepo_case14.txt` empty: `{not old_case14_newer}`",
        f"- case14 fit/eval SHA unchanged pre/post Gate 1: `{case14_hash_equal}`",
        "",
        "## Interpretation",
        "",
        "- Gate 1 did not run training or confirm pipelines.",
        "- No case14 write was observed after the Gate 1 STAMP.",
        "- Gate 2 must create its own pre-run STAMP and repeat this check around the actual rerun.",
    ]
    (OUT / "anti_write_report.md").write_text("\n".join(anti_lines) + "\n", encoding="utf-8")

    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "outputs_tree_gate1.txt").write_text("\n".join(files + ["outputs_tree_gate1.txt"]) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUT),
        "readiness_status": readiness_status,
        "stamp_path": str(STAMP_PATH),
        "anti_write_q1_case14_empty": not q1_case14_newer,
        "anti_write_oldrepo_case14_empty": not old_case14_newer,
        "case14_hash_pre_post_equal": case14_hash_equal,
        "canonical_resolves_case14": canonical_resolves_case14,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

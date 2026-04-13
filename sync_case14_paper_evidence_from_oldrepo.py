import hashlib
import json
import shutil
from pathlib import Path

SRC_ROOT = Path("/home/pang/projects/DDET-MTD")
DST_ROOT = Path(".").resolve()

# 只同步论文级有效证据，不带旧坏 significance v1/v2
DIRS_TO_SYNC = [
    "metric/case14/phase3_oracle_ablation_merged",
    "metric/case14/phase3_external_baseline_bundle",
    "metric/case14/phase3_paper_bundle",
    "metric/case14/phase3_significance_v3",
    "metric/case14/phase3_foundation_audit",
    "metric/case14/phase3_recompute_guard",
    "metric/case14/phase3_repro_spotcheck_v1",
    "metric/case14/phase3_repro_spotcheck_v2",
]

MANIFEST_PATH = DST_ROOT / "metric/case14/case14_paper_evidence_import_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = {
        "source_root": str(SRC_ROOT),
        "dest_root": str(DST_ROOT),
        "copied": [],
        "already_same": [],
        "conflicts": [],
        "missing_sources": [],
    }

    for rel_dir in DIRS_TO_SYNC:
        src_dir = SRC_ROOT / rel_dir
        if not src_dir.exists():
            manifest["missing_sources"].append(rel_dir)
            continue

        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue

            rel_file = src_file.relative_to(SRC_ROOT)
            dst_file = DST_ROOT / rel_file
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            src_hash = sha256_file(src_file)

            if dst_file.exists():
                dst_hash = sha256_file(dst_file)
                if dst_hash == src_hash:
                    manifest["already_same"].append({
                        "relative_path": str(rel_file),
                        "sha256": src_hash,
                    })
                else:
                    manifest["conflicts"].append({
                        "relative_path": str(rel_file),
                        "src_sha256": src_hash,
                        "dst_sha256": dst_hash,
                        "src_path": str(src_file),
                        "dst_path": str(dst_file),
                    })
                continue

            shutil.copy2(src_file, dst_file)
            manifest["copied"].append({
                "relative_path": str(rel_file),
                "sha256": src_hash,
                "src_path": str(src_file),
                "dst_path": str(dst_file),
            })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("sync complete")
    print("copied:", len(manifest["copied"]))
    print("already_same:", len(manifest["already_same"]))
    print("conflicts:", len(manifest["conflicts"]))
    print("missing_sources:", len(manifest["missing_sources"]))
    print("manifest:", MANIFEST_PATH)

    if manifest["conflicts"]:
        print("\n[conflicts]")
        for item in manifest["conflicts"]:
            print(json.dumps(item, ensure_ascii=False, indent=2))

    if manifest["missing_sources"]:
        print("\n[missing_sources]")
        for item in manifest["missing_sources"]:
            print(item)


if __name__ == "__main__":
    main()
import json
import os
from pathlib import Path

current_root = Path(".").resolve()
candidate_roots = []
for p in [
    current_root,
    Path("/home/pang/projects/DDET-MTD"),
    Path("/home/pang/projects/DDET-MTD-q1-case39"),
]:
    if p.exists() and p not in candidate_roots:
        candidate_roots.append(p)

targets = [
    "metric/case14/phase3_confirm_combined_v1_v2/aggregate_summary_merged.json",
    "metric/case14/phase3_oracle_ablation_merged/aggregate_summary.json",
    "metric/case14/phase3_external_baseline_bundle/external_baseline_bundle_summary.json",
    "metric/case14/phase3_paper_bundle/paper_bundle_summary.json",
    "metric/case14/phase3_significance_v3/significance_summary.json",
    "metric/case14/phase3_foundation_audit/foundation_audit_summary.json",
    "metric/case14/phase3_recompute_guard/recompute_guard_summary.json",
    "metric/case14/phase3_repro_spotcheck_v1/repro_spotcheck_summary.json",
    "metric/case14/phase3_repro_spotcheck_v2/repro_spotcheck_summary.json",
]

keywords = [
    "oracle_ablation",
    "external_baseline",
    "paper_bundle",
    "significance",
    "foundation_audit",
    "recompute_guard",
    "repro_spotcheck",
]

def walk_matches(root: Path, keywords):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = Path(dirpath) / fn
            rel = full.relative_to(root)
            rel_s = str(rel)
            low = rel_s.lower()
            if "metric/case14" not in low:
                continue
            if any(k in low for k in keywords):
                out.append(rel_s)
    return sorted(set(out))

print("\n=== roots ===")
for r in candidate_roots:
    print(r)

confirm_path = current_root / "metric/case14/phase3_confirm_combined_v1_v2/aggregate_summary_merged.json"
print("\n=== confirm_merged source_paths ===")
print("confirm exists:", confirm_path.exists())
if confirm_path.exists():
    with open(confirm_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for p in obj.get("source_paths", []):
        pp = Path(p)
        print({
            "source_path": str(pp),
            "exists": pp.exists(),
            "parent_exists": pp.parent.exists(),
        })

for root in candidate_roots:
    print(f"\n=== exact target check @ {root} ===")
    for rel in targets:
        p = root / rel
        print({
            "relative": rel,
            "exists": p.exists(),
            "absolute": str(p),
        })

    print(f"\n=== fuzzy search @ {root} ===")
    matches = walk_matches(root, keywords)
    if not matches:
        print("(no matches)")
    else:
        for m in matches:
            print(m)
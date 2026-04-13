import json
from pathlib import Path
from typing import Any

root = Path(".").resolve()

FILES = {
    "confirm_merged": root / "metric/case14/phase3_confirm_combined_v1_v2/aggregate_summary_merged.json",
    "oracle_ablation_merged": root / "metric/case14/phase3_oracle_ablation_merged/aggregate_summary.json",
    "external_baseline_bundle": root / "metric/case14/phase3_external_baseline_bundle/external_baseline_bundle_summary.json",
    "paper_bundle": root / "metric/case14/phase3_paper_bundle/paper_bundle_summary.json",
    "significance_v3": root / "metric/case14/phase3_significance_v3/significance_summary.json",
    "foundation_audit": root / "metric/case14/phase3_foundation_audit/foundation_audit_summary.json",
    "recompute_guard": root / "metric/case14/phase3_recompute_guard/recompute_guard_summary.json",
    "repro_spotcheck_v1": root / "metric/case14/phase3_repro_spotcheck_v1/repro_spotcheck_summary.json",
    "repro_spotcheck_v2": root / "metric/case14/phase3_repro_spotcheck_v2/repro_spotcheck_summary.json",
    "bad_significance_v1": root / "metric/case14/phase3_significance/significance_summary.json",
    "bad_significance_v2": root / "metric/case14/phase3_significance_v2/significance_summary.json",
}

METHOD_NAMES = [
    "phase3_proposed",
    "phase3_oracle_upgrade",
    "oracle_protected_ec",
    "phase3_reference",
    "oracle_fused_ec",
    "best_threshold_family",
    "topk_expected_consequence",
]

METRIC_KEYS = [
    "mean_recall",
    "mean_unnecessary",
    "mean_cost",
    "mean_served_ratio",
    "recall",
    "unnecessary",
    "cost",
    "served_ratio",
]

SCALAR_KEYWORDS = (
    "winner",
    "variant",
    "holdout",
    "family",
    "slot",
    "budget",
    "delta",
    "p_value",
    "pvalue",
    "ci",
    "pass",
    "status",
    "error",
    "n_holdouts",
    "objective",
)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def collect_scalars(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out.update(collect_scalars(v, p))
            elif is_scalar(v):
                kl = k.lower()
                if any(word in kl for word in SCALAR_KEYWORDS):
                    out[p] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.update(collect_scalars(v, p))
    return out

def summarize_method_blocks(obj: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    hits: list[tuple[str, dict[str, Any]]] = []

    if isinstance(obj, dict):
        methods_present = [m for m in METHOD_NAMES if m in obj and isinstance(obj[m], dict)]
        if methods_present:
            block: dict[str, Any] = {}
            for m in methods_present:
                child = obj[m]
                block[m] = {k: child.get(k) for k in METRIC_KEYS if k in child}
            if any(v for v in block.values()):
                hits.append((prefix or "<root>", block))

        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            hits.extend(summarize_method_blocks(v, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            hits.extend(summarize_method_blocks(v, p))

    return hits

def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

for name, path in FILES.items():
    print_header(name)
    print("path:", path)
    print("exists:", path.exists())

    if not path.exists():
        continue

    if path.suffix.lower() != ".json":
        print("skip: non-json")
        continue

    try:
        obj = load_json(path)
    except Exception as e:
        print("load_error:", repr(e))
        continue

    print("\n[scalar fields]")
    scalars = collect_scalars(obj)
    if not scalars:
        print("(none found)")
    else:
        for k, v in sorted(scalars.items()):
            print(f"{k}: {v}")

    print("\n[method blocks]")
    blocks = summarize_method_blocks(obj)
    if not blocks:
        print("(none found)")
    else:
        seen = set()
        for loc, block in blocks:
            key = json.dumps(block, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            print(f"\n@ {loc}")
            print(json.dumps(block, ensure_ascii=False, indent=2))
import json
from pathlib import Path
from typing import Any

root = Path(".").resolve()

FILES = {
    "stage_compare_significance": root / "metric/case39_compare/stage_compare_significance_v2.json",
    "measure_exact_match_audit": root / "metric/case39/preflight/case39_measure_v2_audit_128_144.json",
    "measure_benchmark": root / "gen_data/case39_bench_parallel_v2_256/parallel_measure_report.json",
}

KEYWORDS = (
    "stage",
    "slot",
    "delta",
    "p_value",
    "pvalue",
    "ci",
    "recall",
    "unnecessary",
    "cost",
    "served",
    "success_exact_equal",
    "z_max_abs_diff",
    "v_max_abs_diff",
    "sec_per_iter",
    "time_per_iter",
    "throughput",
    "status",
)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def collect(obj: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out.update(collect(v, p))
            elif is_scalar(v):
                kl = k.lower()
                if any(word in kl for word in KEYWORDS):
                    out[p] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.update(collect(v, p))
    return out

for name, path in FILES.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("path:", path)
    print("exists:", path.exists())
    if not path.exists():
        continue
    try:
        obj = load_json(path)
    except Exception as e:
        print("load_error:", repr(e))
        continue
    vals = collect(obj)
    for k, v in sorted(vals.items()):
        print(f"{k}: {v}")
from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))

p = Path("metric/case39/q1_top_sprint_20260424_094946/gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py")
spec = importlib.util.spec_from_file_location("g5", p)
g5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(g5)
for d in g5.G3.build_method_defs():
    print(d.get("method"), d.get("group"), d.get("variant"), d.get("cfg").slot_budget, list(d.keys()))

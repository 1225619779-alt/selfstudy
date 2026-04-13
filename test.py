import sys
from pathlib import Path

# >>> 就是加了下面这一行，把所有 print 的内容输出到 search_results.log 中 <<<
sys.stdout = open("search_results.log", "w", encoding="utf-8")

ROOT = Path(".").resolve()
TERMS = [
    "ttl", "ang_no", "ang_str", "severity_true",
    "expected_consequence_hat", "severity_hat",
    "W_max", "max_wait", "wait_steps"
]
SUFFIXES = {".py", ".md", ".json", ".m", ".txt", ".yaml", ".yml", ".ipynb"}

def show_hits(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    lines = text.splitlines()
    hit_ids = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(term.lower() in low for term in TERMS):
            hit_ids.append(i)
    if not hit_ids:
        return
    print(f"\n===== {path} =====")
    shown = set()
    for i in hit_ids:
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            if j in shown:
                continue
            shown.add(j)
            print(f"{j+1:4d}: {lines[j]}")

for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in SUFFIXES:
        show_hits(p)
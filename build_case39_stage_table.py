import json
from pathlib import Path

root = Path(".").resolve()

STAGES = [
    (
        "transfer_frozen_dev",
        "main_result",
        root / "metric/case39/postrun_audits/20260409_231456/summary.json",
    ),
    (
        "source_fixed_replay",
        "mechanism_isolation",
        root / "metric/case39_source_fixed_replay/postrun_bundle/summary.json",
    ),
    (
        "source_anchor",
        "repair_attempt",
        root / "metric/case39_source_anchor/postrun_bundle/summary.json",
    ),
    (
        "local_protected",
        "protocol_internal_negative_control",
        root / "metric/case39_localretune_protectedec/postrun_bundle/summary.json",
    ),
    (
        "local_unconstrained",
        "stress_test_out_of_protocol",
        root / "metric/case39_localretune/postrun_bundle/summary.json",
    ),
]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_slot_metrics(obj, slot: int):
    merged = obj.get("merged_8_holdouts", {})
    for key in (str(slot), slot):
        if key in merged and "phase3_oracle_upgrade" in merged[key]:
            x = merged[key]["phase3_oracle_upgrade"]
            return {
                "mean_recall": x.get("mean_recall"),
                "mean_unnecessary": x.get("mean_unnecessary"),
                "mean_cost": x.get("mean_cost"),
                "mean_served_ratio": x.get("mean_served_ratio"),
            }
    return None

rows = []
for stage_name, role, path in STAGES:
    row_base = {
        "stage_name": stage_name,
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "stage_meta": None,
        "label_meta": None,
        "reference_label": None,
        "native_case39_stage": None,
    }

    if not path.exists():
        rows.append({**row_base, "slot": 1})
        rows.append({**row_base, "slot": 2})
        continue

    obj = load_json(path)
    row_base["stage_meta"] = obj.get("stage")
    row_base["label_meta"] = obj.get("label")
    row_base["reference_label"] = obj.get("reference_label")
    row_base["native_case39_stage"] = obj.get("native_case39_stage")

    for slot in (1, 2):
        m = get_slot_metrics(obj, slot)
        r = {**row_base, "slot": slot}
        if m:
            r.update(m)
        else:
            r.update({
                "mean_recall": None,
                "mean_unnecessary": None,
                "mean_cost": None,
                "mean_served_ratio": None,
            })
        rows.append(r)

csv_path = root / "case39_stage_table_clean.csv"
md_path = root / "case39_stage_table_clean.md"

with open(csv_path, "w", encoding="utf-8") as f:
    headers = [
        "stage_name", "role", "slot",
        "mean_recall", "mean_unnecessary", "mean_cost", "mean_served_ratio",
        "stage_meta", "label_meta", "reference_label", "native_case39_stage",
        "path", "exists"
    ]
    f.write(",".join(headers) + "\n")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            if v is None:
                vals.append("")
            else:
                s = str(v).replace('"', '""')
                if "," in s:
                    s = f'"{s}"'
                vals.append(s)
        f.write(",".join(vals) + "\n")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("| stage | role | slot | recall | unnecessary | cost | served_ratio | stage_meta | label_meta | reference_label | native_case39_stage |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for r in rows:
        f.write(
            f"| {r['stage_name']} | {r['role']} | {r['slot']} | "
            f"{r.get('mean_recall')} | {r.get('mean_unnecessary')} | "
            f"{r.get('mean_cost')} | {r.get('mean_served_ratio')} | "
            f"{r.get('stage_meta')} | {r.get('label_meta')} | "
            f"{r.get('reference_label')} | {r.get('native_case39_stage')} |\n"
        )

print("wrote:", csv_path)
print("wrote:", md_path)
print("\n--- markdown preview ---\n")
print(md_path.read_text(encoding="utf-8"))
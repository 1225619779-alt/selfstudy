from pathlib import Path
import numpy as np
import csv
import re

METRIC_DIR = Path("metric/case14")
OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)

def mean_safe(x):
    if x is None:
        return None
    if len(x) == 0:
        return None
    arr = np.array(x, dtype=float)
    return float(np.mean(arr))

def flatten_grouped_metric(metric_dict, settings):
    values = []
    if not isinstance(metric_dict, dict):
        return values
    for s in settings:
        values.extend(metric_dict.get(s, []))
    return values

def parse_setting_key(s):
    # s 形如 '(1,0.2)'
    m = re.match(r"\((\d+),([0-9.]+)\)", s)
    if not m:
        return (999, 999.0)
    return (int(m.group(1)), float(m.group(2)))

def extract_tag(filename):
    # 例如 metric_event_trigger_mode_0_0.03_1.1.npy -> 0.03_1.1
    m = re.match(r"metric_event_trigger_mode_0_(.+)\.npy", filename)
    return m.group(1) if m else filename

def get_group(metric_obj, setting):
    if isinstance(metric_obj, dict):
        return metric_obj.get(setting, [])
    return None

files = sorted(METRIC_DIR.glob("metric_event_trigger_mode_0_*.npy"))

if not files:
    print("No metric_event_trigger_mode_0_*.npy files found.")
    raise SystemExit(1)

overall_rows = []
detail_rows = []

for f in files:
    m = np.load(f, allow_pickle=True).item()
    tag = extract_tag(f.name)

    if "recover_deviation" not in m or not isinstance(m["recover_deviation"], dict):
        print(f"[skip] {f.name}: recover_deviation missing or not dict")
        continue

    settings = sorted(m["recover_deviation"].keys(), key=parse_setting_key)

    # 每组明细
    for s in settings:
        tp_total = len(m["TP_DDD"][s]) if isinstance(m.get("TP_DDD"), dict) else None
        triggered = len(m["recover_deviation"][s])

        pre_mean = mean_safe(get_group(m.get("pre_deviation"), s))
        recover_mean = mean_safe(get_group(m.get("recover_deviation"), s))

        row = {
            "tag": tag,
            "file": f.name,
            "setting": s,
            "total_samples": tp_total,
            "triggered_samples": triggered,
            "trigger_rate": (triggered / tp_total) if tp_total else None,
            "pre_dev_mean": pre_mean,
            "recover_dev_mean": recover_mean,
            "recover_improvement": (pre_mean - recover_mean) if (pre_mean is not None and recover_mean is not None) else None,
            "fail_rate": mean_safe(get_group(m.get("fail"), s)),
            "stage1_eff_mean": mean_safe(get_group(m.get("mtd_stage_one_eff"), s)),
            "stage2_eff_mean": mean_safe(get_group(m.get("mtd_stage_two_eff"), s)),
            "stage1_hid_mean": mean_safe(get_group(m.get("mtd_stage_one_hidden"), s)),
            "stage2_hid_mean": mean_safe(get_group(m.get("mtd_stage_two_hidden"), s)),
            "obj1_mean": mean_safe(get_group(m.get("obj_one"), s)),
            "obj2_mean": mean_safe(get_group(m.get("obj_two"), s)),
            "cost_no_mtd_mean": mean_safe(get_group(m.get("cost_no_mtd"), s)),
            "cost_mtd1_mean": mean_safe(get_group(m.get("cost_with_mtd_one"), s)),
            "cost_mtd2_mean": mean_safe(get_group(m.get("cost_with_mtd_two"), s)),
            "worst_primal_mean": mean_safe(get_group(m.get("worst_primal"), s)),
            "worst_dual_mean": mean_safe(get_group(m.get("worst_dual"), s)),
            "varrho_summary_mean": mean_safe(get_group(m.get("varrho_summary"), s)),
            "x_ratio_stage1_mean": mean_safe(get_group(m.get("x_ratio_stage_one"), s)),
            "x_ratio_stage2_mean": mean_safe(get_group(m.get("x_ratio_stage_two"), s)),
        }
        detail_rows.append(row)

    # 整体汇总（把所有组拼起来看）
    total_samples_all = 0
    triggered_all = 0
    for s in settings:
        total_samples_all += len(m["TP_DDD"][s]) if isinstance(m.get("TP_DDD"), dict) else 0
        triggered_all += len(m["recover_deviation"][s])

    pre_all = flatten_grouped_metric(m.get("pre_deviation"), settings)
    recover_all = flatten_grouped_metric(m.get("recover_deviation"), settings)
    fail_all = flatten_grouped_metric(m.get("fail"), settings)
    eff1_all = flatten_grouped_metric(m.get("mtd_stage_one_eff"), settings)
    eff2_all = flatten_grouped_metric(m.get("mtd_stage_two_eff"), settings)
    hid1_all = flatten_grouped_metric(m.get("mtd_stage_one_hidden"), settings)
    hid2_all = flatten_grouped_metric(m.get("mtd_stage_two_hidden"), settings)
    cost_no_all = flatten_grouped_metric(m.get("cost_no_mtd"), settings)
    cost1_all = flatten_grouped_metric(m.get("cost_with_mtd_one"), settings)
    cost2_all = flatten_grouped_metric(m.get("cost_with_mtd_two"), settings)
    obj1_all = flatten_grouped_metric(m.get("obj_one"), settings)
    obj2_all = flatten_grouped_metric(m.get("obj_two"), settings)

    overall_rows.append({
        "tag": tag,
        "file": f.name,
        "group_count": len(settings),
        "total_samples_all": total_samples_all,
        "triggered_samples_all": triggered_all,
        "trigger_rate_all": (triggered_all / total_samples_all) if total_samples_all else None,
        "pre_dev_mean_all": mean_safe(pre_all),
        "recover_dev_mean_all": mean_safe(recover_all),
        "recover_improvement_all": (mean_safe(pre_all) - mean_safe(recover_all)) if (mean_safe(pre_all) is not None and mean_safe(recover_all) is not None) else None,
        "fail_rate_all": mean_safe(fail_all),
        "opf_converge_rate_all": mean_safe(m.get("post_mtd_opf_converge")),
        "stage1_eff_mean_all": mean_safe(eff1_all),
        "stage2_eff_mean_all": mean_safe(eff2_all),
        "stage1_hid_mean_all": mean_safe(hid1_all),
        "stage2_hid_mean_all": mean_safe(hid2_all),
        "obj1_mean_all": mean_safe(obj1_all),
        "obj2_mean_all": mean_safe(obj2_all),
        "cost_no_mtd_mean_all": mean_safe(cost_no_all),
        "cost_mtd1_mean_all": mean_safe(cost1_all),
        "cost_mtd2_mean_all": mean_safe(cost2_all),
        "stage1_time_mean_all": mean_safe(m.get("mtd_stage_one_time")),
        "stage2_time_mean_all": mean_safe(m.get("mtd_stage_two_time")),
    })

# 写 overall csv
overall_csv = OUT_DIR / "event_trigger_overall_summary.csv"
with overall_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
    writer.writeheader()
    writer.writerows(overall_rows)

# 写 detail csv
detail_csv = OUT_DIR / "event_trigger_detail_summary.csv"
with detail_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
    writer.writeheader()
    writer.writerows(detail_rows)

# 终端打印一个精简版 overview
print("\n=== Overall summary ===")
for row in overall_rows:
    print(
        f"{row['tag']:>10s} | "
        f"trigger={row['trigger_rate_all']:.4f} | "
        f"improve={row['recover_improvement_all']:.6f} | "
        f"fail={row['fail_rate_all']:.4f} | "
        f"opf={row['opf_converge_rate_all']:.4f} | "
        f"t1={row['stage1_time_mean_all']:.4f}s | "
        f"t2={row['stage2_time_mean_all']:.4f}s"
    )

print(f"\nSaved overall csv: {overall_csv}")
print(f"Saved detail csv : {detail_csv}")
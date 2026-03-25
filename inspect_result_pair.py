import os
import numpy as np
from datetime import datetime

# ===== 改成你现在实际在 compare 里用的两个文件 =====
baseline_path = "metric/case14/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy"
gated_path = "metric/case14/metric_event_trigger_tau_0.021_mode_0_0.03_1.1.npy"
# ==============================================


def fmt_mtime(path):
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def load_result(path):
    data = np.load(path, allow_pickle=True).item()
    return data


def print_file_info(name, path, data):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print("abs path      :", os.path.abspath(path))
    print("modified time :", fmt_mtime(path))
    print("keys          :", list(data.keys()))

    for key in sorted(data["TP_DDD"].keys()):
        tp = data["TP_DDD"][key]
        trig = data.get("trigger_after_verification", {}).get(key, [])
        skip = data.get("skip_by_verification", {}).get(key, [])

        print(f"\n--- {key} ---")
        print("TP_DDD len:", len(tp))
        print("TP_DDD    :", tp)
        print("n_DDD_alarm:", sum(tp))

        print("trigger_after_verification len:", len(trig))
        print("trigger_after_verification    :", trig)
        print("skip_by_verification len      :", len(skip))
        print("skip_by_verification          :", skip)


def compare_tp_lists(baseline, gated):
    print("\n" + "=" * 100)
    print("PAIRWISE CHECK: TP_DDD equality")
    print("=" * 100)

    for key in sorted(baseline["TP_DDD"].keys()):
        tp_b = baseline["TP_DDD"][key]
        tp_g = gated["TP_DDD"][key]

        same = (tp_b == tp_g)
        print(f"\n--- {key} ---")
        print("same TP_DDD list?:", same)

        if not same:
            min_len = min(len(tp_b), len(tp_g))
            diff_pos = [i for i in range(min_len) if tp_b[i] != tp_g[i]]
            print("different positions:", diff_pos)

            if len(tp_b) != len(tp_g):
                print("different lengths:", len(tp_b), len(tp_g))


def main():
    baseline = load_result(baseline_path)
    gated = load_result(gated_path)

    print_file_info("BASELINE FILE", baseline_path, baseline)
    print_file_info("GATED FILE", gated_path, gated)

    compare_tp_lists(baseline, gated)


if __name__ == "__main__":
    main()
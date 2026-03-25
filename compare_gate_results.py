import numpy as np

baseline_path = "metric/case14/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy"
gated_path = "metric/case14/metric_event_trigger_tau_0.021_mode_0_0.03_1.1.npy"


def safe_mean(x):
    if len(x) == 0:
        return None
    return float(np.mean(x))


def safe_sum(x):
    if len(x) == 0:
        return 0.0
    return float(np.sum(x))


def summarize_one_result(name, data):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    keys = sorted(data["TP_DDD"].keys())

    total_alarm_all = 0
    total_trigger_all = 0

    for key in keys:
        tp_list = data["TP_DDD"][key]
        n_total = len(tp_list)
        n_alarm = int(sum(tp_list))

        verify_list = data.get("verify_score", {}).get(key, [])
        trig_list = data.get("trigger_after_verification", {}).get(key, [])
        skip_list = data.get("skip_by_verification", {}).get(key, [])

        n_verify = len(verify_list)
        n_trigger = int(sum(trig_list)) if len(trig_list) > 0 else 0
        n_skip = int(sum(skip_list)) if len(skip_list) > 0 else 0

        total_alarm_all += n_alarm
        total_trigger_all += n_trigger

        trigger_rate_after_alarm = (n_trigger / n_alarm) if n_alarm > 0 else None
        skip_rate_after_alarm = (n_skip / n_alarm) if n_alarm > 0 else None
        trigger_reduction_after_alarm = 1.0 - trigger_rate_after_alarm if trigger_rate_after_alarm is not None else None

        fail_list = data["fail"][key]
        fail_rate_triggered = safe_mean(fail_list)
        fail_per_alarm = (safe_sum(fail_list) / n_alarm) if n_alarm > 0 else None

        obj_one_mean = safe_mean(data["obj_one"][key])
        obj_two_mean = safe_mean(data["obj_two"][key])

        x_ratio_stage_one_mean = safe_mean(data["x_ratio_stage_one"][key])
        x_ratio_stage_two_mean = safe_mean(data["x_ratio_stage_two"][key])

        eff_one_mean = safe_mean(data["mtd_stage_one_eff"][key])
        eff_two_mean = safe_mean(data["mtd_stage_two_eff"][key])

        hidden_one_mean = safe_mean(data["mtd_stage_one_hidden"][key])
        hidden_two_mean = safe_mean(data["mtd_stage_two_hidden"][key])

        cost_no_mtd = np.array(data["cost_no_mtd"][key], dtype=float)
        cost_with_mtd_one = np.array(data["cost_with_mtd_one"][key], dtype=float)
        cost_with_mtd_two = np.array(data["cost_with_mtd_two"][key], dtype=float)

        delta_cost_one = cost_with_mtd_one - cost_no_mtd if len(cost_no_mtd) > 0 else np.array([])
        delta_cost_two = cost_with_mtd_two - cost_no_mtd if len(cost_no_mtd) > 0 else np.array([])

        delta_cost_one_mean_triggered = safe_mean(delta_cost_one.tolist())
        delta_cost_two_mean_triggered = safe_mean(delta_cost_two.tolist())

        delta_cost_one_per_alarm = (safe_sum(delta_cost_one.tolist()) / n_alarm) if n_alarm > 0 else None
        delta_cost_two_per_alarm = (safe_sum(delta_cost_two.tolist()) / n_alarm) if n_alarm > 0 else None

        verify_mean = safe_mean(verify_list)

        print(f"\n--- {key} ---")
        print(f"n_total_samples                 = {n_total}")
        print(f"n_DDD_alarm                     = {n_alarm}")
        print(f"n_verify_scored                 = {n_verify}")
        print(f"n_trigger_after_verification    = {n_trigger}")
        print(f"n_skip_by_verification          = {n_skip}")
        print(f"trigger_rate_after_alarm        = {trigger_rate_after_alarm}")
        print(f"skip_rate_after_alarm           = {skip_rate_after_alarm}")
        print(f"trigger_reduction_after_alarm   = {trigger_reduction_after_alarm}")
        print(f"verify_score_mean               = {verify_mean}")

        print(f"fail_rate_triggered             = {fail_rate_triggered}")
        print(f"fail_per_alarm                  = {fail_per_alarm}")

        print(f"obj_one_mean                    = {obj_one_mean}")
        print(f"obj_two_mean                    = {obj_two_mean}")

        print(f"x_ratio_stage_one_mean          = {x_ratio_stage_one_mean}")
        print(f"x_ratio_stage_two_mean          = {x_ratio_stage_two_mean}")

        print(f"mtd_stage_one_eff_mean          = {eff_one_mean}")
        print(f"mtd_stage_two_eff_mean          = {eff_two_mean}")

        print(f"mtd_stage_one_hidden_mean       = {hidden_one_mean}")
        print(f"mtd_stage_two_hidden_mean       = {hidden_two_mean}")

        print(f"delta_cost_one_mean_triggered   = {delta_cost_one_mean_triggered}")
        print(f"delta_cost_two_mean_triggered   = {delta_cost_two_mean_triggered}")
        print(f"delta_cost_one_per_alarm        = {delta_cost_one_per_alarm}")
        print(f"delta_cost_two_per_alarm        = {delta_cost_two_per_alarm}")

    stage_one_time_total = safe_sum(data["mtd_stage_one_time"])
    stage_two_time_total = safe_sum(data["mtd_stage_two_time"])

    stage_one_time_mean_triggered = safe_mean(data["mtd_stage_one_time"])
    stage_two_time_mean_triggered = safe_mean(data["mtd_stage_two_time"])

    stage_one_time_per_alarm = (stage_one_time_total / total_alarm_all) if total_alarm_all > 0 else None
    stage_two_time_per_alarm = (stage_two_time_total / total_alarm_all) if total_alarm_all > 0 else None

    print("\n[global metrics]")
    print(f"total_DDD_alarm               = {total_alarm_all}")
    print(f"total_trigger_after_gate      = {total_trigger_all}")
    print(f"overall_trigger_rate          = {total_trigger_all / total_alarm_all if total_alarm_all > 0 else None}")
    print(f"post_mtd_opf_converge_mean    = {safe_mean(data['post_mtd_opf_converge'])}")
    print(f"residual_no_att_mean          = {safe_mean(data['residual_no_att'])}")
    print(f"stage_one_time_total          = {stage_one_time_total}")
    print(f"stage_two_time_total          = {stage_two_time_total}")
    print(f"stage_one_time_mean_triggered = {stage_one_time_mean_triggered}")
    print(f"stage_two_time_mean_triggered = {stage_two_time_mean_triggered}")
    print(f"stage_one_time_per_alarm      = {stage_one_time_per_alarm}")
    print(f"stage_two_time_per_alarm      = {stage_two_time_per_alarm}")


def main():
    baseline = np.load(baseline_path, allow_pickle=True).item()
    gated = np.load(gated_path, allow_pickle=True).item()

    summarize_one_result("BASELINE / TAU=-1.0", baseline)
    summarize_one_result("GATED / TAU=0.021", gated)


if __name__ == "__main__":
    main()
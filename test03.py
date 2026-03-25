import numpy as np

path = "metric/case14/metric_event_trigger_mode_0_0.03_1.1.npy"
data = np.load(path, allow_pickle=True).item()

print(data.keys())
key = "(1,0.2)"

print("TP_DDD len:", len(data["TP_DDD"][key]))
print("verify_score len:", len(data["verify_score"][key]))
print("trigger_after_verification len:", len(data["trigger_after_verification"][key]))
print("skip_by_verification len:", len(data["skip_by_verification"][key]))

print("TP_DDD:", data["TP_DDD"][key])
print("trigger_after_verification:", data["trigger_after_verification"][key])
print("skip_by_verification:", data["skip_by_verification"][key])
print("verify_score:", data["verify_score"][key])
print("obj_one len:", len(data["obj_one"][key]))
print("obj_two len:", len(data["obj_two"][key]))
print("fail len:", len(data["fail"][key]))
print("cost_no_mtd len:", len(data["cost_no_mtd"][key]))
print("cost_with_mtd_one len:", len(data["cost_with_mtd_one"][key]))
print("cost_with_mtd_two len:", len(data["cost_with_mtd_two"][key]))
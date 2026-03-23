import numpy as np

x = np.load(r'.\metric\case14\metric_ddd_0.005_0.1_0.1_1000_50.npy', allow_pickle=True).item()

print("FPR =", np.mean(x["FP_DDD"]))

print("\nTPR by attack setting:")
for k, v in x["TP_DDD"].items():
    arr = np.array(v, dtype=float)
    print(k, "->", arr.mean())

print("\nRecovery improvement:")
for k in x["recover_deviation"].keys():
    rec = np.array(x["recover_deviation"][k], dtype=float)
    pre = np.array(x["pre_deviation"][k], dtype=float)
    print(k, "recover_mean =", rec.mean(), "pre_mean =", pre.mean())
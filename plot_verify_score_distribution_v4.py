import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def _flatten_scores(x):
    if isinstance(x, dict):
        vals = []
        for _, v in x.items():
            arr = np.asarray(v, dtype=float).reshape(-1)
            vals.append(arr)
        if len(vals) == 0:
            return np.array([], dtype=float)
        arr = np.concatenate(vals, axis=0)
    else:
        arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]



def _savefig(path_base: Path):
    plt.tight_layout()
    plt.savefig(str(path_base) + ".png", dpi=400, bbox_inches="tight")
    plt.savefig(str(path_base) + ".pdf", bbox_inches="tight")
    plt.close()



def _parse_tau_summary(path: str):
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    for name in ["tau_main", "tau_strict", "rounded_main_3dp", "rounded_strict_3dp"]:
        m = re.search(rf"{name}[^=]*=\s*([0-9.]+)", text)
        if m:
            out[name] = float(m.group(1))
    return out



def _load_clean_scores(path: str):
    obj = np.load(path, allow_pickle=True).item()
    if isinstance(obj, dict):
        for key in ["score_phys_l2", "verify_score_clean_false_alarm", "verify_score", "scores"]:
            if key in obj:
                return _flatten_scores(obj[key])
    raise KeyError(f"Could not find clean-score key in {path}")



def _load_attack_scores(path: str):
    obj = np.load(path, allow_pickle=True).item()
    if isinstance(obj, dict):
        for key in ["score_phys_l2", "verify_score_attack_true_alarm", "verify_score", "scores"]:
            if key in obj:
                raw = obj[key]
                if not isinstance(raw, dict):
                    raise TypeError(f"Attack score key {key} is not a dict in {path}")
                return {k: _flatten_scores(v) for k, v in raw.items()}
    raise KeyError(f"Could not find attack-score dict key in {path}")



def main():
    parser = argparse.ArgumentParser(
        description="Plot verify-score distributions using explicit taus or a validation-calibration summary."
    )
    parser.add_argument("--clean_scores", default="metric/case14/metric_clean_alarm_scores_full.npy")
    parser.add_argument("--attack_scores", default="metric/case14/metric_attack_alarm_scores_200.npy")
    parser.add_argument("--outdir", default="metric/case14/plots_verify_score_v4")
    parser.add_argument("--tau_summary", default="", help="Optional summary txt from validation-only calibration.")
    parser.add_argument("--tau_main", type=float, default=None)
    parser.add_argument("--tau_strict", type=float, default=None)
    parser.add_argument("--use_rounded_from_summary", action="store_true", help="Use rounded_main_3dp / rounded_strict_3dp if present.")
    parser.add_argument("--title_suffix", default="")
    args = parser.parse_args()

    tau_main = args.tau_main
    tau_strict = args.tau_strict
    if args.tau_summary:
        parsed = _parse_tau_summary(args.tau_summary)
        if tau_main is None:
            tau_main = parsed.get("rounded_main_3dp" if args.use_rounded_from_summary else "tau_main")
        if tau_strict is None:
            tau_strict = parsed.get("rounded_strict_3dp" if args.use_rounded_from_summary else "tau_strict")
    if tau_main is None or tau_strict is None:
        raise ValueError("You must provide taus explicitly or via --tau_summary.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clean_scores = _load_clean_scores(args.clean_scores)
    attack_scores_by_group = _load_attack_scores(args.attack_scores)
    attack_all = np.concatenate([v for v in attack_scores_by_group.values() if len(v) > 0], axis=0)

    plt.figure(figsize=(6.8, 4.3))
    bins = np.linspace(0.0, max(clean_scores.max(initial=0), attack_all.max(initial=0)) * 1.05, 40)
    plt.hist(clean_scores, bins=bins, alpha=0.65, density=True, label="Clean false alarms")
    plt.hist(attack_all, bins=bins, alpha=0.65, density=True, label="Attack true alarms")
    plt.axvline(tau_main, linestyle="--", linewidth=1.4, label=f"Main OP τ={tau_main:.3f}")
    plt.axvline(tau_strict, linestyle="-.", linewidth=1.4, label=f"Strict OP τ={tau_strict:.3f}")
    plt.xlabel(r"Verification score $\|c_{recover,no-ref}\|_2$")
    plt.ylabel("Density")
    title = "Verify-score distributions under clean and attack alarms"
    if args.title_suffix:
        title += " " + args.title_suffix
    plt.title(title)
    plt.legend(frameon=False)
    _savefig(outdir / "verify_score_hist_clean_vs_attack")

    labels = ["clean"] + list(attack_scores_by_group.keys())
    values = [clean_scores] + [attack_scores_by_group[k] for k in attack_scores_by_group.keys()]
    plt.figure(figsize=(8.8, 4.5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.axhline(tau_main, linestyle="--", linewidth=1.2, label=f"Main OP τ={tau_main:.3f}")
    plt.axhline(tau_strict, linestyle="-.", linewidth=1.2, label=f"Strict OP τ={tau_strict:.3f}")
    plt.ylabel(r"Verification score $\|c_{recover,no-ref}\|_2$")
    plt.title("Per-group verify-score distribution")
    plt.legend(frameon=False)
    _savefig(outdir / "verify_score_boxplot_by_group")

    def _ecdf(arr):
        arr = np.sort(arr)
        y = np.arange(1, len(arr) + 1) / len(arr)
        return arr, y

    plt.figure(figsize=(6.8, 4.3))
    x1, y1 = _ecdf(clean_scores)
    x2, y2 = _ecdf(attack_all)
    plt.plot(x1, y1, linewidth=1.8, label="Clean false alarms")
    plt.plot(x2, y2, linewidth=1.8, label="Attack true alarms")
    plt.axvline(tau_main, linestyle="--", linewidth=1.2, label=f"Main OP τ={tau_main:.3f}")
    plt.axvline(tau_strict, linestyle="-.", linewidth=1.2, label=f"Strict OP τ={tau_strict:.3f}")
    plt.xlabel(r"Verification score $\|c_{recover,no-ref}\|_2$")
    plt.ylabel("Empirical CDF")
    plt.title("Empirical CDF of verify score")
    plt.legend(frameon=False)
    _savefig(outdir / "verify_score_cdf")

    summary = []
    summary.append(f"clean_count = {len(clean_scores)}")
    summary.append(f"attack_count = {len(attack_all)}")
    summary.append(f"tau_main_used = {tau_main}")
    summary.append(f"tau_strict_used = {tau_strict}")
    summary.append(f"clean_mean = {float(np.mean(clean_scores)):.6f}")
    summary.append(f"clean_median = {float(np.median(clean_scores)):.6f}")
    summary.append(f"clean_p90 = {float(np.percentile(clean_scores, 90)):.6f}")
    summary.append(f"clean_p95 = {float(np.percentile(clean_scores, 95)):.6f}")
    summary.append(f"attack_mean = {float(np.mean(attack_all)):.6f}")
    summary.append(f"attack_median = {float(np.median(attack_all)):.6f}")
    for k, arr in attack_scores_by_group.items():
        if len(arr) == 0:
            continue
        summary.append(f"{k}: count={len(arr)} mean={float(np.mean(arr)):.6f} median={float(np.median(arr)):.6f}")
    with open(outdir / "verify_score_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print(f"Saved figures to: {outdir}")
    print(f"Saved summary to: {outdir / 'verify_score_summary.txt'}")


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

ATTACK_KEYS = ['(1,0.2)', '(1,0.3)', '(2,0.2)', '(2,0.3)', '(3,0.2)', '(3,0.3)']


def _to_1d_finite(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _ecdf(arr):
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', default='metric/case14/metric_clean_alarm_scores_full.npy')
    parser.add_argument('--attack', default='metric/case14/metric_attack_alarm_scores_200.npy')
    parser.add_argument('--outdir', default='metric/case14/plots_verify_score')
    parser.add_argument('--tau_main', type=float, default=0.021050716339785475)
    parser.add_argument('--tau_strict', type=float, default=0.03010798927389737)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean = np.load(args.clean, allow_pickle=True).item()
    attack = np.load(args.attack, allow_pickle=True).item()

    clean_scores = _to_1d_finite(clean['score_phys_l2'])
    attack_scores_dict = attack['score_phys_l2']
    attack_scores_all = []
    for k in ATTACK_KEYS:
        attack_scores_all.extend(_to_1d_finite(attack_scores_dict[k]).tolist())
    attack_scores_all = np.asarray(attack_scores_all, dtype=float)

    # Figure 1: histogram overlay
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, max(clean_scores.max(), attack_scores_all.max()) * 1.05, 40)
    plt.hist(clean_scores, bins=bins, alpha=0.6, density=True, label='Clean false alarms')
    plt.hist(attack_scores_all, bins=bins, alpha=0.6, density=True, label='Attack true alarms (all groups)')
    plt.axvline(args.tau_main, linestyle='--', linewidth=1.5, label=f'tau_main={args.tau_main:.5f}')
    plt.axvline(args.tau_strict, linestyle='-.', linewidth=1.5, label=f'tau_strict={args.tau_strict:.5f}')
    plt.xlabel('verify_score = ||c_recover,no-ref||_2')
    plt.ylabel('Density')
    plt.title('Verify-score distribution: clean vs. attack')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'verify_score_hist_clean_vs_attack.png'), dpi=300)
    plt.close()

    # Figure 2: ECDF
    plt.figure(figsize=(8, 5))
    x_clean, y_clean = _ecdf(clean_scores)
    x_attack, y_attack = _ecdf(attack_scores_all)
    plt.plot(x_clean, y_clean, label='Clean false alarms')
    plt.plot(x_attack, y_attack, label='Attack true alarms (all groups)')
    plt.axvline(args.tau_main, linestyle='--', linewidth=1.5, label=f'tau_main={args.tau_main:.5f}')
    plt.axvline(args.tau_strict, linestyle='-.', linewidth=1.5, label=f'tau_strict={args.tau_strict:.5f}')
    plt.xlabel('verify_score = ||c_recover,no-ref||_2')
    plt.ylabel('ECDF')
    plt.title('Verify-score ECDF: clean vs. attack')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'verify_score_ecdf_clean_vs_attack.png'), dpi=300)
    plt.close()

    # Figure 3: per-group boxplot
    labels = ['clean'] + ATTACK_KEYS
    values = [clean_scores] + [_to_1d_finite(attack_scores_dict[k]) for k in ATTACK_KEYS]
    plt.figure(figsize=(10, 5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.axhline(args.tau_main, linestyle='--', linewidth=1.5, label=f'tau_main={args.tau_main:.5f}')
    plt.axhline(args.tau_strict, linestyle='-.', linewidth=1.5, label=f'tau_strict={args.tau_strict:.5f}')
    plt.ylabel('verify_score = ||c_recover,no-ref||_2')
    plt.title('Verify-score by clean/attack group')
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'verify_score_boxplot_by_group.png'), dpi=300)
    plt.close()

    print('Saved figures to:', args.outdir)
    print('clean_count =', len(clean_scores), 'attack_count =', len(attack_scores_all))
    print('clean_mean =', float(np.mean(clean_scores)), 'clean_p90 =', float(np.percentile(clean_scores, 90)), 'clean_p95 =', float(np.percentile(clean_scores, 95)))


if __name__ == '__main__':
    main()

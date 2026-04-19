import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from paper_worldline import ATTACK_SCORE_METRIC, CLEAN_SCORE_METRIC, TAU_MAIN, TAU_STRICT


def _flatten_scores(x):
    """Return a 1D finite float array from a list/ndarray or dict of lists."""
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
    plt.savefig(str(path_base) + '.png', dpi=400, bbox_inches='tight')
    plt.savefig(str(path_base) + '.pdf', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot verify-score distributions using fixed operating-point taus.')
    parser.add_argument('--clean_scores', default=CLEAN_SCORE_METRIC)
    parser.add_argument('--attack_scores', default=ATTACK_SCORE_METRIC)
    parser.add_argument('--outdir', default='metric/case14/plots_verify_score_v3')
    parser.add_argument('--tau_main', type=float, default=TAU_MAIN, help='Main operating-point threshold.')
    parser.add_argument('--tau_strict', type=float, default=TAU_STRICT, help='Strict operating-point threshold.')
    parser.add_argument('--title_suffix', default='')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clean = np.load(args.clean_scores, allow_pickle=True).item()
    attack = np.load(args.attack_scores, allow_pickle=True).item()

    clean_scores = _flatten_scores(clean['score_phys_l2'])

    attack_scores_by_group = {}
    for k, v in attack['score_phys_l2'].items():
        attack_scores_by_group[k] = _flatten_scores(v)

    attack_all = np.concatenate([v for v in attack_scores_by_group.values() if len(v) > 0], axis=0)

    # Figure 1: overall histogram clean vs attack
    plt.figure(figsize=(6.8, 4.3))
    bins = np.linspace(0.0, max(clean_scores.max(initial=0), attack_all.max(initial=0)) * 1.05, 40)
    plt.hist(clean_scores, bins=bins, alpha=0.65, density=True, label='Clean false alarms')
    plt.hist(attack_all, bins=bins, alpha=0.65, density=True, label='Attack true alarms')
    plt.axvline(args.tau_main, linestyle='--', linewidth=1.4, label=f'Main OP τ={args.tau_main:.3f}')
    plt.axvline(args.tau_strict, linestyle='-.', linewidth=1.4, label=f'Strict OP τ={args.tau_strict:.3f}')
    plt.xlabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.ylabel('Density')
    title = 'Verify-score distributions under clean and attack alarms'
    if args.title_suffix:
        title += ' ' + args.title_suffix
    plt.title(title)
    plt.legend(frameon=False)
    _savefig(outdir / 'verify_score_hist_clean_vs_attack')

    # Figure 2: per-group boxplot
    labels = ['clean'] + list(attack_scores_by_group.keys())
    values = [clean_scores] + [attack_scores_by_group[k] for k in attack_scores_by_group.keys()]
    plt.figure(figsize=(8.8, 4.5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.axhline(args.tau_main, linestyle='--', linewidth=1.2, label=f'Main OP τ={args.tau_main:.3f}')
    plt.axhline(args.tau_strict, linestyle='-.', linewidth=1.2, label=f'Strict OP τ={args.tau_strict:.3f}')
    plt.ylabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.title('Per-group verify-score distribution')
    plt.legend(frameon=False)
    _savefig(outdir / 'verify_score_boxplot_by_group')

    # Figure 3: CDF
    def _ecdf(arr):
        arr = np.sort(arr)
        y = np.arange(1, len(arr) + 1) / len(arr)
        return arr, y

    plt.figure(figsize=(6.8, 4.3))
    x1, y1 = _ecdf(clean_scores)
    x2, y2 = _ecdf(attack_all)
    plt.plot(x1, y1, linewidth=1.8, label='Clean false alarms')
    plt.plot(x2, y2, linewidth=1.8, label='Attack true alarms')
    plt.axvline(args.tau_main, linestyle='--', linewidth=1.2, label=f'Main OP τ={args.tau_main:.3f}')
    plt.axvline(args.tau_strict, linestyle='-.', linewidth=1.2, label=f'Strict OP τ={args.tau_strict:.3f}')
    plt.xlabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of verify score')
    plt.legend(frameon=False)
    _savefig(outdir / 'verify_score_cdf')

    summary = []
    summary.append(f'clean_count = {len(clean_scores)}')
    summary.append(f'attack_count = {len(attack_all)}')
    summary.append(f'tau_main_used = {args.tau_main}')
    summary.append(f'tau_strict_used = {args.tau_strict}')
    summary.append(f'clean_mean = {float(np.mean(clean_scores)):.6f}')
    summary.append(f'clean_median = {float(np.median(clean_scores)):.6f}')
    summary.append(f'clean_p90 = {float(np.percentile(clean_scores, 90)):.6f}')
    summary.append(f'clean_p95 = {float(np.percentile(clean_scores, 95)):.6f}')
    summary.append(f'attack_mean = {float(np.mean(attack_all)):.6f}')
    summary.append(f'attack_median = {float(np.median(attack_all)):.6f}')
    for k, arr in attack_scores_by_group.items():
        if len(arr) == 0:
            continue
        summary.append(f'{k}: count={len(arr)} mean={float(np.mean(arr)):.6f} median={float(np.median(arr)):.6f}')

    with open(outdir / 'verify_score_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary) + '\n')

    print(f'Saved figures to: {outdir}')
    print(f'Saved summary to: {outdir / "verify_score_summary.txt"}')
    print(f'clean_count = {len(clean_scores)} attack_count = {len(attack_all)}')
    print(f'tau_main_used = {args.tau_main} tau_strict_used = {args.tau_strict}')


if __name__ == '__main__':
    main()


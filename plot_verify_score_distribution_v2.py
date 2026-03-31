import argparse
import os
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ATTACK_ORDER = ['(1,0.2)', '(1,0.3)', '(2,0.2)', '(2,0.3)', '(3,0.2)', '(3,0.3)']


def _as_1d_finite(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _first_key(d: Dict[str, Any], preferred: str | None = None) -> str:
    keys = list(d.keys())
    if not keys:
        raise KeyError('Empty dict encountered.')
    if preferred is not None and preferred in d:
        return preferred
    return keys[0]


def _extract_clean_scores(clean: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    raw = clean.get('score_phys_l2')
    if isinstance(raw, dict):
        preferred = clean.get('group_key', None)
        key = _first_key(raw, preferred)
        return _as_1d_finite(raw[key]), key
    return _as_1d_finite(raw), 'flat'


def _extract_attack_scores(attack: Dict[str, Any], ordered_keys: Iterable[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    raw = attack.get('score_phys_l2')
    if not isinstance(raw, dict):
        arr = _as_1d_finite(raw)
        return {'all': arr}, arr

    out: Dict[str, np.ndarray] = {}
    pooled: List[np.ndarray] = []
    used = set()
    for key in ordered_keys:
        if key in raw:
            arr = _as_1d_finite(raw[key])
            out[key] = arr
            pooled.append(arr)
            used.add(key)
    for key, value in raw.items():
        if key in used:
            continue
        arr = _as_1d_finite(value)
        out[key] = arr
        pooled.append(arr)
    if pooled:
        all_arr = np.concatenate(pooled)
    else:
        all_arr = np.asarray([], dtype=float)
    return out, all_arr


def _ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.asarray([], dtype=float)
    return x, y


def _savefig(fig: plt.Figure, outdir: str, stem: str, dpi: int) -> None:
    png_path = os.path.join(outdir, f'{stem}.png')
    pdf_path = os.path.join(outdir, f'{stem}.pdf')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot publish-ready verify-score distributions.')
    parser.add_argument('--clean', default='metric/case14/metric_clean_alarm_scores_full.npy')
    parser.add_argument('--attack', default='metric/case14/metric_attack_alarm_scores_200.npy')
    parser.add_argument('--outdir', default='metric/case14/plots_verify_score')
    parser.add_argument('--tau_main', type=float, default=None)
    parser.add_argument('--tau_strict', type=float, default=None)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean = np.load(args.clean, allow_pickle=True).item()
    attack = np.load(args.attack, allow_pickle=True).item()

    clean_scores, clean_key = _extract_clean_scores(clean)
    attack_by_group, attack_scores_all = _extract_attack_scores(attack, DEFAULT_ATTACK_ORDER)

    if clean_scores.size == 0:
        raise RuntimeError('No finite clean verify scores were found.')
    if attack_scores_all.size == 0:
        raise RuntimeError('No finite attack verify scores were found.')

    tau_main = float(args.tau_main) if args.tau_main is not None else float(np.percentile(clean_scores, 90))
    tau_strict = float(args.tau_strict) if args.tau_strict is not None else float(np.percentile(clean_scores, 95))

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': args.dpi,
        'savefig.dpi': args.dpi,
    })

    max_x = max(float(clean_scores.max()), float(attack_scores_all.max())) * 1.05
    bins = np.linspace(0.0, max_x, 32)

    # 1) Histogram overlay
    fig = plt.figure(figsize=(3.5, 2.6))
    plt.hist(clean_scores, bins=bins, alpha=0.55, density=True, label='Clean false alarms')
    plt.hist(attack_scores_all, bins=bins, alpha=0.55, density=True, label='Attack true alarms')
    plt.axvline(tau_main, linestyle='--', linewidth=1.2, label=fr'$\tau_{{main}}={tau_main:.5f}$')
    plt.axvline(tau_strict, linestyle='-.', linewidth=1.2, label=fr'$\tau_{{strict}}={tau_strict:.5f}$')
    plt.xlabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.ylabel('Density')
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend(frameon=False)
    _savefig(fig, args.outdir, 'verify_score_hist_clean_vs_attack', args.dpi)

    # 2) ECDF
    fig = plt.figure(figsize=(3.5, 2.6))
    x_clean, y_clean = _ecdf(clean_scores)
    x_attack, y_attack = _ecdf(attack_scores_all)
    plt.plot(x_clean, y_clean, label='Clean false alarms')
    plt.plot(x_attack, y_attack, label='Attack true alarms')
    plt.axvline(tau_main, linestyle='--', linewidth=1.2, label=fr'$\tau_{{main}}$')
    plt.axvline(tau_strict, linestyle='-.', linewidth=1.2, label=fr'$\tau_{{strict}}$')
    plt.xlabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.ylabel('ECDF')
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend(frameon=False, loc='lower right')
    _savefig(fig, args.outdir, 'verify_score_ecdf_clean_vs_attack', args.dpi)

    # 3) Boxplot by group
    labels = ['clean'] + list(attack_by_group.keys())
    values = [clean_scores] + [attack_by_group[k] for k in attack_by_group.keys()]
    fig = plt.figure(figsize=(7.16, 2.8))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.axhline(tau_main, linestyle='--', linewidth=1.2, label=fr'$\tau_{{main}}={tau_main:.5f}$')
    plt.axhline(tau_strict, linestyle='-.', linewidth=1.2, label=fr'$\tau_{{strict}}={tau_strict:.5f}$')
    plt.ylabel(r'Verification score $\|c_{recover,no-ref}\|_2$')
    plt.xticks(rotation=20)
    plt.grid(alpha=0.25, linewidth=0.5, axis='y')
    plt.legend(frameon=False)
    _savefig(fig, args.outdir, 'verify_score_boxplot_by_group', args.dpi)

    summary_path = os.path.join(args.outdir, 'verify_score_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'clean_group_key: {clean_key}\n')
        f.write(f'clean_count: {len(clean_scores)}\n')
        f.write(f'attack_count: {len(attack_scores_all)}\n')
        f.write(f'clean_mean: {float(np.mean(clean_scores))}\n')
        f.write(f'clean_median: {float(np.median(clean_scores))}\n')
        f.write(f'clean_p90: {float(np.percentile(clean_scores, 90))}\n')
        f.write(f'clean_p95: {float(np.percentile(clean_scores, 95))}\n')
        f.write(f'tau_main_used: {tau_main}\n')
        f.write(f'tau_strict_used: {tau_strict}\n')
        for key, arr in attack_by_group.items():
            if arr.size == 0:
                continue
            f.write(f'{key}_count: {len(arr)}\n')
            f.write(f'{key}_median: {float(np.median(arr))}\n')
            f.write(f'{key}_p10: {float(np.percentile(arr, 10))}\n')
            f.write(f'{key}_p90: {float(np.percentile(arr, 90))}\n')

    print('Saved figures to:', args.outdir)
    print('Saved summary to:', summary_path)
    print('clean_count =', len(clean_scores), 'attack_count =', len(attack_scores_all))
    print('tau_main_used =', tau_main, 'tau_strict_used =', tau_strict)


if __name__ == '__main__':
    main()

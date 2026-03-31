import argparse
import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f'Cannot find any of columns: {list(candidates)}')


def _savefig(fig: plt.Figure, outdir: str, stem: str, dpi: int) -> None:
    png_path = os.path.join(outdir, f'{stem}.png')
    pdf_path = os.path.join(outdir, f'{stem}.pdf')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot publish-ready matched-budget ablation figures.')
    parser.add_argument('--csv', default='metric/case14/metric_gate_ablation_summary.csv')
    parser.add_argument('--outdir', default='metric/case14/plots_gate_ablation')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    budget_col = _pick_column(df, ['budget_label'])
    score_col = _pick_column(df, ['score_name'])
    umtd_col = _pick_column(df, ['uMTD_rate'])
    fail_col = _pick_column(df, ['fail_per_alarm'])
    t1_col = _pick_column(df, ['stage_one_time_per_alarm'])
    t2_col = _pick_column(df, ['stage_two_time_per_alarm'])
    c1_col = _pick_column(df, ['delta_cost_one_per_alarm'])
    c2_col = _pick_column(df, ['delta_cost_two_per_alarm'])
    ret_col = _pick_column(df, ['attack_retention_overall'])
    weak_col = _pick_column(df, ['weak_retention'])
    strong_col = _pick_column(df, ['strong_retention'])

    sub = df[df[score_col] != 'always_trigger'].copy()
    if sub.empty:
        raise RuntimeError('No matched-budget rows found in ablation CSV.')

    score_map = {
        'detector_loss': 'Detector loss gate',
        'proposed_phys_score': 'Recovery-aware gate',
    }
    sub['label'] = [f"{b}\n{score_map.get(s, s)}" for b, s in zip(sub[budget_col], sub[score_col])]

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

    x = list(range(len(sub)))
    labels: List[str] = sub['label'].tolist()

    # 1) retention
    fig = plt.figure(figsize=(7.16, 2.8))
    width = 0.24
    plt.bar([i - width for i in x], sub[weak_col], width=width, label='Weak retention')
    plt.bar(x, sub[ret_col], width=width, label='Overall retention')
    plt.bar([i + width for i in x], sub[strong_col], width=width, label='Strong retention')
    plt.xticks(x, labels)
    plt.ylabel('Retention')
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25, linewidth=0.5, axis='y')
    plt.legend(frameon=False, ncol=3, loc='upper center')
    _savefig(fig, args.outdir, 'ablation_retention', args.dpi)

    # 2) cost burden
    fig = plt.figure(figsize=(7.16, 2.8))
    width = 0.35
    plt.bar([i - width/2 for i in x], sub[c1_col], width=width, label='Stage-I cost / false alarm')
    plt.bar([i + width/2 for i in x], sub[c2_col], width=width, label='Stage-II cost / false alarm')
    plt.xticks(x, labels)
    plt.ylabel('Incremental operating cost')
    plt.grid(alpha=0.25, linewidth=0.5, axis='y')
    plt.legend(frameon=False)
    _savefig(fig, args.outdir, 'ablation_cost_burden', args.dpi)

    # 3) time burden
    fig = plt.figure(figsize=(7.16, 2.8))
    width = 0.35
    plt.bar([i - width/2 for i in x], sub[t1_col], width=width, label='Stage-I time / false alarm')
    plt.bar([i + width/2 for i in x], sub[t2_col], width=width, label='Stage-II time / false alarm')
    plt.xticks(x, labels)
    plt.ylabel('Defense time')
    plt.grid(alpha=0.25, linewidth=0.5, axis='y')
    plt.legend(frameon=False)
    _savefig(fig, args.outdir, 'ablation_time_burden', args.dpi)

    # 4) deployment and failure burden
    fig = plt.figure(figsize=(7.16, 2.8))
    width = 0.35
    plt.bar([i - width/2 for i in x], sub[umtd_col], width=width, label='uMTD rate')
    plt.bar([i + width/2 for i in x], sub[fail_col], width=width, label='Failure / false alarm')
    plt.xticks(x, labels)
    plt.ylabel('Rate')
    plt.grid(alpha=0.25, linewidth=0.5, axis='y')
    plt.legend(frameon=False)
    _savefig(fig, args.outdir, 'ablation_deployment_failure', args.dpi)

    print('Saved figures to:', args.outdir)


if __name__ == '__main__':
    main()

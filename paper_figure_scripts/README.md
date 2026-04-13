# paper_figure_scripts_v6

This bundle does two things:

1. Generates the quantitative figures (Fig. 2--4 and Supplementary Fig. S1) from `paper_tables/`.
2. Provides a vector-native TikZ source for Fig. 1, which can be included directly in the IEEEtran paper.

## Why TikZ for Fig. 1?
The process diagram is better handled as vector art than as a plotting-library chart. This keeps fonts aligned with the paper, reduces crowding, and produces cleaner PDF output for IEEE submission.

## Usage
From the repo root:

```bash
python paper_figure_scripts_v6/build_all_paper_figures.py
```

This writes:

```bash
paper_figures_v6/
```

It also copies the TikZ source for Fig. 1 into `paper_figures_v6/fig01_online_decision_pipeline_tikz.tex`.

## Main-text figures in the draft
- Fig. 1: `fig01_online_decision_pipeline_tikz.tex` (TikZ, included directly in LaTeX)
- Fig. 2: `fig02_case14_main_blind_confirm.(pdf|png)`
- Fig. 3: `fig03_case14_ablation.(pdf|png)`
- Fig. 4: `fig04_case39_stage_ladder.(pdf|png)`

## Supplementary figure
- Fig. S1: `supp_case14_significance_ci.(pdf|png)`

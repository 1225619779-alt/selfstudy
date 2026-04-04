#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="metric/case14/plots_paper_v2"
TAU_MAIN="0.033183758162"
TAU_STRICT="0.036751313717"

python make_fig2_clean_tradeoff_v2.py --output_dir "$OUT_DIR"
python make_fig3_verify_score_v2.py --output_dir "$OUT_DIR" --tau_main "$TAU_MAIN"
# If you want the sensitivity threshold line too, uncomment the next line and use that PDF/PNG in appendix.
python make_fig3_verify_score_v2.py --output_dir "$OUT_DIR" --tau_main "$TAU_MAIN" --tau_strict "$TAU_STRICT" --show_strict
python make_fig4_mixed_main_v2.py --output_dir "$OUT_DIR" --baseline metric/case14/metric_mixed_timeline_tau_-1.0.npy --gated metric/case14/metric_mixed_timeline_tau_0.033184.npy

echo "Saved all v2 paper figures into: $OUT_DIR"

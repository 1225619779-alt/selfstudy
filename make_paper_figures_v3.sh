#!/usr/bin/env bash
set -euo pipefail

python make_fig2_clean_tradeoff_v3.py
python make_fig3_verify_score_v3.py
python make_fig4_mixed_main_v3.py

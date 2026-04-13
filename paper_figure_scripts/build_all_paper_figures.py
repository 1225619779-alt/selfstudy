from __future__ import annotations

from pathlib import Path
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_utils import find_repo_root, output_dir  # noqa: E402
from fig02_case14_main_blind_confirm import build_figure as build_fig2  # noqa: E402
from fig03_case14_ablation import build_figure as build_fig3  # noqa: E402
from fig04_case39_stage_ladder import build_figure as build_fig4  # noqa: E402
from supp_case14_significance_ci import build_figure as build_s1  # noqa: E402


def main() -> None:
    repo_root = find_repo_root()
    out_dir = output_dir(repo_root)
    # Copy the vector-native TikZ source for Fig. 1 to the output folder.
    src = SCRIPT_DIR / "fig01_online_decision_pipeline_tikz.tex"
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    print(f"wrote: {dst}")

    build_fig2(repo_root)
    build_fig3(repo_root)
    build_fig4(repo_root)
    build_s1(repo_root)
    print("all figures written under paper_figures_v6/")


if __name__ == "__main__":
    main()

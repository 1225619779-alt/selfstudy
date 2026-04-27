# Gate8 Case14 TRBG Compatibility Stats

TRBG-source is applied as a fixed burden guard on top of the original proposed safeguarded scheduler; no case14 test holdout is used for parameter selection.

Input provenance note: the q1-case39 worktree contains the case14 confirm manifests and summaries but not the raw `phase3_confirm_blind_v*/banks` files. Missing raw case14 holdout banks were read from `/home/pang/projects/DDET-MTD/metric/case14/.../banks/` with identical manifest-relative names; `gate8_case14_input_path_resolution.csv` records this fallback.

| B | recall_retention | unnecessary_delta | cost_delta | backend_fail_delta |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.9587 | 16.5000 | -0.6263 | -1.5000 |
| 2 | 0.9729 | 29.7500 | 0.5895 | 0.1250 |

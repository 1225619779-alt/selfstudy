# Matched-burden Diagnostic

Matched-burden rows are post-hoc diagnostics, not deployable baselines and not a new scheduler family.

| comparator | match | B | source_sched | comp_sched | delta |
| --- | --- | --- | --- | --- | --- |
| topk_expected_consequence | backend_fail | 1 | 0.0760 | 0.0723 | 0.0037 |
| topk_expected_consequence | backend_fail | 2 | 0.1750 | 0.1322 | 0.0429 |
| topk_expected_consequence | clean_service | 1 | 0.0848 | 0.0723 | 0.0125 |
| topk_expected_consequence | clean_service | 2 | 0.1772 | 0.1322 | 0.0451 |
| topk_expected_consequence | cost | 1 | 0.0804 | 0.0723 | 0.0081 |
| topk_expected_consequence | cost | 2 | 0.1550 | 0.1322 | 0.0229 |
| topk_expected_consequence | served_count | 1 | 0.0793 | 0.0723 | 0.0070 |
| topk_expected_consequence | served_count | 2 | 0.1659 | 0.1322 | 0.0338 |
| winner_replay | backend_fail | 1 | 0.0586 | 0.0629 | -0.0042 |
| winner_replay | backend_fail | 2 | 0.1540 | 0.1264 | 0.0276 |
| winner_replay | clean_service | 1 | 0.0462 | 0.0629 | -0.0167 |
| winner_replay | clean_service | 2 | 0.1131 | 0.1264 | -0.0133 |
| winner_replay | cost | 1 | 0.0542 | 0.0629 | -0.0087 |
| winner_replay | cost | 2 | 0.1503 | 0.1264 | 0.0239 |
| winner_replay | served_count | 1 | 0.0572 | 0.0629 | -0.0057 |
| winner_replay | served_count | 2 | 0.1503 | 0.1264 | 0.0239 |

- If the source-frozen advantage disappears under matched burden, its gain is mainly bought by more service burden.
- If it remains, source-frozen has ordering advantage at the same burden.

# Funnel Summary

This summary is computed by replaying the existing holdout decisions with current artifacts; no new family or retune was run.

| stage | slot | variant | recall | unnecessary | cost/step | served_ratio | expired | backend_fail | delay_p95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| anchored_retune | 1 | phase3_oracle_upgrade | 0.0175 | 0.1250 | 0.0433 | 0.0749 | 194.7500 | 10.3750 | 9.7000 |
| anchored_retune | 1 | phase3_proposed | 0.0009 | 0.1250 | 0.0050 | 0.0104 | 206.0000 | 1.5000 | 7.1875 |
| anchored_retune | 1 | topk_expected_consequence | 0.0723 | 14.7500 | 0.3528 | 0.5175 | 101.0000 | 94.7500 | 8.3750 |
| anchored_retune | 2 | phase3_oracle_upgrade | 0.0543 | 1.1250 | 0.1855 | 0.3007 | 151.1250 | 48.1250 | 8.5250 |
| anchored_retune | 2 | phase3_proposed | 0.0009 | 0.6250 | 0.0104 | 0.0233 | 203.6250 | 3.8750 | 8.4313 |
| anchored_retune | 2 | topk_expected_consequence | 0.1322 | 17.5000 | 0.4960 | 0.6869 | 62.1250 | 124.0000 | 5.6750 |
| native_safeguarded_retune | 1 | phase3_oracle_upgrade | 0.0155 | 0.0000 | 0.0371 | 0.0574 | 198.0000 | 7.3750 | 9.8250 |
| native_safeguarded_retune | 1 | phase3_proposed | 0.0009 | 0.1250 | 0.0050 | 0.0104 | 206.0000 | 1.5000 | 7.1875 |
| native_safeguarded_retune | 1 | topk_expected_consequence | 0.0723 | 14.7500 | 0.3528 | 0.5175 | 101.0000 | 94.7500 | 8.3750 |
| native_safeguarded_retune | 2 | phase3_oracle_upgrade | 0.0156 | 0.0000 | 0.0404 | 0.0678 | 196.8750 | 8.5000 | 9.8125 |
| native_safeguarded_retune | 2 | phase3_proposed | 0.0009 | 0.6250 | 0.0104 | 0.0233 | 203.6250 | 3.8750 | 8.4313 |
| native_safeguarded_retune | 2 | topk_expected_consequence | 0.1322 | 17.5000 | 0.4960 | 0.6869 | 62.1250 | 124.0000 | 5.6750 |
| native_unconstrained_retune | 1 | phase3_oracle_upgrade | 0.0000 | 0.0000 | 0.0004 | 0.0055 | 207.1250 | 0.6250 | 2.5000 |
| native_unconstrained_retune | 1 | phase3_proposed | 0.0009 | 0.1250 | 0.0050 | 0.0104 | 206.0000 | 1.5000 | 7.1875 |
| native_unconstrained_retune | 1 | topk_expected_consequence | 0.0723 | 14.7500 | 0.3528 | 0.5175 | 101.0000 | 94.7500 | 8.3750 |
| native_unconstrained_retune | 2 | phase3_oracle_upgrade | 0.0833 | 0.2500 | 0.1870 | 0.3188 | 149.0000 | 46.2500 | 8.5812 |
| native_unconstrained_retune | 2 | phase3_proposed | 0.0009 | 0.6250 | 0.0104 | 0.0233 | 203.6250 | 3.8750 | 8.4313 |
| native_unconstrained_retune | 2 | topk_expected_consequence | 0.1322 | 17.5000 | 0.4960 | 0.6869 | 62.1250 | 124.0000 | 5.6750 |
| source_frozen | 1 | phase3_oracle_upgrade | 0.0917 | 13.5000 | 0.3881 | 0.5613 | 89.5000 | 102.7500 | 7.9500 |
| source_frozen | 1 | phase3_proposed | 0.0927 | 15.8750 | 0.4262 | 0.5877 | 82.6250 | 109.6250 | 8.0812 |
| source_frozen | 1 | topk_expected_consequence | 0.0941 | 16.0000 | 0.4265 | 0.5922 | 82.6250 | 109.5000 | 7.9937 |
| source_frozen | 2 | phase3_oracle_upgrade | 0.1772 | 17.5000 | 0.5575 | 0.7422 | 49.8750 | 130.1250 | 6.2188 |
| source_frozen | 2 | phase3_proposed | 0.1756 | 17.5000 | 0.5784 | 0.7643 | 45.0000 | 135.0000 | 6.7375 |
| source_frozen | 2 | topk_expected_consequence | 0.1676 | 17.5000 | 0.5670 | 0.7555 | 47.0000 | 134.0000 | 6.2125 |
| winner_replay | 1 | phase3_oracle_upgrade | 0.0629 | 7.8750 | 0.2808 | 0.4529 | 118.5000 | 77.8750 | 8.7250 |
| winner_replay | 1 | phase3_proposed | 0.0009 | 0.1250 | 0.0050 | 0.0104 | 206.0000 | 1.5000 | 7.1875 |
| winner_replay | 1 | topk_expected_consequence | 0.0723 | 14.7500 | 0.3528 | 0.5175 | 101.0000 | 94.7500 | 8.3750 |
| winner_replay | 2 | phase3_oracle_upgrade | 0.1264 | 11.5000 | 0.4570 | 0.6903 | 68.8750 | 117.8750 | 6.4312 |
| winner_replay | 2 | phase3_proposed | 0.0009 | 0.6250 | 0.0104 | 0.0233 | 203.6250 | 3.8750 | 8.4313 |
| winner_replay | 2 | topk_expected_consequence | 0.1322 | 17.5000 | 0.4960 | 0.6869 | 62.1250 | 124.0000 | 5.6750 |

## Current Main Stage

- Source-frozen oracle average recall across B=1/B=2 rows: `0.1344`.
- Source-frozen oracle average expired jobs per holdout-budget: `69.6875`.
- Source-frozen oracle average backend fail count per holdout-budget: `116.4375`.

# Full-native Case39 Gate 2 Summary

This table uses the fixed 8 case39 holdouts, B=1/2, and Wmax=10. All rows were generated from explicit Gate 1 manifests; canonical case39 fit/eval were not used.

| method | B | recall | unnecessary | cost | served_ratio | backend_fail | delay_p95 | served_attack_mass | clean_served |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchored_retune | 1 | 0.0175 | 0.1250 | 0.0433 | 0.0626 | 10.3750 | 9.7000 | 6.6188 | 0.1250 |
| anchored_retune | 2 | 0.0543 | 1.1250 | 0.1855 | 0.2737 | 48.1250 | 8.5250 | 24.5000 | 1.1250 |
| incumbent_queue_aware | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| incumbent_queue_aware | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| native_safeguarded_retune | 1 | 0.0155 | 0.0000 | 0.0371 | 0.0477 | 7.3750 | 9.8250 | 5.6500 | 0.0000 |
| native_safeguarded_retune | 2 | 0.0156 | 0.0000 | 0.0404 | 0.0535 | 8.5000 | 9.8125 | 6.4500 | 0.0000 |
| native_unconstrained_retune | 1 | 0.0000 | 0.0000 | 0.0004 | 0.0031 | 0.6250 | 2.5000 | 0.5375 | 0.0000 |
| native_unconstrained_retune | 2 | 0.0833 | 0.2500 | 0.1870 | 0.2819 | 46.2500 | 8.5812 | 30.5312 | 0.2500 |
| phase3_oracle_upgrade | 1 | 0.0000 | 0.0000 | 0.0004 | 0.0031 | 0.6250 | 2.5000 | 0.5375 | 0.0000 |
| phase3_oracle_upgrade | 2 | 0.0833 | 0.2500 | 0.1870 | 0.2819 | 46.2500 | 8.5812 | 30.5312 | 0.2500 |
| phase3_proposed | 1 | 0.0009 | 0.1250 | 0.0050 | 0.0085 | 1.5000 | 7.1875 | 0.9937 | 0.1250 |
| phase3_proposed | 2 | 0.0009 | 0.6250 | 0.0104 | 0.0200 | 3.8750 | 8.4313 | 2.1875 | 0.6250 |
| source_frozen_transfer | 1 | 0.0917 | 13.5000 | 0.3881 | 0.5658 | 102.7500 | 7.9500 | 44.5375 | 13.5000 |
| source_frozen_transfer | 2 | 0.1772 | 17.5000 | 0.5575 | 0.7546 | 130.1250 | 6.2188 | 67.8125 | 17.5000 |
| static_threshold | 1 | 0.0108 | 2.8750 | 0.0384 | 0.0490 | 8.8750 | 2.0000 | 3.7375 | 2.8750 |
| static_threshold | 2 | 0.0149 | 2.8750 | 0.0414 | 0.0514 | 8.8750 | 1.2187 | 4.3000 | 2.8750 |
| topk_expected_consequence | 1 | 0.0723 | 14.7500 | 0.3528 | 0.5102 | 94.7500 | 8.3750 | 40.1125 | 14.7500 |
| topk_expected_consequence | 2 | 0.1322 | 17.5000 | 0.4960 | 0.6972 | 124.0000 | 5.6750 | 60.4750 | 17.5000 |
| winner_replay | 1 | 0.0629 | 7.8750 | 0.2808 | 0.4282 | 77.8750 | 8.7250 | 32.1688 | 7.8750 |
| winner_replay | 2 | 0.1264 | 11.5000 | 0.4570 | 0.6649 | 117.8750 | 6.4312 | 59.9187 | 11.5000 |

## Failures

No baseline failed.

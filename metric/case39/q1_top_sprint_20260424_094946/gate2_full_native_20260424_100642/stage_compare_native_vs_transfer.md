# Stage Compare: Native vs Transfer

Rows separate source-frozen transfer from explicit full-native case39 operating points.

| method | B | group | full_native | recall | cost | unnecessary | backend_fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchored_retune | 1 | full_native_source_anchored | True | 0.0175 | 0.0433 | 0.1250 | 10.3750 |
| anchored_retune | 2 | full_native_source_anchored | True | 0.0543 | 0.1855 | 1.1250 | 48.1250 |
| incumbent_queue_aware | 1 | full_native_diagnostic_baseline | True | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| incumbent_queue_aware | 2 | full_native_diagnostic_baseline | True | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| native_safeguarded_retune | 1 | full_native_safeguarded | True | 0.0155 | 0.0371 | 0.0000 | 7.3750 |
| native_safeguarded_retune | 2 | full_native_safeguarded | True | 0.0156 | 0.0404 | 0.0000 | 8.5000 |
| native_unconstrained_retune | 1 | full_native_unconstrained | True | 0.0000 | 0.0004 | 0.0000 | 0.6250 |
| native_unconstrained_retune | 2 | full_native_unconstrained | True | 0.0833 | 0.1870 | 0.2500 | 46.2500 |
| phase3_oracle_upgrade | 1 | full_native_gate2_screen | True | 0.0000 | 0.0004 | 0.0000 | 0.6250 |
| phase3_oracle_upgrade | 2 | full_native_gate2_screen | True | 0.0833 | 0.1870 | 0.2500 | 46.2500 |
| phase3_proposed | 1 | full_native_baseline | True | 0.0009 | 0.0050 | 0.1250 | 1.5000 |
| phase3_proposed | 2 | full_native_baseline | True | 0.0009 | 0.0104 | 0.6250 | 3.8750 |
| source_frozen_transfer | 1 | source_frozen_transfer | False | 0.0917 | 0.3881 | 13.5000 | 102.7500 |
| source_frozen_transfer | 2 | source_frozen_transfer | False | 0.1772 | 0.5575 | 17.5000 | 130.1250 |
| static_threshold | 1 | full_native_diagnostic_baseline | True | 0.0108 | 0.0384 | 2.8750 | 8.8750 |
| static_threshold | 2 | full_native_diagnostic_baseline | True | 0.0149 | 0.0414 | 2.8750 | 8.8750 |
| topk_expected_consequence | 1 | full_native_baseline | True | 0.0723 | 0.3528 | 14.7500 | 94.7500 |
| topk_expected_consequence | 2 | full_native_baseline | True | 0.1322 | 0.4960 | 17.5000 | 124.0000 |
| winner_replay | 1 | full_native_with_source_config | True | 0.0629 | 0.2808 | 7.8750 | 77.8750 |
| winner_replay | 2 | full_native_with_source_config | True | 0.1264 | 0.4570 | 11.5000 | 117.8750 |

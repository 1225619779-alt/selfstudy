# Gate 5 Confirm Summary

Alpha=0 / fail_cap=1.00 reproduction max_abs_diff: `0.00000000`; passed=`True`.

| method | B | recall | backend_fail | cost | recover_fail |
| --- | --- | --- | --- | --- | --- |
| TRBG-native-burden | 1 | 0.0928 | 100.3750 | 0.3659 | 6.6250 |
| TRBG-native-burden | 2 | 0.1707 | 125.0000 | 0.5198 | 7.7500 |
| TRBG-source | 1 | 0.0928 | 93.0000 | 0.3452 | 6.1250 |
| TRBG-source | 2 | 0.1704 | 120.3750 | 0.5128 | 7.6250 |
| native_safeguarded_retune | 1 | 0.0155 | 7.3750 | 0.0371 | 0.6250 |
| native_safeguarded_retune | 2 | 0.0156 | 8.5000 | 0.0404 | 0.7500 |
| native_unconstrained_retune | 1 | 0.0000 | 0.6250 | 0.0004 | 0.2500 |
| native_unconstrained_retune | 2 | 0.0833 | 46.2500 | 0.1870 | 2.7500 |
| source_frozen_transfer | 1 | 0.0917 | 102.7500 | 0.3881 | 6.1250 |
| source_frozen_transfer | 2 | 0.1772 | 130.1250 | 0.5575 | 8.1250 |
| topk_expected_consequence | 1 | 0.0723 | 94.7500 | 0.3528 | 5.8750 |
| topk_expected_consequence | 2 | 0.1322 | 124.0000 | 0.4960 | 7.6250 |
| winner_replay | 1 | 0.0629 | 77.8750 | 0.2808 | 4.6250 |
| winner_replay | 2 | 0.1264 | 117.8750 | 0.4570 | 7.1250 |

Selected TRBG confirm rows are fixed by dev-selected alpha/cap; full grid rows are diagnostic only and were not used for selection.

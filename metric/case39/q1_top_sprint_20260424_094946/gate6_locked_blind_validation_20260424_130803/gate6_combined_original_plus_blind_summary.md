# Combined Original Plus Blind Summary

Gate 6 blind is the primary validation set. Combined original+blind is secondary increased-sample evidence only and was not used to reselect alpha/cap.

| set | method | B | recall | backend | cost |
| --- | --- | --- | --- | --- | --- |
| gate5_original_internal | TRBG-source | 1 | 0.0928 | 93.0000 | 0.3452 |
| gate5_original_internal | TRBG-source | 2 | 0.1704 | 120.3750 | 0.5128 |
| gate5_original_internal | source_frozen_transfer | 1 | 0.0917 | 102.7500 | 0.3881 |
| gate5_original_internal | source_frozen_transfer | 2 | 0.1772 | 130.1250 | 0.5575 |
| gate5_original_internal | topk_expected_consequence | 1 | 0.0723 | 94.7500 | 0.3528 |
| gate5_original_internal | topk_expected_consequence | 2 | 0.1322 | 124.0000 | 0.4960 |
| gate5_original_internal | winner_replay | 1 | 0.0629 | 77.8750 | 0.2808 |
| gate5_original_internal | winner_replay | 2 | 0.1264 | 117.8750 | 0.4570 |
| gate6_blind | TRBG-source | 1 | 0.0891 | 81.8750 | 0.3302 |
| gate6_blind | TRBG-source | 2 | 0.1650 | 106.6250 | 0.4721 |
| gate6_blind | source_frozen_transfer | 1 | 0.0964 | 95.5000 | 0.4149 |
| gate6_blind | source_frozen_transfer | 2 | 0.1664 | 117.2500 | 0.5319 |
| gate6_blind | topk_expected_consequence | 1 | 0.0738 | 81.6250 | 0.3326 |
| gate6_blind | topk_expected_consequence | 2 | 0.1310 | 113.1250 | 0.4840 |
| gate6_blind | winner_replay | 1 | 0.0658 | 74.7500 | 0.2793 |
| gate6_blind | winner_replay | 2 | 0.1262 | 101.5000 | 0.3996 |

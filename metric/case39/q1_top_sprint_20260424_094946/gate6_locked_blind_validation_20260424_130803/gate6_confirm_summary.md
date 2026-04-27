# Gate 6 Confirm Summary

| method | B | recall | backend | cost | recover |
| --- | --- | --- | --- | --- | --- |
| TRBG-native-burden | 1 | 0.0917 | 88.2500 | 0.3594 | 5.5000 |
| TRBG-native-burden | 2 | 0.1653 | 114.3750 | 0.5009 | 8.7500 |
| TRBG-source | 1 | 0.0891 | 81.8750 | 0.3302 | 5.5000 |
| TRBG-source | 2 | 0.1650 | 106.6250 | 0.4721 | 8.5000 |
| native_safeguarded_retune | 1 | 0.0148 | 4.2500 | 0.0185 | 0.7500 |
| native_safeguarded_retune | 2 | 0.0218 | 5.0000 | 0.0267 | 0.8750 |
| native_unconstrained_retune | 1 | 0.0013 | 0.2500 | 0.0010 | 0.0000 |
| native_unconstrained_retune | 2 | 0.0830 | 39.0000 | 0.1806 | 4.0000 |
| source_frozen_transfer | 1 | 0.0964 | 95.5000 | 0.4149 | 5.8750 |
| source_frozen_transfer | 2 | 0.1664 | 117.2500 | 0.5319 | 8.2500 |
| topk_expected_consequence | 1 | 0.0738 | 81.6250 | 0.3326 | 5.5000 |
| topk_expected_consequence | 2 | 0.1310 | 113.1250 | 0.4840 | 7.7500 |
| winner_replay | 1 | 0.0658 | 74.7500 | 0.2793 | 5.3750 |
| winner_replay | 2 | 0.1262 | 101.5000 | 0.3996 | 6.7500 |

- Recall retention vs source: `0.9666`.
- Backend_fail reduction vs source: `0.1140`.
- Cost change vs source: `-0.0722`.
- Recall remains above topk: `True`.

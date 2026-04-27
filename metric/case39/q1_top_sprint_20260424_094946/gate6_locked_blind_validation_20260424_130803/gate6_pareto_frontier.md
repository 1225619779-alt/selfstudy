# Gate 6 Pareto Frontier

| method | B | recall | backend | cost | dom_cost | dom_backend |
| --- | --- | --- | --- | --- | --- | --- |
| TRBG-native-burden | 1 | 0.0917 | 88.2500 | 0.3594 | False | False |
| TRBG-source | 1 | 0.0891 | 81.8750 | 0.3302 | False | False |
| native_safeguarded_retune | 1 | 0.0148 | 4.2500 | 0.0185 | False | False |
| native_unconstrained_retune | 1 | 0.0013 | 0.2500 | 0.0010 | False | False |
| source_frozen_transfer | 1 | 0.0964 | 95.5000 | 0.4149 | False | False |
| topk_expected_consequence | 1 | 0.0738 | 81.6250 | 0.3326 | True | False |
| winner_replay | 1 | 0.0658 | 74.7500 | 0.2793 | False | False |
| TRBG-native-burden | 2 | 0.1653 | 114.3750 | 0.5009 | False | False |
| TRBG-source | 2 | 0.1650 | 106.6250 | 0.4721 | False | False |
| native_safeguarded_retune | 2 | 0.0218 | 5.0000 | 0.0267 | False | False |
| native_unconstrained_retune | 2 | 0.0830 | 39.0000 | 0.1806 | False | False |
| source_frozen_transfer | 2 | 0.1664 | 117.2500 | 0.5319 | False | False |
| topk_expected_consequence | 2 | 0.1310 | 113.1250 | 0.4840 | True | True |
| winner_replay | 2 | 0.1262 | 101.5000 | 0.3996 | False | False |

1. TRBG-source not dominated in recall-cost: `True`.
2. TRBG-source not dominated in recall-backend_fail: `True`.
3. TRBG-source remains useful if it is between source-frozen and topk/winner with lower burden and retained recall.
4. Gate 6 direction is compared explicitly against Gate 5 in the combined summary.
5. B=1 backend reduction `0.1427`, B=2 backend reduction `0.0906`.

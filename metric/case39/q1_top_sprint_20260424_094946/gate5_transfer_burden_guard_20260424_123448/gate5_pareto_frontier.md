# Gate 5 Pareto Frontier

| method | B | recall | backend | cost | dom_rc | dom_rb |
| --- | --- | --- | --- | --- | --- | --- |
| TRBG-native-burden | 1 | 0.0928 | 100.3750 | 0.3659 | False | False |
| TRBG-source | 1 | 0.0928 | 93.0000 | 0.3452 | False | False |
| native_safeguarded_retune | 1 | 0.0155 | 7.3750 | 0.0371 | False | False |
| native_unconstrained_retune | 1 | 0.0000 | 0.6250 | 0.0004 | False | False |
| source_frozen_transfer | 1 | 0.0917 | 102.7500 | 0.3881 | False | False |
| topk_expected_consequence | 1 | 0.0723 | 94.7500 | 0.3528 | False | False |
| winner_replay | 1 | 0.0629 | 77.8750 | 0.2808 | False | False |
| TRBG-native-burden | 2 | 0.1707 | 125.0000 | 0.5198 | False | False |
| TRBG-source | 2 | 0.1704 | 120.3750 | 0.5128 | False | False |
| native_safeguarded_retune | 2 | 0.0156 | 8.5000 | 0.0404 | False | False |
| native_unconstrained_retune | 2 | 0.0833 | 46.2500 | 0.1870 | False | False |
| source_frozen_transfer | 2 | 0.1772 | 130.1250 | 0.5575 | False | False |
| topk_expected_consequence | 2 | 0.1322 | 124.0000 | 0.4960 | False | False |
| winner_replay | 2 | 0.1264 | 117.8750 | 0.4570 | False | False |

## Answers

1. TRBG-source dominated flags: `{'recall_cost': [False, False], 'recall_backend': [False, False]}`.
2. TRBG-native-burden dominated flags: `{'recall_cost': [False, False], 'recall_backend': [False, False]}`.
3. TRBG movement is interpreted by success criteria, not by test-selected candidate choice.
4. Useful middle point: TRBG-source `moderate_success`, TRBG-native-burden `moderate_success`.
5. More reasonable candidate: `TRBG-source`.
6. Native information is considered useful for low-dimensional burden calibration only if TRBG-native-burden beats TRBG-source on the locked success criteria; this does not rescue full local retune.

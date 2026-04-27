# Gate9 Case39 Original 8-Holdout Secondary Stats

This is secondary diagnostic only, not primary confirmatory evidence, because fail-only was motivated by Gate8 diagnostics.

| method | B | recall | backend_fail | cost | unnecessary | recover_fail |
| --- | --- | --- | --- | --- | --- | --- |
| TRBG-source | 1 | 0.0928 | 93.0000 | 0.3452 | 7.2500 | 6.1250 |
| TRBG-source | 2 | 0.1704 | 120.3750 | 0.5128 | 10.0000 | 7.6250 |
| TRFG-native-fail | 1 | 0.0967 | 89.6250 | 0.3568 | 7.6250 | 6.1250 |
| TRFG-native-fail | 2 | 0.1726 | 115.7500 | 0.4790 | 8.6250 | 7.2500 |
| TRFG-source | 1 | 0.0950 | 97.6250 | 0.3691 | 8.5000 | 5.8750 |
| TRFG-source | 2 | 0.1794 | 123.5000 | 0.5219 | 11.3750 | 7.5000 |
| native_safeguarded_retune | 1 | 0.0155 | 7.3750 | 0.0371 | 0.0000 | 0.6250 |
| native_safeguarded_retune | 2 | 0.0156 | 8.5000 | 0.0404 | 0.0000 | 0.7500 |
| native_unconstrained_retune | 1 | 0.0000 | 0.6250 | 0.0004 | 0.0000 | 0.2500 |
| native_unconstrained_retune | 2 | 0.0833 | 46.2500 | 0.1870 | 0.2500 | 2.7500 |
| source_frozen_transfer | 1 | 0.0917 | 102.7500 | 0.3881 | 13.5000 | 6.1250 |
| source_frozen_transfer | 2 | 0.1772 | 130.1250 | 0.5575 | 17.5000 | 8.1250 |
| topk_expected_consequence | 1 | 0.0723 | 94.7500 | 0.3528 | 14.7500 | 5.8750 |
| topk_expected_consequence | 2 | 0.1322 | 124.0000 | 0.4960 | 17.5000 | 7.6250 |
| winner_replay | 1 | 0.0629 | 77.8750 | 0.2808 | 7.8750 | 4.6250 |
| winner_replay | 2 | 0.1264 | 117.8750 | 0.4570 | 11.5000 | 7.1250 |

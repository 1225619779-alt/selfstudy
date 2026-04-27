# Gate 5 Calibration Summary

Selection used only source case14 train/val for TRBG-source and explicit native case39 train/val for TRBG-native-burden.

| mode | alpha | cap | dev_recall | source_dev_recall | dev_backend_fail | dev_cost | guard_failed_on_dev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRBG-source | 1.0 | 1.0 | 0.7342 | 0.7776 | 4.0000 | 15.5256 | False |
| TRBG-native-burden | 2.0 | 1.0 | 0.0843 | 0.0754 | 211.0000 | 0.3268 | False |

All primary and secondary grid candidates are written to CSV; only fail_cap_quantile=1.00 alpha candidates were eligible for primary selection.

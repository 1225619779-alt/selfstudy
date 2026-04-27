# Gate 5 Proxy Robustness Selected

| method | B | proxy | score | backend | recover |
| --- | --- | --- | --- | --- | --- |
| TRBG-native-burden | 1 | additive_proxy | 0.0883 | 100.3750 | 6.6250 |
| TRBG-native-burden | 1 | backend_success_proxy | 0.2957 | 100.3750 | 6.6250 |
| TRBG-native-burden | 1 | burden_proxy | -0.7069 | 100.3750 | 6.6250 |
| TRBG-native-burden | 1 | product_proxy | 0.0928 | 100.3750 | 6.6250 |
| TRBG-native-burden | 1 | recovery_aware_proxy | 0.0967 | 100.3750 | 6.6250 |
| TRBG-native-burden | 1 | success_burden_proxy | -7.3447 | 100.3750 | 6.6250 |
| TRBG-native-burden | 2 | additive_proxy | 0.1593 | 125.0000 | 7.7500 |
| TRBG-native-burden | 2 | backend_success_proxy | 0.5424 | 125.0000 | 7.7500 |
| TRBG-native-burden | 2 | burden_proxy | -0.9335 | 125.0000 | 7.7500 |
| TRBG-native-burden | 2 | product_proxy | 0.1707 | 125.0000 | 7.7500 |
| TRBG-native-burden | 2 | recovery_aware_proxy | 0.1781 | 125.0000 | 7.7500 |
| TRBG-native-burden | 2 | success_burden_proxy | -9.3313 | 125.0000 | 7.7500 |
| TRBG-source | 1 | additive_proxy | 0.0871 | 93.0000 | 6.1250 |
| TRBG-source | 1 | backend_success_proxy | 0.3021 | 93.0000 | 6.1250 |
| TRBG-source | 1 | burden_proxy | -0.6622 | 93.0000 | 6.1250 |
| TRBG-source | 1 | product_proxy | 0.0928 | 93.0000 | 6.1250 |
| TRBG-source | 1 | recovery_aware_proxy | 0.0968 | 93.0000 | 6.1250 |
| TRBG-source | 1 | success_burden_proxy | -6.7500 | 93.0000 | 6.1250 |
| TRBG-source | 2 | additive_proxy | 0.1586 | 120.3750 | 7.6250 |
| TRBG-source | 2 | backend_success_proxy | 0.5458 | 120.3750 | 7.6250 |
| TRBG-source | 2 | burden_proxy | -0.9053 | 120.3750 | 7.6250 |
| TRBG-source | 2 | product_proxy | 0.1704 | 120.3750 | 7.6250 |
| TRBG-source | 2 | recovery_aware_proxy | 0.1777 | 120.3750 | 7.6250 |
| TRBG-source | 2 | success_burden_proxy | -9.0628 | 120.3750 | 7.6250 |
| source_frozen_transfer | 1 | additive_proxy | 0.0872 | 102.7500 | 6.1250 |
| source_frozen_transfer | 1 | backend_success_proxy | 0.2998 | 102.7500 | 6.1250 |
| source_frozen_transfer | 1 | burden_proxy | -0.7285 | 102.7500 | 6.1250 |
| source_frozen_transfer | 1 | product_proxy | 0.0917 | 102.7500 | 6.1250 |
| source_frozen_transfer | 1 | recovery_aware_proxy | 0.0956 | 102.7500 | 6.1250 |
| source_frozen_transfer | 1 | success_burden_proxy | -7.6374 | 102.7500 | 6.1250 |
| source_frozen_transfer | 2 | additive_proxy | 0.1643 | 130.1250 | 8.1250 |
| source_frozen_transfer | 2 | backend_success_proxy | 0.5600 | 130.1250 | 8.1250 |
| source_frozen_transfer | 2 | burden_proxy | -0.9739 | 130.1250 | 8.1250 |
| source_frozen_transfer | 2 | product_proxy | 0.1772 | 130.1250 | 8.1250 |
| source_frozen_transfer | 2 | recovery_aware_proxy | 0.1850 | 130.1250 | 8.1250 |
| source_frozen_transfer | 2 | success_burden_proxy | -9.8270 | 130.1250 | 8.1250 |
| topk_expected_consequence | 1 | additive_proxy | 0.0685 | 94.7500 | 5.8750 |
| topk_expected_consequence | 1 | backend_success_proxy | 0.2252 | 94.7500 | 5.8750 |
| topk_expected_consequence | 1 | burden_proxy | -0.6564 | 94.7500 | 5.8750 |
| topk_expected_consequence | 1 | product_proxy | 0.0723 | 94.7500 | 5.8750 |
| topk_expected_consequence | 1 | recovery_aware_proxy | 0.0754 | 94.7500 | 5.8750 |
| topk_expected_consequence | 1 | success_burden_proxy | -7.1163 | 94.7500 | 5.8750 |
| topk_expected_consequence | 2 | additive_proxy | 0.1247 | 124.0000 | 7.6250 |
| topk_expected_consequence | 2 | backend_success_proxy | 0.4111 | 124.0000 | 7.6250 |
| topk_expected_consequence | 2 | burden_proxy | -0.8989 | 124.0000 | 7.6250 |
| topk_expected_consequence | 2 | product_proxy | 0.1322 | 124.0000 | 7.6250 |
| topk_expected_consequence | 2 | recovery_aware_proxy | 0.1379 | 124.0000 | 7.6250 |
| topk_expected_consequence | 2 | success_burden_proxy | -9.2459 | 124.0000 | 7.6250 |

- Selected TRBG decisions are fixed before proxy rescoring.
- Improvement, if any, should be framed as burden/recovery robustness diagnostic unless locked success criteria are met.

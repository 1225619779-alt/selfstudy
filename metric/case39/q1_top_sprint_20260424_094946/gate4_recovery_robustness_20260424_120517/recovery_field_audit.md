# Recovery / Burden Field Audit

Mixed banks contain `recover_fail` and `backend_fail`; raw banks expose `recover_fail` but not `backend_fail`. Exact `service_time` / `service_cost` names are absent in raw npy payloads, but scheduler arrays derive them from `stage_one_time + stage_two_time` and `delta_cost_one + delta_cost_two`.

| metric | value |
| --- | --- |
| mixed_recover_frequency_mean | 0.0407 |
| mixed_backend_frequency_mean | 0.6360 |
| recover_backend_corr | 0.1507 |
| recover_product_corr | -0.0059 |
| recover_additive_corr | -0.0052 |
| recover_service_time_corr | -0.0370 |
| recover_service_cost_corr | -0.0806 |
| served_recover_attack_rate | 0.0492 |
| served_recover_clean_rate | 0.0852 |

## Recover Fail x Backend Fail

| recover_fail | backend_fail | count |
| --- | --- | --- |
| 0 | 0 | 1079 |
| 0 | 1 | 1836 |
| 1 | 0 | 0 |
| 1 | 1 | 120 |

## Decision

- `recover_fail` availability: `True`.
- `recover_fail` sparsity flag: `False`.
- `recover_fail` vs `backend_fail` served-attack-rate correlation: `0.2190`.
- Label decision: `main_text_robustness_label`.
- Conservative use: recovery-aware proxy is field-supported enough for a main-text robustness check, but not as a new trained method.

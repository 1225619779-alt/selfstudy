# Gate9 Fresh Runtime Report

Reduced fresh physical-solver sanity check; not full 8-bank statistical validation.

- attempted banks: `4`
- completed banks: `4`
- serial execution: `True`

| bank_id | complete | completed_steps | exit_code | mean_step_sec | p95_step_sec | backend_fail | recover_fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate9_sparse_front_0 | True | 240 | 0 | 13.2352 | 49.6408 | 96 | 3 |
| gate9_late_mixed_1 | True | 240 | 0 | 28.0862 | 98.3813 | 106 | 5 |
| gate9_alternating_short_2 | True | 240 | 0 | 15.6984 | 70.4969 | 75 | 2 |
| gate9_dense_middle_3 | True | 240 | 0 | 10.5609 | 61.1760 | 34 | 3 |

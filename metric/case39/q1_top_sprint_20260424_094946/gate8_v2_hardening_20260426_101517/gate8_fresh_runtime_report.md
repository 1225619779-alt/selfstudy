# Gate8 Fresh Runtime Report

This is a reduced fresh physical-solver sanity check, not full statistical validation.

- attempted banks: `4`
- completed banks: `4`
- steps per bank: `240`
- concurrency cap: `2`

| bank_id | complete | completed_steps | exit_code | mean_step_sec | p95_step_sec | backend_fail | recover_fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reduced_interleaved_0 | True | 240 | 0 | 66.3264 | 327.4091 | 74 | 3 |
| reduced_tailheavy_1 | True | 240 | 0 | 163.1738 | 206.0554 | 57 | 4 |
| reduced_cleanheavy_2 | True | 240 | 0 | 15.8336 | 64.2055 | 70 | 4 |
| reduced_attackpulse_3 | True | 240 | 0 | 28.7427 | 147.2957 | 63 | 2 |

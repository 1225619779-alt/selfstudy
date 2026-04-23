# Case39 Failure-Aware Statistics

This report keeps failure denominators explicit.

## Clean

### baseline

- metric_path: `metric/case39/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy`
- tau_verify: `-1.000000`
- total_alarms: `819`
- total_triggers: `819`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `819`
- backend_metric_fail_count: `0`
- backend_metric_fail_rate_among_triggers: `0.000000`
- backend_metric_fail_rate_among_alarms: `0.000000`
- backend_mtd_fail_count: `792`
- backend_mtd_fail_rate_among_triggers: `0.967033`
- backend_mtd_fail_rate_among_alarms: `0.967033`
- stage_one_time_success_only_mean: `9.947360` over `819` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `1.701035` over `819` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.850488` over `762` finite samples, nan_count=`0`
- delta_cost_two_success_only_mean: `-0.568006` over `761` finite samples, nan_count=`0`
- legacy_fail_per_alarm: `0.967033`
- legacy_stage_two_time_per_alarm: `1.530378`

### main

- metric_path: `metric/case39/metric_event_trigger_clean_tau_0.013_mode_0_0.03_1.1.npy`
- tau_verify: `0.013000`
- total_alarms: `819`
- total_triggers: `758`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `758`
- backend_metric_fail_count: `0`
- backend_metric_fail_rate_among_triggers: `0.000000`
- backend_metric_fail_rate_among_alarms: `0.000000`
- backend_mtd_fail_count: `745`
- backend_mtd_fail_rate_among_triggers: `0.982850`
- backend_mtd_fail_rate_among_alarms: `0.909646`
- stage_one_time_success_only_mean: `7.518317` over `758` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `0.562690` over `758` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.788538` over `709` finite samples, nan_count=`0`
- delta_cost_two_success_only_mean: `0.745729` over `709` finite samples, nan_count=`0`
- legacy_fail_per_alarm: `0.909646`
- legacy_stage_two_time_per_alarm: `0.468113`

### strict

- metric_path: `metric/case39/metric_event_trigger_clean_tau_0.016_mode_0_0.03_1.1.npy`
- tau_verify: `0.016000`
- total_alarms: `819`
- total_triggers: `681`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `681`
- backend_metric_fail_count: `0`
- backend_metric_fail_rate_among_triggers: `0.000000`
- backend_metric_fail_rate_among_alarms: `0.000000`
- backend_mtd_fail_count: `669`
- backend_mtd_fail_rate_among_triggers: `0.982379`
- backend_mtd_fail_rate_among_alarms: `0.816850`
- stage_one_time_success_only_mean: `5.354934` over `681` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `0.489185` over `681` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.738602` over `638` finite samples, nan_count=`0`
- delta_cost_two_success_only_mean: `0.710193` over `638` finite samples, nan_count=`0`
- legacy_fail_per_alarm: `0.816850`
- legacy_stage_two_time_per_alarm: `0.364102`

## Attack

### baseline

- metric_path: `metric/case39/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy`
- tau_verify: `-1.000000`
- total_alarms: `292`
- total_triggers: `292`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `268`
- backend_metric_fail_count: `24`
- backend_metric_fail_rate_among_triggers: `0.082192`
- backend_metric_fail_rate_among_alarms: `0.082192`
- backend_mtd_fail_count: `229`
- backend_mtd_fail_rate_among_triggers: `0.784247`
- backend_mtd_fail_rate_among_alarms: `0.784247`
- stage_one_time_success_only_mean: `6.087153` over `292` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `12.154338` over `292` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.866754` over `270` finite samples, nan_count=`22`
- delta_cost_two_success_only_mean: `0.537009` over `268` finite samples, nan_count=`24`
- overall_arr: `1.000000`

### main

- metric_path: `metric/case39/metric_event_trigger_tau_0.013_mode_0_0.03_1.1.npy`
- tau_verify: `0.013000`
- total_alarms: `292`
- total_triggers: `290`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `266`
- backend_metric_fail_count: `24`
- backend_metric_fail_rate_among_triggers: `0.082759`
- backend_metric_fail_rate_among_alarms: `0.082192`
- backend_mtd_fail_count: `228`
- backend_mtd_fail_rate_among_triggers: `0.786207`
- backend_mtd_fail_rate_among_alarms: `0.780822`
- stage_one_time_success_only_mean: `6.528410` over `290` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `13.164438` over `290` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.863117` over `268` finite samples, nan_count=`22`
- delta_cost_two_success_only_mean: `0.538360` over `266` finite samples, nan_count=`24`
- overall_arr: `0.993151`

### strict

- metric_path: `metric/case39/metric_event_trigger_tau_0.016_mode_0_0.03_1.1.npy`
- tau_verify: `0.016000`
- total_alarms: `292`
- total_triggers: `288`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `unknown`
- backend_metric_success_count: `265`
- backend_metric_fail_count: `23`
- backend_metric_fail_rate_among_triggers: `0.079861`
- backend_metric_fail_rate_among_alarms: `0.078767`
- backend_mtd_fail_count: `226`
- backend_mtd_fail_rate_among_triggers: `0.784722`
- backend_mtd_fail_rate_among_alarms: `0.773973`
- stage_one_time_success_only_mean: `8.188220` over `288` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `14.528734` over `288` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.865062` over `267` finite samples, nan_count=`21`
- delta_cost_two_success_only_mean: `0.539094` over `265` finite samples, nan_count=`23`
- overall_arr: `0.986301`

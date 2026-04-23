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
- recovery_error_policy: `skip`
- backend_metric_success_count: `761`
- backend_metric_fail_count: `58`
- backend_metric_fail_rate_among_triggers: `0.070818`
- backend_metric_fail_rate_among_alarms: `0.070818`
- backend_mtd_fail_count: `789`
- backend_mtd_fail_rate_among_triggers: `0.963370`
- backend_mtd_fail_rate_among_alarms: `0.963370`
- stage_one_time_success_only_mean: `8.044997` over `819` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `1.362096` over `819` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.850488` over `762` finite samples, nan_count=`57`
- delta_cost_two_success_only_mean: `0.784288` over `761` finite samples, nan_count=`58`
- legacy_fail_per_alarm: `0.963370`
- legacy_stage_two_time_per_alarm: `1.362096`

### main

- metric_path: `metric/case39/metric_event_trigger_clean_tau_0.013319196253_mode_0_0.03_1.1.npy`
- tau_verify: `0.013319`
- total_alarms: `819`
- total_triggers: `754`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `skip`
- backend_metric_success_count: `705`
- backend_metric_fail_count: `49`
- backend_metric_fail_rate_among_triggers: `0.064987`
- backend_metric_fail_rate_among_alarms: `0.059829`
- backend_mtd_fail_count: `739`
- backend_mtd_fail_rate_among_triggers: `0.980106`
- backend_mtd_fail_rate_among_alarms: `0.902320`
- stage_one_time_success_only_mean: `8.144084` over `754` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `0.698837` over `754` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.794578` over `705` finite samples, nan_count=`49`
- delta_cost_two_success_only_mean: `0.751526` over `705` finite samples, nan_count=`49`
- legacy_fail_per_alarm: `0.902320`
- legacy_stage_two_time_per_alarm: `0.643374`

### strict

- metric_path: `metric/case39/metric_event_trigger_clean_tau_0.016153267226_mode_0_0.03_1.1.npy`
- tau_verify: `0.016153`
- total_alarms: `819`
- total_triggers: `676`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `skip`
- backend_metric_success_count: `633`
- backend_metric_fail_count: `43`
- backend_metric_fail_rate_among_triggers: `0.063609`
- backend_metric_fail_rate_among_alarms: `0.052503`
- backend_mtd_fail_count: `663`
- backend_mtd_fail_rate_among_triggers: `0.980769`
- backend_mtd_fail_rate_among_alarms: `0.809524`
- stage_one_time_success_only_mean: `8.347419` over `676` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `0.659990` over `676` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `0.733448` over `633` finite samples, nan_count=`43`
- delta_cost_two_success_only_mean: `0.712166` over `633` finite samples, nan_count=`43`
- legacy_fail_per_alarm: `0.809524`
- legacy_stage_two_time_per_alarm: `0.544753`

## Attack

### baseline

- metric_path: `metric/case39/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.npy`
- tau_verify: `-1.000000`
- total_alarms: `292`
- total_triggers: `292`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `count_as_non_trigger_non_skip`
- backend_metric_success_count: `0`
- backend_metric_fail_count: `292`
- backend_metric_fail_rate_among_triggers: `1.000000`
- backend_metric_fail_rate_among_alarms: `1.000000`
- backend_mtd_fail_count: `229`
- backend_mtd_fail_rate_among_triggers: `0.784247`
- backend_mtd_fail_rate_among_alarms: `0.784247`
- stage_one_time_success_only_mean: `6.399666` over `292` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `12.574490` over `292` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `nan` over `0` finite samples, nan_count=`292`
- delta_cost_two_success_only_mean: `nan` over `0` finite samples, nan_count=`292`
- overall_arr: `1.000000`

### main

- metric_path: `metric/case39/metric_event_trigger_tau_0.013319196253_mode_0_0.03_1.1.npy`
- tau_verify: `0.013319`
- total_alarms: `292`
- total_triggers: `290`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `count_as_non_trigger_non_skip`
- backend_metric_success_count: `0`
- backend_metric_fail_count: `290`
- backend_metric_fail_rate_among_triggers: `1.000000`
- backend_metric_fail_rate_among_alarms: `0.993151`
- backend_mtd_fail_count: `228`
- backend_mtd_fail_rate_among_triggers: `0.786207`
- backend_mtd_fail_rate_among_alarms: `0.780822`
- stage_one_time_success_only_mean: `6.413142` over `290` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `12.545351` over `290` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `nan` over `0` finite samples, nan_count=`290`
- delta_cost_two_success_only_mean: `nan` over `0` finite samples, nan_count=`290`
- overall_arr: `0.993151`

### strict

- metric_path: `metric/case39/metric_event_trigger_tau_0.016153267226_mode_0_0.03_1.1.npy`
- tau_verify: `0.016153`
- total_alarms: `292`
- total_triggers: `288`
- recovery_error_count: `0`
- recovery_error_rate_among_alarms: `0.000000`
- recovery_error_policy: `count_as_non_trigger_non_skip`
- backend_metric_success_count: `0`
- backend_metric_fail_count: `288`
- backend_metric_fail_rate_among_triggers: `1.000000`
- backend_metric_fail_rate_among_alarms: `0.986301`
- backend_mtd_fail_count: `226`
- backend_mtd_fail_rate_among_triggers: `0.784722`
- backend_mtd_fail_rate_among_alarms: `0.773973`
- stage_one_time_success_only_mean: `6.424032` over `288` finite samples, nan_count=`0`
- stage_two_time_success_only_mean: `12.632471` over `288` finite samples, nan_count=`0`
- delta_cost_one_success_only_mean: `nan` over `0` finite samples, nan_count=`288`
- delta_cost_two_success_only_mean: `nan` over `0` finite samples, nan_count=`288`
- overall_arr: `0.986301`

# Clean main results write-up template

## One-sentence claim

In the clean false-alarm setting, the proposed recovery-aware verification trigger does not change the front-end DDD false-alarm rate, but it substantially reduces false-alarm-induced unnecessary backend MTD deployment and the associated failure, time, and operating-cost burden.

## Table caption

**Table X.** Clean false-alarm paired comparison of the backend burden under three operating points: baseline (always trigger), the main verification threshold ($\tau=0.021$), and a stricter threshold ($\tau=0.030$). All rows are evaluated on the same clean test set and the same DDD false alarms.

## Results paragraph template

Table X reports the clean false-alarm results under three operating points. The front-end DDD false-alarm rate remains unchanged across all rows because the proposed method does not modify the detector itself. Compared with the always-trigger baseline, the main operating point ($\tau=0.021$) reduces the unnecessary backend MTD deployment rate, the backend failure rate per false alarm, the mean stage-I/II defense time per false alarm, and the mean stage-I/II incremental operating cost per false alarm. When the threshold is increased to the stricter operating point ($\tau=0.030$), the backend deployment rate is further reduced, indicating a more conservative trigger policy.

## Metric names to use in the paper

- `false_alarm_rate` -> Front-end FAR / DDD false-alarm rate
- `trigger_rate` -> Backend MTD deployment rate among alarms
- `skip_rate` -> Alarm rejection rate
- `useless_mtd_rate` -> Unnecessary MTD deployment rate
- `fail_per_alarm` -> Backend failure rate per false alarm
- `stage_one_time_per_alarm` -> Mean stage-I defense time per false alarm
- `stage_two_time_per_alarm` -> Mean stage-II defense time per false alarm
- `delta_cost_one_per_alarm` -> Mean stage-I incremental operating cost per false alarm
- `delta_cost_two_per_alarm` -> Mean stage-II incremental operating cost per false alarm

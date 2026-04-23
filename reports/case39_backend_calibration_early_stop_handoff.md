# Case39 Backend Calibration Early-Stop Handoff

- git_head: `371463f7de5bb07eb60393fdade5d6b642bc511e`
- completed_candidate_count: `16`
- targeted_candidate_count: `3`

## Consistency
- metadata_mismatch_candidates: `0`
- clean_trigger_count_main_unique: `[541]`
- clean_trigger_count_strict_unique: `[512]`
- attack_trigger_count_main_unique: `[266]`
- attack_overall_arr_main_unique: `[0.9743589743589743]`
- attack_trigger_count_strict_unique: `[243]`
- attack_overall_arr_strict_unique: `[0.8901098901098901]`

## Best Current Candidate
- candidate: `xf_0.5_var_0.015_up_1.10_mr_15.json`
- clean main fail/alarm: `0.752669039146`
- clean main fail/trigger: `0.781885397412`
- clean main trigger_count: `541`
- clean main stage_two_time/alarm: `12.619476459629`
- attack main overall_arr: `0.974358974359`
- attack main trigger_count: `266`
- attack main backend_mtd_fail_rate_among_triggers: `0.612781954887`

## Targeted x=0.5, var=0.015
- `xf_0.5_var_0.015_up_1.00_mr_15.json`: clean fail/alarm=`0.752669039146`, clean fail/trigger=`0.781885397412`, stage_two_time/alarm=`14.302505711219`, attack overall_arr=`0.974358974359`, attack backend_mtd_fail_rate_among_triggers=`0.612781954887`
- `xf_0.5_var_0.015_up_1.05_mr_15.json`: clean fail/alarm=`0.752669039146`, clean fail/trigger=`0.781885397412`, stage_two_time/alarm=`13.301282523366`, attack overall_arr=`0.974358974359`, attack backend_mtd_fail_rate_among_triggers=`0.612781954887`
- `xf_0.5_var_0.015_up_1.10_mr_15.json`: clean fail/alarm=`0.752669039146`, clean fail/trigger=`0.781885397412`, stage_two_time/alarm=`12.619476459629`, attack overall_arr=`0.974358974359`, attack backend_mtd_fail_rate_among_triggers=`0.612781954887`

## From-Scratch Spot Rerun
- reference candidate: `xf_0.2_var_0.015_up_1.05_mr_15.json`
- cached candidate trigger_count: `541`
- spot rerun trigger_count: `541`
- cached candidate fail_count: `428`
- spot rerun fail_count: `428`
- cached candidate fail/alarm: `0.761565836299`
- spot rerun fail/alarm: `0.761565836299`
- cached candidate fail/trigger: `0.791127541590`
- spot rerun fail/trigger: `0.791127541590`
- cached candidate stage_one_time/alarm: `5.074347149435`
- spot rerun stage_one_time/alarm: `4.406706931328`
- cached candidate stage_two_time/alarm: `5.116305115384`
- spot rerun stage_two_time/alarm: `4.766349624484`

## Recommendation
- case39 = stress-test / limitation evidence
- case14 remains the main detailed evidence
- do not claim robust end-to-end case39 backend success

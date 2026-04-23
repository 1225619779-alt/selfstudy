# Paper Compact Tables

## Case14 Clean Exact
| label | total_clean_sample | total_DDD_alarm | total_trigger_after_verification | trigger_rate | useless_mtd_rate | fail_per_alarm | stage_one_time_per_alarm | stage_two_time_per_alarm | delta_cost_one_per_alarm | delta_cost_two_per_alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 7021 | 1179 | 1179 | 1.000000 | 0.167925 | 0.105174 | 0.605506 | 4.838323 | 29.604553 | 4.161843 |
| main | 7021 | 1179 | 82 | 0.069550 | 0.011679 | 0.021204 | 0.068860 | 0.270593 | 1.596107 | 0.414595 |
| strict | 7021 | 1179 | 72 | 0.061069 | 0.010255 | 0.017812 | 0.080962 | 0.398795 | 1.342565 | 0.293321 |

## Case14 Mixed Timeline
| label | total_DDD_alarm | total_trigger_after_gate | total_skip_by_gate | total_backend_fail | final_cumulative_stage_time | final_cumulative_delta_cost |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 141 | 141 | 0 | 10 | 753.869646 | 5190.730326 |
| main | 141 | 58 | 83 | 4 | 595.989086 | 1860.467609 |
| strict | 141 | 51 | 90 | 4 | 436.349683 | 1531.508583 |

## Case14 Gate Comparator
| budget_label | score_name | trigger_rate | fail_per_alarm | stage_two_time_per_alarm | delta_cost_two_per_alarm | attack_retention_overall | strong_retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main OP | detector_loss | 0.069550 | 0.005937 | 0.581301 | 1.160728 | 0.901361 | 0.993333 |
| Main OP | proposed_phys_score | 0.069550 | 0.021204 | 0.417818 | 0.414595 | 0.806122 | 0.980000 |
| Strict OP | detector_loss | 0.061069 | 0.005089 | 0.547639 | 1.113798 | 0.897959 | 0.993333 |
| Strict OP | proposed_phys_score | 0.061069 | 0.017812 | 0.399588 | 0.293321 | 0.758503 | 0.966667 |

## Case39 Stress Benchmark Clean Exact
| label | total_clean_sample | total_DDD_alarm | total_trigger_after_verification | trigger_rate | useless_mtd_rate | fail_per_alarm | stage_two_time_per_alarm | delta_cost_two_per_alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 7021 | 819 | 819 | 1.000000 | 0.116650 | 0.963370 | 1.362096 | 0.728746 |
| main | 7021 | 819 | 754 | 0.920635 | 0.107392 | 0.902320 | 0.643374 | 0.646918 |
| strict | 7021 | 819 | 676 | 0.825397 | 0.096283 | 0.809524 | 0.544753 | 0.550429 |

## Case39 Stress Benchmark Attack Exact
| label | overall_arr | protected_min_arr |
| --- | --- | --- |
| main | 0.993151 | 0.980000 |
| strict | 0.986301 | 0.980000 |

## Case39 Gate Comparator
| budget_label | score_name | trigger_rate | fail_per_alarm | stage_two_time_per_alarm | delta_cost_two_per_alarm | attack_retention_overall | strong_retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main OP | detector_loss | 0.925519 | 0.896215 | 1.385167 | 0.684659 | 1.000000 | 1.000000 |
| Main OP | proposed_phys_score | 0.925519 | 0.909646 | 0.720996 | 0.645570 | 0.993151 | 0.993289 |
| Strict OP | detector_loss | 0.831502 | 0.807082 | 1.167734 | 0.610034 | 0.989726 | 0.993289 |
| Strict OP | proposed_phys_score | 0.831502 | 0.816850 | 0.658331 | 0.553239 | 0.986301 | 0.993289 |

## Case39 Minimal Score Ablation
| regime | score_family | tau_valid | valid_clean_trigger_count | test_clean_trigger_count | test_attack_trigger_count | test_attack_overall_arr | test_attack_protected_min_arr | test_clean_backend_mtd_fail_count | test_clean_stage_two_time_per_alarm | test_clean_delta_cost_two_per_alarm | test_clean_fail_per_alarm | test_clean_fail_per_trigger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | angle_l2 | 0.013772 | 536.000000 | 737.000000 | 290.000000 | 0.993151 | 0.980000 | 722.000000 | 0.643374 | 0.630997 | 0.881563 | 0.979647 |
| main | angle_linf | 0.006390 | 535.000000 | 740.000000 | 289.000000 | 0.989726 | 0.980000 | 720.000000 | 0.885198 | 0.623076 | 0.879121 | 0.972973 |
| main | joint_angle_vmag_l2 | 0.014163 | 535.000000 | 735.000000 | 290.000000 | 0.993151 | 0.980000 | 720.000000 | 0.643374 | 0.638818 | 0.879121 | 0.979592 |
| main | detector_loss | 0.075536 | 0.000000 | 0.000000 | 257.000000 | 0.880137 | 0.918367 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| strict | angle_l2 | 0.014391 | 533.000000 | 721.000000 | 290.000000 | 0.993151 | 0.980000 | 707.000000 | 0.591687 | 0.586250 | 0.863248 | 0.980583 |
| strict | angle_linf | 0.007032 | 522.000000 | 698.000000 | 289.000000 | 0.989726 | 0.980000 | 682.000000 | 0.761588 | 0.618072 | 0.832723 | 0.977077 |
| strict | joint_angle_vmag_l2 | 0.014592 | 533.000000 | 720.000000 | 290.000000 | 0.993151 | 0.980000 | 706.000000 | 0.591687 | 0.586146 | 0.862027 | 0.980556 |
| strict | detector_loss | 0.734464 | 0.000000 | 0.000000 | 248.000000 | 0.849315 | 0.877551 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |

## Positioning
- case14 remains the main detailed evidence.
- case39 is an additional stress-test / limitation benchmark.
- Detector-loss gate retains slightly more attack alarms, while the recovery-aware physical score is more backend-aware in stage-two burden/cost.
- Do not claim robust end-to-end case39 backend success.

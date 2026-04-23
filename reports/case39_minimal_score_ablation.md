# Case39 Minimal Score Ablation

Validation-only selection policy:
- Main: overall ARR >= 0.90 and protected-group ARR >= 0.95
- Strict: overall ARR >= 0.85 and protected-group ARR >= 0.90

| regime | score_family | tau_valid | valid_clean_trigger_count | test_clean_trigger_count | test_attack_trigger_count | test_attack_overall_arr | test_attack_protected_min_arr | test_clean_backend_mtd_fail_count | test_clean_stage_two_time_per_alarm | test_clean_delta_cost_two_per_alarm | test_clean_fail_per_alarm | test_clean_fail_per_trigger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | angle_l2 | 0.013772 | 536 | 737 | 290 | 0.993151 | 0.980000 | 722 | 0.643374 | 0.630997 | 0.881563 | 0.979647 |
| main | angle_linf | 0.006390 | 535 | 740 | 289 | 0.989726 | 0.980000 | 720 | 0.885198 | 0.623076 | 0.879121 | 0.972973 |
| main | joint_angle_vmag_l2 | 0.014163 | 535 | 735 | 290 | 0.993151 | 0.980000 | 720 | 0.643374 | 0.638818 | 0.879121 | 0.979592 |
| main | detector_loss | 0.075536 | 0 | 0 | 257 | 0.880137 | 0.918367 | 0 | 0.000000 | 0.000000 | 0.000000 | nan |
| strict | angle_l2 | 0.014391 | 533 | 721 | 290 | 0.993151 | 0.980000 | 707 | 0.591687 | 0.586250 | 0.863248 | 0.980583 |
| strict | angle_linf | 0.007032 | 522 | 698 | 289 | 0.989726 | 0.980000 | 682 | 0.761588 | 0.618072 | 0.832723 | 0.977077 |
| strict | joint_angle_vmag_l2 | 0.014592 | 533 | 720 | 290 | 0.993151 | 0.980000 | 706 | 0.591687 | 0.586146 | 0.862027 | 0.980556 |
| strict | detector_loss | 0.734464 | 0 | 0 | 248 | 0.849315 | 0.877551 | 0 | 0.000000 | 0.000000 | 0.000000 | nan |

Notes:
- `angle_l2` is the current main score.
- `angle_linf` uses max abs angle recovery deviation on non-reference buses.
- `joint_angle_vmag_l2` uses sqrt(||dtheta||_2^2 + ||d|V|||_2^2).
- Burden metrics are reported per clean alarm, not per triggered alarm.
- Count columns are included to make failure denominators explicit.

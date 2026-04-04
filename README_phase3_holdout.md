# Phase 3 holdout protocol (cleaner train/val/test workflow)

This bundle is for a cleaner workflow after the earlier development rounds already used `mixed_bank_eval.npy` as feedback.

## What this protocol does

- `train_bank`: fit posterior/service/severity models
- `val_bank`: tune thresholds / adaptive-threshold / proposed scheduler, and pick regime
- `test_bank`: final holdout evaluation only

This removes the biggest leakage risk from the earlier workflow: choosing regimes on the same bank that later gets reported as the final result.

## Step 0: make a fresh, unseen holdout test bank

Use a **new** seed and offset that you have not used in earlier development.

Example:

```bash
python evaluation_mixed_timeline.py \
  --tau_verify -1 \
  --schedule "clean:120;att-1-0.2:60;clean:60;att-2-0.2:60;clean:60;att-3-0.3:60;clean:120" \
  --seed_base 20260411 \
  --start_offset 480 \
  --output metric/case14/mixed_bank_test_holdout.npy
```

## Step 1: validation-only regime ranking (does NOT touch test)

```bash
python select_regime_phase3_val.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --train_bank metric/case14/mixed_bank_fit.npy \
  --val_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --decision_step_group_list 1 2 \
  --busy_time_quantile_list 0.35 0.50 0.65 \
  --use_cost_budget_modes off on \
  --cost_budget_quantile_list 0.50 0.60 \
  --output metric/case14/phase3_val_regime_ranking.json
```

Pick **one** regime from this validation ranking.

## Step 2: final holdout test for the chosen regime

Example A: no cost budget

```bash
python evaluation_budget_scheduler_phase3_holdout.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --train_bank metric/case14/mixed_bank_fit.npy \
  --val_bank metric/case14/mixed_bank_eval.npy \
  --test_bank metric/case14/mixed_bank_test_holdout.npy \
  --slot_budget_list 1 2 \
  --decision_step_group 1 \
  --busy_time_quantile 0.50 \
  --output metric/case14/budget_scheduler_phase3_holdout_nocost.npy
```

Example B: with cost budget

```bash
python evaluation_budget_scheduler_phase3_holdout.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --train_bank metric/case14/mixed_bank_fit.npy \
  --val_bank metric/case14/mixed_bank_eval.npy \
  --test_bank metric/case14/mixed_bank_test_holdout.npy \
  --slot_budget_list 1 2 \
  --decision_step_group 1 \
  --busy_time_quantile 0.65 \
  --use_cost_budget \
  --cost_budget_window_steps 20 \
  --cost_budget_quantile 0.60 \
  --output metric/case14/budget_scheduler_phase3_holdout_cost.npy
```

## Important note

Do **not** keep changing code after looking at the holdout test result. If you change the method again, generate a new unseen holdout bank.

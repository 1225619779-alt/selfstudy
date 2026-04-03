
# Phase 3: consequence-aware hard-constrained backend scheduler

This phase does two things beyond phase2_hard:

1. It replaces the old `value_proxy ~= verify_score` shortcut with a **learned expected consequence term**.
2. It adds a **regime sweep** so you can find which overload regime best exposes the scheduling advantage.

## New idea

For each alarm job, fit:

- `p_hat(signal)` from clean/attack banks
- `tau_hat(signal)`, `cost_hat(signal)`, `fail_hat(signal)` from fit mixed bank
- `severity_hat(signal, attack)` from attack jobs in fit mixed bank

Then define

- `expected_consequence_hat = p_hat * severity_hat`  (default `--consequence_mode conditional`)

The proposed policy `proposed_ca_vq_hard` ranks pending alarms using:

- expected consequence
- clean-risk penalty `(1 - p_hat)`
- age / urgency
- predicted busy-time / cost / fail risk
- hard server capacity
- optional rolling hard cost budget

## Files

- `evaluation_budget_scheduler_phase3.py`
- `scheduler/calibration.py`
- `scheduler/policies_phase3.py`
- `sweep_regimes_phase3.py`

Copy them into your repo root as:
- `evaluation_budget_scheduler_phase3.py`
- `sweep_regimes_phase3.py`
- `scheduler/calibration.py`
- `scheduler/policies_phase3.py`

(If you already have a `scheduler/calibration.py`, back it up first or merge carefully.)

## Recommended first run

Use the same banks you already generated.

```bash
python evaluation_budget_scheduler_phase3.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --max_wait_steps 10 \
  --busy_time_quantile 0.50 \
  --output metric/case14/budget_scheduler_phase3_ca.npy
```

## Second run: add hard cost budget

```bash
python evaluation_budget_scheduler_phase3.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --max_wait_steps 10 \
  --busy_time_quantile 0.50 \
  --use_cost_budget \
  --cost_budget_window_steps 20 \
  --cost_budget_quantile 0.60 \
  --output metric/case14/budget_scheduler_phase3_ca_cost.npy
```

## Third run: sweep regimes

```bash
python sweep_regimes_phase3.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --decision_step_group_list 1 2 \
  --busy_time_quantile_list 0.35 0.50 0.65 \
  --use_cost_budget_modes off on \
  --cost_budget_quantile_list 0.50 0.60 \
  --output metric/case14/phase3_regime_sweep.json
```

## What to look for

Main fields in the compact summary:

- `weighted_attack_recall_no_backend_fail`
- `unnecessary_mtd_count`
- `queue_delay_p95`
- `average_service_cost_per_step`
- `pred_expected_consequence_served_ratio`

Key comparison targets:

- `threshold_verify_fifo`
- `threshold_expected_consequence_fifo`
- `adaptive_threshold_verify_fifo`
- `topk_expected_consequence`
- `static_expected_consequence_cost`
- `proposed_ca_vq_hard`

## Practical interpretation

Good sign:
- `proposed_ca_vq_hard` beats `threshold_expected_consequence_fifo` in recall by a visible margin
- while keeping `unnecessary_mtd_count` very low
- and without exploding `queue_delay_p95`

Bad sign:
- `topk_expected_consequence` and `static_expected_consequence_cost` stay almost identical
- `proposed_ca_vq_hard` still collapses in `slot_budget=1`
- or adding cost budget destroys the advantage completely

If that happens, the next step is to collect richer recovery-side consequence features from the repo and stop relying on `ang_no * ang_str` as the only evaluation-side severity.

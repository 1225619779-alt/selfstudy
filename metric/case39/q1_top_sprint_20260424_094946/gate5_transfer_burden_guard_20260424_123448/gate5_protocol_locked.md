# Gate 5 Locked Protocol

This protocol was written before generating Gate 5 calibration/confirm results.

## Primary Baselines

- `source_frozen_transfer`.
- `topk_expected_consequence`.
- `winner_replay`.
- `native_safeguarded_retune`.
- `native_unconstrained_retune`.

## Diagnostic Extension

- Name: `transfer_regularized_burden_guard`.
- TRBG-source: source-frozen score plus burden guard selected only on case14 source train/val.
- TRBG-native-burden: source-frozen score plus burden guard selected only on explicit native case39 train/val.
- TRBG-source tests fully source-frozen burden control.
- TRBG-native-burden tests whether native information is useful only for a low-dimensional burden guard, without full native retuning.

## Allowed Guard Inputs

- Predicted backend_fail probability.
- Predicted service_cost.
- Predicted service_time.
- Predicted attack posterior.
- Predicted consequence score already used by source-frozen.

Forbidden: actual backend_fail, actual recover_fail, actual test labels, and test holdout aggregate results.

## Guard Construction

- Keep original source-frozen score weights and regime unchanged.
- `Bhat = z(pred_fail_prob) + z(pred_service_cost) + z(pred_service_time)`.
- `S_guard = S_source - alpha * Bhat`.
- alpha grid: `{0.0, 0.1, 0.25, 0.5, 1.0, 2.0}`.
- fail_cap_quantile grid: `{0.80, 0.90, 0.95, 1.00}`.
- If fail_cap_quantile < 1.00, jobs above the calibration pred_fail_prob quantile are ineligible while uncapped queue candidates exist; this is secondary diagnostic only.

## Candidate Grid

- Primary soft-guard grid: alpha only, fail_cap_quantile = 1.00.
- Secondary diagnostic grid: alpha x fail_cap_quantile.
- All candidates are reported.

## Dev Selection Rule

- For each calibration mode, choose exactly one primary candidate before test confirm.
- Among candidates whose dev recall is at least 90% of source_frozen dev recall, choose the candidate with the lowest dev backend_fail.
- If no candidate satisfies 90% recall retention, choose alpha=0.0 and mark `guard_failed_on_dev=true`.
- Tie break: lower cost, then lower unnecessary, then lower alpha.
- One alpha/cap pair is selected per calibration mode and reused for B=1 and B=2.

## Confirm Success Criteria

- Strong success: backend_fail decreases by at least 15%, recall remains at least 90%, cost does not increase, and the point is not Pareto-dominated by source_frozen/topk/winner in recall-cost or recall-backend_fail.
- Moderate success: recall remains higher than topk, backend_fail or cost decreases versus source_frozen, and the point is Pareto-efficient in at least one plane.
- Failure: recall falls below topk without substantial burden reduction, backend_fail/cost do not improve, or results depend on test-set cherry-picking.

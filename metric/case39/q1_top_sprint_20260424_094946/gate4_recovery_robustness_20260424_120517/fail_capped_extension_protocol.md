# Fail-capped Source-frozen Extension Protocol

Status: preregistered design only. Do not execute until Gate 4 robustness is reviewed.

## Candidate Variant

- Name: `source_frozen_fail_capped`.
- Keep source-frozen score weights, winner variant, budgets, Wmax, and regime unchanged.
- Add only a pre-registered fail/burden gate before service admission.
- Gate inputs allowed: predicted fail probability, predicted service cost, predicted service time.
- No true backend_fail, recover_fail, or test-holdout labels may enter deployment scoring.

## Candidate Cap Grid

- `pred_fail_prob_cap`: `[0.50, 0.60, 0.70, 0.80]`.
- `pred_service_cost_cap_quantile`: `[0.75, 0.85, 0.95, None]` computed on selection split only.
- `pred_service_time_cap_quantile`: `[0.75, 0.85, 0.95, None]` computed on selection split only.
- `burden_penalty_lambda`: `[0.00, 0.05, 0.10, 0.20]` if using penalty instead of hard cap.

## Selection Split

- Primary selection: source train/val for transfer-preserving variant.
- Optional explicit native-val selection: allowed only if reported as a separate native-val-selected diagnostic.
- The frozen 8 test holdouts cannot select cap, penalty, or winner.

## Fixed Endpoints

- Primary endpoints: recall, backend_fail, cost, unnecessary, served_ratio, delay_p95.
- Scheduler-conditioned endpoints: served_attack_mass, backend_success_attack_mass, clean service count.
- Robustness endpoints: product_proxy, additive_proxy, backend_success_proxy, recovery_aware_proxy, success_burden_proxy.

## Success Criteria

- Primary: backend_fail decreases by at least 15% or 20% versus source-frozen while recall drops by no more than 5% or 10%.
- Secondary: not dominated by source-frozen, topk_expected_consequence, or winner_replay on recall-backend_fail and recall-cost Pareto frontiers.
- All cap-grid outcomes must be shown; no cherry-picking.

## Promotion Rule

- v2 main experiment only if it satisfies primary criteria on selection split and confirms on all 8 frozen holdouts without test-selected tuning.
- Future work only if it reduces burden but fails recall retention or Pareto non-domination.

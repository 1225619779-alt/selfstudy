# Gate 6 Locked Protocol

This protocol is written before blind confirm results are generated.

- Locked method: `TRBG-source`.
- alpha = `1.0`.
- fail_cap_quantile = `1.00`.
- Source-frozen weights and regime unchanged.
- No test selection; new blind holdouts are confirm-only.
- budgets = `1, 2`.
- Wmax = `10`.

## Primary Baselines

- `source_frozen_transfer`.
- `TRBG-source locked`.
- `topk_expected_consequence`.
- `winner_replay`.
- `native_safeguarded_retune`.
- `native_unconstrained_retune`.

## Optional Diagnostic

- `TRBG-native-burden locked`, alpha = `2.0`, cap = `1.00`.

## Primary Endpoints

- recall.
- backend_fail.
- cost.
- recover_fail.
- unnecessary.
- served_attack_mass.
- backend_success_attack_mass.

## Validation Criteria

- Primary: TRBG-source retains at least 95% of source_frozen recall averaged over B=1/B=2, reduces backend_fail by at least 5%, does not increase cost, and remains above topk recall.
- Stronger: backend_fail reduction at least 10% with recall retention at least 95%.
- Failure: recall falls below topk, backend_fail/cost do not improve, or any test-set choice occurs.

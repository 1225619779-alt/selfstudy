# Gate9 Protocol Locked

This protocol is locked before any Gate9 confirm result is generated.

## Objective

Gate9 validates a simpler fail-only transfer guard suggested by Gate8 component diagnostics, without using Gate8 confirm results as a selector. The goal is to determine whether this guard can replace TRBG-source as the v2 main method and whether it fixes 14-bus backward compatibility.

## Candidate Definitions

- Primary candidate: `TRFG-source` = Transfer-Regularized Fail Guard, source calibrated.
- Secondary diagnostic candidate: `TRFG-native-fail` = source-frozen score plus fail-only guard calibrated on explicit native case39 train/val.
- Locked formula: `S_fail = S_source - alpha * z(pred_fail_prob)`.
- `alpha=0.0` must reproduce `source_frozen_transfer` within numerical tolerance.

## Forbidden Inputs

The guard must not use:

- actual backend_fail;
- actual recover_fail;
- actual test labels;
- test holdout aggregate results;
- Gate8 confirm outcome as a selector.

## Alpha Grid

`alpha in {0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0}`.

## Calibration Sources

- `TRFG-source` normalization and selection use case14 source train/val only.
- `TRFG-native-fail` normalization and selection use explicit native case39 train/val only and is diagnostic only.
- Gate1-Gate8 original case39 eight holdouts are secondary diagnostic only, not primary confirmatory evidence.

## Source-Side Selection Rule

On case14 source dev/val, among alpha candidates satisfying:

- dev recall retention vs original proposed safeguarded is at least `0.98`;
- dev unnecessary is no more than original proposed unnecessary plus `5%`;
- dev cost does not increase by more than `5%`;

select the alpha with the lowest dev backend_fail. If backend_fail is unavailable or too sparse, select lowest dev cost, then lowest unnecessary. If no alpha satisfies the constraints, select `alpha=0.0` and mark `fail_guard_failed_source_dev=true`.

## Confirmatory Validation

Primary new validation uses four new 240-step reduced fresh case39 physical-solver banks with new seeds, offsets, and schedules that are not identical to Gate8 reduced banks. These banks are not calibration data and are not used for alpha selection.

Fixed confirm methods:

- `source_frozen_transfer`;
- `TRBG-source`, alpha `1.0`, cap `1.00`;
- `TRFG-source`, alpha selected from source dev;
- `TRFG-native-fail`, alpha selected from native dev, diagnostic only;
- `topk_expected_consequence`;
- `winner_replay`;
- `native_safeguarded_retune`;
- `native_unconstrained_retune`.

## Fresh Success Criteria

Primary success:

- TRFG-source recall retention vs source_frozen_transfer is at least `0.95`;
- backend_fail reduction vs source_frozen_transfer is at least `0.08`;
- cost does not increase;
- unnecessary does not materially increase;
- recall is not materially below topk_expected_consequence;
- TRFG-source is not Pareto-dominated by source_frozen, topk, winner_replay, or TRBG-source in recall-cost and recall-backend_fail planes.

Strong success:

- recall retention is at least `0.97`;
- backend_fail reduction is at least `0.10`;
- cost decreases;
- TRFG-source also passes 14-bus compatibility.

Failure:

- 14-bus compatibility fails badly;
- or fresh recall falls below topk by a material margin;
- or backend_fail/cost do not improve;
- or any test-set selection occurs.

## Interpretation Boundaries

- No case39 result may be written as native case39 success.
- Fail-only guard is not a recovery-aware method.
- Physical sanity, if available, is a post-hoc deviation sanity check and not an OPF consequence label.


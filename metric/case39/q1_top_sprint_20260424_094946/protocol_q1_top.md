# Q1-top Shadow Revision Evidence Sprint Protocol

## Scope

This sprint is a shadow evidence pack for major revision or resubmission planning. It does not overwrite submitted-result artifacts, does not open a new scheduling family, and does not relabel current case39 as native larger-system success.

All new outputs are isolated under:

`metric/case39/q1_top_sprint_20260424_094946/`

## Separation Rule

- Source-frozen transfer evidence must use explicit case14 train/validation banks and case39 target clean/attack/test banks.
- Full-native case39 evidence must use explicit native case39 train/validation banks and case39 clean/attack/test banks.
- Manifests and directories must make the source-frozen and full-native regimes distinguishable without relying on canonical symlink interpretation.

## Primary Endpoints

The primary endpoints are fixed before any Gate 2+ experiment:

- `recall`
- `unnecessary`
- `cost`
- `served_ratio`
- `backend_fail`
- `delay_p95`

## Case39 Scheduler-conditioned Endpoints

Scheduler-conditioned endpoints are fixed as:

- `verified_attack_mass`
- `admitted_attack_mass`
- `served_attack_mass`
- `backend_success_attack_mass`

## Consequence Labels

Post-hoc consequence labels are fixed as:

- `product_proxy = max(ang_no, 0) * max(ang_str, 0)`
- `additive_proxy = max(ang_no, 0) + max(ang_str, 0)`
- `backend_success_proxy = product_proxy * 1[backend_fail == 0]`
- `recovery_aware_proxy = product_proxy * 1[recover_fail == 0]`, if `recover_fail` is available
- `burden_proxy = normalized service_cost + normalized service_time + backend_fail penalty + recover_fail penalty`, if the required fields are available

## Fixed Evaluation Design

- Budgets: `B = 1, 2`
- Maximum wait: `Wmax = 10`
- Holdouts: frozen 8 existing case39 holdouts
- Stage comparison set: source-frozen transfer, winner replay, anchored retune, native safeguarded retune, native unconstrained retune, phase3 proposed, phase3 oracle upgrade, top-k expected consequence, incumbent queue-aware, and static threshold where existing implementations/artifacts allow direct reuse

## No Cherry-picking Rule

All Gate 2+ results are reportable within this sprint, including negative, null, or failed results. If a method fails to run, the failure reason must be recorded in the relevant summary rather than omitted. Operating points must not be selected using test-holdout outcomes after the fact.

## Stopping Gates

- Gate 0 ends after this protocol is written.
- Gate 1 ends after provenance manifests, hashes, anti-write checks, and readiness status are written.
- Gate 2, Gate 3, Gate 4, and Gate 5 must not start until the previous gate has been reviewed.

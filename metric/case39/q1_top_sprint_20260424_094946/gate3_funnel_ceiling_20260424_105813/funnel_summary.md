# Case39 Funnel Summary

Definitions: `verified` means DDD/scheduler-visible alarms; `queued` equals accepted into the scheduler queue after any threshold gate. Diagnostic rows do not relabel source-frozen as native success.

| variant | B | detector | served_mass | backend_success_mass | sched_recall | absolute | backend_fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_safeguarded_retune | 1 | 0.9711 | 5.6500 | 1.7188 | 0.0155 | 0.0150 | 7.3750 |
| native_safeguarded_retune | 2 | 0.9711 | 6.4500 | 1.7562 | 0.0156 | 0.0152 | 8.5000 |
| native_unconstrained_retune | 1 | 0.9711 | 0.5375 | 0.0000 | 0.0000 | 0.0000 | 0.6250 |
| native_unconstrained_retune | 2 | 0.9711 | 30.5312 | 9.5438 | 0.0833 | 0.0809 | 46.2500 |
| source_frozen_transfer | 1 | 0.9711 | 44.5375 | 10.2937 | 0.0917 | 0.0889 | 102.7500 |
| source_frozen_transfer | 2 | 0.9711 | 67.8125 | 19.9687 | 0.1772 | 0.1720 | 130.1250 |
| topk_expected_consequence | 1 | 0.9711 | 40.1125 | 8.2938 | 0.0723 | 0.0702 | 94.7500 |
| topk_expected_consequence | 2 | 0.9711 | 60.4750 | 15.0812 | 0.1322 | 0.1283 | 124.0000 |
| winner_replay | 1 | 0.9711 | 32.1687 | 7.0688 | 0.0629 | 0.0610 | 77.8750 |
| winner_replay | 2 | 0.9711 | 59.9187 | 14.2687 | 0.1264 | 0.1226 | 117.8750 |

## Answers

1. Largest recall loss is `verified -> served`; source-frozen average detector ceiling is `0.9711`.
2. B=1 and B=2 share the upstream detector ceiling, but B=1 has stronger capacity/queue pressure while B=2 shifts more of the loss to backend success.
3. Source-frozen's advantage versus safeguarded, unconstrained, anchored, and winner replay occurs mainly after verification: it serves more attack mass before backend filtering.
4. Full-native local retune collapses primarily at service/admission-score operating point: verified attacks are admitted to the queue but little attack mass is actually served, especially for safeguarded and B=1 unconstrained.
5. Backend_fail burden is dominated by served attack jobs rather than served clean jobs; clean backend failures are small relative to attack backend failures in these traces.

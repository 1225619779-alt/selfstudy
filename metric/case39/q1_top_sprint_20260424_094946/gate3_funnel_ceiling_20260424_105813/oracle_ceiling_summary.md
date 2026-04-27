# Oracle Ceiling Summary

These diagnostic oracles use hindsight truth labels and are ceilings only, not deployable baselines and not new scheduler families.

| B | detector | capacity | backend_success | source_gap_capacity | source_gap_backend | native_best_gap_capacity | native_best_gap_backend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9711 | 0.3870 | 0.1108 | 0.2981 | 0.0219 | 0.2588 | 0.0057 |
| 2 | 0.9711 | 0.6264 | 0.1904 | 0.4544 | 0.0183 | 0.4039 | 0.0283 |

- `verified-oracle top severity` prioritizes true product severity within verified alarms.
- `backend-success oracle` uses true backend_fail only to estimate an upper bound.
- `capacity-only oracle` estimates how much verified attack mass can be served under the same arrival, B, and Wmax.

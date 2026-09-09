# Numerical records for the KBS V8 manuscript

**Selective Backend Service for Grid-Defense Alarms: Gating, Ordering, and Planning**  
Jing Zhang and Wenbo Pang

This directory contains 21 machine-readable result tables and supporting analysis records accompanying manuscript V8 and its numerical supplement, Supplementary Data S1. This is a data-only addition to the repository. The repository's other directories are historical materials; their presence does not mean that all versions or claims therein are part of this manuscript's data release.

## Contents

| Directory | Files | Content |
| --- | ---: | --- |
| `Supplementary_Data_S1` | 9 | Replay inventory, comparison means, paired and pattern differences, validation selection, component results, and offline bounds |
| `Supplementary_Planning_Comparison` | 3 | Two saved planning-comparison tables and their figure-data JSON |
| `Supplementary_Event_Diagnostic` | 7 | Clean-side arm and endpoint records, paired differences, recovery summaries, common-tail equivalence, frozen-policy readout, and timing |
| `Supplementary_Objective_Sensitivity` | 2 | Ten-row post-hoc objective-sensitivity table and corresponding saved-comparison means |

The 21 CSV/JSON files retain the exact bytes of the numerical supplement. `SHA256SUMS.txt` records the SHA-256 hashes of those files and this README. The commit-specific URL identifies the released version independently of later changes to the repository's main branch.

## Reading the records

- Coverage fractions are multiplied by 100 to obtain percentages. Differences between percentage-valued coverage rates are percentage points.
- `P02_ordering_D_minus_S_16.csv` stores D minus S. A plot described as S minus D uses the opposite sign. Component differences are variant minus Safeguarded.
- `B` denotes concurrent backend capacity. The internal baseline-family identifier `family=B` is not a second capacity variable.
- S or S0 denotes Safeguarded; D or ELIGIBLE_FIFO is the qualified FIFO comparison; A_Q25_FIFO is A-Q25; ATC_START denotes deadline-aware dispatch; ROLLING_CURRENT_QUEUE_NONDELAY denotes current-queue rolling planning.
- Full busy-step occupancy includes completion beyond the arrival horizon. Waiting D95 is summarized from per-replay percentiles, not from a pooled request sample. Undefined waiting values for empty served sets are not zeros.
- Reported replay means weight replays equally. The eight historical replays overlap and are not independent source replications.
- Offline feasible schedules and upper bounds use retrospective information. They concern the specified recorded objective and resource caps, not confidence intervals or online access to future outcomes.

## Scope of the additional analyses

The saved planning comparison uses 32 complete historical traces: two methods at two capacities over eight replays. Short development traces do not enter these eight-replay means. Equal mean occupancy is not equal occupancy on every replay, and identical served sets need not have identical start times.

The event diagnostic covers two exposed, event-triggered clean development cases. Matched endpoint rows compare no service, earliest application, and delayed application of the same candidate. These are clean-side-effect records, not attack-conditioned service-benefit or independent-source confirmation. Logical endpoint rows are not necessarily distinct new physical calls. Snapshot OPF objective differences are not energy bills, and timing fields are not measured deployment latency.

The objective-sensitivity table is an algebraic, post-hoc analysis of existing count-based and proxy-weighted coverage. It does not change policies, retune thresholds, or introduce new evaluation workloads.

`P02_validation_freeze.json` retains historical code hashes and local path, size, and timestamp metadata as provenance. Those local paths are identifiers, not paths that readers must possess. No provenance fields or scientific values were silently removed from the numerical records.

## Release boundary

This release provides numerical result tables and selected analysis/diagnostic records. It is not a complete release of raw candidate banks, model checkpoints, archived physical simulators, or all execution traces. It does not establish that the full experiment pipeline can be reproduced solely from these 21 files. The manuscript and its Supplementary Material define the analyses and their evidential scope. No new experiments were run to create this data release.

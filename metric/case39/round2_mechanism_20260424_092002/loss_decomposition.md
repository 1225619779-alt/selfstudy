# Loss Decomposition

The scheduler denominator is DDD-alarm attack severity. The raw mixed banks also expose attack steps that never became DDD alarm jobs; those are reported separately as not-alarm loss.

| stage | slot | not-alarm severity | expired jobs | horizon unserved | backend_fail | served_attack | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchored_retune | 1 | 3.2937 | 194.5000 | 0.2500 | 10.3750 | 12.8750 | 0.0175 |
| anchored_retune | 2 | 3.2937 | 150.8750 | 0.2500 | 48.1250 | 55.5000 | 0.0543 |
| native_safeguarded_retune | 1 | 3.2937 | 197.7500 | 0.2500 | 7.3750 | 9.7500 | 0.0155 |
| native_safeguarded_retune | 2 | 3.2937 | 196.6250 | 0.2500 | 8.5000 | 10.8750 | 0.0156 |
| native_unconstrained_retune | 1 | 3.2937 | 206.8750 | 0.2500 | 0.6250 | 0.6250 | 0.0000 |
| native_unconstrained_retune | 2 | 3.2937 | 148.7500 | 0.2500 | 46.2500 | 58.5000 | 0.0833 |
| source_frozen | 1 | 3.2937 | 89.5000 | 0.0000 | 102.7500 | 104.7500 | 0.0917 |
| source_frozen | 2 | 3.2937 | 49.8750 | 0.0000 | 130.1250 | 140.3750 | 0.1772 |
| winner_replay | 1 | 3.2937 | 118.3750 | 0.1250 | 77.8750 | 81.3750 | 0.0629 |
| winner_replay | 2 | 3.2937 | 68.7500 | 0.1250 | 117.8750 | 127.3750 | 0.1264 |

## Answers

- The largest absolute loss before scheduling is the attack mass that never becomes a DDD-alarm job. Within the scheduler-visible jobs, the dominant B=1 loss is queue pressure: admitted jobs remain queued until TTL/horizon expiration rather than being immediately served.
- Source-frozen keeps materially higher recall than anchored/protected local retunes because it admits and serves more attack jobs under the frozen case14 operating point, accepting higher clean-service and cost burden.
- Winner replay uses native train/val with the source winner and falls between source-frozen and local retunes, which indicates that calibration-bank shift itself explains part of the recall drop.
- Anchored and safeguarded local retunes are conservative: fewer served clean jobs and lower cost, but also sharply lower served attack severity.
- Native unconstrained retune partially recovers B=2 recall but remains unstable at B=1, so B=1 and B=2 bottlenecks are not identical. B=1 is capacity/queue constrained; B=2 is more sensitive to tuned operating point and proxy ranking.

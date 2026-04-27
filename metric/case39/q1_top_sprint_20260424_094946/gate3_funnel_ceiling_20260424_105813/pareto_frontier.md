# Pareto Frontier

| variant | B | recall | cost | backend | rc | rb |
| --- | --- | --- | --- | --- | --- | --- |
| native_safeguarded_retune | 1 | 0.0155 | 0.0371 | 7.3750 | True | True |
| native_unconstrained_retune | 1 | 0.0000 | 0.0004 | 0.6250 | False | False |
| source_frozen_transfer | 1 | 0.0917 | 0.3881 | 102.7500 | True | True |
| topk_expected_consequence | 1 | 0.0723 | 0.3528 | 94.7500 | True | True |
| winner_replay | 1 | 0.0629 | 0.2808 | 77.8750 | True | True |
| native_safeguarded_retune | 2 | 0.0156 | 0.0404 | 8.5000 | True | True |
| native_unconstrained_retune | 2 | 0.0833 | 0.1870 | 46.2500 | True | True |
| source_frozen_transfer | 2 | 0.1772 | 0.5575 | 130.1250 | True | True |
| topk_expected_consequence | 2 | 0.1322 | 0.4960 | 124.0000 | True | True |
| winner_replay | 2 | 0.1264 | 0.4570 | 117.8750 | True | True |

- Source-frozen is not dominated by top-k or winner replay because it has higher recall, but it is a high-burden point.
- Top-k is the closest full-native high-recall point and is more burden-efficient.
- Native safeguarded is best read as low-cost low-service low-recall, not as an effective operating point.

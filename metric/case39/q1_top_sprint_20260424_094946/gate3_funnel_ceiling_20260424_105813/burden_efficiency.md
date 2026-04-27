# Burden-efficiency

| variant | B | recall/cost | recall/backend | attack_precision | backend_success_attack |
| --- | --- | --- | --- | --- | --- |
| native_safeguarded_retune | 1 | 0.4184 | 0.0021 | 1.0000 | 0.2436 |
| native_safeguarded_retune | 2 | 0.3865 | 0.0018 | 1.0000 | 0.2184 |
| native_unconstrained_retune | 1 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| native_unconstrained_retune | 2 | 0.4454 | 0.0018 | 0.9957 | 0.2137 |
| source_frozen_transfer | 1 | 0.2362 | 0.0009 | 0.8858 | 0.1468 |
| source_frozen_transfer | 2 | 0.3179 | 0.0014 | 0.8892 | 0.1968 |
| topk_expected_consequence | 1 | 0.2049 | 0.0008 | 0.8618 | 0.1304 |
| topk_expected_consequence | 2 | 0.2665 | 0.0011 | 0.8798 | 0.1678 |
| winner_replay | 1 | 0.2240 | 0.0008 | 0.9118 | 0.1382 |
| winner_replay | 2 | 0.2766 | 0.0011 | 0.9172 | 0.1639 |

- Source-frozen buys recall with materially higher cost and backend_fail.
- Top-k is the closest full-native high-recall point and is less burdensome, but it remains below source-frozen recall.
- Native safeguarded is a low-cost, low-service, low-recall point rather than an effective operating point.

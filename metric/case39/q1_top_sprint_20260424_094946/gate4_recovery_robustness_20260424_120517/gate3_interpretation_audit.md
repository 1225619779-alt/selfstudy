# Gate 3 Interpretation Audit

| status | wording |
| --- | --- |
| accepted | Detector ceiling = 0.9711 means upstream detector is not the main bottleneck. |
| corrected | The main bottleneck should be stated as post-verification service plus backend-success loss, not detector ceiling. |
| accepted | Source-frozen is close to backend-success oracle but far from capacity oracle. |
| corrected | Matched-burden ordering advantage primarily holds against topk_expected_consequence. |
| corrected | Against winner_replay, source-frozen is not stably ahead under B=1 matched-burden diagnostics. |
| corrected | Topk has lower absolute burden, but should not be called more burden-efficient without qualification because recall/cost and recall/backend ratios do not uniformly beat source-frozen. |
| accepted | Source-frozen is a high-recall/high-burden Pareto operating point, not a dominant winner. |
| rejected | Any wording that treats current case39 as native success or larger-system native success. |

Corrected conclusion: Gate 3 supports `transfer regularization mechanism + backend stress-test limitation`, not `native success`.

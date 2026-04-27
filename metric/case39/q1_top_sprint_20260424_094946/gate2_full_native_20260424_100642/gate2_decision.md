# Gate 2 Decision

- full-native local retune still collapses: `True`
- any full-native operating point beats source-frozen transfer on recall: `False`
- best full-native B=1 by post-hoc recall: `topk_expected_consequence` recall `0.0723`
- best full-native B=2 by post-hoc recall: `topk_expected_consequence` recall `0.1322`
- source advantage: source-frozen has higher recall and served attack mass, but it also carries higher cost and backend-fail burden.
- source burden versus the post-hoc best full-native recall baseline: `{'B1_cost_delta_vs_best': 0.03533870109272641, 'B2_cost_delta_vs_best': 0.06147868867723577, 'B1_clean_delta_vs_best': -1.25, 'B2_clean_delta_vs_best': 0.0, 'B1_backend_fail_delta_vs_best': 8.0, 'B2_backend_fail_delta_vs_best': 6.125}`; clean service is not consistently higher versus top-k.
- full-native result supports Q1-level native mechanism claim: `False`
- main gap: Full-native canonical route is explicit and runnable, but full-native operating points do not beat source-frozen transfer; evidence supports transfer regularization/stress behavior, not native success.
- recommend Gate 3 funnel/ceiling: `True`

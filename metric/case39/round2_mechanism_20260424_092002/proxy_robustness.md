# Proxy Robustness

This is post-hoc re-scoring under fixed service decisions. It does not change scheduler decisions.

| proxy | slot | ordering by oracle proxy recall |
|---|---:|---|
| additive_proxy | 1 | source_frozen (0.4623) > winner_replay (0.3474) > anchored_retune (0.0629) > native_safeguarded_retune (0.0520) > native_unconstrained_retune (0.0039) |
| additive_proxy | 2 | source_frozen (0.6641) > winner_replay (0.5931) > native_unconstrained_retune (0.2884) > anchored_retune (0.2498) > native_safeguarded_retune (0.0591) |
| backend_success_weighted_proxy | 1 | source_frozen (0.2998) > winner_replay (0.2015) > anchored_retune (0.0564) > native_safeguarded_retune (0.0503) > native_unconstrained_retune (0.0000) |
| backend_success_weighted_proxy | 2 | source_frozen (0.5600) > winner_replay (0.3980) > native_unconstrained_retune (0.2612) > anchored_retune (0.1682) > native_safeguarded_retune (0.0502) |
| burden_proxy | 1 | source_frozen (0.5526) > winner_replay (0.4353) > anchored_retune (0.0686) > native_safeguarded_retune (0.0541) > native_unconstrained_retune (0.0039) |
| burden_proxy | 2 | source_frozen (0.7348) > winner_replay (0.6746) > native_unconstrained_retune (0.3065) > anchored_retune (0.2971) > native_safeguarded_retune (0.0620) |
| current_product_proxy | 1 | source_frozen (0.4019) > winner_replay (0.2944) > anchored_retune (0.0598) > native_safeguarded_retune (0.0515) > native_unconstrained_retune (0.0045) |
| current_product_proxy | 2 | source_frozen (0.6128) > winner_replay (0.5390) > native_unconstrained_retune (0.2712) > anchored_retune (0.2203) > native_safeguarded_retune (0.0593) |
| recovery_aware_proxy | 1 | source_frozen (0.4027) > winner_replay (0.2914) > anchored_retune (0.0601) > native_safeguarded_retune (0.0514) > native_unconstrained_retune (0.0029) |
| recovery_aware_proxy | 2 | source_frozen (0.6140) > winner_replay (0.5363) > native_unconstrained_retune (0.2714) > anchored_retune (0.2188) > native_safeguarded_retune (0.0581) |

## Answers

- Stage ordering is broadly stable for source-frozen versus safeguarded/anchored local retunes under product, additive, backend-success, and recovery-aware proxies.
- Source-frozen remains ahead of local protected retune under the tested proxy recalls, but the exact gap is proxy dependent.
- `phase3_oracle_upgrade` is a reasonable operating point when the goal is lower clean-service/cost than phase3/topk at similar source-frozen recall; it is not a fully native case39 success proof.
- Conclusions about `physical consequence` remain weak because the current scheduler target is still proxy-based. The burden proxy is the most direct warning that cost/time/fail penalties can change interpretation.
- `recover_fail` is present in mixed banks and can be used in the next version, but it is not currently in the scheduler objective or primary recall denominator.

# Gate6 Decision Reaudit

- Strictly locked/no-test-selection: `True`, per `gate6_decision.md` and protocol.
- New blind banks source: `deterministic_recombined_stress_banks`; file log confirms deterministic recombined case39 blind-v3 stress banks.
- New blind results were not used to tune alpha/cap; TRBG-source remained locked at alpha=1.0, cap=1.00.
- Gate6 TRBG-source recall retention: `0.9666`.
- Gate6 backend_fail reduction: `0.1140`.
- Gate6 cost delta: `-0.0722`.
- Gate6 recover_fail delta: `-0.0625`.
- Primary criterion reached: `True`.
- Stronger criterion reached: `True`.
- Fresh physical-solver validation: `False`.
- v2 main-text candidate support: `Yes, if described as locked burden guard with recombined stress replication, not fresh blind validation`.

## Source Decision Excerpt

```text
# Gate 6 Decision

1. Gate 6 was strictly locked and no-test-selection: `True`.
2. Gate 5 W/L/T direction needed correction: `True`.
3. TRBG-source replicated on new blind holdouts under primary criterion: `True`.
4. Recall retention vs source-frozen: `0.9666`.
5. Backend_fail reduction vs source-frozen: `0.1140`.
6. Cost change vs source-frozen: `-0.0722`.
7. Recover_fail change vs source-frozen: `-0.0625`.
8. Recall vs topk: above=`True`.
9. Pareto status: recall-cost not dominated `True`, recall-backend not dominated `True`.
10. Upgrade from moderate internal success to robust v2 main method: `True`; stronger criterion met `True`.
11. If not upgraded, TRBG-source should be appendix/future work; if upgraded, it is main text as locked burden guard, not native success.
12. Evidence strong enough to enter manuscript rewrite pack: `True`.
```

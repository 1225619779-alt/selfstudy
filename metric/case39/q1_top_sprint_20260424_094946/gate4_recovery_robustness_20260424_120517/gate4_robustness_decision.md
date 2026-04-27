# Gate 4 Robustness Decision

1. Source-frozen remains a high consequence-recall / high burden point across product, additive, backend-success, recovery-aware, and success-burden diagnostics.
2. Source-frozen relative to topk is stable on high-is-better consequence proxies: `False`.
3. Source-frozen relative to winner_replay is stable on high-is-better consequence proxies: `False`; interpret winner comparisons more cautiously than topk comparisons.
4. Native local retune collapse persists across consequence proxies: `True`.
5. `recovery_aware_proxy` does not create a new winner; it mainly validates that the ordering is not an artifact of product severity alone.
6. `burden_proxy` weakens any dominant-winner claim for source-frozen: `True`.
7. Current paper should not say fully recovery-aware or physical-consequence-aware as a method claim; safe wording is `proxy-consequence-guided` with recovery/burden robustness diagnostics.
8. Main text: product/additive/backend-success/recovery-aware robustness and high-burden Pareto interpretation. Appendix: raw recovery-field audit, pure burden-proxy ranks, and fail-capped extension protocol.
9. `recover_fail` is field-supported enough for a main-text robustness check, but still should not be framed as a newly trained recovery-aware method.

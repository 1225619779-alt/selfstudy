# Gate9 Final Decision

## Summary

Gate9 was executed as a pre-registered fail-only guard validation. It did not justify replacing TRBG-source as the v2 main method.

TRFG-source selected `alpha=0.25` from case14 source train/val only. `alpha=0.0` exactly reproduced the alpha-zero source/original proposed behavior in the source-dev check. Gate8 confirm results and Gate1-Gate8 case39 holdouts were not used to select alpha.

## Direct Answers

1. Gate9 was strictly no-test-selection: yes; alpha selection used source dev for TRFG-source and explicit native train/val for the diagnostic TRFG-native-fail.
2. Alpha=0 reproduced source-frozen/source proposed behavior: yes, `alpha0_exact_reproduces_original_proposed=true`.
3. TRFG-source selected alpha: `0.25`.
4. TRFG-source passed 14-bus compatibility: no, it failed the strict unnecessary threshold.
5. TRFG-source is much less unnecessary than TRBG-source on 14-bus: yes, unnecessary is lower than TRBG-source by `16.125` at B=1 and `31.5` at B=2.
6. Fresh validation recall retention vs source-frozen: `0.7500`.
7. Fresh backend_fail reduction vs source-frozen: `0.1457`.
8. Fresh cost/recover_fail/unnecessary directions vs source-frozen: cost `-0.0581`, recover_fail `-0.2500`, unnecessary `-10.6250`.
9. TRFG-source recall was above or close to topk on average: average recall delta vs topk is `+0.0494`, but this is not enough to overcome the failed source-retention criterion.
10. TRFG-source Pareto status: not stable enough; it is not Pareto-efficient in both required planes/budgets and `trfg_not_pareto_dominated_in_at_least_one_plane_each_budget=false`.
11. TRFG-source can replace TRBG-source as v2 main method: no.
12. If not replacing TRBG-source, TRFG-source should be appendix diagnostic or future work, not the main method.
13. Physical sanity is only partially available: state/residual and backend/service fields exist, but OPF/branch-flow/voltage-deviation consequence fields are not available without new logging.
14. The release_candidate removes ambiguous canonical case39 fit/eval symlinks from the candidate structure and explicitly separates transfer/native/Q1 sprint evidence.
15. Formal v2 rewrite can proceed, but Gate9 should be written as a negative/diagnostic hardening result rather than a method replacement.

## Interpretation

TRFG-source is attractive because it fixes much of the TRBG-source 14-bus unnecessary explosion and reduces fresh backend/cost burden versus source-frozen. However, the fresh recall-retention failure is decisive under the locked protocol. The result is useful evidence for reviewers because it shows the team did not opportunistically swap methods after Gate8; the simpler guard was pre-registered and tested, then rejected as a main-method replacement.

## Recommendation

Do not replace TRBG-source with TRFG-source in the v2 main method. Use Gate9 to support a rigorous ablation/hardening narrative: fail-only is simpler and lowers burden, but it gives up too much source-frozen recall on new fresh validation and still narrowly fails 14-bus compatibility. The next publishable path is either to keep TRBG-source as the locked burden-guard extension with caveats, or open a new pre-registered Gate10 for a stricter fail guard that explicitly enforces 14-bus compatibility and recall-retention constraints before any fresh confirm.


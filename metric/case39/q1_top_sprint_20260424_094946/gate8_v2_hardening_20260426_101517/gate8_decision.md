# Gate8 Final Decision

## 1. 14-bus backward compatibility

TRBG-source does not pass the pre-registered 14-bus compatibility success criteria. Average recall retention versus the original proposed safeguarded scheduler is `0.9658`, which clears the 95% retention floor but not the 98% strong bar. More importantly, unnecessary interventions increase substantially: `+16.5` at B=1 and `+29.75` at B=2. Cost/backend burden is mixed rather than uniformly better: B=1 improves cost and backend_fail, while B=2 increases cost by `+0.5895` and backend_fail by `+0.1250`.

Decision: TRBG-source should not be written as a paper-wide replacement for the original 14-bus proposed safeguarded scheduler. It is better framed as a case39 scale-up burden guard motivated by the case39 backend burden diagnosis.

## 2. Fresh full-solver extension

Gate8 used the reduced fresh protocol because two additional 540-step full banks were resource-unacceptable given Gate7 runtime. Four new 240-step case39 physical-solver banks completed successfully: `reduced_interleaved_0`, `reduced_tailheavy_1`, `reduced_cleanheavy_2`, and `reduced_attackpulse_3`.

This supports Gate7 direction as small fresh sanity evidence, not full statistical validation. TRBG-source retains `0.9757` of source-frozen recall on average, reduces backend_fail by `0.2439`, lowers cost by `0.1190`, and reduces recover_fail by `0.3750` actions on average. No serious reverse result appears. However, TRBG-source does not exceed topk_expected_consequence recall on every bank/budget pair, and its average recall is slightly below topk in the reduced fresh summary.

Decision: the fresh extension supports the burden-reduction direction of TRBG-source versus source-frozen, but it does not support a dominant-recall claim over topk.

## 3. Component ablation

The component ablation supports the general burden-guard idea but does not prove that the full three-component TRBG formula is uniquely best. At alpha=1, full TRBG reduces backend_fail by `0.0849` versus source-frozen, while fail-only reduces backend_fail by `0.1112`; cost-time-only is not useful for backend_fail reduction (`-0.0123`). The main backend_fail reduction driver is the fail-probability component.

Decision: fail-only looks like a simpler and possibly stronger diagnostic component, but Gate8 is not authorized to replace locked TRBG-source based on confirm/test ablation. Any replacement would require a new pre-registered gate with train/val selection and fresh confirm.

## 4. Repo release cleanup risk

The largest release risk remains path ambiguity: canonical `metric/case39/mixed_bank_fit.npy` and `metric/case39/mixed_bank_eval.npy` still resolve to case14 assets. The dry-run scan found `48` scripts/docs mentioning canonical case39 fit/eval paths. Release cleanup should split `metric/case39_transfer/`, `metric/case39_native/`, and `metric/case39_q1_sprint/`, and README must mark old pre-fix artifacts invalid or caution-only.

## 5. V2 readiness

Gate8 is enough to enter formal v2 rewrite only with conservative positioning. The safe main line is: transfer-regularized post-detection defense scheduling with a locked low-dimensional burden guard, evaluated as case14-compatible-with-caveats plus case39 bridge/stress and reduced fresh sanity evidence.

Do not write native case39 success, recovery-aware method, physical-consequence-aware scheduler, or dominant winner across all baselines/proxies. The evidence most likely to improve SCI Q1-top success remains a pre-registered follow-up that tests the simpler fail-only guard and/or a larger fresh physical-solver validation set.

## Direct Answers

1. TRBG-source compatibility with 14-bus original confirm: limited; it fails the Gate8 compatibility criterion because unnecessary interventions rise materially.
2. TRBG-source should be framed as a case39 scale-up extension, not a paper-wide method upgrade.
3. Fresh extension completed: yes, `4/4` reduced 240-step banks.
4. Fresh extension supports Gate7 direction versus source-frozen on burden reduction and recall retention.
5. Component ablation supports burden guarding but not full TRBG as uniquely best.
6. A simpler fail-only component is stronger on backend_fail reduction in diagnostic ablation.
7. Replacing the main method is not allowed in this gate because TRBG-source was locked and ablation was diagnostic confirm, not a pre-registered selector.
8. Repo release maximum risk remains canonical case39 fit/eval resolving to case14.
9. Formal v2 rewrite can start, but with TRBG-source as a conservative locked burden-guard extension.
10. Missing evidence with largest Q1-top impact: a new pre-registered fail-only/fail-guard comparison or larger fresh physical-solver statistical validation.


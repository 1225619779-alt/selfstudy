# Gate 5 Decision

1. Gate 5 followed no-test-selection: selected alpha/cap came only from source or native train/val before holdout confirm.
2. Alpha=0 and fail_cap_quantile=1.00 reproduced source-frozen: `True` with max_abs_diff `0.0000000000`.
3. TRBG-source status: `moderate_success`.
4. TRBG-native-burden status: `moderate_success`.
5. Recommended v2 main candidate: `TRBG-source`.
6. If both statuses are failure, Gate 5 belongs in appendix/future work only.
7. Mainline upgrade to `transfer-regularized scheduling with low-dimensional burden guard` is justified only if recommended candidate is not `none_appendix_or_future_work`; current recommendation: `TRBG-source`.
8. Backend burden reduction source/native: `0.0837` / `0.0322`.
9. Recall retention source/native: `0.9789` / `0.9799`.
10. Recover_fail delta source/native: `-0.2500` / `0.0625`.
11. SCI Q1-top v2 method closure requires strong or at least clean moderate success plus manuscript reframing away from native success.
12. Proceed to manuscript rewrite pack only if the selected recommendation is accepted as the v2 direction.

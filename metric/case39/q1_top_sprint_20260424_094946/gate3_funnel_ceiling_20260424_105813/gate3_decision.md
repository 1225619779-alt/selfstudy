# Gate 3 Decision

1. Low absolute recall is partly upstream-limited by verified-alarm ceiling (`0.9711`), but substantial post-verification/backend loss remains.
2. Source-frozen average gap to capacity oracle is `0.3762` absolute recall.
3. Source-frozen average gap to backend-success oracle is `0.0201` absolute recall.
4. Source-frozen is not just cost: matched-burden diagnostics retain some ordering advantage versus top-k, though the gap shrinks materially.
5. Matched-cost source-frozen remains ahead of top-k across both budgets: `True`.
6. Matched-backend-fail source-frozen remains ahead of top-k across both budgets: `True`.
7. Backend_fail is a major burden and is dominated by served attack jobs rather than clean jobs.
8. Full-native local retune collapses at the service/operating-point stage: verified attacks are present, but little attack mass is served before backend filtering.
9. Best current wording: `transfer regularization mechanism` plus `backend stress-test limitation`; not `native case39 success`.
10. Gate 4 consequence/recovery robustness is recommended.
11. A small fail-capped source-frozen variant is worth testing only as a future extension or v2 experiment, not as the current submitted main result.

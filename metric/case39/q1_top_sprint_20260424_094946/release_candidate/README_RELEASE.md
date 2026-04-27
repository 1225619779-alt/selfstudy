# Release Candidate README

This release candidate separates case39 transfer evidence, full-native case39 evidence, and Q1 sprint audit artifacts.

- `case39_transfer/`: source-frozen transfer means case14 train/val -> case39 target.
- `case39_native/`: full-native means explicit native case39 train/val.
- `case39_q1_sprint/`: Gate0-Gate9 audit, mechanism, fresh sanity, and rewrite evidence.
- Ambiguous canonical `metric/case39/mixed_bank_fit.npy` / `mixed_bank_eval.npy` symlinks are not placed in this release candidate.
- Gate6 is recombined stress replication.
- Gate7/Gate8/Gate9 fresh outputs are sanity evidence, not full 8-bank statistical validation unless enough banks accumulate.

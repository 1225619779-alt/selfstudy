# Path Ambiguity Scan

- `metric/case39/mixed_bank_fit.npy` currently resolves to case14.
- `metric/case39/mixed_bank_eval.npy` currently resolves to case14.
- Gate2-Gate6 avoided these canonical paths via explicit manifests.
- Release must either remove these symlinks or replace them with native case39 banks plus explicit transfer namespace.

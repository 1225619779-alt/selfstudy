# Gate 1 Provenance Report

- Sprint directory: `metric/case39/q1_top_sprint_20260424_094946`
- STAMP: `/tmp/case39_q1_top_sprint_20260424_094946.stamp`
- Branch: `q1_case39_expansion`
- Commit: `66c0e407fde30868fefbb6fc12afaf06d0d09f6c`
- Source-frozen manifest: `source_frozen_transfer_manifest.json`
- Full-native manifest: `full_native_case39_manifest.json`

## Source-frozen Transfer Separation

- `source_case = case14`
- `source_train_bank = metric/case14/mixed_bank_fit.npy`
- `source_val_bank = metric/case14/mixed_bank_eval.npy`
- `target_case = case39`
- target clean/attack/test banks are explicit case39 paths.
- Interpretation: bridge transfer / stress-test evidence, not native train/val evidence.

## Full-native Case39 Separation

- `native_train_bank = metric/case39_localretune/mixed_bank_fit_native.npy`
- `native_val_bank = metric/case39_localretune/mixed_bank_eval_native.npy`
- clean/attack/test banks are explicit case39 paths.
- The manifest forbids using canonical `metric/case39/mixed_bank_fit.npy` and `metric/case39/mixed_bank_eval.npy` as train/val inputs.

## Canonical Case39 Fit/Eval

- canonical fit resolves to `/home/pang/projects/DDET-MTD-q1-case39/metric/case14/mixed_bank_fit.npy`.
- canonical eval resolves to `/home/pang/projects/DDET-MTD-q1-case39/metric/case14/mixed_bank_eval.npy`.
- canonical still resolves to case14: `True`.

## Hash Evidence

- case14 fit/eval pre/post SHA equal: `True`.
- Pre-hash manifest: `hash_pre_gate1.json`.
- Post-hash manifest: `hash_post_gate1.json`.

## Readiness

- Readiness status: `MANIFEST_NATIVE_READY_CANONICAL_STILL_CASE14`.
- Practical Gate 2 route: use the explicit full-native manifest. Do not rely on canonical case39 fit/eval until they are cleaned or bypassed by manifest arguments.

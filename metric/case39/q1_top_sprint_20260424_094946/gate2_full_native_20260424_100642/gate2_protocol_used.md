# Gate 2 Protocol Used

- Gate 1 full-native manifest: `metric/case39/q1_top_sprint_20260424_094946/full_native_case39_manifest.json`
- Gate 1 source-frozen transfer manifest: `metric/case39/q1_top_sprint_20260424_094946/source_frozen_transfer_manifest.json`
- Gate 2 compatible full-native manifest: `metric/case39/q1_top_sprint_20260424_094946/gate2_full_native_20260424_100642/gate2_full_native_manifest_used.json`
- Full-native train bank: `metric/case39_localretune/mixed_bank_fit_native.npy`
- Full-native val bank: `metric/case39_localretune/mixed_bank_eval_native.npy`
- Canonical `metric/case39/mixed_bank_fit.npy` was not used.
- Canonical `metric/case39/mixed_bank_eval.npy` was not used.
- Source-frozen transfer uses `source_case=case14` and `target_case=case39` from `source_frozen_transfer_manifest.json`.
- Budgets: `B = 1, 2`.
- `Wmax = 10`.
- Frozen holdouts: `8`.
- Winner/config selection used train/val only; test holdouts were not used to select winners.
- Missing or failed baselines are retained in `baseline_status.csv` and in summary files.

## Baseline Definitions

- `source_frozen_transfer`: case14 train/val transfer config evaluated on case39 holdouts.
- `winner_replay`: source winner/config replayed with native case39 train/val prediction models.
- `anchored_retune`: existing source-anchored native train/val screen config replayed on the same holdouts.
- `native_safeguarded_retune`: forced `oracle_protected_ec` native train/val screen config.
- `native_unconstrained_retune`: full native oracle-family screen without the protected-only guard.
- `phase3_proposed`: native baseline `proposed_ca_vq_hard` selected on native val.
- `phase3_oracle_upgrade`: native val-selected oracle-family upgrade.
- `topk_expected_consequence`: static top-k by predicted expected consequence under native predictions.
- `incumbent_queue_aware`: tuned non-consequence queue-aware `proposed_vq_hard` diagnostic baseline.
- `static_threshold`: tuned static verify-score FIFO threshold diagnostic baseline.

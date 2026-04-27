# Repo Release Cleanup Plan

Do not change the main repo layout in Gate7. For release, split ambiguous case39 artifacts into explicit namespaces.

- `metric/case39_transfer/`: source-frozen transfer artifacts using case14 train/val and case39 target/holdouts.
- `metric/case39_native/`: full-native case39 train/val/local-retune artifacts using native fit/eval.
- `metric/case39_q1_sprint/`: audit/sprint evidence packs, including Gate0-Gate7 outputs.

Release README must state that canonical `metric/case39/mixed_bank_fit.npy` and `mixed_bank_eval.npy` must not ambiguously resolve to case14.
Release README must also mark old pre-fix attack-side summaries as invalid or caution-only.

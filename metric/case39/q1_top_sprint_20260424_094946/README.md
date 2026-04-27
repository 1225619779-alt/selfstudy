# Q1 Case39 Sprint Evidence Pack

This directory contains the Gate0-Gate9 evidence sprint for the `q1_case39_expansion` branch.

The pack is intended for Pro review. It preserves the submitted-version evidence and adds separately namespaced audit, mechanism, fresh-solver sanity, and release-candidate materials. The safe interpretation remains conservative:

- case39 evidence is bridge transfer / stress-test evidence, not native case39 success;
- the consequence target is proxy-consequence-guided, not physical-consequence-aware or recovery-aware;
- Gate6 is locked recombined stress replication, not fresh physical-solver validation;
- Gate7-Gate9 fresh banks are sanity evidence, not full 8-bank statistical validation;
- canonical `metric/case39/mixed_bank_fit.npy` and `metric/case39/mixed_bank_eval.npy` remain ambiguous in the working repo and are separated only in `release_candidate/`.

## Main Outputs

| Gate | Directory / file | Purpose | Current decision |
| --- | --- | --- | --- |
| Round 1 | `../audit_round1_20260424_002926/` | case39 provenance/readiness/definition audit | case39 not native-ready; bridge/stress evidence only |
| Round 2 | `../round2_mechanism_20260424_092002/` | claim repair, funnel, statistics, proxy robustness | source-frozen high-recall/high-burden point |
| Gate0-Gate1 | `protocol_q1_top.md`, `source_frozen_transfer_manifest.json`, `full_native_case39_manifest.json` | preregistration and manifest separation | source-frozen and native case39 explicitly separated |
| Gate2 | `gate2_full_native_20260424_100642/` | full-native train/val/confirm comparison | full-native local retune still collapses |
| Gate3 | `gate3_funnel_ceiling_20260424_105813/` | funnel, ceiling, burden diagnostics | bottleneck is post-verification service/backend success |
| Gate4 | `gate4_recovery_robustness_20260424_120517/` | proxy/recovery robustness | proxy-consequence-guided only |
| Gate5 | `gate5_transfer_burden_guard_20260424_123448/` | TRBG-source extension | moderate internal success |
| Gate6 | `gate6_locked_blind_validation_20260424_130803/` | locked recombined blind validation and stats repair | stress replication, not fresh solver |
| Gate7 | `gate7_evidence_lock_20260425_215908/` | evidence lock, one-bank fresh sanity, rewrite pack | enter v2 rewrite with caveats |
| Gate8 | `gate8_v2_hardening_20260426_101517/` | 14-bus compatibility, fresh extension, ablation | TRBG remains case39 extension; not paper-wide upgrade |
| Gate9 | `gate9_fail_only_guard_20260426_224426/` | pre-registered fail-only validation | TRFG-source rejected as main-method replacement |
| Release | `release_candidate/` | release cleanup candidate | transfer/native/sprint split, no ambiguous canonical symlink in candidate |

## Gate0-Gate9 Added Code Files

### Round 2

- `../round2_mechanism_20260424_092002/generate_round2_pack.py`

### Gate1-Gate5

- `gate1_provenance_cleanup.py`
- `gate2_full_native_20260424_100642/gate2_full_native_rerun.py`
- `gate3_funnel_ceiling_20260424_105813/gate3_funnel_ceiling.py`
- `gate4_recovery_robustness_20260424_120517/gate4_recovery_robustness.py`
- `gate5_transfer_burden_guard_20260424_123448/gate5_transfer_burden_guard.py`

### Gate6

- `gate6_locked_blind_validation_20260424_130803/gate6_locked_blind_validation.py`
- `gate6b_fullsolver_blind_20260424_194059/gate6b_fullsolver_manager.py`
- `gate6b_fullsolver_blind_20260424_194059/launch_gate6b.sh`
- `gate6c_checkpoint_fullsolver_20260425_0820/checkpointed_evaluation_mixed_timeline.py`
- `gate6c_checkpoint_fullsolver_20260425_0820/run_gate6c_checkpointed.sh`
- `gate6c_checkpoint_fullsolver_20260425_0820/summarize_gate6c_partial.py`

### Gate7

- `gate7_evidence_lock_20260425_215908/gate7_build_static_reports.py`
- `gate7_evidence_lock_20260425_215908/gate7_fresh_confirm.py`
- `gate7_evidence_lock_20260425_215908/inspect_method_defs.py`
- `gate7_evidence_lock_20260425_215908/resume_checkpointed_evaluation_mixed_timeline.py`
- `gate7_evidence_lock_20260425_215908/run_fresh_resume.sh`

### Gate8

- `gate8_v2_hardening_20260426_101517/checkpointed_evaluation_mixed_timeline.py`
- `gate8_v2_hardening_20260426_101517/gate8_reduced_fresh_manager.py`
- `gate8_v2_hardening_20260426_101517/gate8_static_and_sim.py`
- `gate8_v2_hardening_20260426_101517/run_gate8_reduced_fresh_manager.sh`

### Gate9

- `gate9_fail_only_guard_20260426_224426/checkpointed_evaluation_mixed_timeline.py`
- `gate9_fail_only_guard_20260426_224426/gate9_fail_only_analysis.py`
- `gate9_fail_only_guard_20260426_224426/gate9_fresh_manager.py`
- `gate9_fail_only_guard_20260426_224426/run_gate9_fresh_manager.sh`

## Key Decisions For Pro

1. Case39 should not be claimed as native larger-system success.
2. Source-frozen case14 -> case39 transfer is a high-recall/high-burden operating point.
3. Full-native case39 local retune remains over-conservative and does not beat source-frozen recall.
4. TRBG-source reduces source-frozen burden enough to be a v2 candidate, but it is better framed as a case39 scale-up extension rather than a paper-wide replacement.
5. Gate8 showed fail-only looked simpler in ablation, so Gate9 pre-registered and tested TRFG-source.
6. Gate9 rejected TRFG-source as main replacement: fresh recall retention was `0.7500`, below the locked `0.95` criterion, despite backend/cost/unnecessary reductions.
7. Physical-consequence sanity is not strong enough for a main claim because OPF/branch-flow/voltage-deviation logging is missing.
8. `release_candidate/` is the proposed cleanup structure for Pro review and does not alter the original experiment directories.

## Important Caveats

- Several case14 confirm raw banks were read from the sibling legacy path `/home/pang/projects/DDET-MTD/metric/case14/.../banks/` because the q1-case39 worktree has case14 manifests/summaries but not those raw bank files. This is recorded in `gate8_case14_input_path_resolution.csv` and `gate9_case14_input_path_resolution.csv`.
- Fresh full-solver runs are expensive and were checkpointed. Gate8 and Gate9 reduced fresh banks are physical-solver sanity checks, not full statistical validation.
- Old pre-fix attack-side artifacts remain invalid or caution-only unless explicitly regenerated after the provenance/path fixes.

## Suggested Review Order

1. Read `gate7_evidence_lock_20260425_215908/gate7_decision.md`.
2. Read `gate8_v2_hardening_20260426_101517/gate8_decision.md`.
3. Read `gate9_fail_only_guard_20260426_224426/gate9_decision.md`.
4. Inspect `release_candidate/README_RELEASE.md` and `release_candidate/canonical_case39_fit_eval_reference_classification.csv`.
5. Check the code files listed above only if the decision summaries raise questions.


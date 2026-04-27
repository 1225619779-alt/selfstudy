# Fresh Full-Solver Resume Protocol

This protocol was written before Gate7 resumes the Gate6c partial bank.

## Priority 1 Attempt

- Resume source: `gate6c_checkpoint_fullsolver_20260425_0820/partials/mixed_bank_test_fresh_checkpointed_540_seed20260711_off1500.partial.npy`.
- Resume point: `218/540` completed steps.
- Case: `DDET_CASE_NAME=case39`.
- Model: `saved_model/case39/checkpoint_rnn.pt`.
- Schedule: `att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120`.
- Seed: `20260711`.
- Start offset: `1500`.
- `tau_verify=-1`.

## Boundaries

- Do not modify detector logic.
- Do not modify backend MTD solver numerical logic.
- Do not modify scheduler family.
- Do not change TRBG-source alpha/cap.
- Use checkpoint/resume and runtime logging only.
- Do not overwrite Gate6c partial; Gate7 writes new partial and final files.

## Evidence Use

If the resumed 540-step bank completes, it is one fresh physical-solver sanity check, not 8-bank statistical validation.

If it does not complete, Gate7 reports the partial runtime evidence and falls back to reduced fresh sanity planning rather than claiming full fresh validation.

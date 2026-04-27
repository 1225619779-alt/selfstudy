# Gate 6c Checkpointed Full-Solver Prefix Protocol

This protocol was written before starting the checkpointed Gate 6c run.

## Goal

Produce useful fresh case39 full-solver evidence before the 21:00 decision window without claiming an 8-bank blind validation.

## Fixed Inputs

- Case: `case39` via explicit `DDET_CASE_NAME=case39`.
- Script: isolated `checkpointed_evaluation_mixed_timeline.py`.
- Base algorithm: copied from `evaluation_mixed_timeline.py` logic; no detector, backend MTD solver, or scheduler-family change.
- Schedule: `att-3-0.35:120;clean:60;att-2-0.25:90;clean:90;att-1-0.15:60;clean:120`.
- Seed: `20260711`.
- Start offset: `1500`.
- `tau_verify=-1`.

## Stop Rule

- Run one 540-step fresh physical-solver bank with checkpointing.
- Save partial output after every completed step.
- Stop when the bank completes or the wall-clock limit is reached.
- External timeout may stop a long in-step backend solve; completed previous steps remain in partial output.

## Evidence Use

- If complete, this is one fresh physical-solver bank, not 8-bank validation.
- If partial, this is feasibility/profiling evidence and may support a checkpoint/resume full-solver plan.
- Neither complete nor partial Gate 6c can be used to reselect TRBG alpha/cap or rewrite Gate 6 as native case39 success.

# Gate 6b Feasibility Decision

## Status

- Gate 6b full-solver smoke was attempted under explicit `DDET_CASE_NAME=case39`.
- Smoke bank: `fresh_burst_frontloaded_0_seed20260711_off1500`.
- Schedule length: `540` steps.
- Timeout: `21641.8` seconds, about `6.0` hours.
- Output bank produced: `False`.
- Residual process after timeout: `False`.

## What Happened

The first smoke attempt without `DDET_CASE_NAME=case39` failed quickly because the repo config defaults to case14. That failure is preserved in `logs/fresh_burst_frontloaded_0_seed20260711_off1500.case14_env_failure.log`.

After fixing the wrapper to set `DDET_CASE_NAME=case39`, `evaluation_mixed_timeline.py` entered the correct case39 path and loaded the case39 model/data successfully. The run reached the full-solver loop but did not finish a single 540-step bank within the 6-hour smoke timeout.

The log shows very slow early progress: `5/540` steps after about `40:31`, followed by long backend MTD solver output. This indicates the blocker is the full recovery/backend MTD solve path, not file paths, manifests, or the scheduler wrapper.

## Decision

Do not run the remaining 7 full-solver banks under the current raw `evaluation_mixed_timeline.py` path.

The observed runtime implies that fresh 8-bank physical-solver validation is not feasible in the current 24-hour window without changing the experiment engineering. Current Gate 6 should remain labeled as locked recombined stress replication, not fresh physical-solver blind validation.

## Safe Optimization Options

These are engineering changes that preserve detector/backend/scheduler algorithms if implemented carefully:

- Add checkpoint/resume to an isolated Gate 6b copy or wrapper so partial per-step results are saved and can be resumed.
- Add per-step runtime JSONL logging around DDD recovery and backend MTD solve calls.
- Save partial `.npz` or `.jsonl` rows every step or every small chunk, then atomically assemble a `.npy` only after successful completion.
- Run a much shorter fresh physical-solver prefix bank first, for example 10 to 20 steps, to profile step-level runtime.
- Only after a complete short prefix succeeds, decide whether a reduced fresh validation design is scientifically acceptable.

## Unsafe Or Evidence-Changing Options

These should not be used for main evidence without Pro approval:

- Reducing `mtd_config["multi_run_no"]`.
- Changing backend solver tolerances or iteration limits.
- Adding a per-MTD timeout and marking timed-out solves as backend failures.
- Raising `tau_verify` to avoid backend solves when the goal is to generate full scheduler-compatible banks.
- Claiming the current Gate 6 recombined banks are fresh physical-solver simulations.

## Recommendation

Wait for Pro before running more full-solver validation. The next reasonable task is either a Gate 6b engineering-only checkpoint/resume wrapper, or a deliberately reduced fresh physical-solver prefix validation with a pre-registered reduced scope.

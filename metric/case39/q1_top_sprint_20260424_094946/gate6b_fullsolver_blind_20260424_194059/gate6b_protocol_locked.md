# Gate 6b Full-Solver Blind Validation Protocol

This protocol is written before running any Gate 6b fresh full-solver bank.

## Purpose

- Test whether fresh `evaluation_mixed_timeline.py` case39 physical-solver blind banks are feasible within the 24h window.
- Do not modify detector, backend MTD solver, scheduler family, TRBG alpha/cap, or any Gate 1-6 result.
- Keep current Gate 6 recombined stress replication separate from Gate 6b fresh full-solver evidence.

## Locked Method Context

- TRBG-source remains locked at `alpha=1.0`, `fail_cap_quantile=1.00`, `calibration_mode=source`.
- Gate 6b bank generation is only provenance/validation infrastructure; it does not select parameters.

## Auto-Continuation Rule

- Run exactly one smoke bank first: `fresh_burst_frontloaded_0_seed20260711_off1500`.
- Per-bank timeout: `21600` seconds; smoke timeout: `21600` seconds.
- If smoke succeeds and `smoke_elapsed_seconds * 8 <= 72000`, automatically run the remaining 7 banks serially.
- If smoke fails, times out, or estimates beyond the limit, stop and report feasibility instead of running the remaining banks.
- No human-in-the-loop decision or test-set selection occurs during auto-continuation.

## Output Policy

- Every bank writes stdout/stderr to `logs/<tag>.log`.
- Runtime and success/failure are recorded in `gate6b_runtime_log.csv`.
- Status is updated in `gate6b_status.json` after every heartbeat and bank completion.
- Successful banks are written only under this Gate 6b directory.
- The runner explicitly sets `DDET_CASE_NAME=case39` for `evaluation_mixed_timeline.py`; this is required because the repo config defaults to case14 when the environment variable is absent.

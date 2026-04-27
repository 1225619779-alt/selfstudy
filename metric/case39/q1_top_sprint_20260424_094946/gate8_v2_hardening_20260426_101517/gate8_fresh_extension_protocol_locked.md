# Gate8 Fresh Extension Protocol Locked

This protocol was written before any Gate8 fresh confirm result is generated.

## Scope

- Objective: add small fresh physical-solver direction evidence beyond the Gate7 one-bank sanity check.
- Primary method remains locked: `TRBG-source`, `alpha=1.0`, `fail_cap_quantile=1.00`.
- No parameter, threshold, score weight, cap, or winner selection is allowed on the Gate8 fresh banks.
- Detector, backend MTD solver numeric logic, scheduler regime, and source-frozen weights are unchanged.
- Interpretation: reduced fresh sanity check if reduced banks are used; not full statistical validation.

## Full-Bank Feasibility Decision

Gate7 showed that a single 540-step fresh physical-solver bank is runtime-heavy. Therefore Gate8 uses the pre-authorized priority-2 reduced protocol instead of starting two additional 540-step banks.

## Reduced Fresh Banks

Four new 240-step case39 physical-solver banks are attempted. Each uses a new seed, a new offset, and a different stress schedule. These banks are not calibration data and are not used to choose any method.

| bank_id | seed | offset | schedule |
| --- | ---: | ---: | --- |
| reduced_interleaved_0 | 20260811 | 2100 | `clean:80;att-1-0.15:20;clean:40;att-2-0.20:20;clean:40;att-3-0.30:20;clean:20` |
| reduced_tailheavy_1 | 20260812 | 2160 | `clean:150;att-1-0.10:15;clean:30;att-2-0.20:15;att-3-0.35:30` |
| reduced_cleanheavy_2 | 20260813 | 2220 | `clean:120;att-1-0.15:20;clean:60;att-2-0.25:20;clean:20` |
| reduced_attackpulse_3 | 20260814 | 2280 | `clean:90;att-3-0.35:30;clean:60;att-1-0.15:30;clean:30` |

## Fixed Confirm Methods

- `source_frozen_transfer`
- `TRBG-source` with `alpha=1.0`, `fail_cap_quantile=1.00`
- `topk_expected_consequence`
- `winner_replay`
- `native_safeguarded_retune`
- `native_unconstrained_retune`

## Runtime Controls

- Checkpoint every 5 steps.
- Concurrency cap is 2 banks at a time to reduce wall time without changing solver settings.
- If a bank does not complete, its partial status and runtime are reported; incomplete banks are not silently converted into full evidence.

## Decision Endpoints

- TRBG-source recall retention vs source-frozen.
- Backend_fail reduction vs source-frozen.
- Cost delta vs source-frozen.
- Recover_fail delta vs source-frozen.
- Recall vs topk_expected_consequence.
- Serious reverse result flag.


# Gate9 Fresh Protocol Locked

This protocol is locked after source-side alpha selection and before any Gate9 fresh confirm result is generated.

## Locked Candidates

- `TRFG-source`: alpha `0.25`, selected only from case14 source train/val.
- `TRFG-native-fail`: alpha `2.0`, selected only from explicit native case39 train/val, diagnostic only.
- `TRBG-source`: alpha `1.0`, cap `1.00`, carried forward as the previous locked burden guard.
- No Gate8 confirm result is used as a selector.

## Fresh Banks

Priority 1 uses four new 240-step reduced fresh case39 physical-solver banks. Seeds, offsets, and schedules are new and not identical to Gate8 reduced banks.

| bank_id | seed | offset | schedule |
| --- | ---: | ---: | --- |
| gate9_sparse_front_0 | 20260911 | 2400 | `clean:100;att-1-0.12:15;clean:80;att-2-0.22:15;clean:30` |
| gate9_late_mixed_1 | 20260912 | 2480 | `clean:140;att-2-0.18:20;clean:40;att-3-0.28:20;clean:20` |
| gate9_alternating_short_2 | 20260913 | 2560 | `clean:60;att-1-0.10:10;clean:50;att-2-0.22:10;clean:50;att-3-0.32:10;clean:50` |
| gate9_dense_middle_3 | 20260914 | 2640 | `clean:90;att-1-0.12:15;att-2-0.18:15;clean:60;att-3-0.30:20;clean:40` |

## Fixed Confirm Methods

- `source_frozen_transfer`
- `TRBG-source`
- `TRFG-source`
- `TRFG-native-fail`
- `topk_expected_consequence`
- `winner_replay`
- `native_safeguarded_retune`
- `native_unconstrained_retune`

## Runtime Controls

- Serial execution is used to avoid the Gate8-observed backend-solver oversubscription long-tail.
- Checkpoint every 5 steps.
- No detector, backend solver, scheduler, tolerance, tau_verify, or timeout-as-fail setting is changed.
- Incomplete banks are reported as incomplete and are not silently converted into full evidence.

## Interpretation

These four reduced banks are fresh physical-solver sanity evidence, not full 8-bank statistical validation. The optional 540-step fresh bank is not started unless resources remain after the four reduced banks complete.


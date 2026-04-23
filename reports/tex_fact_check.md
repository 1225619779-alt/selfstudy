# TeX Fact Check

Current HEAD: `371463f7de5bb07eb60393fdade5d6b642bc511e`

Primary TeX source checked:
- `/mnt/c/Users/GA/Downloads/gate_only_epsr_final_draft.tex`

Primary result sources checked:
- `/home/pang/projects/DDET-MTD/reports/paper_compact_tables.md`
- `/home/pang/projects/DDET-MTD/reports/case39_failure_aware_statistics.md`
- `/home/pang/projects/DDET-MTD/reports/case39_minimal_score_ablation.md`

## Safe Statements

### Case14 clean exact

TeX locations:
- Abstract: line 33
- Main paragraph: lines 193-207

Checked against `paper_compact_tables.md`:
- `trigger_rate`: `1.000000 -> 0.069550`
- `useless_mtd_rate (UDR)`: `0.167925 -> 0.011679`
- `fail_per_alarm`: `0.105174 -> 0.021204`
- `stage_two_time_per_alarm`: `4.838323 -> 0.270593`
- `delta_cost_two_per_alarm`: `4.161843 -> 0.414595`

Rounded forms currently used in TeX are correct:
- `1.0000 -> 0.0696`
- `0.1679 -> 0.0117`
- `0.1052 -> 0.0212`
- `4.8383 -> 0.2706`
- `4.1618 -> 0.4146`

### Case14 mixed timeline

TeX locations:
- Main paragraph: line 214
- Table: lines 216-229

Checked against `paper_compact_tables.md`:
- trigger count: `141 -> 58`
- backend fail: `10 -> 4`
- cumulative time: `753.869646 -> 595.989086`
- cumulative delta cost: `5190.730326 -> 1860.467609`

Rounded TeX forms are correct:
- `141 -> 58`
- `10 -> 4`
- `753.87 -> 595.99`
- `5190.73 -> 1860.47`

### Case39 stress benchmark

TeX locations:
- Setup framing: lines 184-185
- Stress benchmark paragraph: lines 286-302
- Discussion: lines 320-322

Checked against `paper_compact_tables.md`:
- main clean `trigger_rate = 0.920635`
- main clean `UDR = 0.107392`
- main clean `fail_per_alarm = 0.902320`
- main clean `stage_two_time_per_alarm = 0.643374`
- main attack `overall_arr = 0.993151`
- strict attack `overall_arr = 0.986301`
- protected min ARR = `0.980000`

Current TeX framing is safe:
- `additional stress benchmark`
- does not claim robust large-system backend success
- explicitly says backend optimizer remains failure-prone

### Case39 failure-aware wording

Checked against `case39_failure_aware_statistics.md`:
- `recovery_error_count = 0`
- clean backend MTD fail remains high
- attack-side backend metric failure is explicitly tracked
- backend robustness is not established

Safe wording:
- recovery did not throw errors in the reported exact-tau case39 runs
- backend MTD failure remains high
- attack-side backend metric failure is tracked explicitly
- do not claim backend robustness

### Minimal score ablation

TeX locations:
- Ablation paragraph and table: lines 261-279
- Discussion: line 328

Checked against `case39_minimal_score_ablation.md`:
- `angle-L2` remains the main method
- `joint angle+|V|-L2` does not show a clear practical advantage
- `angle-Linf` is similar but not clearly better
- current TeX does not claim universal optimality

This framing is safe.

## Risky Or Over-Broad Statements

### Case14 comparator `T^(2)` must not be mixed with Case14 clean exact `T^(2)`

TeX location:
- Comparator paragraph: line 236
- Comparator table: lines 246-253

Exact clean main result:
- `stage_two_time_per_alarm = 0.270593`

Comparator main proposed physical score:
- `stage_two_time_per_alarm = 0.417818`

Comparator main detector-loss score:
- `stage_two_time_per_alarm = 0.581301`

All three numbers are valid in their own contexts, but they are not interchangeable.

Instruction:
- Do not write a sentence that directly contrasts case14 clean exact `0.2706` with comparator `0.4178` or `0.5813` as if they were the same evaluation path.
- In the comparator subsection, only emphasize retention vs stage-II burden/cost tradeoff.

Safe wording:
- detector-loss gating retains more attack alarms
- recovery-aware gating yields lower stage-II backend burden/cost than detector-loss under matched clean-trigger budgets

Unsafe wording:
- the comparator confirms the exact main stage-II time is `0.4178`
- the comparator directly reproduces the case14 clean exact stage-II number

### Detector-loss row in minimal score ablation should not be promoted into a main fairness table

Checked against `case39_minimal_score_ablation.md`:
- main `detector_loss`: `valid_clean_trigger_count = 0`, `test_clean_trigger_count = 0`, `test_attack_overall_arr = 0.880137`
- strict `detector_loss`: `valid_clean_trigger_count = 0`, `test_clean_trigger_count = 0`, `test_attack_overall_arr = 0.849315`

Instruction:
- This row is informative as a calibration instability signal.
- It should not be presented as the main fairness comparison for score families.

## Forbidden Claims

The following claims should not appear anywhere in the paper:
- `robust larger benchmark`
- `strict superiority over detector-loss gate`
- `universal optimality of angle-L2`
- `case39 proves end-to-end backend robustness`

## Recommended Writing Boundary

Safe summary:
- `case14` remains the main detailed evidence.
- `case39` is an additional stress-test / limitation benchmark.
- detector-loss and recovery-aware gating form a tradeoff, not a dominance relation.
- angle-L2 is the practical main method in the current setting, not a universally optimal score.

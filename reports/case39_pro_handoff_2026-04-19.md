# Case39 Handoff for Pro Review

## Repo State

This branch contains the local DDET-MTD extension work used to:

1. unify the case14 paper worldline,
2. add real case39 support,
3. run a full case39 pipeline through measurement generation, training, clean calibration, clean evaluation, attack evaluation, matched-budget comparator, and backend-failure audit.

The branch intentionally does **not** include large raw artifacts such as:

- `gen_data/case39/*.npy`
- `saved_model/case39/checkpoint_rnn.pt`
- `logs/*`
- large metric `.npy` payloads

Instead, it includes the code, orchestration scripts, and compact text/JSON/CSV summaries needed to inspect the work.

## Major Code Changes

### Worldline / paper-run cleanup

- `paper_worldline.py`
- `evaluation_event_trigger_clean.py`
- `evaluation_mixed_timeline.py`
- `compare_gate_results_clean.py`
- `compare_mixed_timeline.py`
- `analyze_gate_ablation.py`
- `export_clean_main_table.py`
- `plot_verify_score_distribution_v3.py`
- `make_fig4_mixed_main_v3.py`
- `run_case14_paper_batch.sh`

### Case39 enablement

- `configs/config.py`
- `configs/nn_setting.py`
- `utils/load_data.py`
- `gen_data/gen_data.py`
- `generate_case_basic_npy.py`
- `generate_case14_basic_npy.py`

### Measurement-bank acceleration / orchestration

- `generate_measurement_bank_parallel.py`
- `summarize_case39_measurement_progress.py`
- `run_case39_measurement_full.sh`
- `run_case39_after_measurement.sh`
- `run_case39_train_full.sh`
- `run_case39_post_train.sh`
- `run_case39_smoke_clean.sh`

### Tau selection / clean suite / support chain

- `run_case39_tau_selection_joint_valid.sh`
- `run_case39_clean_suite_from_tau.sh`
- `run_case39_post_tau.sh`
- `run_case39_attack_support_from_clean.sh`
- `run_case39_post_clean_suite.sh`
- `run_case39_perf_probe.sh`
- `run_case39_post_support_perf_probe.sh`

### Attack-side evaluation hardening

- `evaluation_event_trigger_attack_cli.py`
- `run_case39_attack_suite_from_tau.sh`

### Backend-failure auditing

- `audit_case39_backend_failures.py`

## Case39 Pipeline Status

Completed:

1. case39 basic data generation
2. case39 measurement-bank generation
3. case39 detector training
4. case39 smoke clean run
5. validation-only joint tau selection
6. case39 clean baseline/main/strict
7. clean alarm score collection
8. attack alarm score collection
9. matched-budget comparator / ablation
10. performance probe
11. case39 attack baseline/main/strict
12. backend-failure audit

## Key Case39 Results

### Validation-only tau selection

Source:

- `metric/case39/tau_selection_joint_valid/tau_selection_summary.txt`

Selected operating points:

- `tau_main = 0.013319196253` (`0.013` rounded)
- `tau_strict = 0.016153267226` (`0.016` rounded)

Validation constraints satisfied:

- main: overall ARR `0.9667`, protected groups all `>= 0.95`
- strict: overall ARR `0.9133`, protected groups all `>= 0.90`

### Clean results

Sources:

- `metric/case39/metric_event_trigger_clean_tau_-1.0_mode_0_0.03_1.1.npy`
- `metric/case39/metric_event_trigger_clean_tau_0.013_mode_0_0.03_1.1.npy`
- `metric/case39/metric_event_trigger_clean_tau_0.016_mode_0_0.03_1.1.npy`

Common clean front-end stats:

- total clean samples: `7021`
- clean DDD alarms: `819`
- false-alarm rate: `0.11665`

Backend burden:

- baseline:
  - trigger rate among clean alarms: `1.00000`
  - uMTD rate: `0.11665`
  - fail per alarm: `0.96703`
  - stage-one time per alarm: `9.17432`
  - stage-two time per alarm: `1.53038`
- main (`tau=0.013`):
  - trigger rate among clean alarms: `0.92552`
  - uMTD rate: `0.10796`
  - fail per alarm: `0.90965`
  - stage-one time per alarm: `6.47497`
  - stage-two time per alarm: `0.46811`
- strict (`tau=0.016`):
  - trigger rate among clean alarms: `0.83150`
  - uMTD rate: `0.09699`
  - fail per alarm: `0.81685`
  - stage-one time per alarm: `4.14647`
  - stage-two time per alarm: `0.36410`

Interpretation:

- clean burden is reduced, but not nearly as dramatically as in case14.
- stage-two burden drops more clearly than stage-one burden.
- backend failure on case39 clean remains very high.

### Attack results

Sources:

- `metric/case39/metric_event_trigger_tau_-1.0_mode_0_0.03_1.1.summary.txt`
- `metric/case39/metric_event_trigger_tau_0.013_mode_0_0.03_1.1.summary.txt`
- `metric/case39/metric_event_trigger_tau_0.016_mode_0_0.03_1.1.summary.txt`

Overall attack ARR:

- baseline: `1.0000`
- main: `0.99315`
- strict: `0.98630`

Groupwise ARR at strict:

- `(1,0.2) = 0.93617`
- `(1,0.3) = 1.00000`
- `(2,0.2) = 1.00000`
- `(2,0.3) = 1.00000`
- `(3,0.2) = 0.98000`
- `(3,0.3) = 1.00000`

Interpretation:

- attack retention remains very high on case39.
- the protected stronger groups are still well preserved.

### Matched-budget comparator

Source:

- `metric/case39/gate_ablation_case39.csv`

Main-budget comparison:

- detector-loss gate:
  - clean budget matched at `758`
  - attack retention overall: `1.0000`
  - stage-two time per alarm: `1.38517`
  - delta cost stage two per alarm: `0.68466`
- proposed physical score:
  - clean budget matched at `758`
  - attack retention overall: `0.99315`
  - stage-two time per alarm: `0.720996`
  - delta cost stage two per alarm: `0.64557`

Strict-budget comparison:

- detector-loss gate:
  - budget `681`
  - attack retention overall: `0.98973`
  - stage-two time per alarm: `1.16773`
  - delta cost stage two per alarm: `0.61003`
- proposed physical score:
  - budget `681`
  - attack retention overall: `0.98630`
  - stage-two time per alarm: `0.65833`
  - delta cost stage two per alarm: `0.55324`

Interpretation:

- detector-loss still wins on retention.
- proposed physical score gives lower stage-two burden and lower cost at matched clean budget.
- therefore the safe claim is not "strictly better gate", but "backend-aware tradeoff with lower stage-two burden under similar retention".

### Performance probe

Source:

- `metric/case39/perf_probe/summary.txt`

Small-sample probe on `collect_clean_alarm_scores.py`:

- threads=1: `211 sec`
- threads=4: `224 sec`
- threads=8: `225 sec`

Outputs were consistent across thread settings.

Interpretation:

- simply increasing BLAS/thread counts does not improve this workload.
- major speedups, if any, will need structural changes rather than thread knobs.

## Backend Failure Audit

Sources:

- `metric/case39/backend_failure_audit.txt`
- `metric/case39/backend_failure_audit.json`

Main observations:

1. clean-side backend failure is dominated by the backend itself, not by post-processing noise.
2. clean-side `obj_one` median is `0.0`.
3. clean-side `obj_two` median is `1000.0`.
4. attack-side `fail_rate_triggered_only` is around `0.784 ~ 0.786`.
5. attack-side `backend_metric_fail_rate` is only around `0.08`.

Interpretation:

- the dominant issue is not merely backend metric evaluation crashing.
- the main issue is that stage-one / stage-two MTD on case39 is often degenerating or failing to produce healthy backend solutions.

## Why This Matters for the Paper

Current evidence now supports:

- the method is not case14-only.
- high attack retention is still achievable on case39.
- the proposed physical verification score can reduce backend stage-two burden at matched clean budget.

Current evidence does **not** yet support a strong Q2-ready claim that the full pipeline is robust on larger systems, because:

- case39 clean burden reduction is modest,
- detector-loss comparator still dominates retention,
- case39 backend MTD itself is highly failure-prone.

## Recommended Next-Step Decision

Please advise which of these should be the main next-stage research move:

1. **Validation-only backend calibration for case39**
   - tune a very small candidate set of backend parameters only on validation,
   - then rerun clean + attack once on test,
   - goal: reduce case39 backend failure while keeping methodology defensible.

2. **Treat backend instability as a limitation and push the paper around gate/comparator evidence**
   - emphasize portability + high retention + stage-two burden reduction,
   - explicitly admit backend MTD degradation on larger systems,
   - avoid claiming robust end-to-end backend success on case39.

3. **Minimal additional score ablation before rewriting**
   - compare current angle-L2 with a tiny set such as angle-Linf and joint angle+|V|,
   - only if this is more likely than backend work to improve the paper’s ceiling.

## Specific Question for Review

Given this branch and the attached compact summaries, what would you prioritize next if the target is a stronger SCI Q2 submission:

- backend calibration / robustness repair,
- limitation-aware rewriting around the current comparator story,
- or one more tightly scoped ablation to improve the score design?

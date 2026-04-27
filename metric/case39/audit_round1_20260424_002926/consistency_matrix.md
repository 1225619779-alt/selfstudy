# Consistency Matrix

## 1. Paper / README definition vs code definition

| Topic | Paper / README definition | Code path / actual implementation | Consistency call | Impacted outputs |
|---|---|---|---|---|
| `severity` truth target | `README_phase3.md` says phase3 replaces `value_proxy ~= verify_score` with a learned expected consequence term, fits `severity_hat(signal, attack)` on attack jobs, and uses `expected_consequence_hat = p_hat * severity_hat` under the default conditional mode. The same README also explicitly says the current evaluation-side severity still relies on `ang_no * ang_str`. | `scheduler/calibration.py` defines `severity_true = max(ang_no, 0) * max(ang_str, 0)` and zeroes it on clean jobs. `fit_attack_severity_models_from_arrays()` and `fit_expected_consequence_models_from_arrays()` both fit directly against that `severity_true`. | Partial match. The predictor form matches the README, but the truth target is still the old attack-parameter proxy, not a richer recovery-side consequence target. | All phase3 tables/summaries using `weighted_attack_recall_no_backend_fail` or `pred_expected_consequence_served_ratio`, including case14 confirm tables and case39 stage summaries. |
| `expected_consequence_hat` predictor | `README_phase3.md` defines conditional expected consequence as `p_hat * severity_hat`. | `evaluation_budget_scheduler_phase3.py` computes `expected_consequence_hat = clip(p_hat) * max(attack_severity_hat, 0)` when `--consequence_mode conditional` (default). `phase3_oracle_family_core.py` and `phase3_oracle_confirm_core.py` reuse the same calibration logic. | Match at the predictor level. | Phase3 ranking logic itself is aligned with the stated formula. |
| `recover_fail` semantics | Paper/README language suggests “consequence-aware” evaluation, which a reader could reasonably interpret as including recovery-side failure. | `scheduler/calibration.py` extracts `recover_fail`, but `scheduler/policies_phase3.py` only stores it in `AlarmJob.meta`. It is not used by `fit_service_models_from_mixed_bank()`, not part of `severity_true`, and not part of the primary summary metrics. | Mismatch / omission. Recovery failure is carried along but not actually scored. | Any claim that phase3 consequence directly reflects recovery failure is overstated. |
| `backend_fail` semantics | README highlights fail-risk prediction as part of the learned scheduler inputs. | `scheduler/calibration.py` fits `backend_fail` via `fit_service_models_from_mixed_bank()`. `scheduler/policies_phase3.py` uses `pred_fail_prob` in the ranking penalty and uses `actual_backend_fail` to zero out `weighted_attack_recall_no_backend_fail`. | Match. | `weighted_attack_recall_no_backend_fail`, `total_backend_fail`, and fail-risk penalties are implemented as described. |
| `service_time` / `service_cost` semantics | README says phase3 fits `tau_hat` and `cost_hat` from the fit mixed bank. | `scheduler/calibration.py` defines `service_time = stage_one_time + stage_two_time` and `service_cost = delta_cost_one + delta_cost_two`. `scheduler/policies_phase3.py` uses predicted time/cost for ranking and actual time/cost for queueing and summary statistics. | Match. | Queue delay, cost per step, utilization, and optional cost-budget metrics. |

## 2. `scheduler/calibration.py` 里 `severity_true` 现在到底怎么实现

- Source: `scheduler/calibration.py`, `mixed_bank_to_alarm_arrays()`
- Inputs pulled from the mixed bank:
  - `ang_no_summary`
  - `ang_str_summary`
  - `recover_fail`
  - `backend_fail`
  - `stage_one_time`
  - `stage_two_time`
  - `delta_cost_one`
  - `delta_cost_two`
- Concrete implementation:
  - `service_time = stage_one_time + stage_two_time`
  - `service_cost = delta_cost_one + delta_cost_two`
  - `severity_true = max(ang_no, 0) * max(ang_str, 0)` on attack rows
  - `severity_true = 0` on clean rows
- Practical consequence:
  - “truth severity” is still driven by injected attack geometry, not by actual recovery burden, not by `recover_fail`, and not by realized service cost.

## 3. `recover_fail` / `backend_fail` / `service_time` / `service_cost` 在哪里进入 evaluation

- `recover_fail`
  - Entry: `scheduler/calibration.py -> mixed_bank_to_alarm_arrays()`
  - Propagation: `scheduler/policies_phase3.py -> build_jobs_from_arrays()` stores it only in `AlarmJob.meta`
  - Use in metrics/policy: none
  - Audit conclusion: currently not part of the reportable phase3 scoring chain

- `backend_fail`
  - Entry: `scheduler/calibration.py -> mixed_bank_to_alarm_arrays()`
  - Model fit: `fit_service_models_from_mixed_bank()` fits `backend_fail` as `fail_given_verify_score`
  - Policy input: `scheduler/policies_phase3.py -> pred_fail_prob`
  - Metric use: `weighted_attack_recall_no_backend_fail` only counts served attack severity when `actual_backend_fail == 0`

- `service_time`
  - Entry: `scheduler/calibration.py -> stage_one_time + stage_two_time`
  - Model fit: `fit_service_models_from_mixed_bank()` fits `time_given_verify_score`
  - Policy input: predicted busy-time penalty and actual busy-step occupancy
  - Metric use: `average_service_time_per_step`, queueing dynamics, server utilization

- `service_cost`
  - Entry: `scheduler/calibration.py -> delta_cost_one + delta_cost_two`
  - Model fit: `fit_service_models_from_mixed_bank()` fits `cost_given_verify_score`
  - Policy input: predicted cost penalty and optional rolling cost budget
  - Metric use: `average_service_cost_per_step`

## 4. Attack-side recompute / summary 的关键调用链路

### A. Native clean / attack bank recompute path

`case39_stage2_native_run.sh`
-> `collect_clean_alarm_scores.py`
-> `metric/case39/metric_clean_alarm_scores_full.npy`

`case39_stage2_native_run.sh`
-> `collect_attack_alarm_scores.py`
-> `metric/case39/metric_attack_alarm_scores_400.npy`

This path is what replaced the old clean/attack symlinks in stage-2 native repair.

### B. Legacy attack-side support summary path

`evaluation_event_trigger_attack_cli.py`
-> writes `metric/<case>/metric_event_trigger_tau_*.npy`
-> `summarize_attack_support_metric.py`
-> prints compact attack-support summary text

This path summarizes gate-triggered attack runs (`trigger_after_verification`, `fail`, stage times, delta costs), but it is not the direct source of the current phase3 confirm aggregate summaries.

### C. Current phase3 confirm / reportable path

`metric/case39/phase3_confirm_blind_v{1,2}/manifest.json`
+ `metric/case14/phase3_oracle_family/screen_train_val_summary.json`
-> `run_phase3_oracle_confirm.py`
-> `phase3_oracle_confirm_core.run_phase3_oracle_confirm()`
-> `metric/case39/phase3_oracle_confirm_v{1,2}_native_clean_attack_test/aggregate_summary.json`
-> `case39_postrun_audit_bundle.sh`
-> `metric/case39/postrun_audits/20260409_231456/summary.json`

Key detail:
- `phase3_oracle_confirm_core.py` reads `dev_screen_summary["selection"]["winner_variant"]`
- The current dev summary winner is `oracle_protected_ec`
- The manifest frozen regime is exactly:
  - `decision_step_group = 1`
  - `busy_time_quantile = 0.65`
  - `use_cost_budget = false`
  - `slot_budget_list = [1, 2]`
  - `max_wait_steps = 10`

## 5. 当前 paper-definition vs code-definition 是否一致

### Consistency conclusion

- Consistent:
  - The scheduler really does rank on a learned expected-consequence predictor.
  - Default conditional mode really is `p_hat * severity_hat`.
  - `backend_fail`, `service_time`, and `service_cost` are genuinely wired into the policy/evaluation loop.

- Not consistent:
  - The code’s `severity_true` is still the primitive `ang_no * ang_str` proxy.
  - `recover_fail` is not part of the primary phase3 score.
  - So “learned expected consequence” in the paper text is only partially realized: the predictor is learned, but the truth target it learns against is still a limited proxy.

### Affected tables / figures / summaries

- Affected:
  - Any table/figure/summary whose main y-axis is `weighted_attack_recall_no_backend_fail`
  - Any table/figure/summary using `pred_expected_consequence_served_ratio`
  - case14 phase3 confirm main tables
  - case39 transfer/stage summaries and stage comparison bundles

- Not directly affected:
  - Native path existence / checkpoint audits
  - STAMP / anti-write provenance checks
  - Measurement exact-match audits

## Bottom-line audit call

- The repo is internally consistent about the *mechanics* of phase3 scheduling.
- The repo is not fully consistent about the *meaning* of “severity” / “expected consequence”.
- For human decision-making, the safe phrasing is:
  - current phase3 numbers are consequence-aware with respect to a learned proxy
  - but that proxy is still grounded in `ang_no * ang_str`, not a richer recovery-side consequence definition

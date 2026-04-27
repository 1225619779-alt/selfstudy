# Audit Summary

## Bottom Line

当前这批 case39 结果**不能**被表述为 native larger-system evidence。更准确的说法是：

- 它是**bridge evidence** on a larger system
- 不是 fully native case39 evidence
- 同时也提供了一部分 limitation / mechanism evidence

原因很直接：

- native case39 clean / attack / test 路径已经补上
- 但 canonical `train_bank` / `val_bank` 仍然 resolve 到 `metric/case14`
- blind confirm 的 fixed winner 仍来自 `metric/case14/phase3_oracle_family/screen_train_val_summary.json`
- 当前没有 usable STAMP，因此 anti-write 证据不足

## Direct Answers

1. **status**

   `BRIDGE_ONLY_CASE14_BACKED`

   采用这个 status 的原因：
   - `native_readiness_audit.json` 直接给出 `BRIDGE_ONLY_CASE14_BACKED`
   - canonical `train_bank` / `val_bank` 仍然是 case14-backed
   - current transfer confirm 仍然是 `native_clean_attack_test_with_frozen_case14_dev`

2. **canonical `metric/case39` 的 clean / attack / train / val 是否真实 resolve 到 case39，而不是 `metric/case14`**

   结论：**不是全都 native；是混合态**

   - `clean_bank`
     - path-level: resolve 到 `metric/case39/metric_clean_alarm_scores_full.npy`
     - content-level: SHA 和 `metric/case14/metric_clean_alarm_scores_full.npy` 不同
     - audit call: native case39 file
   - `attack_bank`
     - path-level: resolve 到 `metric/case39/metric_attack_alarm_scores_400.npy`
     - content-level: SHA 和 `metric/case14/metric_attack_alarm_scores_400.npy` 不同
     - audit call: native case39 file
   - `train_bank`
     - path-level: symlink resolve 到 `metric/case14/mixed_bank_fit.npy`
     - content-level: SHA 与 case14 完全相同
     - audit call: case14 bridge contamination
   - `val_bank`
     - path-level: symlink resolve 到 `metric/case14/mixed_bank_eval.npy`
     - content-level: SHA 与 case14 完全相同
     - audit call: case14 bridge contamination

3. **`saved_model/case39/checkpoint_rnn.pt` 是否存在**

   存在。

   - path: `/home/pang/projects/DDET-MTD-q1-case39/saved_model/case39/checkpoint_rnn.pt`
   - repo inventory size: `1232693` bytes

4. **`anti_write_q1_case14.txt` 和 `anti_write_oldrepo_case14.txt` 是否为空**

   都**不为空**。

   当前写入内容是：
   - `anti-write 证据不足: no usable current-run STAMP found under /tmp/case39_* at audit time.`

   这表示：
   - 本轮没有 usable current-run STAMP
   - 因此 anti-write 无法被证明成立
   - 本轮只能给出“证据不足”，不能给出“未写入 case14”的强证明

5. **parallel vs sequential measurement 是否 exact-match**

   **成立。**

   本轮重新审计了 5 个 slice：

   - `0:16`
   - `128:144`
   - `4096:4112`
   - `20000:20016`
   - `34000:34016`

   结果：
   - `success_exact_equal = true` for all 5 slices
   - `z_allclose_rtol1e-7_atol1e-9 = true` for all 5 slices
   - `v_allclose_rtol1e-7_atol1e-9 = true` for all 5 slices
   - observed `v_max_abs_diff` is only floating-point noise scale (`<= 1.31e-14`)

6. **当前 merged 8 holdouts 的 stage 是什么，B=1/B=2 的主要指标摘要是什么**

   stage:
   - `native_clean_attack_test_with_frozen_case14_dev`

   current merged 8-holdout winner summary (`phase3_oracle_upgrade`, i.e. fixed `oracle_protected_ec`):

   - `slot_budget = 1`
     - recall: `0.0916625`
     - unnecessary: `13.5`
     - cost: `0.388106625`
     - served_ratio: `0.5612625`
   - `slot_budget = 2`
     - recall: `0.177225`
     - unnecessary: `17.5`
     - cost: `0.5574813750000001`
     - served_ratio: `0.7422124999999999`

   reference comparison on the same merged stage:
   - `phase3_proposed`
     - B=1 recall `0.0927125`, unnecessary `15.875`, cost `0.42615125`
     - B=2 recall `0.175625`, unnecessary `17.5`, cost `0.578423125`
   - `topk_expected_consequence`
     - B=1 recall `0.09415`, unnecessary `16.0`, cost `0.426519375`
     - B=2 recall `0.16765`, unnecessary `17.5`, cost `0.56700075`

7. **paper-definition vs code-definition 的 severity mismatch 是否存在**

   **存在。**

   - predictor-level:
     - yes, code does implement the README-style `expected_consequence_hat = p_hat * severity_hat` default path
   - truth-target-level:
     - no, code truth severity is still `max(ang_no, 0) * max(ang_str, 0)`
     - `recover_fail` is not part of the primary phase3 scoring chain

   所以更准确的说法应当是：
   - “phase3 ranks with a learned expected-consequence predictor”
   - 但它学习和评估的 truth target 仍然是受限 proxy，不是 richer recovery consequence

8. **基于以上证据，当前 case39 更应该被表述为哪一种**

   `bridge evidence`

   理由：
   - clean / attack / test side 已经是 native case39
   - 但 train / val calibration 以及 winner selection 仍然 case14-backed
   - 因而它更像“case14-frozen stack transferred onto a larger native testbed”
   - 还不能叫 fully native case39 evidence

## Red Flags

- <span style="color:red">RED FLAG</span> `status != NATIVE_CASE39_READY`。当前 status 是 `BRIDGE_ONLY_CASE14_BACKED`。
- <span style="color:red">RED FLAG</span> canonical case39 assets 里 `train_bank` 和 `val_bank` 实际 resolve 到 `metric/case14`。
- checkpoint **不是**红旗。本轮确认 `saved_model/case39/checkpoint_rnn.pt` 存在。
- <span style="color:red">RED FLAG</span> `anti_write_q1_case14.txt` 与 `anti_write_oldrepo_case14.txt` 都非空，因为当前没有 usable STAMP，anti-write 证据不足。
- measurement exact-match **不是**红旗。本轮 5 个 slice 都成立。
- <span style="color:red">RED FLAG</span> paper 与 code 的 severity 定义存在 mismatch：predictor 是 learned EC，但 truth severity 仍是 `ang_no * ang_str` proxy。
- <span style="color:red">RED FLAG</span> pre-fix attack-side summary 已失效。例子：winner merged recall 在 `slot_budget=1` 上从 pre-fix 约 `0.72035` 跌到 post-fix `0.0916625`；`slot_budget=2` 上从约 `0.9724625` 跌到 `0.177225`。这些 pre-fix artifacts 不能再当可信科学证据。
- 当前未发现直接证据表明**当前** paper 表/图仍在读取这些 invalid pre-fix case39 aggregate summaries；当前 paper pipeline 指向的是 `metric/case39/postrun_audits/20260409_231456/summary.json` 与 `metric/case39_compare/stage_compare_significance_v2.json`。

## Audit Call

- 当前 case39 结果不能支撑 “native larger-system evidence” 这一定性。
- 当前 case39 结果可以支撑：
  - bridge evidence
  - larger-system transfer limitation evidence
  - mechanism / stress-test support evidence

- 当前 case39 结果不应支撑：
  - fully native case39 main-paper headline claim
  - “all canonical case39 assets are native” 的说法
  - “anti-write 已证伪 contamination” 的说法

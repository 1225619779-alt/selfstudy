# case39 第一批 bridge patch + 第一组正式实验顺序

## 0. 应用 patch

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python /mnt/data/case39_first_batch_patch.py .
```

## 1. 建议先留一个时间戳，后面检查是否误写回 case14

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
STAMP=/tmp/case39_bridge_$(date +%s).stamp
touch "$STAMP"
```

## 2. 先生成 manifest（只写 manifest，不触发 raw test-bank generation）

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python make_phase3_confirm_manifest.py \
  --workdir . \
  --case_name case39 \
  --output_dir metric/case39/phase3_confirm_blind_v1 \
  --manifest_only

python make_phase3_confirm_manifest_v2.py \
  --workdir . \
  --case_name case39 \
  --output_dir metric/case39/phase3_confirm_blind_v2 \
  --manifest_only
```

## 3. staging/import canonical case39 banks（纯迁移，不走 raw generation）

把下面的 `<LOCAL_CASE39_* >` 替换成你本地已经准备好的 case39 资产路径。

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python prepare_case_bank_assets.py \
  --case_name case39 \
  --out_root metric/case39 \
  --clean_src  <LOCAL_CASE39_ASSET_ROOT>/metric_clean_alarm_scores_full.npy \
  --attack_src <LOCAL_CASE39_ASSET_ROOT>/metric_attack_alarm_scores_400.npy \
  --train_src  <LOCAL_CASE39_ASSET_ROOT>/mixed_bank_fit.npy \
  --val_src    <LOCAL_CASE39_ASSET_ROOT>/mixed_bank_eval.npy \
  --mode symlink
```

如果你已经有一个单独的 smoke test bank，也可以一并挂进去：

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python prepare_case_bank_assets.py \
  --case_name case39 \
  --out_root metric/case39 \
  --clean_src  <LOCAL_CASE39_ASSET_ROOT>/metric_clean_alarm_scores_full.npy \
  --attack_src <LOCAL_CASE39_ASSET_ROOT>/metric_attack_alarm_scores_400.npy \
  --train_src  <LOCAL_CASE39_ASSET_ROOT>/mixed_bank_fit.npy \
  --val_src    <LOCAL_CASE39_ASSET_ROOT>/mixed_bank_eval.npy \
  --test_src   <LOCAL_CASE39_ASSET_ROOT>/mixed_bank_test_smoke.npy \
  --mode symlink \
  --force
```

## 4. staging/import blind confirm holdout test banks

这里要求你已经有 case39 的 blind holdout `mixed_bank_test_*.npy`。

### v1 holdouts

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python prepare_case_bank_assets.py \
  --case_name case39 \
  --out_root metric/case39 \
  --skip_canonical \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --holdout_src_dir <LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR> \
  --mode symlink
```

### v2 holdouts

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python prepare_case_bank_assets.py \
  --case_name case39 \
  --out_root metric/case39 \
  --skip_canonical \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --holdout_src_dir <LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR> \
  --mode symlink
```

## 5. generic wiring smoke（不作为 reportable 主结果）

需要你在第 3 步已经挂了 `mixed_bank_test_smoke.npy`。

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python evaluation_budget_scheduler_phase3_holdout.py \
  --clean_bank metric/case39/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case39/metric_attack_alarm_scores_400.npy \
  --train_bank metric/case39/mixed_bank_fit.npy \
  --val_bank metric/case39/mixed_bank_eval.npy \
  --test_bank metric/case39/mixed_bank_test_smoke.npy \
  --slot_budget_list 1 2 \
  --decision_step_group 1 \
  --busy_time_quantile 0.65 \
  --max_wait_steps 10 \
  --output metric/case39/budget_scheduler_phase3_holdout_smoke.npy
```

## 6. 为 blind confirm 生成 baseline holdout summaries

因为 manifest 里现在已经有 test_bank 路径了，而且你已经在第 4 步把这些 holdout bank staged 到位，所以这里不需要 raw generation；脚本会跳过已存在的 test_bank，只补 baseline holdout summaries。

### v1

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python make_phase3_confirm_manifest.py \
  --workdir . \
  --case_name case39 \
  --output_dir metric/case39/phase3_confirm_blind_v1
```

### v2

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python make_phase3_confirm_manifest_v2.py \
  --workdir . \
  --case_name case39 \
  --output_dir metric/case39/phase3_confirm_blind_v2
```

## 7. 跑真正更接近 reportable 的 oracle_confirm 主链

这里需要你提供 case14 冻结 winner 的 dev-screen summary。通常就是你之前 `oracle_protected_ec` 胜出的那份 summary json。

### v1

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v1/manifest.json \
  --dev_screen_summary <CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON> \
  --output metric/case39/phase3_oracle_confirm_v1
```

### v2

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
python run_phase3_oracle_confirm.py \
  --manifest metric/case39/phase3_confirm_blind_v2/manifest.json \
  --dev_screen_summary <CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON> \
  --output metric/case39/phase3_oracle_confirm_v2
```

## 8. 合并 v1 + v2 的 aggregate summaries（手工检查版）

最少先看这两个文件：

```bash
cat metric/case39/phase3_oracle_confirm_v1/aggregate_summary.json | head -n 40
cat metric/case39/phase3_oracle_confirm_v2/aggregate_summary.json | head -n 40
```

建议重点看：
- `formal_decision_summary`
- `slot_budget_aggregates.1`
- `slot_budget_aggregates.2`

## 9. 防回归检查：确认没有写回 metric/case14

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
find metric/case14 -type f -newer "$STAMP" -print

grep -R "metric/case14" \
  metric/case39/phase3_confirm_blind_v1 \
  metric/case39/phase3_confirm_blind_v2 \
  metric/case39/*.json -n
```

两条命令都应该尽量没有输出。

## 10. 找 case14 frozen winner summary 的实用命令

如果你一时忘了那份 summary 放哪，可以先在本地搜：

```bash
cd /home/pang/projects/DDET-MTD-q1-case39
find metric/case14 -type f \( -name "*.json" -o -name "*.md" \) | grep -E "oracle|summary|screen|winner"
```

## 11. 当前阶段的硬边界

- 不开新 family
- 不跑 `select_regime_phase3_val.py` 当 reportable 主结果
- 不做 case39 retune
- `winner = oracle_protected_ec`
- regime 固定：`decision_step_group=1, busy_time_quantile=0.65, use_cost_budget=false, slot_budget_list=[1,2], max_wait_steps=10`

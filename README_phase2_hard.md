
# Budget Scheduler Phase 2 (Hard-Constrained Queueing)

这版是对 Phase 1 的结构性修正。

## 为什么要升级
Phase 1 的问题在于：`service_time` 只被记账，没有真正占用未来时隙。  
结果就是：

- 每步通常只有 1 个 job
- `slot_budget=1` 几乎不形成真实拥塞
- `fifo/random/topk/static_value_cost` 常常表现一样
- 这不是一个真正的 backend scheduling 环境

Phase 2 改成了：

- **hard-constrained busy servers**：每个 backend slot 都是一个 server
- **actual service time -> busy steps**：job 启动后会占用 server 多个 step
- **optional rolling cost budget**：可选的滚动窗口硬成本预算
- **adaptive threshold baseline**：补一个“在线调阈值但不做显式队列调度”的强基线
- **dynamic pressure-aware scheduler**：`proposed_vq_hard`

## 新脚本
- `evaluation_budget_scheduler_hard.py`
- `scheduler/calibration.py`
- `scheduler/policies_hard.py`

## 推荐先跑的命令
先保持和你 Phase 1 一样的 banks，不加 cost budget，先看“硬排队”是否把竞争做出来：

```bash
python evaluation_budget_scheduler_hard.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --max_wait_steps 10 \
  --busy_time_quantile 0.50 \
  --output metric/case14/budget_scheduler_phase2_hard.npy
```

如果结果里：

- `max_queue_len` 还是几乎 0
- `server_utilization` 很低
- `always_fifo/random/topk_verify` 还几乎一样

再试更强一些的拥塞版本：

```bash
python evaluation_budget_scheduler_hard.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --max_wait_steps 10 \
  --busy_time_quantile 0.35 \
  --decision_step_group 2 \
  --output metric/case14/budget_scheduler_phase2_hard_g2.npy
```

## 想把“成本预算”也打开
第二轮再开：

```bash
python evaluation_budget_scheduler_hard.py \
  --clean_bank metric/case14/metric_clean_alarm_scores_full.npy \
  --attack_bank metric/case14/metric_attack_alarm_scores_400.npy \
  --fit_bank metric/case14/mixed_bank_fit.npy \
  --eval_bank metric/case14/mixed_bank_eval.npy \
  --slot_budget_list 1 2 \
  --max_wait_steps 10 \
  --busy_time_quantile 0.50 \
  --use_cost_budget \
  --cost_budget_window_steps 20 \
  --cost_budget_quantile 0.60 \
  --output metric/case14/budget_scheduler_phase2_hard_cost.npy
```

## 参数含义
### `--busy_time_quantile`
把 fit bank 的 service time 分布的某个分位数当成一个 backend busy-time unit。  
例如设为 `0.50`，就表示：

- 大约 service time 在中位数附近的 job -> 占用 1 个 server step
- 更重的 job -> 占用 2、3、... 个 step

这一步的目的，是把连续 service time 变成真正的 server occupancy。

### `--decision_step_group`
把原始 timeline 的多个 step 合并成一个 decision epoch。  
这是可选项。只有在你发现 Phase 2 仍然没有足够竞争时，再开它。

### `--use_cost_budget`
打开滚动窗口硬成本预算。  
如果不开，Phase 2 仍然是一个真正的 **time-capacity constrained queueing** 环境。  
如果打开，就进一步变成 **time + cost constrained**。

## 结果里最该看的指标
- `weighted_attack_recall_no_backend_fail`
- `unnecessary_mtd_count`
- `queue_delay_p95`
- `max_queue_len`
- `server_utilization`
- `budget_blocked_starts`
- `mean_threshold_used`（对 adaptive threshold 有用）

## 你应该期待什么现象
### 好现象
- `always_fifo/random/topk_verify` 不再完全一样
- `queue_delay_p95` 和 `max_queue_len` 明显上来
- `threshold_verify_fifo` 仍然强，但在重拥塞下不再一枝独秀
- `adaptive_threshold_verify_fifo` 比静态 threshold 更强
- `proposed_vq_hard` 在某些预算档位下开始赢 `adaptive_threshold_verify_fifo`

### 坏现象
- 所有策略还是一样
- 队列始终几乎为 0
- `proposed_vq_hard` 只是在 recall 和 unnecessary 之间做很差的交换

如果出现坏现象，不要直接否定方向，先检查：
1. `busy_time_quantile` 是否太大  
2. `decision_step_group` 是否还是 1  
3. schedule 是否本身太稀疏

## 这版还不是最终一区稿
这版重点是把“在线受约束调度环境”做实。  
它比 Phase 1 更接近论文主问题，但还不是最终版本。

后续还需要：
- consequence proxy 从 `verify_score` 升级到 recovery/vector-based proxy
- 第二系统（case39 或更大）
- offline oracle / 更强上界
- 更系统的 burst / budget-shock 实验设计

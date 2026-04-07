# case39 最小 bridge patch 规格说明（基于交接卡）

> 说明：当前会话未挂载 `/home/pang/projects/DDET-MTD-q1-case39`，因此下面给的是**可直接映射到真实仓库的最小改动规格**，目标是：
> 1. 不改 `metric/case14` 冻结资产；
> 2. 新增 `metric/case39`；
> 3. 只把 `case root / bank path / manifest generation / runner entry` 参数化；
> 4. 不重写 phase3 policy shell。

---

## 一、最小改动原则

### 必须保持不动
- `metric/case14/**`
- case14 已冻结的冠军选择、holdout、audit/repro 资产
- phase3 policy shell 的策略逻辑

### 允许做的 bridge 改动
- 把写死的 `case14` 从 **路径/入口参数** 上抽出来，变成 `case_name`
- runner / manifest generator / bank resolver 支持 `case39`
- 新建 `metric/case39` 所需最小目录骨架

### 不要做的事
- 不引入新的 wrapper / overload / state-adm 家族
- 不在 bridge 阶段扩展新的 policy 变体
- 不复制 case14 的结果资产到 case39 当“默认输入”

---

## 二、先定位这四类写死点

优先在 repo 根目录执行：

```bash
git grep -nE 'metric/case14|/case14|\bcase14\b'
git grep -nE 'manifest|bank|runner' -- '*.py' '*.sh' '*.json' '*.yaml' '*.yml'
```

把命中点归为四类：
1. **case root resolver**：任何 `repo_root / "metric" / "case14"`
2. **bank path resolver**：任何从 `metric/case14/.../bank` 取输入的地方
3. **manifest generation**：任何输出 manifest 到 `metric/case14/...` 的地方
4. **runner entry**：任何 CLI / shell / python 入口默认跑 case14 的地方

bridge 只需要改这四类。

---

## 三、建议的最小参数化形态

如果仓库里已经有统一 config/path 模块，**优先在已有模块上扩展**；如果没有，再加一个很薄的 resolver 文件。

### 1）统一 case 选择

```python
# 伪代码：放到现有 config / path / common utils 中
from __future__ import annotations
import os
from pathlib import Path

SUPPORTED_CASES = {"case14", "case39"}


def resolve_case_name(case_name: str | None = None) -> str:
    case = case_name or os.environ.get("CASE_NAME") or "case14"
    if case not in SUPPORTED_CASES:
        raise ValueError(f"Unsupported case: {case}")
    return case


def resolve_metric_case_root(repo_root: Path, case_name: str | None = None) -> Path:
    case = resolve_case_name(case_name)
    root = repo_root / "metric" / case
    return root
```

关键点：
- 默认值仍是 `case14`，保证旧链路不炸
- case39 通过 `--case case39` 或环境变量 `CASE_NAME=case39` 打通

### 2）bank path 统一从 case root 派生

把这类写死：

```python
repo_root / "metric" / "case14" / "bank"
```

改成：

```python
case_root = resolve_metric_case_root(repo_root, case_name)
bank_root = case_root / "bank"
```

如果仓库里 bank 还有子层级（例如 `bank/raw`、`bank/processed`、`bank/blind`），都从 `bank_root` 继续派生，不再在后面重新拼 `case14`。

### 3）manifest generation 从 case root 派生

把这类写死：

```python
manifest_out = repo_root / "metric" / "case14" / "manifests" / manifest_name
```

改成：

```python
case_root = resolve_metric_case_root(repo_root, case_name)
manifest_out = case_root / "manifests" / manifest_name
manifest_out.parent.mkdir(parents=True, exist_ok=True)
```

### 4）runner entry 只加 case 参数，不改 phase3 shell 逻辑

Python 入口：

```python
parser.add_argument("--case", default="case14", choices=["case14", "case39"])
```

调用链只需要把 `args.case` 往下透传到：
- repo/case root resolver
- bank resolver
- manifest generator
- output root resolver

Shell 入口如果原来是：

```bash
python xxx.py --policy phase3
```

最小 bridge 改成：

```bash
CASE_NAME="${CASE_NAME:-case14}"
python xxx.py --case "${CASE_NAME}" --policy phase3
```

注意：
- **不要重写 phase3 shell 的策略判断分支**
- 只加 `CASE_NAME` 透传即可

---

## 四、`metric/case39` 最小目录骨架

按 runner/manifest/bank 实际期望的目录建最小骨架。典型最小版：

```text
metric/
  case39/
    bank/
      .gitkeep
    manifests/
      .gitkeep
    outputs/
      .gitkeep
```

如果现有 runner 还依赖别的固定子目录（例如 `tmp/`, `logs/`, `cache/`, `reports/`），**只创建空目录骨架**，不要复制 case14 的内容进去。

如果 case39 现在已经有 4 个文件，就保持已有文件不动，只补齐 runner 真正会写入/读取的缺口目录。

---

## 五、推荐的实际替换模式

### 模式 A：路径写死

**替换前**
```python
CASE14_ROOT = REPO_ROOT / "metric" / "case14"
BANK_DIR = CASE14_ROOT / "bank"
MANIFEST_DIR = CASE14_ROOT / "manifests"
```

**替换后**
```python
case_root = resolve_metric_case_root(REPO_ROOT, case_name)
BANK_DIR = case_root / "bank"
MANIFEST_DIR = case_root / "manifests"
```

### 模式 B：函数内部隐式写死 case14

**替换前**
```python
def load_bank(repo_root: Path):
    bank_root = repo_root / "metric" / "case14" / "bank"
    ...
```

**替换后**
```python
def load_bank(repo_root: Path, case_name: str = "case14"):
    bank_root = resolve_metric_case_root(repo_root, case_name) / "bank"
    ...
```

### 模式 C：manifest builder 无 case 参数

**替换前**
```python
def build_manifest(repo_root: Path, tag: str):
    out = repo_root / "metric" / "case14" / "manifests" / f"{tag}.json"
```

**替换后**
```python
def build_manifest(repo_root: Path, tag: str, case_name: str = "case14"):
    out = resolve_metric_case_root(repo_root, case_name) / "manifests" / f"{tag}.json"
```

### 模式 D：runner 入口没有 case 概念

**替换前**
```python
def main():
    run_phase3()
```

**替换后**
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case14", choices=["case14", "case39"])
    args = parser.parse_args()
    run_phase3(case_name=args.case)
```

---

## 六、最小 smoke 通过标准

bridge 完成后，至少要看到下面 4 件事成立：

1. `--case case39` 不再触发任何 `metric/case14` 路径写入
2. manifest 能生成到 `metric/case39/manifests/...`
3. bank 读取走的是 `metric/case39/bank/...`
4. runner 输出落到 `metric/case39/...`，且 phase3 shell 无需重写策略逻辑

建议再加一个“防回归断言”：

```python
assert "metric/case14" not in str(resolved_output_path)
```

这个断言适合加在 case39 smoke test 或 audit 脚本里，不要塞进 case14 冻结资产本体。

---

## 七、最小运行顺序（bridge 后）

下面是**最小运行顺序**，优先验证“能跑通”，不是一上来跑全量实验：

### Step 1：补齐 case39 目录骨架
```bash
mkdir -p metric/case39/bank metric/case39/manifests metric/case39/outputs
```

### Step 2：跑 case39 manifest 生成 smoke
```bash
python <manifest_entry>.py --case case39 <其余最小必要参数>
```

验收：
- `metric/case39/manifests/` 下出现目标 manifest
- 内容里不再引用 `metric/case14`

### Step 3：跑 bank resolve / audit smoke
```bash
python <bank_or_audit_entry>.py --case case39 <最小必要参数>
```

验收：
- 读取入口指向 `metric/case39/bank`
- 不回落到 case14 bank

### Step 4：跑 runner 最小样本 smoke
```bash
python <runner_entry>.py --case case39 --policy phase3 <最小样本参数>
```
或 shell：
```bash
CASE_NAME=case39 bash <existing_phase3_shell>.sh <最小样本参数>
```

验收：
- 输出路径落在 `metric/case39/...`
- phase3 shell 本身没有被重写，只是透传了 case

### Step 5：跑对比链路（在 smoke 通过后）
按你交接卡的目标，后续再进入：
1. `phase3`
2. `oracle_protected_ec`
3. `best-threshold`
4. `topk`

这里建议先从**最小 slice / 单 family / 小 budget**开始，确认 case39 桥接没有路径污染，再跑全量。

---

## 八、我建议你在真实仓库里优先改的文件类型

按概率从高到低排查：
- `*runner*.py`
- `*manifest*.py`
- `*audit*.py`
- `*path*.py`, `*config*.py`, `*common*.py`
- 现有 phase3 shell / launch shell

只要这些入口完成参数化，通常就足够 bridge；不需要大范围重构。

---

## 九、最小提交说明（commit message 可直接用）

```text
bridge(case39): parameterize case root/bank/manifest/runner without touching frozen case14 assets
```

更细一点也可以拆成两提交：
1. `bridge(case39): add case-aware path resolution for bank and manifests`
2. `bridge(case39): thread --case through runners and create metric/case39 skeleton`

---

## 十、真正落库时的裁剪原则

如果某个写死点只影响 case14 历史审计、复现实验、冻结报告，不必强行改；
只改**会阻止 case39 bank/manifest/runner 打通**的最短路径。

也就是说，本轮 bridge 的完成定义不是“仓库里完全没有 case14 字符串”，而是：
- case39 能独立 resolve root
- case39 能生成 manifest
- case39 能读写 bank / outputs
- runner 能以 `--case case39` 跑通


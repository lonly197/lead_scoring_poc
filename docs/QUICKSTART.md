# 快速开始指南

本文档提供销售线索智能评级 POC 项目的快速入门指南。

---

## 环境准备

```bash
# 使用 uv 安装依赖
uv sync

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 配置数据路径
```

---

## 数据诊断

在训练前，可使用诊断脚本检查数据格式：

```bash
# 检查数据格式和关键列
uv run python scripts/diagnose_data.py ./data/202603.tsv
```

**诊断功能**：

- 检测分隔符和列数
- 验证关键列是否存在（线索创建时间、线索唯一ID 等）
- 显示 OHAB 评级分布、是否仍保留 O 级
- 检查 `is_final_ordered` 是否存在及其分布
- 输出 `线索创建时间` 范围，便于确认是否会触发自动 OOT
- 列出所有列名和样本值（方便调试映射问题）

---

## 运行训练

### 原数据格式训练（train_arrive, train_ohab）

适用于原格式数据（如 `20260308-v2.csv`，逗号分隔，有表头，目标变量预计算）。

```bash
# 方式 1：前台运行（推荐用于调试）
uv run python scripts/train_arrive.py

# 方式 2：后台运行（推荐用于长时间训练）
uv run python scripts/run.py train_arrive --daemon

# 方式 3：指定参数后台运行
uv run python scripts/run.py train_arrive --daemon \
    --preset high_quality \
    --time-limit 3600 \
    --data-path ./data/20260308-v2.csv

# 可选：试驾预测训练
uv run python scripts/run.py train_test_drive --daemon --preset high_quality

# 可选：OHAB 评级训练
uv run python scripts/run.py train_ohab --daemon --preset high_quality
```

### HAB 评级训练与验证（推荐 POC 主流程）

适用于新格式数据（如 `202602~03.tsv`，跨度覆盖 2 月和 3 月）。当前推荐将 `train_ohab.py` 用作 **H/A/B 智能评级主入口**：

- 默认 `--label-mode hab`，仅训练 `H/A/B`
- `O` 视为已成交/已锁单状态，不进入常规评级桶
- 训练完成后会在模型目录输出：
  - `feature_importance.*`
  - `business_dimension_contribution.*`
  - `predictions_test.csv`
  - `hab_bucket_summary.*`
  - `evaluation_summary.json`

**推荐切分策略**：

对 `202602~03.tsv` 这类跨月数据，建议固定手动 OOT 切分，保证每次汇报口径一致。

```bash
# 推荐：HAB 评级训练（固定 OOT 口径）
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --preset high_quality \
    --num-bag-folds 5 \
    --label-mode hab \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

**训练完成后查看状态**：

```bash
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_ohab -f
```

### 验证 HAB 结果

训练结束后，直接对同一份数据跑验证脚本。验证脚本会自动读取模型元数据里的：

- `split_info`
- `label_policy`
- `label_mode`
- `decision_policy`

因此不需要手动再指定阈值。

```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv
```

验证目录默认输出到 `outputs/validation/`，重点关注：

- `evaluation_report.txt`
- `hab_bucket_summary.csv`
- `lead_actions.csv`
- `monotonicity_check.json`

其中：

- `hab_bucket_summary.csv`：看 `H/A/B` 三桶的 14 天到店率、试驾率是否形成单调分层
- `lead_actions.csv`：可直接给业务看，包含 `预测HAB + 建议SOP + 原因1-3`

### 生成客户版 Markdown 报告

如果要把训练和验证结果整理成面向客户的汇报文档，继续执行：

```bash
uv run python scripts/generate_business_report.py \
    --model-dir ./outputs/models/ohab_model \
    --validation-dir ./outputs/validation \
    --output-path ./outputs/reports/hab_poc_report.md
```

### 服务器最短闭环

在服务器上完成一次最短闭环，按下面顺序执行即可：

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 训练 HAB 模型
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --preset high_quality \
    --num-bag-folds 5 \
    --label-mode hab \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20

# 3. 验证
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv

# 4. 生成客户版报告
uv run python scripts/generate_business_report.py \
    --model-dir ./outputs/models/ohab_model \
    --validation-dir ./outputs/validation \
    --output-path ./outputs/reports/hab_poc_report.md
```

---

## 监控后台任务

```bash
# 查看运行中的任务
uv run python scripts/monitor.py status

# 查看任务详情
uv run python scripts/monitor.py detail train_arrive

# 查看任务日志（显示最新任务的日志）
uv run python scripts/monitor.py log train_arrive

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive -f

# 停止指定任务
uv run python scripts/monitor.py stop train_arrive

# 一键停止所有运行中的任务
uv run python scripts/monitor.py stop --all

# 列出所有任务（包括已完成）
uv run python scripts/monitor.py list
```

---

## 生成 Top-K 名单

```bash
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/arrive_model \
    --data-path ./data/20260308-v2.csv \
    --k 100 500 1000
```

---

## 模型推理

### 使用已训练模型预测

```python
from autogluon.tabular import TabularPredictor

# 加载模型
predictor = TabularPredictor.load("outputs/models/arrive_model")

# 预测新数据
import pandas as pd
new_data = pd.read_csv("new_leads.csv")

# 预测类别
predictions = predictor.predict(new_data)

# 预测概率
probabilities = predictor.predict_proba(new_data)
```

### 命令行快速查看

```bash
# 查看模型信息
uv run python -c "
from autogluon.tabular import TabularPredictor
p = TabularPredictor.load('outputs/models/arrive_model')
print('最佳模型:', p.model_best)
print('评估指标:', p.eval_metric)
"
```

### 验证脚本

```bash
# 使用验证脚本（默认读取 ./outputs/models/ohab_model）
uv run python scripts/validate_model.py --data-path ./data/202602~03.tsv

# 显式指定模型路径，避免误用旧的 ohab_oot 目录
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv
```

**验证行为说明**：
- 若模型元数据中存在 `split_info`，脚本会自动按训练时记录的 `valid_end` 只评估 OOT 测试集，避免混入训练段和验证段数据。
- 若验证数据中存在 `is_final_ordered`，脚本会额外输出“AI 评级 vs 最终下定”的业务转化统计，但该列不会参与模型特征。

---

## 更多文档

- [FAQ 常见问题](FAQ.md)
- [训练脚本说明](TRAINING.md)
- [配置说明](CONFIGURATION.md)
- [架构说明](ARCHITECTURE.md)

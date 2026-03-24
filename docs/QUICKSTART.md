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
- 显示 OHAB 评级分布
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

### OOT 验证训练（train_arrive_oot, train_ohab_oot）

适用于新格式数据（如 `202603.tsv`，Tab 分隔，无表头，需计算目标变量）。

**数据要求**：
- 文件格式：`.tsv` 或 `.csv`（自动检测分隔符）
- 自动启用数据适配器（`auto_adapt=True`）
- 目标变量自动从原始时间字段派生

**OOT 三层时间切分**：
```
训练集: < 2026-03-11
验证集: 2026-03-11 ~ 2026-03-16
测试集: >= 2026-03-16（有完整 7 天观察期）
```

```bash
# 到店预测（7天窗口）- 后台运行
uv run python scripts/run.py train_arrive_oot --daemon \
    --data-path ./data/202603.tsv \
    --preset high_quality \
    --time-limit 3600

# OHAB 评级 - 后台运行
uv run python scripts/run.py train_ohab_oot --daemon \
    --data-path ./data/202603.tsv \
    --preset high_quality \
    --time-limit 3600

# 自定义 OOT 切分日期
uv run python scripts/run.py train_arrive_oot --daemon \
    --data-path ./data/202603.tsv \
    --train-end 2026-03-10 \
    --valid-end 2026-03-15

# 前台运行（调试用）
uv run python scripts/train_arrive_oot.py --data-path ./data/202603.tsv
```

**查看 OOT 训练状态**：

```bash
# 查看运行状态
uv run python scripts/monitor.py status

# 查看日志
uv run python scripts/monitor.py log train_arrive_oot -f
uv run python scripts/monitor.py log train_ohab_oot -f

# 停止任务
uv run python scripts/monitor.py stop train_arrive_oot
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
# 使用验证脚本
uv run python scripts/validate_model.py

# 使用新数据验证
uv run python scripts/validate_model.py --data-path /path/to/new_data.csv
```

---

## 更多文档

- [FAQ 常见问题](FAQ.md)
- [训练脚本说明](TRAINING.md)
- [配置说明](CONFIGURATION.md)
- [架构说明](ARCHITECTURE.md)
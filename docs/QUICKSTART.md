# 快速开始指南

5 分钟上手销售线索智能评级 POC 项目。

---

## 环境准备

```bash
# 安装依赖
uv sync

# 复制环境变量配置
cp .env.example .env
```

---

## 一键训练

```bash
# 试驾预测模型（三模型集成）
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# 下订预测模型（三模型集成）
uv run python scripts/run.py train order_after_drive --daemon --included-model-types CAT

# HAB 评级模型
uv run python scripts/run.py train ohab --daemon
```

---

## 监控任务

```bash
# 查看状态
uv run python scripts/run.py monitor status

# 跟踪日志
uv run python scripts/run.py monitor log train_ensemble -f

# 停止任务
uv run python scripts/run.py monitor stop --all
```

---

## 验证与预测

```bash
# 验证模型
uv run python scripts/run.py validate --model-path outputs/models/test_drive_ensemble

# 预测（中等模式，推荐）
uv run python scripts/predict.py \
    --mode medium \
    --ensemble-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/test.parquet \
    --output ./predictions.csv \
    --include-ohab
```

---

## 预测模式

| 模式 | 描述 | 业务匹配度 |
|------|------|-----------|
| `simple` | 单模型（14天试驾概率） | 部分 |
| `medium` | 试驾三模型集成（7/14/21天） | 完整匹配试驾前阶段 |
| `advanced` | 试驾+下订双阶段 | 完整匹配全部规则 |

---

## OHAB 评级说明

| 评级 | 定义 | 说明 |
|------|------|------|
| O | 已成交 | 已订车/已成交 |
| H | 7天内试驾/下订 | 高意向 |
| A | 14天内试驾/下订 | 中意向 |
| B | 21天内试驾/下订 | 低意向 |

---

## 更多文档

| 文档 | 说明 |
|------|------|
| [CONFIGURATION.md](CONFIGURATION.md) | 配置参数详解 |
| [TRAINING.md](TRAINING.md) | 训练脚本说明 |
| [SCRIPTS.md](SCRIPTS.md) | 脚本完整索引 |
| [FAQ.md](FAQ.md) | 常见问题 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 故障排查 |
# 训练脚本说明

本文档介绍训练脚本的使用方法和参数。

---

## 统一入口

```bash
uv run python scripts/run.py train <task> [options]
```

### 任务类型

| 任务 | 目标变量 | 说明 |
|------|----------|------|
| `ensemble` | 试驾标签_7/14/21天 | 三模型集成（推荐） |
| `order_after_drive` | 下订标签_7/14/21天 | 下订预测三模型 |
| `ohab` | 线索评级结果 | HAB 评级 |
| `test_drive` | 试驾标签_14天 | 单模型 |
| `arrive` | 到店标签_14天 | 单模型 |

---

## 常用命令

```bash
# 试驾预测（三模型集成，推荐）
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# 下订预测（三模型集成）
uv run python scripts/run.py train order_after_drive --daemon --included-model-types CAT

# HAB 评级
uv run python scripts/run.py train ohab --daemon

# 仅训练 CatBoost
uv run python scripts/run.py train test_drive --daemon --included-model-types CAT

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon --parallel --max-workers 3

# 提前拆分数据文件
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet
```

---

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 数据文件路径 | `.env` 配置 |
| `--train-path` | 训练集路径（提前拆分模式） | - |
| `--test-path` | 测试集路径（提前拆分模式） | - |
| `--preset` | 模型预设 | `good_quality` |
| `--time-limit` | 训练时间限制（秒） | 5400 |
| `--included-model-types` | 模型类型（CAT/GBM/XGB） | 空（全部） |
| `--parallel` | 并行训练三模型（仅 ensemble） | false |
| `--max-workers` | 并行进程数 | 3 |
| `--daemon` | 后台运行 | false |

---

## 预设选择

| 预设 | 磁盘需求 | 训练时间 | 适用场景 |
|------|----------|----------|----------|
| `medium_quality` | ~1G | ~15分钟 | 快速验证 |
| `good_quality` | ~2G | ~30分钟 | 16GB 服务器推荐 |
| `high_quality` | ~4G | ~1小时 | 更大机器 |

---

## 训练档位

| 档位 | 适用场景 | 关键配置 |
|------|----------|----------|
| `server_16g_compare` | 16GB 服务器（默认） | good_quality + 3 folds + two_stage |
| `server_16g_fast` | 快速验证 | medium_quality + 0 folds |
| `lab_full_quality` | 大内存机器 | high_quality + 5 folds |

```bash
uv run python scripts/run.py train ohab --daemon \
    --training-profile server_16g_compare
```

---

## 模型验证

```bash
# 验证模型
uv run python scripts/run.py validate --model-path outputs/models/test_drive_ensemble

# 指定测试集
uv run python scripts/run.py validate \
    --model-path outputs/models/test_drive_ensemble \
    --test-path ./data/test.parquet
```

---

## 标签泄漏验证

所有训练脚本自动验证并排除兄弟标签：

```python
# 训练脚本中的验证逻辑
leaks = validate_no_label_leakage(list(train_df.columns), target_label)
if leaks:
    raise ValueError(f"检测到泄漏标签: {leaks}")
```

| 标签组 | 包含标签 |
|--------|----------|
| `test_drive_label_group` | 试驾标签_7天/14天/21天/30天 |
| `arrive_label_group` | 到店标签_7天/14天/30天 |
| `order_label_group` | 下订标签_7天/14天/21天 |
| `ohab_label_group` | label_OHAB, 线索评级结果 |
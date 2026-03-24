# 训练脚本说明

本文档详细介绍销售线索智能评级 POC 项目的训练脚本及其使用方法。

---

## 脚本概览

本项目提供五个独立的训练脚本，对应不同的预测任务。**它们之间没有依赖关系，可以独立运行。**

### 执行优先级

```
┌─────────────────────────────────────────────────────────────┐
│  train_arrive.py        ← 核心任务，POC 主指标，优先执行     │
├─────────────────────────────────────────────────────────────┤
│  train_arrive_oot.py    ← 核心任务（OOT验证，新数据格式）   │
│  train_test_drive.py    ← 辅助指标，可选执行                │
│  train_ohab.py          ← 辅助任务，可选执行                │
│  train_ohab_oot.py      ← 辅助任务（OOT验证，新数据格式）   │
└─────────────────────────────────────────────────────────────┘
```

### 详细对比

| 脚本 | 目标变量 | 任务类型 | 数据格式 | 优先级 |
|------|----------|----------|----------|--------|
| `train_arrive.py` | `到店标签_14天` | 二分类 | 原格式 | **核心** |
| `train_arrive_oot.py` | `到店标签_7天` | 二分类 | 新格式（自动适配） | **核心** |
| `train_test_drive.py` | `试驾标签_14天` | 二分类 | 原格式 | 辅助 |
| `train_ohab.py` | `线索评级_试驾前` | 多分类 | 原格式 | 辅助 |
| `train_ohab_oot.py` | `线索评级_试驾前` | 多分类 | 新格式（自动适配） | 辅助 |

---

## OOT 脚本特性

`train_arrive_oot.py` 和 `train_ohab_oot.py` 是针对新数据格式的 OOT 验证版本：

| 特性 | 说明 |
|------|------|
| 数据适配 | 自动启用 `auto_adapt=True`，支持无表头、Tab 分隔的新数据 |
| OOT 切分 | 三层时间切分：训练集、验证集、测试集 |
| 目标变量 | 自动从原始时间字段派生（如 `到店时间 - 线索创建时间`） |
| 后台运行 | 通过 `run.py --daemon` 支持 |

---

## 命令行参数

所有训练脚本支持相似的参数：

```bash
# 原格式脚本参数
uv run python scripts/train_arrive.py \
    --data-path ./data/your_data.csv \    # 数据文件路径
    --target 到店标签_14天 \               # 目标变量
    --preset high_quality \                # AutoGluon 预设
    --time-limit 3600 \                    # 训练时间限制（秒）
    --test-size 0.2 \                      # 测试集比例
    --output-dir ./outputs/models/arrive_model

# OOT 脚本额外参数
uv run python scripts/train_arrive_oot.py \
    --data-path ./data/202603.tsv \
    --target 到店标签_7天 \
    --train-end 2026-03-11 \               # 训练集截止日期
    --valid-end 2026-03-16 \               # 验证集截止日期
    --preset high_quality \
    --time-limit 3600
```

### 数据路径指定

支持三种方式指定数据文件：

```bash
# 方式 1：使用默认路径（.env 中的 DATA_PATH）
uv run python scripts/train_arrive.py

# 方式 2：命令行指定数据文件（推荐）
uv run python scripts/train_arrive.py --data-path /path/to/data.csv

# 方式 3：同时指定数据和目标变量
uv run python scripts/train_arrive.py \
    --data-path ./data/custom_data.csv \
    --target 到店标签_30天
```

**参数优先级**：命令行参数 > `.env` 环境变量 > `config/config.py` 默认值

### 预设选项

| 预设 | 磁盘需求 | 训练时间 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| `medium_quality` | ~1G | ~15分钟 | 中等 | 快速验证、磁盘紧张 |
| `good_quality` | ~2G | ~30分钟 | 良好 | 平衡方案 |
| `high_quality` | ~4G | ~1小时 | 高 | **推荐，生产使用** |
| `best_quality` | ~8G | ~4小时 | 最高 | 最终优化 |

---

## 评估指标

### 核心指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| ROC-AUC | 整体区分能力 | 模型区分正负类的能力 |
| Top-K 命中率 | Top-K 中的实际到店比例 | `命中数 / K` |
| Lift | 模型比随机好多少 | `命中率 / 基线转化率` |

### Lift 解读

```
Lift = 2.0  →  Top-K 的命中率是随机的 2 倍
Lift = 1.0  →  与随机相同，模型无效
Lift > 1.5  →  模型有实际业务价值
```

### 示例报告

```json
{
  "topk_metrics": {
    "top_100": {"hit_rate": 0.45, "lift": 3.2},
    "top_500": {"hit_rate": 0.38, "lift": 2.7},
    "top_1000": {"hit_rate": 0.32, "lift": 2.3}
  }
}
```

---

## 磁盘空间管理

### 训练前检查

训练脚本会在启动时自动检查磁盘空间，并在空间不足时给出警告：

```
磁盘状态: 剩余 1.2G / 需要 4.0G (high_quality)
WARNING: 磁盘空间不足！建议使用 medium_quality preset
```

### 磁盘空间建议

| 预设 | 最低磁盘空间 | 推荐磁盘空间 |
|------|--------------|--------------|
| `medium_quality` | 2G | 3G+ |
| `good_quality` | 3G | 5G+ |
| `high_quality` | 5G | 8G+ |
| `best_quality` | 10G | 15G+ |

### 训练后自动清理

训练完成后会自动清理非最佳模型，释放磁盘空间：

```
模型目录大小: 1250.5 MB
清理完成: 释放 520.3 MB
```

### 手动清理命令

```bash
# 清理失败的模型目录
rm -rf outputs/models/ohab_model/

# 清理 AutoGluon 缓存
rm -rf ~/.cache/autogluon/
rm -rf /tmp/autogluon*

# 清理 Ray 会话缓存
rm -rf /tmp/ray/
```
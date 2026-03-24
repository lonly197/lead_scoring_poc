# 训练脚本说明

本文档详细介绍销售线索智能评级 POC 项目的训练脚本及其使用方法。

---

## 脚本概览

本项目提供统一的智能自适应训练脚本。**脚本会自动识别数据时间跨度并选择最佳切分策略。**

### 核心脚本列表

| 脚本 | 目标变量 | 任务类型 | 核心特性 |
|------|----------|----------|----------|
| `train_arrive.py` | `到店标签_14天` | 二分类 | 智能自适应 OOT/随机切分，输出 Top-K/Lift |
| `train_ohab.py` | `线索评级_试驾前` | 多分类 | 智能自适应 OOT/随机切分，多类别权重平衡 |
| `train_test_drive.py` | `试驾标签_14天` | 二分类 | 支持所有特征工程与自适应切分 |

---

## 🚀 智能自适应切分逻辑

为了平衡“单日快照数据”与“跨月长周期数据”的处理需求，脚本内置了智能探查机制：

1.  **自动探查**：启动时自动计算 `线索创建时间` 的时间跨度。
2.  **自适应选择**：
    *   **OOT 模式 (跨度 $\ge 14$ 天)**：自动按照 `70%训练 / 15%验证 / 15%测试` 的比例进行时间轴切分。模拟“用过去预测未来”。
    *   **随机切分模式 (跨度 $< 14$ 天)**：自动降级为分层随机切分（80/20），并提取测试集样本 ID 作为“防泄漏指纹”。
3.  **防泄漏闭环**：无论哪种模式，切分元数据都会保存在 `feature_metadata.json` 中，`validate_model.py` 会自动识别并实施物理隔离。

---

## 命令行参数

所有训练脚本支持相似的参数：

```bash
# 统一自适应脚本参数
uv run python scripts/train_arrive.py \
    --data-path ./data/202602_03.csv \      # 数据文件路径
    --target 到店标签_14天 \               # 目标变量
    --preset high_quality \                # AutoGluon 预设
    --time-limit 3600 \                    # 训练时间限制（秒）
    --num-bag-folds 5                      # 交叉验证折数
    --output-dir ./outputs/models/arrive_model

# 手动指定 OOT 日期（可选）
uv run python scripts/train_arrive.py \
    --data-path ./data/202603.tsv \
    --train-end 2026-03-11 \               # 训练集截止日期
    --valid-end 2026-03-16 
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

### --time-limit 参数详解

`--time-limit` 是 AutoGluon 的核心参数，控制模型训练的总时间（秒）。

**工作原理**：

AutoGluon 是"时间驱动"的 AutoML 框架，在时间限制内自动尝试多种模型（XGBoost、LightGBM、CatBoost、神经网络等），时间到后返回最佳模型。

**时间与预设匹配**：

| time-limit | 推荐预设 | 适用场景 |
|------------|----------|----------|
| 300 (5分钟) | `medium_quality` | 快速验证、调试流程 |
| 1800 (30分钟) | `good_quality` | 初步评估模型效果 |
| 3600 (1小时) | `high_quality` | 生产级模型（默认值） |
| 7200+ (2小时+) | `best_quality` | 追求极致性能 |

**重要**：时间应与预设匹配。过短时间配合高质量预设会导致模型训练不充分；过长时间配合低质量预设则浪费资源。

---

## 当前数据集分析

### 数据概况（202603.tsv）

| 项目 | 数值 |
|------|------|
| 总数据量 | 286,823 行 |
| 原始列数 | 46 列 |
| 派生列数 | 10 列（目标变量 + 时间特征） |
| 时间范围 | 2026-03-01 ~ 2026-03-23 |

### OHAB 评级分布

```
线索评级结果 分布:
H      112,012 (39%)  - 高意向
A       87,175 (30%)  - 中等意向
B       10,805 (4%)   - 低意向
O            12 (0.006%)  - 已成交
Unknown    76,819 (27%)  - 未评级
```

### ⚠️ 数据质量问题：O 级样本不足

**问题**：O 级（已订车/已成交）仅有 12 个样本，极度不平衡。

**影响**：
- 12 个样本无法学习有效模式
- 多分类模型可能忽略 O 级或过拟合
- 评估指标不可靠

**建议方案**：

| 方案 | 适用场景 | 实现方式 |
|------|----------|----------|
| **降级为三分类** | O 级无预测价值 | 过滤 O 级，预测 H/A/B |
| **合并为二分类** | 只需区分高/低意向 | H/A = 高意向，B/O = 低意向 |
| **等待数据沉淀** | O 级有业务价值 | 收集更多成交数据后再训练 |

### 推荐训练配置

基于当前数据集（286,823 行，过滤后 ~210,000 行）：

```bash
# 推荐配置（平衡时间和质量）
uv run python scripts/train_ohab_oot.py \
    --data-path ./data/202603.tsv \
    --preset good_quality \
    --time-limit 1800

# 快速验证配置（调试用）
uv run python scripts/train_ohab_oot.py \
    --data-path ./data/202603.tsv \
    --preset medium_quality \
    --time-limit 600
```

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
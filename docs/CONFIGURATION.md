# 配置说明

本文档说明项目的配置方式和数据字段。

---

## 环境变量（`.env`）

```bash
# 数据配置
DATA_PATH=./data/202602~03.tsv

# 通用模型配置
MODEL_PRESET=good_quality
TIME_LIMIT=3600
RANDOM_SEED=42

# HAB/OHAB 训练配置（16GB 服务器）
OHAB_TRAINING_PROFILE=server_16g_compare
OHAB_MODEL_PRESET=good_quality
OHAB_TIME_LIMIT=5400
OHAB_EVAL_METRIC=balanced_accuracy
OHAB_NUM_BAG_FOLDS=3
OHAB_LABEL_MODE=hab
OHAB_ENABLE_MODEL_COMPARISON=true
OHAB_BASELINE_FAMILY=gbm
OHAB_MEMORY_LIMIT_GB=
OHAB_FIT_STRATEGY=sequential
OHAB_EXCLUDED_MODEL_TYPES=RF,XT,KNN,FASTAI,NN_TORCH
OHAB_NUM_FOLDS_PARALLEL=1
OHAB_MAX_MEMORY_RATIO=0.7
OHAB_GENERATE_PLOTS=false
OHAB_SPLIT_MODE=random
OHAB_AUTO_OOT_MIN_DAYS=90
OHAB_PIPELINE_MODE=two_stage
OHAB_SPLIT_GROUP_MODE=phone_or_lead
OHAB_FEATURE_PROFILE=auto_scorecard
OHAB_ENABLE_AUTO_DEGRADE=true
OHAB_ENABLE_PREP_CACHE=true
OHAB_FORCE_REBUILD_CACHE=false
OHAB_ENABLE_RETRY_ON_MEMORY_ERROR=true

# 输出配置
OUTPUT_DIR=./outputs
```

---

## 参数优先级

```
命令行参数 > .env 环境变量 > config/config.py 默认值
```

---

## OHAB 配置详解

| 环境变量 | 说明 | 推荐值 |
|----------|------|--------|
| `OHAB_TRAINING_PROFILE` | 训练档位 | `server_16g_compare` |
| `OHAB_MODEL_PRESET` | 模型预设 | `good_quality` |
| `OHAB_TIME_LIMIT` | 总训练时长（秒） | `5400` |
| `OHAB_EVAL_METRIC` | 训练阶段选模指标 | `balanced_accuracy` |
| `OHAB_NUM_BAG_FOLDS` | Bagging 折数 | `3` |
| `OHAB_LABEL_MODE` | 评级模式 | `hab` |
| `OHAB_ENABLE_MODEL_COMPARISON` | 保留基线模型对比 | `true` |
| `OHAB_BASELINE_FAMILY` | 基线模型家族 | `gbm` |
| `OHAB_MEMORY_LIMIT_GB` | 训练内存上限 | 留空自动探测 |
| `OHAB_FIT_STRATEGY` | 模型训练策略 | `sequential` |
| `OHAB_EXCLUDED_MODEL_TYPES` | 排除的高内存模型 | `RF,XT,KNN,FASTAI,NN_TORCH` |
| `OHAB_NUM_FOLDS_PARALLEL` | 并行折数 | `1` |
| `OHAB_MAX_MEMORY_RATIO` | 单模型内存比例上限 | `0.7` |
| `OHAB_GENERATE_PLOTS` | 生成 PNG 图表 | `false` |
| `OHAB_SPLIT_MODE` | 切分模式 | `random` |
| `OHAB_AUTO_OOT_MIN_DAYS` | 自动 OOT 最小跨度 | `90` |
| `OHAB_PIPELINE_MODE` | 流水线模式 | `two_stage` |
| `OHAB_SPLIT_GROUP_MODE` | 分组键策略 | `phone_or_lead` |
| `OHAB_FEATURE_PROFILE` | 特征筛选配置 | `auto_scorecard` |
| `OHAB_ENABLE_AUTO_DEGRADE` | 资源风险自动降级 | `true` |
| `OHAB_ENABLE_PREP_CACHE` | 预处理缓存 | `true` |
| `OHAB_FORCE_REBUILD_CACHE` | 强制重建缓存 | `false` |
| `OHAB_ENABLE_RETRY_ON_MEMORY_ERROR` | 内存失败自动重试 | `true` |

---

## 内置训练档位

| 档位 | 适用场景 | 关键配置 |
|------|----------|----------|
| `server_16g_compare` | 16GB 服务器正式推荐档 | good_quality + 3 folds + two_stage |
| `server_16g_fast` | 快速验证 | medium_quality + 0 folds |
| `server_16g_probe_nn_torch` | 神经网络实验 | good_quality + 0 folds + 仅 NN_TORCH |
| `server_16g_compare_balanced` | 业务导向档 | balanced_accuracy + two_stage |
| `lab_full_quality` | 大内存机器 | high_quality + 5 folds |

---

## 标签分组系统

训练某时间窗口模型时，同组内其他时间窗口标签必须排除。

### 分组定义

| 标签组 | 包含标签 |
|--------|----------|
| `test_drive_label_group` | 试驾标签_7天/14天/21天/30天 |
| `arrive_label_group` | 到店标签_7天/14天/30天 |
| `order_label_group` | 下订标签_7天/14天/21天 |
| `ohab_label_group` | label_OHAB, 线索评级结果 |

### 辅助函数

```python
from config.config import get_sibling_labels, get_excluded_columns

# 训练 7 天试驾模型时，需排除 14/21 天标签
siblings = get_sibling_labels("试驾标签_7天")
# 返回: ["试驾标签_14天", "试驾标签_21天", "试驾标签_30天"]

excluded = get_excluded_columns("试驾标签_7天")
# 返回: id_columns + leakage_columns + sibling_labels
```

---

## 特征配置（`config/config.py`）

自动排除的字段：
- **ID 类**：线索唯一ID、客户ID、手机号_脱敏 等
- **目标泄漏**：到店时间、试驾时间、订单状态 等
- **兄弟标签**：同组内其他时间窗口标签（自动排除）
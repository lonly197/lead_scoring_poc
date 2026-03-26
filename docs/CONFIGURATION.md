# 配置说明

本文档详细说明销售线索智能评级 POC 项目的配置方式和数据字段。

---

## 环境变量（`.env`）

```bash
# 数据配置
DATA_PATH=./data/202602~03.tsv
# 通用脚本遗留参数；train_ohab 主流程默认不使用这个比例
TRAIN_TEST_SPLIT_RATIO=0.2

# 通用模型配置（到店/试驾等脚本）
MODEL_PRESET=good_quality
TIME_LIMIT=3600
RANDOM_SEED=42

# 通用任务默认目标变量
TARGET_LABEL=到店标签_14天

# HAB/OHAB 训练推荐配置（16GB 服务器，默认两阶段）
OHAB_TRAINING_PROFILE=server_16g_compare
OHAB_MODEL_PRESET=good_quality
OHAB_TIME_LIMIT=5400
OHAB_EVAL_METRIC=balanced_accuracy
OHAB_NUM_BAG_FOLDS=3
OHAB_LABEL_MODE=hab
OHAB_ENABLE_MODEL_COMPARISON=true
OHAB_BASELINE_FAMILY=gbm
# 建议留空，交给 train_ohab.py 按当前可用内存自动推导；如需固定，建议 8.5-9
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

# 输出配置
OUTPUT_DIR=./outputs
```

---

## 特征配置（`config/config.py`）

自动排除的字段：
- **ID 类**：线索唯一ID、客户ID、手机号_脱敏 等
- **目标泄漏**：到店时间、试驾时间、订单状态 等
- **其他目标变量**：到店标签_7天、试驾标签_14天 等

---

## 数据字段分类

### 原格式 (20260308-v2.csv，60列)

| 类别 | 字段数 | 示例 |
|------|--------|------|
| 目标变量 | 8 | 到店标签_14天、试驾标签_14天、线索评级结果、成交标签 |
| ID 列 | 6 | 线索唯一ID、客户ID、手机号_脱敏 |
| 渠道特征 | 5 | 一级~四级渠道名称、线索类型 |
| 客户特征 | 5 | 客户性别、所在城市/省份、首触意向车型、预算区间 |
| 门店特征 | 2 | 经销店代码、经销店名称 |
| 时间特征 | 5 | 线索创建时间、分配时间、线索创建星期几、线索创建小时 |
| 通话特征 | 6 | 通话次数、通话总时长、是否接通 等 |
| 业务特征 | 8 | 首触线索是否及时外呼、SOP开口标签、意向金支付状态 等 |
| AI分析特征 | 5 | 客户是否主动询问交车时间/购车权益/金融政策 等 |
| 历史统计 | 3 | 历史订单次数、历史到店次数、历史试驾次数 |
| 目标泄漏 | 6 | 到店时间、试驾时间、下订时间、订单状态、战败原因 等 |

### 新格式 (202603.csv，46列)

| 类别 | 字段数 | 说明 |
|------|--------|------|
| 原始字段 | 46 | 缺少省份、经销店代码/名称、订单状态等 |
| 目标变量 | - | 需从原始时间字段派生（适配器自动处理） |

**注意**：AI分析特征（5列）可能存在空值，模型训练时会自动处理。

---

## 参数优先级

```
命令行参数 > .env 环境变量 > config/config.py 默认值
```

### HAB/OHAB 训练专用配置

`train_ohab.py` 会优先读取 `OHAB_*` 环境变量。推荐把 16GB 服务器默认档直接写入 `.env`，避免团队继续误用通用 `MODEL_PRESET/TIME_LIMIT` 的高资源口径。对 OHAB 训练而言，`--training-profile` / `OHAB_*` 的优先级高于通用模型变量。

| 环境变量 | 说明 | 推荐值 |
|----------|------|--------|
| `OHAB_TRAINING_PROFILE` | 训练档位 | `server_16g_compare` |
| `OHAB_MODEL_PRESET` | 模型预设 | `good_quality` |
| `OHAB_TIME_LIMIT` | 总训练时长（秒） | `5400` |
| `OHAB_EVAL_METRIC` | 训练阶段选模指标 | `balanced_accuracy` |
| `OHAB_NUM_BAG_FOLDS` | Bagging 折数 | `3` |
| `OHAB_LABEL_MODE` | 评级模式 | `hab` |
| `OHAB_ENABLE_MODEL_COMPARISON` | 单阶段模式下是否保留基线模型对比 | `true` |
| `OHAB_BASELINE_FAMILY` | 基线模型家族 | `gbm` |
| `OHAB_MEMORY_LIMIT_GB` | 训练内存上限 | 建议留空自动探测；如需固定，建议 `8.5-9` |
| `OHAB_FIT_STRATEGY` | 模型训练策略 | `sequential` |
| `OHAB_EXCLUDED_MODEL_TYPES` | 排除的高内存模型 | `RF,XT,KNN,FASTAI,NN_TORCH` |
| `OHAB_NUM_FOLDS_PARALLEL` | 并行折数 | `1` |
| `OHAB_MAX_MEMORY_RATIO` | 单模型内存使用比例上限 | `0.7` |
| `OHAB_GENERATE_PLOTS` | 是否生成 PNG 图表 | `false` |
| `OHAB_SPLIT_MODE` | HAB 默认切分模式 | `random` |
| `OHAB_AUTO_OOT_MIN_DAYS` | 自动启用 OOT 的最小跨度（天） | `90` |
| `OHAB_PIPELINE_MODE` | 训练流水线模式 | `two_stage` |
| `OHAB_SPLIT_GROUP_MODE` | 随机切分分组键策略 | `phone_or_lead` |
| `OHAB_FEATURE_PROFILE` | 特征筛选配置 | `auto_scorecard` |

### 内置训练档位

| 档位 | 适用场景 | 关键配置 |
|------|----------|----------|
| `server_16g_compare` | 16GB 服务器正式推荐档 | `good_quality + 3 folds + balanced_accuracy + two_stage + random group split` |
| `server_16g_fast` | 快速验证流程 | `medium_quality + 0 folds` |
| `server_16g_probe_nn_torch` | 16GB 服务器受控实验档 | `good_quality + 0 folds + 仅恢复 NN_TORCH` |
| `server_16g_compare_balanced` | 与正式档相同的业务导向档 | `balanced_accuracy + two_stage` |
| `lab_full_quality` | 更大机器的高质量训练 | `high_quality + 5 folds` |

### 资源参数自动探测

从当前版本开始，`train_ohab.py` 在未显式指定资源参数时，会先探测当前机器状态，再推导默认值：

- 自动探测 `cpu_count`
- 自动探测当前 `available_memory_gb`
- 自动生成更保守的 `memory_limit_gb`
- 自动生成 `num_folds_parallel`

规则层面遵循：

- 普通场景优先自动探测
- 高级场景再用命令行显式覆盖

也就是说，`OHAB_MEMORY_LIMIT_GB` 更适合作为“按需覆盖项”，而不是必须填写的基础配置；`OHAB_NUM_FOLDS_PARALLEL` 对 16GB 服务器建议固定为 `1`。

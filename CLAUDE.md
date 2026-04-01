# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
uv sync

# ====================
# 数据转换（CSV/Excel → Parquet）
# ====================

# 转换单个文件（推荐用于大文件）
uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv

# 批量转换目录
uv run python scripts/convert_to_parquet.py ./data --batch

# ====================
# 数据加载模式
# ====================

# 动态拆分模式（默认）
uv run python scripts/run.py train test_drive --daemon

# 提前拆分模式（支持 .csv/.tsv/.parquet）
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet

# ====================
# 常用训练命令
# ====================

# 试驾预测模型（三模型集成，默认使用统一分割数据）
uv run python scripts/run.py train ensemble --daemon \
    --included-model-types CAT

# 下订预测模型（三模型集成，默认使用统一分割数据）
uv run python scripts/run.py train order_after_drive --daemon \
    --included-model-types CAT

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon --parallel

# 验证模型
uv run python scripts/run.py validate \
    --model-path ./outputs/models/test_drive_model

# 预测（纯推理，无需标签）
uv run python scripts/predict.py \
    --model-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions.csv

# 预测 + OHAB 评级（O/H/A/B/N）
uv run python scripts/predict.py \
    --model-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions.csv \
    --include-ohab

# 预测（包含原始数据列）
uv run python scripts/predict.py \
    --model-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions_full.csv \
    --include-original

# 监控任务
uv run python scripts/run.py monitor status
uv run python scripts/run.py monitor log train_test_drive -f
uv run python scripts/run.py monitor stop --all

# 数据诊断
uv run python scripts/diagnose_data.py ./data/202603.tsv
```

# ====================
# 数据合并（线索宽表 + DMP 行为）
# ====================

# 合并 Excel 多 Sheet + 关联 DMP 行为数据
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表_完整.parquet

# 启用脱敏处理（品牌关键词替换 + ID掩码）
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表_脱敏.parquet \
    --desensitize

# 合并 + 随机拆分训练/测试集（输出 train/test 文件）
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表 \
    --split \
    --split-mode random \
    --split-target 线索评级结果

# 合并 + OOT 时间切分（推荐用于3个月数据）
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表 \
    --split \
    --split-mode oot \
    --split-time-column 线索创建时间 \
    --split-cutoff 2026-03-01

# 输出: 线索宽表_train.parquet, 线索宽表_test.parquet
```

# ====================
# 数据管道（推荐）
# ====================

# 一键执行完整管道
uv run python scripts/pipeline/run_pipeline.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/final

# 跳过脱敏步骤
uv run python scripts/pipeline/run_pipeline.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/final \
    --skip desensitize

# 单步执行：数据清洗
uv run python scripts/pipeline/run_pipeline.py \
    --step clean \
    --input ./data/merged.parquet \
    --output ./data/cleaned.parquet

# 单步执行：OOT 拆分
uv run python scripts/pipeline/run_pipeline.py \
    --step split \
    --input ./data/desensitized.parquet \
    --output ./data/final \
    -- --mode oot --time-column 线索创建时间

# 输出: final_train.parquet, final_test.parquet
```

## 统一数据源方案（推荐）

**从单一数据源生成试驾预测和下订预测的训练/测试集**，确保数据切分一致。

```bash
# 生成统一分割数据
uv run python scripts/pipeline/06_split_unified.py \
    --input ./data/线索宽表_合并_补充试驾.parquet \
    --output ./data/unified_split \
    --time-column 线索创建时间 \
    --cutoff 2026-03-01

# 输出文件:
# - train.parquet / test.parquet         → 试驾预测用（全量线索）
# - train_driven.parquet / test_driven.parquet → 下订预测用（已试驾子集）
```

**训练命令**（默认使用统一分割数据）：

```bash
# 试驾预测模型（7/14/21天三模型集成）
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# 下订预测模型（7/14/21天三模型集成）
uv run python scripts/run.py train order_after_drive --daemon --included-model-types CAT
```

**数据架构**：

```
线索宽表_合并_补充试驾.parquet (1,199,453 行)
    ↓ OOT 时间切分（2026-03-01）
    ├── train.parquet (791,546 行) → 试驾预测训练
    ├── test.parquet (407,907 行) → 试驾预测测试
    ├── train_driven.parquet (47,413 行) → 下订预测训练
    └── test_driven.parquet (22,067 行) → 下订预测测试
```

## 数据管道架构

```
scripts/pipeline/
├── 01_merge.py        # 数据合并（Excel + DMP）
├── 02_profile.py      # 数据探查（缺失值、分布、建议）
├── 03_clean.py        # 数据清洗（异常值、偏斜、高基数）
├── 04_desensitize.py  # 数据脱敏（品牌、ID、手机号）
├── 05_split.py        # 数据拆分（random/oot/auto）
├── 06_split_unified.py # 统一数据拆分（试驾+下订共用）
└── run_pipeline.py    # 统一运行器
```

**管道流程**：
```
Excel + DMP → merged.parquet → profile.md → cleaned.parquet → desensitized.parquet → train/test.parquet
```

**统一数据源流程**：
```
线索宽表_合并_补充试驾.parquet → unified_split/
                                  ├── train.parquet / test.parquet（试驾预测）
                                  └── train_driven.parquet / test_driven.parquet（下订预测）
```

## 入口架构

```
run.py (一级入口)
├── train <task>     → train_model.py (二级入口)
│   ├── arrive       → train_arrive.py
│   ├── test_drive   → train_test_drive.py
│   ├── ohab         → train_ohab.py
│   ├── ensemble     → train_test_drive_ensemble.py  # 试驾预测（三模型）
│   └── order_after_drive → train_order_after_drive.py  # 下订预测（三模型）
└── validate         → validate_model.py (二级入口)
    ├── arrive       → validate_arrive_model.py
    ├── test_drive   → validate_test_drive_model.py
    └── ohab         → validate_ohab_model.py
```

## 模型选择配置

### 指定训练模型类型（提升效率）

通过 `--included-model-types` 参数可指定训练的模型类型：

| 模型类型 | 说明 | 训练时间 | ROC-AUC |
|----------|------|---------|---------|
| `CAT` | CatBoost（推荐） | ~60s | ~0.998 |
| `GBM` | LightGBM | ~30s | ~0.996 |
| `XGB` | XGBoost | ~25s | ~0.995 |
| `CAT,GBM` | 多模型 | ~90s | ~0.998 |

### 配置方式

**方式一：命令行参数**
```bash
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT
```

**方式二：环境变量（.env 文件）**
```bash
# .env
MODEL_INCLUDED_TYPES=CAT
```

### 训练脚本优先级

```
train_test_drive_ensemble.py  → 核心任务（试驾预测三模型集成）
train_order_after_drive.py    → 核心任务（下订预测三模型集成）
train_test_drive.py           → 辅助任务（单模型试驾预测）
train_ohab.py                 → 辅助任务（OHAB 评级）
train_arrive.py               → 辅助任务（到店预测）
```

## 数据格式适配

项目支持多种数据格式，通过 `auto_adapt=True` 自动适配：

| 格式 | 文件 | 特点 |
|------|------|------|
| Parquet | `*.parquet` | **推荐**：加载快、体积小、自带类型 |
| 新格式 | `202603.tsv` | Tab分隔，无表头，46列 |
| 原格式 | `20260308-v2.csv` | 逗号分隔，有表头，60列 |

关键代码：`src/data/adapter.py` 定义了格式检测和列映射。修改时注意：
- `线索创建时间` 在新格式索引 4
- `线索评级结果`（OHAB）在新格式索引 26

**目标标签自动计算**（`adapter.calculate_target_labels()`）：

| 标签 | 计算逻辑 | 用途 |
|------|---------|------|
| `试驾标签_7天` | 试驾时间 - 线索创建时间 ≤ 7天 | 试驾预测 |
| `试驾标签_14天` | 试驾时间 - 线索创建时间 ≤ 14天 | 试驾预测 |
| `试驾标签_21天` | 试驾时间 - 线索创建时间 ≤ 21天 | 试驾预测 |
| `下订标签_7天` | 下订时间 - 试驾时间 ≤ 7天 | 下订预测 |
| `下订标签_14天` | 下订时间 - 试驾时间 ≤ 14天 | 下订预测 |
| `下订标签_21天` | 下订时间 - 试驾时间 ≤ 21天 | 下订预测 |

## DuckDB 加速（大文件优化）

对于大文件（>100MB），可使用 DuckDB 加速数据加载和切分：

### DataLoader DuckDB 加载

```python
from src.data.loader import DataLoader

# 启用 DuckDB 加速加载 Parquet
loader = DataLoader(
    "./data/large.parquet",
    use_duckdb=True,
    duckdb_memory_limit="4GB",
)
df = loader.load()
```

### DuckDB 版 OOT 切分

```python
from src.data.loader import split_data_oot_duckdb, split_data_oot_three_way_duckdb

# 二层切分
train_df, test_df, metadata = split_data_oot_duckdb(
    "./data/large.parquet",
    time_column="线索创建时间",
    cutoff_date="2026-03-01",
)

# 三层切分
train_df, valid_df, test_df, metadata = split_data_oot_three_way_duckdb(
    "./data/large.parquet",
    time_column="线索创建时间",
    train_end="2026-03-11",
    valid_end="2026-03-16",
)
```

**性能对比**：
| 场景 | pandas | DuckDB | 提升 |
|------|--------|--------|------|
| 1GB Parquet 加载 | ~15s | ~3s | 5x |
| OOT 时间切分 | 全量加载 | 按需过滤 | 避免OOM |

**注意事项**：
- DuckDB 加速仅适用于 Parquet 文件
- 最终输出仍为 pandas DataFrame（AutoGluon 要求）
- 小文件（<100MB）收益不明显

## AutoML 预处理

**不要手动进行以下处理**，模型框架会自动处理：
- 类别编码（自动类别特征处理）
- 缺失值填充（自动填充）
- 异常值/偏斜分布（自动变换）
- 类别不平衡（`sample_weight="balance_weight"`）

**需要手动处理的**：
- 时间特征提取（`线索创建星期几`、`线索创建小时`）在 `loader.py` 中实现

**验证脚本注意事项**：
- `FeatureEngineer` 只接受 `time_columns` 和 `numeric_columns` 参数
- 验证脚本应自动匹配训练时的特征工程配置，避免硬编码参数导致不兼容

## 配置优先级

```
命令行参数 > .env 环境变量 > config/config.py 默认值
```

## OOT 训练最佳实践

**避免数据泄露**：
- 使用 `tuning_data` 参数传入验证集，而非合并到 `train_data`
- 模型框架会用验证集做模型选择，但不参与训练
- 当启用 bagging（如 `num_bag_folds > 0`）时，`LeadScoringPredictor` 会自动设置 `use_bag_holdout=True` 以保证兼容性
- 验证集性能才能真正反映泛化能力

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/run.py` | 一级入口：统一调度器 |
| `scripts/train_model.py` | 二级入口：训练路由器 |
| `scripts/validate_model.py` | 二级入口：验证路由器 |
| `scripts/predict.py` | 模型预测：纯推理，无需标签 |
| `scripts/convert_to_parquet.py` | 数据格式转换：CSV/TSV → Parquet |
| `scripts/merge_data.py` | 数据合并：线索宽表 + DMP 行为 |
| `scripts/pipeline/06_split_unified.py` | 统一数据拆分：试驾+下订共用 |
| `config/config.py` | 配置管理：ID 列、泄漏字段、特征定义 |
| `src/data/adapter.py` | 数据格式适配：列映射、目标变量计算 |
| `src/data/loader.py` | 数据加载：特征工程、OOT 时间切分 |
| `src/data/json_extractor.py` | JSON 特征提取：跟进详情解析 |
| `src/models/predictor.py` | 模型封装：训练、清理 |
| `src/inference/hab_deriver.py` | HAB 推导逻辑（三模型模式） |

## 数据质量警告

当前数据集 O 级（已成交）样本仅 12 个，极度不平衡。建议：
- 降级为三分类（H/A/B）
- 或合并为二分类（高意向/低意向）

详见 `docs/TRAINING.md` 的数据集分析部分。

## 项目约定

### 数据处理优化
- **DuckDB 优先**：管道脚本优先使用 DuckDB SQL 向量化处理，避免 pandas 全量加载
- **Pipeline 一致**：所有 step 子脚本使用相同的优化逻辑和错误处理模式

### 文档规范
- **简洁一致**：文档描述简洁，与代码实现保持同步
- **CHANGELOG.md**：仅记录关键变更节点，供 agent 理解项目上下文

## 故障排查

```bash
# KeyError: '线索创建时间' → 列映射错误
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 服务器代码未同步
git pull && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

详细故障排查指南见 `docs/TROUBLESHOOTING.md`。
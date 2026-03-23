# 销售线索智能评级 POC 项目

基于 AutoGluon TabularPredictor 的销售线索智能评级预测系统。

## 项目目标

- **主指标**：到店预测（Top-K 高优线索到店率）
- **目标变量**：`到店标签_14天`（14 天内到店标签）
- **业务目标**：输出线索评分排名，支持销售资源优先分配

> **选择说明**：POC 方案的核心业务规则是 OHAB 评级，但当前数据中 OHAB 评级 85% 为 Unknown，成交标签无正样本，试驾标签仅 1.4%。综合考虑数据可用性，选择到店预测作为 POC 主指标，后续迭代再扩展到 OHAB 评级预测。

## 快速开始

### 环境准备

```bash
# 使用 uv 安装依赖
uv sync

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 配置数据路径
```

### 运行训练

#### 原数据格式训练（train_arrive, train_ohab）

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

#### OOT 验证训练（train_arrive_oot, train_ohab_oot）

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

### 监控后台任务

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

### 生成 Top-K 名单

```bash
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/arrive_model \
    --data-path ./data/20260308-v2.csv \
    --k 100 500 1000
```

---

## 训练脚本说明

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

### OOT 脚本特性

`train_arrive_oot.py` 和 `train_ohab_oot.py` 是针对新数据格式的 OOT 验证版本：

| 特性 | 说明 |
|------|------|
| 数据适配 | 自动启用 `auto_adapt=True`，支持无表头、Tab 分隔的新数据 |
| OOT 切分 | 三层时间切分：训练集、验证集、测试集 |
| 目标变量 | 自动从原始时间字段派生（如 `到店时间 - 线索创建时间`） |
| 后台运行 | 通过 `run.py --daemon` 支持 |

### 命令行参数

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

#### 数据路径指定

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

**预设选项**（`--preset`）：

| 预设 | 磁盘需求 | 训练时间 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| `medium_quality` | ~1G | ~15分钟 | 中等 | 快速验证、磁盘紧张 |
| `good_quality` | ~2G | ~30分钟 | 良好 | 平衡方案 |
| `high_quality` | ~4G | ~1小时 | 高 | **推荐，生产使用** |
| `best_quality` | ~8G | ~4小时 | 最高 | 最终优化 |

---

## 项目结构

```
lead_scoring_poc/
├── pyproject.toml          # uv 项目配置
├── .env.example            # 环境变量模板
├── .gitignore
├── README.md
│
├── config/
│   ├── __init__.py
│   └── config.py           # 配置管理（路径、参数、特征定义）
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── adapter.py       # 数据格式适配器（自动检测、目标变量计算）
│   │   └── loader.py        # 数据加载、特征工程、数据划分
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py    # AutoGluon 模型封装（含磁盘清理）
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py      # Top-K/Lift 评估指标
│   └── utils/
│       ├── __init__.py
│       └── helpers.py      # 工具函数（含磁盘空间检查）
│
├── scripts/
│   ├── run.py               # 统一启动脚本（支持后台运行）
│   ├── monitor.py           # 任务监控脚本（支持 stop --all）
│   ├── train_arrive.py      # 到店预测训练（核心）
│   ├── train_test_drive.py  # 试驾预测训练
│   ├── train_ohab.py        # OHAB 评级训练
│   └── generate_topk.py     # Top-K 名单生成
│
├── notebooks/
│   └── exploration.ipynb   # 数据探索笔记本
│
├── outputs/                # 输出目录（自动创建）
│   ├── models/             # 保存的模型
│   ├── reports/            # 评估报告（JSON/图表）
│   ├── logs/               # 训练日志
│   ├── .process/           # 进程信息（用于监控）
│   └── topk_lists/         # Top-K 名单（CSV）
│
└── data/                   # 数据文件（不提交到 git）
```

---

## 核心功能详解

### 1. 到店预测（核心任务）

预测线索在 14 天内到店的概率，输出 Top-K 高优线索名单。

**业务价值**：
- 销售资源优先分配给高概率线索
- 提升线索转化效率
- 减少无效跟进成本

**输出**：
- 模型：`outputs/models/arrive_model/`
- 报告：`outputs/models/arrive_model/reports/arrive_model_report.json`
- 名单：`outputs/models/arrive_model/reports/arrive_model_top1000.csv`
- 图表：`outputs/models/arrive_model/feature_importance.png`

### 2. 试驾预测（辅助任务）

预测线索在 14 天内完成试驾的概率。

**业务价值**：
- 识别试驾意向强的线索
- 优化试驾资源安排

### 3. OHAB 评级（辅助任务）

多分类任务，预测线索的 OHAB 评级（O/H/A/B）。

**业务价值**：
- 自动化线索分级
- 对比人工评级准确性

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

---

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| AutoGluon | 自动化机器学习 | >= 1.2.0 |
| Pandas | 数据处理 | >= 2.0.0 |
| Scikit-learn | 评估指标 | >= 1.3.0 |
| Matplotlib/Seaborn | 可视化 | >= 3.7.0 |
| Ray | 分布式计算后端 | AutoGluon 依赖 |

### AutoGluon 兼容性说明

本项目充分利用 AutoGluon 内置的预处理能力，避免重复处理：

| 功能 | 处理方 | 说明 |
|------|--------|------|
| 类别编码 | AutoGluon | `CategoryFeatureGenerator` 自动处理 |
| 缺失值填充 | AutoGluon | `FillNaFeatureGenerator` 自动处理 |
| 异常值/偏斜分布 | AutoGluon | `QuantileTransformer` 自动处理 |
| 类别不平衡 | AutoGluon | `sample_weight="balance_weight"` |
| 时间特征提取 | 自定义代码 | AutoGluon 不自动提取 day_of_week、hour |

**重要**：不要手动进行类别编码或缺失值填充，交给 AutoGluon 自动处理。

---

## 配置说明

### 环境变量（`.env`）

```bash
# 数据配置
DATA_PATH=./data/20260308-v2.csv
TRAIN_TEST_SPLIT_RATIO=0.2

# 模型配置
MODEL_PRESET=high_quality
TIME_LIMIT=3600
RANDOM_SEED=42

# 目标变量（中文）
TARGET_LABEL=到店标签_14天

# 输出配置
OUTPUT_DIR=./outputs
```

### 特征配置（`config/config.py`）

自动排除的字段：
- **ID 类**：线索唯一ID、客户ID、手机号_脱敏 等
- **目标泄漏**：到店时间、试驾时间、订单状态 等
- **其他目标变量**：到店标签_7天、试驾标签_14天 等

#### 数据字段分类

**原格式 (20260308-v2.csv，60列)**：

| 类别 | 字段数 | 示例 |
|------|--------|------|
| 目标变量 | 8 | 到店标签_14天、试驾标签_14天、线索评级_试驾前、成交标签 |
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

**新格式 (202603.csv，46列)**：

| 类别 | 字段数 | 说明 |
|------|--------|------|
| 原始字段 | 46 | 缺少省份、经销店代码/名称、订单状态等 |
| 目标变量 | - | 需从原始时间字段派生（适配器自动处理） |

**注意**：AI分析特征（5列）可能存在空值，模型训练时会自动处理。

---

## 常见问题

### Q: 数据适配器是什么？如何使用？

**数据适配器**用于自动适配不同格式的数据文件：

| 数据格式 | 示例文件 | 特点 |
|----------|----------|------|
| 原格式 | `20260308-v2.csv` | 逗号分隔，有表头，60列 |
| 新格式 | `202603.csv` | Tab分隔，无表头，46列 |

**重要**：适配器**默认关闭**，需要显式启用。这是为了确保不干扰后续新数据的正常加载。

```python
from src.data.loader import DataLoader

# 默认行为：不启用适配，直接加载
loader = DataLoader("data/your_data.csv")
df = loader.load()

# 启用适配：处理新格式数据（如 202603.csv）
loader = DataLoader("data/202603.csv", auto_adapt=True)
df = loader.load()

# 查看检测到的格式
print(loader.get_data_format())
```

**适配器功能**：
1. ✅ 自动检测分隔符（逗号/Tab）
2. ✅ 自动识别有无表头
3. ✅ 为无表头数据提供列名映射
4. ✅ 从原始时间字段计算目标变量（到店标签、试驾标签等）
5. ✅ 自动衍生时间特征（星期几、小时）

**下一阶段**：当数仓开发同事导出新格式数据后，可直接使用 `auto_adapt=False` 或删除适配代码。

### Q: 支持哪些文件格式？

| 格式 | 分隔符 | 扩展名 | 支持状态 |
|------|--------|--------|----------|
| CSV（逗号） | `,` | `.csv` | ✅ 支持 |
| TSV（Tab） | `\t` | `.tsv`, `.csv` | ✅ 支持 |
| Parquet | - | `.parquet` | ✅ 支持 |

**重要**：文件内容格式优先于扩展名。例如，`.csv` 文件如果是 Tab 分隔，适配器会自动识别。

### Q: 三个脚本需要按顺序执行吗？

**不需要。** 三个脚本完全独立，没有依赖关系。推荐先运行 `train_arrive.py`，因为到店预测是 POC 的核心指标。

### Q: 训练需要多长时间？

使用 `high_quality` 预设，约 1 小时（7.3万条数据）。可通过 `--time-limit` 调整。

### Q: 如何选择预设？

| 预设 | 磁盘需求 | 训练时间 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| `medium_quality` | ~1G | ~15分钟 | 中等 | 快速验证、磁盘紧张 |
| `good_quality` | ~2G | ~30分钟 | 良好 | 平衡方案 |
| `high_quality` | ~4G | ~1小时 | 高 | **推荐，生产使用** |
| `best_quality` | ~8G | ~4小时 | 最高 | 最终优化 |

### Q: 磁盘空间不足怎么办？

1. 使用更轻量的 preset：`--preset medium_quality`
2. 清理残留文件：`rm -rf outputs/models/ohab_model/`
3. 清理缓存：`rm -rf ~/.cache/autogluon/ /tmp/ray/`

### Q: 如何使用已训练的模型？

```python
from src.models.predictor import LeadScoringPredictor

# 加载模型
predictor = LeadScoringPredictor.load('./outputs/models/arrive_model')

# 预测概率
y_proba = predictor.get_positive_proba(new_data)

# 查看磁盘占用
disk_usage = predictor.get_disk_usage()
print(f"模型大小: {disk_usage['total_size_mb']:.1f} MB")

# 清理非最佳模型释放空间
result = predictor.cleanup(keep_best_only=True)
print(f"释放空间: {result['freed_mb']:.1f} MB")
```

### Q: 如何处理类别不平衡问题？

训练脚本已默认启用 `sample_weight="balance_weight"`，自动平衡类别权重：

```python
# 在训练脚本中已自动配置
predictor = LeadScoringPredictor(
    label="线索评级_试驾前",
    sample_weight="balance_weight",  # 自动平衡 O/H/A/B 权重
    weight_evaluation=True,
)
```

**效果**：H 类（17%）和 O 类（1.9%）的召回率会得到提升。

### Q: 如何使用 OOT 时间切分验证？

当有跨时间段数据时，可使用时间切分而非随机切分：

```python
from src.data.loader import split_data_oot

train_df, test_df = split_data_oot(
    df=df,
    target_label="到店标签_14天",
    time_column="线索创建时间",
    cutoff_date="2026-02-01",  # 此日期之前为训练集，之后为测试集
)
```

**优势**：更好地模拟真实预测场景（用历史数据预测未来）。

### Q: 新数据格式如何生成目标变量？

新数据格式（如 `202603.csv`）缺少预计算的目标变量，适配器会自动从原始时间字段派生：

| 目标变量 | 计算规则 |
|----------|----------|
| `到店标签_7天` | `到店时间 - 线索创建时间 <= 7天` |
| `到店标签_14天` | `到店时间 - 线索创建时间 <= 14天` |
| `到店标签_30天` | `到店时间 - 线索创建时间 <= 30天` |
| `试驾标签_14天` | `试驾时间 - 线索创建时间 <= 14天` |
| `试驾标签_30天` | `试驾时间 - 线索创建时间 <= 30天` |
| `线索评级_试驾前` | 直接使用 `线索评级结果` 列 |
| `成交标签` | `下订时间 IS NOT NULL` |

**验证数据质量**：

```python
from src.data.loader import DataLoader

loader = DataLoader("data/202603.csv")
df = loader.load()

# 检查目标变量
print(f"到店率 (14天): {df['到店标签_14天'].mean()*100:.2f}%")
print(f"试驾率 (14天): {df['试驾标签_14天'].mean()*100:.2f}%")
print(f"OHAB分布: {df['线索评级_试驾前'].value_counts().to_dict()}")
```

### Q: 如何使用自己的数据文件？

确保数据文件包含配置中定义的字段，然后：

```bash
# 方式 1：命令行指定（推荐）
uv run python scripts/train_arrive.py --data-path /path/to/your_data.csv

# 方式 2：修改 .env 文件
DATA_PATH=/path/to/your_data.csv
```

**数据文件要求**：
- 格式：CSV/TSV（UTF-8 编码）
- 分隔符：逗号或Tab均可（**自动检测**）
- 表头：有无均可（**自动适配**）
- 目标变量：预计算或原始时间字段均可（**自动派生**）
- 建议包含配置中的特征字段，缺失字段会自动处理

### Q: 如何后台运行长时间训练？

```bash
# 后台启动训练
uv run python scripts/run.py train_arrive --daemon

# 查看运行状态
uv run python scripts/monitor.py status

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive -f

# 一键停止所有任务
uv run python scripts/monitor.py stop --all
```

**日志文件位置**：`outputs/logs/{task_name}_{timestamp}.log`

**进程信息位置**：`outputs/.process/{task_name}_{pid}.json`

### Q: 如何运行 OOT 验证训练？

OOT（Out-of-Time）验证适用于新格式数据，使用三层时间切分：

```bash
# 后台运行到店预测（7天窗口）
uv run python scripts/run.py train_arrive_oot --daemon \
    --data-path ./data/202603.tsv

# 后台运行 OHAB 评级
uv run python scripts/run.py train_ohab_oot --daemon \
    --data-path ./data/202603.tsv

# 查看训练状态
uv run python scripts/monitor.py status

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive_oot -f
```

**OOT 切分说明**：

| 数据集 | 时间范围 | 用途 |
|--------|----------|------|
| 训练集 | < 2026-03-11 | 模型训练 |
| 验证集 | 2026-03-11 ~ 2026-03-16 | 超参数调优 |
| 测试集 | >= 2026-03-16 | 最终评估 |

**为什么使用 7 天窗口？**

数据时间范围 2026-03-01 ~ 2026-03-23，共 23 天。使用 7 天预测窗口：
- 测试集从 3 月 16 日开始，到 3 月 23 日有完整 7 天观察期
- 确保测试集标签完整，避免数据泄漏

### Q: 训练失败后如何排查？

1. 查看日志：`uv run python scripts/monitor.py log train_arrive`
2. 检查磁盘空间：`df -h`
3. 查看任务状态：`uv run python scripts/monitor.py list`
4. 清理残留文件后重试

---

## 注意事项

1. **环境要求**：Python 3.9-3.12，推荐 3.11
2. **内存需求**：AutoGluon 训练约需 16GB+ 内存
3. **磁盘需求**：根据 preset 需要 2-10G 空间（见上文表格）
4. **GPU 可选**：`best_quality` 预设建议使用 GPU
5. **数据安全**：`.env` 文件不上传 git
6. **后台任务**：使用 `--daemon` 后台运行，用 `monitor.py` 管理
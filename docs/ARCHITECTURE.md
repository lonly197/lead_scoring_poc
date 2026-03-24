# 架构说明

本文档介绍销售线索智能评级 POC 项目的技术架构和核心功能。

---

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| AutoGluon | 自动化机器学习 | >= 1.2.0 |
| Pandas | 数据处理 | >= 2.0.0 |
| Scikit-learn | 评估指标 | >= 1.3.0 |
| Matplotlib/Seaborn | 可视化 | >= 3.7.0 |
| Ray | 分布式计算后端 | AutoGluon 依赖 |

---

## AutoGluon 兼容性说明

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
│   ├── train_arrive.py      # 到店预测训练 (统一自适应版)
│   ├── train_test_drive.py  # 试驾预测训练 (统一自适应版)
│   ├── train_ohab.py        # OHAB 评级训练 (统一自适应版)
│   └── generate_topk.py     # Top-K 名单生成
...
---

## 核心设计：智能切分与防泄漏闭环

本项目在工程上实现了严谨的数据隔离机制，防止指标虚高。

### 1. 🚀 智能自适应切分 (`src/data/loader.py`)

脚本自动探查 `线索创建时间` 的跨度：
- **跨度 $\ge 14$ 天**：自动激活 **OOT (Out-of-Time)** 模式。按 `70% / 15% / 15%` 的时间比例划分为训练、验证和测试集。这模拟了“用过去预测未来”的真实业务场景。
- **跨度 $< 14$ 天**：自动降级为 **分层随机切分 (Random Split)** 模式（80/20）。适用于单日快照数据的快速实验。

### 2. 🛡️ 防泄漏指纹系统 (`test_ids`)

为了防止在随机切分模式下，`validate_model.py` 误用训练集数据进行验证，我们实现了指纹闭环：
- **记录**：在训练降级为随机切分时，`train_*.py` 会提取所有测试集样本的 `线索唯一ID` 并加密持久化至 `feature_metadata.json`。
- **识别**：`validate_model.py` 启动时会自动读取元数据。
- **隔离**：验证脚本会强制仅加载这些特定的“指纹”样本。即使给验证脚本一个包含训练集的全量文件，它也能精准地通过指纹实现**物理隔离**，确保结果真实。

### 3. 🚿 特征脱水机制

在 `config/config.py` 中定义了严格的 `leakage_columns` 黑名单：
- **原理**：自动拦截一切在“定级时点”之后产生的后验信息（如成交标签、最终到店时间、后续评级变化等）。
- **价值**：确保模型是基于客户的“意向信号”进行预测，而非根据“成交结果”进行反推。
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

## 核心功能

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
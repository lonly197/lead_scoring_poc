# 架构说明

本文档介绍销售线索智能评级 POC 项目的技术架构和核心功能。

---

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| AutoML 框架 | 自动化机器学习 | AutoGluon >= 1.2.0 |
| Pandas | 数据处理 | >= 2.0.0 |
| Scikit-learn | 评估指标 | >= 1.3.0 |
| Matplotlib/Seaborn | 可视化 | >= 3.7.0 |
| Ray | 分布式计算后端 | 框架依赖 |

---

## 模型框架兼容性说明

本项目充分利用 AutoML 框架内置的预处理能力，避免重复处理：

| 功能 | 处理方 | 说明 |
|------|--------|------|
| 类别编码 | 自动处理 | 框架自动识别并编码类别特征 |
| 缺失值填充 | 自动处理 | 框架自动处理缺失值 |
| 异常值/偏斜分布 | 自动处理 | 框架自动进行数据变换 |
| 类别不平衡 | 自动处理 | `sample_weight="balance_weight"` |
| 时间特征提取 | 自定义代码 | 框架不自动提取 day_of_week、hour |

**重要**：不要手动进行类别编码或缺失值填充，交给框架自动处理。

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
│   │   └── predictor.py    # 模型封装（含磁盘清理）
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

当前 HAB 主流程默认采用随机分组切分：
- **默认模式**：按手机号优先、线索唯一 ID 回退的分组键做 `70% / 15% / 15%` 切分，优先避免同客泄漏。
- **显式 OOT**：仅当用户手动指定 `train_end/valid_end`，或将 `split_mode` 配为 `auto_oot/manual_oot` 时，才启用时间切分。
- **自动 OOT 门槛**：自动 OOT 的最小跨度默认提升到 `90` 天，避免旧版短跨度数据被误判为适合 OOT。

### 2. 🛡️ 防泄漏指纹系统 (`test_ids`)

为了防止随机切分下同客样本进入不同集合，我们实现了分组键闭环：
- **记录**：训练时会提取测试集分组键并持久化至 `feature_metadata.json`。
- **识别**：`validate_model.py` 启动时会自动读取元数据。
- **隔离**：验证脚本会强制仅加载这些特定分组键对应的样本。即使给验证脚本一个包含训练集的全量文件，它也能实现**物理隔离**，确保结果真实。

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

### 3. OHAB/HAB 评级（核心业务分层任务）

当前默认以 `H/A/B` 为主流程，内部采用两阶段 `H vs 非H + A vs B` 流水线；`O` 不进入常规 HAB 训练主路径。

**业务价值**：
- 自动化线索分级
- 对比人工评级准确性

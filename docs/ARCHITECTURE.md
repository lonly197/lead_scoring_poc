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

## 模型框架预处理说明

AutoGluon 自动处理以下预处理，无需手动编码：

| 功能 | 说明 |
|------|------|
| 类别编码 | 自动识别并编码 |
| 缺失值填充 | 自动处理 |
| 异常值/偏斜分布 | 自动变换 |
| 类别不平衡 | `sample_weight="balance_weight"` |

**需手动处理**：时间特征提取（day_of_week、hour）在 `loader.py` 中实现。

---

## 项目结构

```
lead_scoring_poc/
├── pyproject.toml          # uv 项目配置
├── .env.example            # 环境变量模板
│
├── config/
│   └── config.py           # 配置管理（路径、参数、特征定义）
│
├── src/
│   ├── pipeline/           # 数据管道核心模块
│   │   ├── utils.py        # 共享工具函数
│   │   └── config.py       # 管道配置
│   ├── data/
│   │   ├── adapter.py      # 数据格式适配器
│   │   ├── loader.py       # 数据加载、特征工程
│   │   └── label_policy.py # 标签计算策略
│   ├── models/
│   │   └── predictor.py    # AutoGluon 模型封装
│   ├── training/
│   │   └── hab_pipeline.py # 两阶段 HAB 流水线
│   ├── inference/
│   │   └── hab_deriver.py  # HAB 概率推导
│   ├── evaluation/
│   │   └── metrics.py      # Top-K/Lift 评估
│   └── utils/
│       └── helpers.py      # 工具函数
│
├── scripts/
│   ├── run.py              # 一级入口：统一调度器
│   ├── train_model.py      # 二级入口：训练路由器
│   ├── validate_model.py   # 二级入口：验证路由器
│   ├── monitor.py          # 任务监控
│   │
│   ├── pipeline/           # 数据管道（推荐）
│   │   ├── 01_merge.py     # 数据合并
│   │   ├── 02_profile.py   # 数据探查
│   │   ├── 03_clean.py     # 数据清洗
│   │   ├── 04_desensitize.py # 数据脱敏
│   │   ├── 05_split.py     # 数据拆分
│   │   └── run_pipeline.py # 统一运行器
│   │
│   ├── train_ohab.py       # OHAB 评级训练
│   ├── train_arrive.py     # 到店预测训练
│   ├── train_test_drive.py # 试驾预测训练
│   ├── train_test_drive_ensemble.py  # 三模型集成训练
│   │
│   ├── validate_ohab_model.py        # OHAB 验证
│   ├── validate_test_drive_model.py  # 试驾验证
│   ├── validate_ensemble.py          # 三模型验证
│   │
│   ├── merge_data.py       # 数据合并（旧版，保留兼容）
│   ├── generate_topk.py    # Top-K 名单生成
│   └── diagnose_data.py    # 数据诊断
│
├── outputs/                # 输出目录（自动创建）
│   ├── models/             # 模型文件
│   ├── validation/         # 验证结果
│   ├── reports/            # 评估报告
│   └── logs/               # 训练日志
│
└── data/                   # 数据文件（不提交到 git）
```

---

## 核心设计

### 1. 智能切分与防泄漏闭环

- **默认模式**：按手机号分组做 `70/15/15` 随机切分，避免同客泄漏
- **测试集指纹**：训练时记录测试集 ID 到 `feature_metadata.json`，验证时强制隔离
- **特征脱水**：`config.py` 定义 `leakage_columns` 黑名单，排除后验特征

### 2. 两阶段 HAB 流水线

内部采用 `H vs 非H` → `A vs B` 两阶段分类，优先保证分层稳定性。

### 3. 三模型集成训练

训练 7/14/21 天试驾预测模型，通过概率阈值推导 H/A/B 评级（`src/inference/hab_deriver.py`）。

### 4. 数据管道

独立的数据处理流水线，每个步骤职责单一：

```
01_merge.py      → 02_profile.py  → 03_clean.py    → 04_desensitize.py → 05_split.py
(合并)              (探查)           (清洗)           (脱敏)              (拆分)
```

| 步骤 | 功能 | 输入 | 输出 |
|------|------|------|------|
| merge | 合并 Excel + DMP | Excel, CSV | merged.parquet |
| profile | 数据探查诊断 | parquet | profile.md |
| clean | 数据清洗（异常值、偏斜、高基数） | parquet | cleaned.parquet |
| desensitize | 数据脱敏（品牌、ID、手机号） | parquet | desensitized.parquet |
| split | 数据拆分（random/oot/auto） | parquet | train.parquet, test.parquet |

**高级清洗功能**：
- 异常值检测（IQR/Z-Score）
- 偏斜分布识别
- 高基数列检测
- 常量列删除

---

## 核心功能

| 任务 | 目标变量 | 说明 |
|------|----------|------|
| OHAB 评级 | `线索评级结果` | **主流程**：两阶段 HAB 流水线 |
| 到店预测 | `到店标签_14天` | 辅助：预测 14 天内到店概率 |
| 试驾预测 | `试驾标签_14天` | 辅助：预测 14 天内试驾概率 |
| 三模型集成 | `试驾标签_7/14/21天` | 从概率推导 H/A/B 评级 |

**业务价值**：
- 自动化线索分级，优化销售资源分配
- H 级（7天内试驾）、A 级（14天内）、B 级（21天内）
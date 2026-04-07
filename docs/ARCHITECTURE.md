# 架构说明

本文档介绍项目的技术架构和核心设计。

---

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| AutoML 框架 | 自动化机器学习 | AutoGluon >= 1.2.0 |
| Pandas | 数据处理 | >= 2.0.0 |
| DuckDB | 大文件处理 | >= 0.10.0 |
| Ray | 分布式计算后端 | 框架依赖 |

---

## 项目结构

```
lead_scoring_poc/
├── config/config.py           # 配置管理
├── src/
│   ├── data/                  # 数据处理
│   │   ├── adapter.py         # 格式适配器
│   │   ├── loader.py          # 数据加载、特征工程
│   │   └── label_policy.py    # 标签计算策略
│   ├── models/
│   │   ├── predictor.py       # AutoGluon 封装
│   │   └── hab_engine.py      # HAB 决策引擎
│   ├── training/
│   │   ├── hab_pipeline.py    # 两阶段 HAB 流水线
│   │   └── pipeline/          # 训练管道组件
│   ├── inference/
│   │   └── hab_deriver.py     # HAB 概率推导
│   └── evaluation/
│       └── metrics.py         # Top-K/Lift 评估
│
├── scripts/
│   ├── run.py                 # 一级入口：统一调度器
│   ├── training/              # 训练脚本
│   ├── validation/            # 验证脚本
│   ├── prediction/            # 预测脚本
│   ├── tools/                 # 工具脚本
│   └── pipeline/              # 数据管道
│
└── outputs/                   # 输出目录（自动创建）
    ├── models/                # 模型文件
    ├── validation/            # 验证结果
    └── logs/                  # 训练日志
```

---

## 核心设计

### 1. 智能切分与防泄漏

- **默认模式**：按手机号分组做 `70/15/15` 随机切分
- **测试集指纹**：训练时记录测试集 ID，验证时强制隔离
- **标签分组系统**：自动排除同组内其他时间窗口标签

### 2. 两阶段 HAB 流水线

```
Stage 1: H vs 非H  →  Stage 2: A vs B
```

### 3. 三模型集成训练

训练 7/14/21 天试驾预测模型，通过概率阈值推导 H/A/B 评级。

### 4. 数据管道

```
01_merge.py → 02_profile.py → 03_clean.py → 04_desensitize.py → 05_split.py
```

---

## 核心功能

| 任务 | 目标变量 | 说明 |
|------|----------|------|
| OHAB 评级 | `线索评级结果` | **主流程**：两阶段 HAB 流水线 |
| 试驾预测 | `试驾标签_7/14/21天` | 三模型集成，推导 H/A/B |
| 下订预测 | `下订标签_7/14/21天` | 试驾后阶段 |
| 到店预测 | `到店标签_14天` | 辅助任务 |

---

## 模型框架预处理

AutoGluon 自动处理：

| 功能 | 说明 |
|------|------|
| 类别编码 | 自动识别并编码 |
| 缺失值填充 | 自动处理 |
| 异常值/偏斜分布 | 自动变换 |
| 类别不平衡 | `sample_weight="balance_weight"` |

**需手动处理**：时间特征提取（在 `loader.py` 中实现）
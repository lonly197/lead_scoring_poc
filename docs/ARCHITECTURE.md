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
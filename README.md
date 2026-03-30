# 销售线索智能评级 POC 项目

基于 AutoML 技术的销售线索智能评级预测系统。

## 项目目标

- **主流程**：HAB 智能评级训练与验证
- **默认目标变量**：`线索评级结果`
- **业务目标**：输出可解释的 `H/A/B` 分层结果，支持销售资源优先分配与 SOP 差异化跟进。

## 核心特性

- **🚀 默认随机分组切分**：`train_ohab` 默认按手机号优先、线索 ID 回退的分组键做随机 `70/15/15` 切分，避免同客泄漏。
- **🛡️ 全链路防泄漏**：训练阶段记录测试集分组键，`validate_model.py` 会自动只评估训练时定义的测试集。
- **🚿 特征脱水技术**：内置严格的后验特征排除机制，彻底杜绝模型"偷看答案"的行为。
- **🧭 两阶段 HAB 流水线**：默认先做 `H vs 非H`，再做 `A vs B`，并在验证集调 `H` 阈值，优先兼顾分类稳定性与业务单调性。
- **⚡ 单模型训练优化**：支持指定训练单一模型（如 CatBoost），训练时间缩短 5 倍，性能损失极小。
- **🧠 资源自适应训练**：`train_ohab` 启动前会自动探测当前 CPU 数和可用内存，自动收敛 `memory_limit_gb` 与 `num_folds_parallel`；显式命令行参数仍可覆盖。

## 快速开始

详细指南请参阅 **[快速开始指南](docs/QUICKSTART.md)**。

```bash
# 安装依赖
uv sync && cp .env.example .env

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

# ====================
# 数据加载模式
# ====================

# 动态拆分模式（默认）
uv run python scripts/run.py train test_drive --daemon

# 提前拆分模式（优先级更高）
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet

# ====================
# 常用命令
# ====================

# 仅训练 CatBoost（推荐，性能最佳）
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT

# 三模型集成训练（7/14/21 天试驾预测 → HAB 推导）
uv run python scripts/run.py train ensemble --daemon

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon --parallel

# 验证模型
uv run python scripts/run.py validate \
    --model-path ./outputs/models/test_drive_model

# 查看任务状态
uv run python scripts/run.py monitor status

# 停止所有任务
uv run python scripts/run.py monitor stop --all
```

## 数据管道架构

```
scripts/pipeline/
├── 01_merge.py        # 数据合并（Excel + DMP）
├── 02_profile.py      # 数据探查（缺失值、分布、建议）
├── 03_clean.py        # 数据清洗（异常值、偏斜、高基数）
├── 04_desensitize.py  # 数据脱敏（品牌、ID、手机号）
├── 05_split.py        # 数据拆分（random/oot/auto）
└── run_pipeline.py    # 统一运行器
```

**管道流程**：
```
Excel + DMP → merged.parquet → profile.md → cleaned.parquet → desensitized.parquet → train/test.parquet
```

## 入口架构

```
run.py (一级入口 - 统一调度器)
│
├── train <task>     → train_model.py (二级入口)
│   ├── arrive       → train_arrive.py
│   ├── test_drive   → train_test_drive.py
│   ├── ohab         → train_ohab.py
│   └── ensemble     → train_test_drive_ensemble.py
│
├── validate         → validate_model.py (二级入口)
│   ├── arrive       → validate_arrive_model.py
│   ├── test_drive   → validate_test_drive_model.py
│   └── ohab         → validate_ohab_model.py
│
└── monitor          → monitor.py (二级入口)
    ├── status       - 查看运行状态
    ├── list         - 列出所有任务
    ├── log <task>   - 查看任务日志
    ├── detail <task>- 查看任务详情
    └── stop <task>  - 停止任务
```

## 模型选择配置

通过 `--included-model-types` 参数指定训练的模型类型：

| 模型类型 | 说明 | 训练时间 | ROC-AUC |
|----------|------|---------|---------|
| `CAT` | CatBoost（推荐） | ~60s | ~0.998 |
| `GBM` | LightGBM | ~30s | ~0.996 |
| `XGB` | XGBoost | ~25s | ~0.995 |
| `CAT,GBM` | 多模型 | ~90s | ~0.998 |

```bash
# 仅训练 CatBoost（推荐）
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT

# 通过环境变量配置
echo "MODEL_INCLUDED_TYPES=CAT" >> .env
```

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 环境准备、运行训练、监控任务 |
| [训练脚本](docs/TRAINING.md) | 脚本说明、命令行参数、评估指标 |
| [配置说明](docs/CONFIGURATION.md) | 环境变量、特征配置、数据字段 |
| [架构说明](docs/ARCHITECTURE.md) | 技术栈、项目结构、核心功能 |
| [常见问题](docs/FAQ.md) | 数据处理、训练相关、故障排查 |

## 环境要求

- Python 3.9-3.12（推荐 3.11）
- 内存：16GB+（16GB 服务器建议使用 `server_16g_compare`，更大机器再用 `lab_full_quality`）
- 磁盘：2-10G（根据 preset）

## 许可证

MIT
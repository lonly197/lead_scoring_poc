# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
uv sync

# 训练模型（核心任务）
uv run python scripts/train_arrive.py --data-path ./data/your_data.csv
uv run python scripts/run.py train_arrive_oot --data-path ./data/202603.tsv  # OOT 验证（兼容任务名）

# 后台运行长时间训练
uv run python scripts/run.py train_arrive --daemon

# 监控后台任务
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_arrive -f
uv run python scripts/monitor.py stop --all

# 数据诊断（调试列映射问题）
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 生成 Top-K 名单
uv run python scripts/generate_topk.py --model-path ./outputs/models/arrive_model --k 100 500
```

## 架构要点

### 训练脚本优先级

```
train_arrive.py      → 核心任务（到店预测）
train_ohab.py        → 辅助任务（OHAB 评级）
train_test_drive.py  → 辅助任务（试驾预测）
```

仓库中实际存在 3 个训练脚本；另有 `train_arrive_oot` / `train_ohab_oot` 两个兼容任务名，通过 `scripts/run.py` 转发到统一训练入口。

### 数据格式适配

项目支持两种数据格式，通过 `auto_adapt=True` 自动适配：

| 格式 | 文件 | 特点 |
|------|------|------|
| 原格式 | `20260308-v2.csv` | 逗号分隔，有表头，60列 |
| 新格式 | `202603.tsv` | Tab分隔，无表头，46列 |

关键代码：`src/data/adapter.py` 定义了 46 列的映射关系。修改时注意 `线索创建时间` 在索引 4，`线索评级结果`（OHAB）在索引 26。

### AutoML 预处理

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

### 配置优先级

```
命令行参数 > .env 环境变量 > config/config.py 默认值
```

### OOT 训练最佳实践

**避免数据泄露**：
- 使用 `tuning_data` 参数传入验证集，而非合并到 `train_data`
- 模型框架会用验证集做模型选择，但不参与训练
- 当启用 bagging（如 `num_bag_folds > 0`）时，`LeadScoringPredictor` 会自动设置 `use_bag_holdout=True` 以保证兼容性
- 验证集性能才能真正反映泛化能力

## 关键文件

| 文件 | 用途 |
|------|------|
| `config/config.py` | 配置管理：ID 列、泄漏字段、特征定义 |
| `src/data/adapter.py` | 数据格式适配：列映射、目标变量计算 |
| `src/data/loader.py` | 数据加载：特征工程、OOT 时间切分 |
| `src/models/predictor.py` | 模型封装：训练、清理 |

## 数据质量警告

当前数据集 O 级（已成交）样本仅 12 个，极度不平衡。建议：
- 降级为三分类（H/A/B）
- 或合并为二分类（高意向/低意向）

详见 `docs/TRAINING.md` 的数据集分析部分。

## 故障排查

```bash
# KeyError: '线索创建时间' → 列映射错误
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 服务器代码未同步
git pull && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

详细故障排查指南见 `docs/TROUBLESHOOTING.md`。

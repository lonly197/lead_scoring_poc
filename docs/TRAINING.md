# 训练脚本说明文档 (Auto-Adapt Edition)

本文档详细介绍销售线索智能评级 POC 项目的训练脚本及其使用方法。

---

## 🚀 统一入口：run.py 透传机制

为了简化操作，项目提供了统一的任务调度入口 `scripts/run.py`。

### 核心特性
- **全参数透传**：`run.py` 仅解析调度相关的核心参数（如 `task` 和 `--daemon`），其他所有传入的参数（如 `--preset`, `--data-path`, `--o-merge-threshold` 等）都会**原封不动地转发**给底层的具体任务脚本。
- **自动解耦**：当子任务脚本新增功能或参数时，无需修改 `run.py` 即可直接通过统一入口调用。
- **后台守护**：使用 `--daemon` 参数可将任务转入后台运行，并通过 `monitor.py` 跟踪进度。

---

## 脚本概览

本项目提供统一的智能自适应训练脚本。当前 `train_ohab.py` 默认采用**随机分组切分 + 两阶段 HAB 流水线**，只有显式配置时才启用 OOT。

### 核心脚本列表

| 脚本 | 目标变量 | 任务类型 | 说明 |
|------|----------|----------|------|
| `train_arrive.py` | `到店标签_14天` | 二分类 | 智能自适应切分，输出 Top-K/Lift |
| `train_ohab.py` | `线索评级结果` | HAB 评级 | 两阶段流水线，多类别权重平衡 |
| `train_test_drive.py` | `试驾标签_14天` | 二分类 | 智能自适应切分 |
| `train_test_drive_ensemble.py` | `试驾标签_7/14/21天` | 三模型集成 | 从概率推导 H/A/B 评级 |

---

## 🚀 切分逻辑

`train_ohab.py` 当前默认优先保证同客隔离与业务分层稳定性：

1.  **默认模式**：`OHAB_SPLIT_MODE=random`，按手机号优先、线索 ID 回退的分组键做 `70%训练 / 15%验证 / 15%测试` 切分。
2.  **显式 OOT**：只有传 `--train-end/--valid-end` 或把 `OHAB_SPLIT_MODE` 设为 `auto_oot/manual_oot` 时，才走时间切分。
3.  **自动 OOT 门槛**：自动 OOT 的最小跨度默认是 `90` 天，而不是旧版的 `14` 天。
4.  **防泄漏闭环**：切分元数据都会保存在 `feature_metadata.json` 中，`validate_model.py` 会自动识别并实施物理隔离。

---

## 命令行参数调用示例

推荐通过 `run.py` 统一入口进行调用，支持参数透传：

```bash
# 到店预测训练
uv run python scripts/run.py train arrive --daemon \
    --data-path ./data/202602_03.csv \
    --preset high_quality \
    --time-limit 3600

# OHAB 评级训练
uv run python scripts/run.py train ohab --daemon \
    --data-path ./data/202603.tsv

# 三模型集成训练（顺序模式，16GB 推荐）
uv run python scripts/run.py train ensemble --daemon

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon --parallel

# 提前拆分数据文件（优先级高于 --data-path）
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet

# 仅训练 CatBoost（推荐）
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--train-path` | 训练集文件（提前拆分模式，支持 .parquet/.csv/.tsv） |
| `--test-path` | 测试集文件（提前拆分模式） |
| `--parallel` | 并行训练三模型（仅 ensemble 任务，需 32GB+ 内存） |
| `--max-workers` | 并行进程数（默认 3） |
| `--included-model-types` | 指定模型类型（如 `CAT` 仅训练 CatBoost） |

---

## 16GB 服务器推荐档

当前 `202602~03.tsv` 规模接近 48 万行。对 16GB 内存服务器，推荐使用 `server_16g_compare` 档位。

```bash
uv run python scripts/run.py train ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare
```

该档位的默认取舍是：

- 使用 `good_quality`
- 默认训练阶段按 `balanced_accuracy` 选模
- 默认使用两阶段 `H vs 非H + A vs B`
- 保留 `LightGBM / CatBoost / XGBoost`
- 默认排除 `RF/XT/KNN/FASTAI/NN_TORCH`
- 固定 `fit_strategy=sequential`
- 固定 `num_folds_parallel=1`
- 默认关闭 OHAB 解释性 PNG 图表，仅保留 CSV/JSON 结构化产物

这套组合针对当前 16GB / 8 核 / CPU-only 服务器做过收敛，优先保证训练稳定性和基线对比可复现性，不建议直接切到 `high_quality`。

### 业务指标受控实验档

如果需要验证“训练阶段直接按业务更关注的 `balanced_accuracy` 选模型”是否会优于当前默认档，可使用受控实验档：

```bash
uv run python scripts/run.py train ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare_balanced \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

该档位当前与正式档同为业务导向口径，主要用于受控回归验证配置稳定性。

### 资源自适应控制

内存控制参数现在支持**所有**训练任务（train_arrive, train_ohab, train_test_drive），帮助在资源受限的环境下稳定运行：

- `--max-memory-ratio`: 最大内存使用比例（当前 OHAB 默认 0.7，建议 0.6-0.8）
- `--exclude-memory-heavy-models`: 排除内存密集型模型（KNN, RF, XT）
- `--num-folds-parallel`: 限制并行训练的 fold 数量

### 16GB 服务器上的受控实验档

如果只想验证神经网络模型是否有增益，不要直接把 `FASTAI` 和所有非树模型一起放开。推荐使用单变量实验档：

```bash
uv run python scripts/run.py train ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_probe_nn_torch \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

该档位只恢复 `NN_TORCH`，保持 `FASTAI/RF/XT` 继续排除，并关闭 bagging。若 `balanced_accuracy` 或 `macro_f1` 没有稳定提升，不建议在当前机器上继续扩展模型池。

### OHAB 补充产物口径

- `train_ohab` 默认生成 `feature_importance.csv/json` 和 `business_dimension_contribution.json`。
- `train_ohab` 默认不生成 `feature_importance.png`、`business_dimension_contribution.png`；如需汇报配图，显式设置 `OHAB_GENERATE_PLOTS=true` 或传 `--generate-plots`。
- 客户报告 `generate_business_report.py` 读取的是结构化 CSV/JSON，不依赖 PNG。
- OHAB 的 Top-K 名单语义固定为按 `P(H)` 排序；命令行生成时必须显式传 `--target-class H`。

---

## 模型验证

验证脚本已拆分为独立文件，按模型类型调用：

| 脚本 | 用途 |
|------|------|
| `validate_ohab_model.py` | OHAB 评级验证 |
| `validate_test_drive_model.py` | 试驾预测验证 |
| `validate_ensemble.py` | 三模型集成验证 |

### 统一入口调用

```bash
# 验证模型（自动识别类型）
uv run python scripts/run.py validate \
    --model-path ./outputs/models/ohab_model

# 指定测试集（提前拆分模式）
uv run python scripts/run.py validate \
    --model-path ./outputs/models/test_drive_model \
    --test-path ./data/test.parquet
```

### 直接调用验证脚本

```bash
# OHAB 验证
uv run python scripts/validate_ohab_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202603.tsv

# 三模型集成验证
uv run python scripts/validate_ensemble.py \
    --model-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/202603.tsv
```

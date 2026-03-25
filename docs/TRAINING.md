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

本项目提供统一的智能自适应训练脚本。**脚本会自动识别数据时间跨度并选择最佳切分策略。**

### 核心脚本列表

| 脚本 | 目标变量 | 任务类型 | 核心特性 |
|------|----------|----------|----------|
| `train_arrive.py` | `到店标签_14天` | 二分类 | 智能自适应 OOT/随机切分，输出 Top-K/Lift |
| `train_ohab.py` | `线索评级_试驾前` | 多分类 | 智能自适应 OOT/随机切分，多类别权重平衡，支持基线模型对比，默认仅输出结构化解释产物 |
| `train_test_drive.py` | `试驾标签_14天` | 二分类 | 支持所有特征工程与自适应切分 |

---

## 🚀 智能自适应切分逻辑

为了平衡“单日快照数据”与“跨月长周期数据”的处理需求，脚本内置了智能探查机制：

1.  **自动探查**：启动时自动计算 `线索创建时间` 的时间跨度。
2.  **自适应选择**：
    *   **OOT 模式 (跨度 $\ge 14$ 天)**：自动按照 `70%训练 / 15%验证 / 15%测试` 的比例进行时间轴切分。模拟“用过去预测未来”。
    *   **随机切分模式 (跨度 $< 14$ 天)**：自动降级为分层随机切分（80/20），并提取测试集样本 ID 作为“防泄漏指纹”。
3.  **防泄漏闭环**：无论哪种模式，切分元数据都会保存在 `feature_metadata.json` 中，`validate_model.py` 会自动识别并实施物理隔离。

---

## 命令行参数调用示例

推荐通过 `run.py` 统一入口进行调用，支持参数透传：

```bash
# 到店预测训练典型调用
uv run python scripts/run.py train_arrive --daemon \
    --data-path ./data/202602_03.csv \      # 数据文件路径
    --target 到店标签_14天 \               # 目标变量
    --preset high_quality \                # AutoGluon 预设
    --time-limit 3600 \                    # 训练时间限制（秒）
    --num-bag-folds 5                      # 交叉验证折数
    --output-dir ./outputs/models/arrive_model

# OHAB 评级任务透传 O 级合并参数
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202603.tsv \
    --o-merge-threshold 60 \
    --report-topk 10,20,30
```

---

## 16GB 服务器推荐档

当前 `202602~03.tsv` 规模接近 48 万行。对 16GB 内存服务器，推荐使用 `server_16g_compare` 档位。

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

该档位的默认取舍是：

- 使用 `good_quality`
- 保留 `LightGBM / CatBoost / XGBoost + WeightedEnsemble`
- 默认排除 `RF/XT/KNN/FASTAI/NN_TORCH`
- 固定 `fit_strategy=sequential`
- 固定 `num_folds_parallel=1`
- 默认关闭 OHAB 解释性 PNG 图表，仅保留 CSV/JSON 结构化产物

这套组合针对当前 16GB / 8 核 / CPU-only 服务器做过收敛，优先保证训练稳定性和基线对比可复现性，不建议直接切到 `high_quality`。

### 业务指标受控实验档

如果需要验证“训练阶段直接按业务更关注的 `balanced_accuracy` 选模型”是否会优于当前默认档，可使用受控实验档：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare_balanced \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

该档位除 `eval_metric=balanced_accuracy` 外，其他资源约束、模型池和对比输出都与 `server_16g_compare` 保持一致，适合做一轮并排 A/B 验证。

### 资源自适应控制

内存控制参数现在支持**所有**训练任务（train_arrive, train_ohab, train_test_drive），帮助在资源受限的环境下稳定运行：

- `--max-memory-ratio`: 最大内存使用比例（默认 0.8，建议 0.6-0.8）
- `--exclude-memory-heavy-models`: 排除内存密集型模型（KNN, RF, XT）
- `--num-folds-parallel`: 限制并行训练的 fold 数量

### 16GB 服务器上的受控实验档

如果只想验证神经网络模型是否有增益，不要直接把 `FASTAI` 和所有非树模型一起放开。推荐使用单变量实验档：

```bash
uv run python scripts/run.py train_ohab --daemon \
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

## 模型验证 (validate_model)

验证模型在测试集上的表现。

### 基础验证
```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202603.tsv
```

> 当前验证脚本会同时保留：
> - `technical_best_model`：AutoGluon 内部优化目标下的最优模型
> - `business_recommended_model`：客户报告和主输出默认采用的业务推荐模型
>
> `predictions.csv`、`lead_actions.csv`、`hab_bucket_summary.csv` 和客户版报告都会默认跟随 `business_recommended_model`。

### 进阶：严格 OOT 验证（防泄露）
在评估 OOT 效果时，必须开启 `--oot-test` 标志，以确保仅评估测试集（时间 >= valid_end）范围内的数据：

```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202603.tsv \
    --oot-test \
    --train-end 2026-03-11 \
    --valid-end 2026-03-16
```

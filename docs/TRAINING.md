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
| `train_ohab.py` | `线索评级_试驾前` | 多分类 | 智能自适应 OOT/随机切分，多类别权重平衡，支持基线模型对比 |
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

### 资源自适应控制

内存控制参数现在支持**所有**训练任务（train_arrive, train_ohab, train_test_drive），帮助在资源受限的环境下稳定运行：

- `--max-memory-ratio`: 最大内存使用比例（默认 0.8，建议 0.6-0.8）
- `--exclude-memory-heavy-models`: 排除内存密集型模型（KNN, RF, XT）
- `--num-folds-parallel`: 限制并行训练的 fold 数量

---

## 模型验证 (validate_model)

验证模型在测试集上的表现。

### 基础验证
```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202603.tsv
```

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

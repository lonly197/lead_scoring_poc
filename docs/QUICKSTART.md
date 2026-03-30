# 快速开始指南

本文档提供销售线索智能评级 POC 项目的快速入门指南。

---

## 环境准备

```bash
# 安装依赖
uv sync

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 配置数据路径（可选，默认已配置）
```

---

## 一键闭环

**前提**：已按 `.env.example` 配置好环境变量。

```bash
# 1. 训练 HAB 模型（后台运行）
uv run python scripts/run.py train ohab --daemon

# 2. 查看训练日志
uv run python scripts/run.py monitor log train_ohab -f

# 3. 验证模型
uv run python scripts/run.py validate --model-path outputs/models/ohab_model

# 4. 生成报告
uv run python scripts/generate_business_report.py
```

**输出位置**：
- 模型：`outputs/models/ohab_model/`
- 验证：`outputs/validation/ohab_validation/`
- 报告：`outputs/reports/hab_poc_report.md`

**其他模型默认输出**：
- 到店验证：`outputs/validation/arrive_validation/`
- 试驾验证：`outputs/validation/test_drive_validation/`

---

## 统一入口命令

### 数据加载模式

| 模式 | 参数 | 说明 |
|------|------|------|
| 动态拆分 | `--data-path` | 默认模式，自动划分训练/测试集 |
| 提前拆分 | `--train-path` + `--test-path` | 优先级更高，适用于已切分的数据 |

支持 `.csv`、`.tsv`、`.parquet` 格式。

### 训练模型

```bash
# 动态拆分模式（默认）
uv run python scripts/run.py train test_drive --daemon

# 提前拆分模式
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet

# 仅训练 CatBoost（推荐）
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT

# 三模型集成训练
uv run python scripts/run.py train ensemble --daemon

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon --parallel --max-workers 3

# HAB 评级模型
uv run python scripts/run.py train ohab --daemon
```

### 验证模型

```bash
# 动态拆分模式
uv run python scripts/run.py validate \
    --model-path outputs/models/test_drive_model

# 提前拆分模式
uv run python scripts/run.py validate \
    --model-path outputs/models/test_drive_model \
    --test-path ./data/test.parquet
```

### 监控任务

```bash
# 查看运行状态
uv run python scripts/run.py monitor status

# 列出所有任务（包括已完成）
uv run python scripts/run.py monitor list

# 查看任务日志
uv run python scripts/run.py monitor log train_test_drive

# 持续跟踪日志
uv run python scripts/run.py monitor log train_test_drive -f

# 查看任务详情
uv run python scripts/run.py monitor detail train_test_drive

# 停止任务
uv run python scripts/run.py monitor stop train_test_drive

# 停止所有任务
uv run python scripts/run.py monitor stop --all
```

---

## 二级入口命令

也可以直接调用二级入口脚本：

### 训练路由器

```bash
uv run python scripts/train_model.py test_drive --daemon
uv run python scripts/train_model.py ensemble --daemon
```

### 验证路由器

```bash
uv run python scripts/validate_model.py --model-path outputs/models/test_drive_model
```

### 监控脚本

```bash
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_test_drive -f
uv run python scripts/monitor.py stop --all
```

---

## 数据合并

合并线索宽表和 DMP 行为数据：

```bash
# 输出 parquet 格式（推荐）
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表_完整.parquet

# 启用脱敏处理（品牌关键词替换 + ID掩码）
uv run python scripts/merge_data.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/线索宽表_脱敏.parquet \
    --desensitize
```

**功能**：
- 合并 Excel 多个数据 Sheet
- 按手机号关联 DMP 行为数据
- 聚合行为特征（行为次数、最近时间、试驾/下单相关次数等）

**脱敏规则**（`--desensitize`）：
- 品牌关键词：广汽丰田/广丰 → 品牌A，广汽 → 集团A，GTMC → 代号G
- ID 字段：保留前2后2位（如 `AB****XY`）
- 文本字段：手机号、身份证正则替换

---

## 高级参数

如需覆盖 `.env` 配置，可使用命令行参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 数据文件路径（动态拆分模式） | `.env` 中的 `DATA_PATH` |
| `--train-path` | 训练集路径（提前拆分模式） | `.env` 中的 `TRAIN_DATA_PATH` |
| `--test-path` | 测试集路径（提前拆分模式） | `.env` 中的 `TEST_DATA_PATH` |
| `--preset` | 模型预设 | `good_quality` |
| `--time-limit` | 训练时间限制（秒） | 5400 |
| `--included-model-types` | 指定训练的模型类型 | 空（使用预设所有模型） |
| `--parallel` | 并行训练三模型（仅 ensemble） | false |
| `--max-workers` | 并行进程数（仅 ensemble） | 3 |

**示例**：

```bash
# 提前拆分模式
uv run python scripts/run.py train test_drive --daemon \
    --train-path ./data/train.parquet \
    --test-path ./data/test.parquet

# 仅训练 CatBoost
uv run python scripts/run.py train test_drive --daemon \
    --included-model-types CAT

# 三模型并行训练（32GB+ 服务器）
uv run python scripts/run.py train ensemble --daemon \
    --parallel --max-workers 3
```

---

## 训练档位说明

| 档位 | 适用场景 | 特点 |
|------|----------|------|
| `server_16g_compare` | 16GB 服务器（默认） | 两阶段流水线，平衡精度与资源 |
| `server_16g_fast` | 快速验证 | 单阶段，耗时短 |
| `server_16g_probe_nn_torch` | 测试神经网络 | 仅启用 NN_TORCH |
| `lab_full_quality` | 大内存机器 | 最高精度，耗时最长 |

---

## 验证结果说明

重点关注以下文件：

| 文件 | 说明 |
|------|------|
| `hab_bucket_summary.csv` | H/A/B 三桶的到店率、试驾率分层 |
| `evaluation_summary.json` | balanced_accuracy / macro_f1 等指标 |
| `predictions_best.csv` | 每条线索的预测结果 |

**HAB 分层验证**：`hab_bucket_summary.csv` 中 H 桶的到店率应高于 A 桶，A 桶应高于 B 桶（单调递减）。

---

## 故障排查

```bash
# KeyError: '线索创建时间' → 列映射错误
uv run python scripts/diagnose_data.py ./data/your_data.tsv

# 清理缓存后重试
rm -rf outputs/cache/
```

---

## 更多文档

- [FAQ 常见问题](FAQ.md)
- [配置说明](CONFIGURATION.md)
- [架构说明](ARCHITECTURE.md)
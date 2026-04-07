# Scripts 文档

本目录包含数据预处理、模型训练、验证评估等所有脚本。

---

## 目录结构

```
scripts/
├── run.py                  # 一级入口（统一调度器）
│
├── training/               # 训练脚本
│   ├── train_model.py      # 二级入口：训练路由器
│   ├── train_ohab.py       # OHAB 评级训练
│   ├── train_test_drive.py # 试驾预测训练
│   ├── train_test_drive_ensemble.py  # 试驾预测三模型集成
│   └── train_order_after_drive.py    # 下订预测三模型集成
│
├── validation/             # 验证脚本
│   ├── validate_model.py   # 二级入口：验证路由器
│   ├── validate_ohab_model.py        # OHAB 验证
│   └── validate_ensemble.py          # 集成验证
│
├── prediction/             # 预测脚本
│   └── predict.py          # 模型预测（纯推理）
│
├── tools/                  # 工具脚本
│   ├── monitor.py          # 后台任务监控
│   ├── diagnose_data.py    # 数据诊断
│   └── generate_topk.py    # Top-K 名单
│
└── pipeline/               # 数据管道脚本
    ├── run_pipeline.py     # 统一管道运行器
    ├── 01_merge.py         # 数据合并
    ├── 02_profile.py       # 数据探查
    ├── 03_clean.py         # 数据清洗
    ├── 04_desensitize.py   # 数据脱敏
    ├── 05_split.py         # 数据拆分
    └── 06_split_unified.py # 统一数据拆分
```

---

## 一级入口：run.py

```bash
# 训练任务
uv run python scripts/run.py train <task> --daemon

# 验证任务
uv run python scripts/run.py validate --model-path ./outputs/models/test_drive_model

# 监控任务
uv run python scripts/run.py monitor status
uv run python scripts/run.py monitor log train_test_drive -f
uv run python scripts/run.py monitor stop --all
```

---

## 数据管道脚本

### 一键执行

```bash
uv run python scripts/pipeline/run_pipeline.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/final
```

### 管道流程

```
01_merge.py → 02_profile.py → 03_clean.py → 04_desensitize.py → 05_split.py
```

| 步骤 | 功能 | 输出 |
|------|------|------|
| merge | 合并 Excel + DMP | merged.parquet |
| profile | 数据探查诊断 | profile.md |
| clean | 数据清洗 | cleaned.parquet |
| desensitize | 数据脱敏 | desensitized.parquet |
| split | 数据拆分 | train.parquet, test.parquet |

### 统一数据拆分（推荐）

```bash
uv run python scripts/pipeline/06_split_unified.py \
    --input ./data/线索宽表_合并_补充试驾.parquet \
    --output ./data/unified_split \
    --time-column 线索创建时间 \
    --cutoff 2026-03-01
```

**输出**：
- `train.parquet` / `test.parquet` → 试驾预测
- `train_driven.parquet` / `test_driven.parquet` → 下订预测

---

## 预测脚本

```bash
# 简单模式：单模型
uv run python scripts/predict.py \
    --mode simple \
    --model-path ./outputs/models/test_drive_model \
    --data-path ./data/test.parquet \
    --output ./predictions.csv

# 中等模式：三模型集成（推荐）
uv run python scripts/predict.py \
    --mode medium \
    --ensemble-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/test.parquet \
    --output ./predictions.csv \
    --include-ohab

# 高等模式：双阶段预测
uv run python scripts/predict.py \
    --mode advanced \
    --drive-ensemble-path ./outputs/models/test_drive_ensemble \
    --order-ensemble-path ./outputs/models/order_after_drive_ensemble \
    --data-path ./data/test.parquet \
    --output ./predictions.csv \
    --include-ohab
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 预测模式（simple/medium/advanced） | simple |
| `--include-ohab` | 包含 OHAB 评级 | false |
| `--include-original` | 包含原始数据列 | false |

---

## 工具脚本

### diagnose_data.py - 数据诊断

```bash
uv run python scripts/diagnose_data.py ./data/202603.parquet
```

### monitor.py - 后台任务监控

```bash
uv run python scripts/tools/monitor.py status
uv run python scripts/tools/monitor.py log train_ohab -f
uv run python scripts/tools/monitor.py stop --all
```

### generate_topk.py - Top-K 名单

```bash
uv run python scripts/tools/generate_topk.py \
    --model-path ./outputs/models/arrive_model \
    --k 100 500
```
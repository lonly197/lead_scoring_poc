# Scripts 文档

本目录包含数据预处理、模型训练、验证评估等所有脚本。

## 目录结构

```
scripts/
├── pipeline/           # 数据管道脚本（推荐）
│   ├── run_pipeline.py # 统一管道运行器
│   ├── 01_merge.py     # 数据合并
│   ├── 02_profile.py   # 数据探查
│   ├── 03_clean.py     # 数据清洗
│   ├── 04_desensitize.py # 数据脱敏
│   ├── 05_split.py     # 数据拆分
│   ├── excel_to_csv.py   # Excel→CSV（xlsx2csv）
│   ├── excel_to_parquet.py # Excel→Parquet（流式）
│   └── merge_parquet.py   # Parquet 合并（DuckDB）
├── convert_to_parquet.py   # CSV/TSV→Parquet
├── merge_data.py           # 数据合并（旧版）
├── diagnose_data.py        # 数据诊断
├── run.py                  # 一级入口
├── train_model.py          # 二级入口：训练
├── validate_model.py       # 二级入口：验证
├── train_*.py              # 各任务训练脚本
├── validate_*.py           # 各任务验证脚本
├── monitor.py              # 后台任务监控
├── generate_*.py           # 报告生成脚本
└── test_adapter.py         # 数据适配测试
```

---

## 数据管道脚本（推荐）

### run_pipeline.py - 统一管道运行器

一键执行完整数据管道：合并 → 探查 → 清洗 → 脱敏 → 拆分。

```bash
# 一键执行完整管道
uv run python scripts/pipeline/run_pipeline.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/final

# 跳过特定步骤
uv run python scripts/pipeline/run_pipeline.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/final \
    --skip desensitize

# 单步执行
uv run python scripts/pipeline/run_pipeline.py \
    --step clean \
    --input ./data/merged.parquet \
    --output ./data/cleaned.parquet
```

**管道流程**：
```
Excel + DMP → merged.parquet → profile.md → cleaned.parquet → desensitized.parquet → train/test.parquet
```

### 01_merge.py - 数据合并

使用 DuckDB 流式合并 Excel 多 Sheet + DMP 行为数据。

```bash
uv run python scripts/pipeline/01_merge.py \
    --excel ./data/线索宽表.xlsx \
    --dmp ./data/DMP行为数据.csv \
    --output ./data/merged.parquet \
    --memory-limit 4GB \
    --threads 4
```

### 02_profile.py - 数据探查

生成数据概览报告，包括缺失值、分布、清洗建议。

```bash
uv run python scripts/pipeline/02_profile.py \
    --input ./data/merged.parquet \
    --target 线索评级结果 \
    --output ./reports/profile.md
```

### 03_clean.py - 数据清洗

处理异常值、偏斜分布、高基数类别。

```bash
uv run python scripts/pipeline/03_clean.py \
    --input ./data/merged.parquet \
    --output ./data/cleaned.parquet \
    --target 线索评级结果
```

### 04_desensitize.py - 数据脱敏

品牌关键词替换、ID掩码、手机号脱敏。

```bash
uv run python scripts/pipeline/04_desensitize.py \
    --input ./data/cleaned.parquet \
    --output ./data/desensitized.parquet
```

### 05_split.py - 数据拆分

支持 random、oot、auto 三种拆分模式。

```bash
# OOT 时间切分（推荐）
uv run python scripts/pipeline/05_split.py \
    --input ./data/desensitized.parquet \
    --output ./data/final \
    --mode oot \
    --time-column 线索创建时间 \
    --cutoff 2026-03-01

# 随机拆分
uv run python scripts/pipeline/05_split.py \
    --input ./data/desensitized.parquet \
    --output ./data/final \
    --mode random \
    --target 线索评级结果
```

---

## 大文件处理脚本

### excel_to_csv.py - Excel 转 CSV（xlsx2csv）

xlsx2csv 比 openpyxl 快 10-50 倍，适合处理大文件。

```bash
# 转换所有 Sheet（每个 Sheet 一个 CSV）
uv run python scripts/pipeline/excel_to_csv.py \
    --input ./data/large.xlsx \
    --output-dir ./data/csv

# 合并所有 Sheet 到单个 CSV
uv run python scripts/pipeline/excel_to_csv.py \
    --input ./data/large.xlsx \
    --output ./data/merged.csv \
    --merge-sheets
```

### excel_to_parquet.py - Excel 转 Parquet（流式）

使用 openpyxl read_only 模式流式读取，分批写入 Parquet，严格控制内存。

```bash
uv run python scripts/pipeline/excel_to_parquet.py \
    --input ./data/large.xlsx \
    --output ./data/output.parquet \
    --batch-size 50000
```

### merge_parquet.py - Parquet 合并（DuckDB）

直接合并 Parquet 格式的线索数据和 DMP 数据，无需 Excel 转换。

```bash
uv run python scripts/pipeline/merge_parquet.py \
    --clue ./data/线索数据.parquet \
    --dmp ./data/DMP数据.parquet \
    --output ./data/merged.parquet \
    --phone-column "手机号（脱敏）"
```

### convert_to_parquet.py - CSV/TSV 转 Parquet

转换 CSV/TSV 文件为 Parquet 格式，提升加载速度和减少存储空间。

```bash
# 转换单个文件
uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv

# 批量转换目录
uv run python scripts/convert_to_parquet.py ./data --batch

# 指定压缩算法
uv run python scripts/convert_to_parquet.py ./data/file.tsv --compression gzip

# 流式处理大文件
uv run python scripts/convert_to_parquet.py ./data/large.tsv --chunksize 100000
```

---

## 训练脚本

### run.py - 一级入口

统一调度器，支持训练、验证、监控子命令。

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

### train_model.py - 二级入口：训练路由器

根据任务名路由到具体训练脚本。

```bash
uv run python scripts/train_model.py test_drive --daemon \
    --data-path ./data/202603.parquet \
    --included-model-types CAT
```

### 各任务训练脚本

| 脚本 | 目标变量 | 说明 |
|------|----------|------|
| `train_test_drive.py` | 试驾标签 | 核心任务：试驾预测 |
| `train_ohab.py` | 线索评级结果 | HAB 评级 |
| `train_arrive.py` | 到店标签 | 到店预测 |
| `train_test_drive_ensemble.py` | 多标签 | 三模型集成 |

---

## 验证脚本

### validate_model.py - 二级入口：验证路由器

自动加载训练时记录的测试集配置，确保防泄漏评估。

```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/test_drive_model \
    --data-path ./data/202603.parquet
```

### 各任务验证脚本

| 脚本 | 说明 |
|------|------|
| `validate_test_drive_model.py` | 试驾模型验证 |
| `validate_ohab_model.py` | HAB 模型验证 |
| `validate_arrive_model.py` | 到店模型验证 |
| `validate_ensemble.py` | 集成模型验证 |

---

## 预测脚本

### predict.py - 模型预测

对输入数据进行预测，将预测结果追加到 DataFrame 中返回。与验证脚本不同，predict.py 不要求目标标签存在，专注于纯推理。

```bash
# 基本用法：输出 ID + 预测结果
uv run python scripts/predict.py \
    --model-path ./outputs/models/test_drive_model \
    --data-path ./data/final_v4_test.parquet \
    --output ./predictions.csv

# 包含原始数据列
uv run python scripts/predict.py \
    --model-path ./outputs/models/test_drive_model \
    --data-path ./data/final_v4_test.parquet \
    --output ./predictions_full.csv \
    --include-original
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径（必需） | - |
| `--data-path` | 数据文件路径（必需） | - |
| `--output` | 输出文件路径 | 不保存 |
| `--include-original` | 包含原始数据列 | False |
| `--id-column` | ID 列名 | 线索唯一ID |

**输出格式**：

| 模式 | 列数 | 内容 |
|------|------|------|
| 默认 | 3 列 | `线索唯一ID`, `预测概率`, `预测标签` |
| `--include-original` | 完整列 | 原始数据 + 预处理列 + 预测结果 |

---

## 其他脚本

### diagnose_data.py - 数据诊断

检查列映射、数据格式、特征分布。

```bash
uv run python scripts/diagnose_data.py ./data/202603.parquet
```

### monitor.py - 后台任务监控

查看后台任务状态、日志，停止任务。

```bash
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_ohab -f
uv run python scripts/monitor.py stop --all
```

### generate_business_report.py - 业务报告生成

生成客户版评级报告。

```bash
uv run python scripts/generate_business_report.py \
    --model-dir ./outputs/models/ohab_model \
    --validation-dir ./outputs/validation \
    --output-path ./outputs/reports/hab_poc_report.md
```

### generate_topk.py - Top-K 名单生成

生成 Top-K 高意向线索名单。

```bash
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/arrive_model \
    --k 100 500
```

---

## 脚本关系图

```
数据管道（推荐）
┌─────────────────────────────────────────────────────────────┐
│  run_pipeline.py                                             │
│  ├── 01_merge.py      ← excel_to_parquet.py / excel_to_csv.py│
│  ├── 02_profile.py                                          │
│  ├── 03_clean.py                                            │
│  ├── 04_desensitize.py                                      │
│  └── 05_split.py                                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  训练入口                                                    │
│  run.py → train_model.py → train_*.py                       │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  验证入口                                                    │
│  run.py → validate_model.py → validate_*.py                 │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  预测入口                                                    │
│  predict.py（纯推理，无需标签）                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 维护日志

| 日期 | 新增脚本 | 说明 |
|------|----------|------|
| 2026-03-31 | `predict.py` | 模型预测脚本（纯推理，无需标签） |
| 2026-03-31 | `excel_to_csv.py` | Excel→CSV 快速转换 |
| 2026-03-31 | `excel_to_parquet.py` | Excel→Parquet 流式处理 |
| 2026-03-31 | `merge_parquet.py` | Parquet 合并（DuckDB） |
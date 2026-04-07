# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
uv sync

# 试驾预测模型（三模型集成）
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# 下订预测模型（三模型集成）
uv run python scripts/run.py train order_after_drive --daemon --included-model-types CAT

# 监控任务
uv run python scripts/run.py monitor status
uv run python scripts/run.py monitor log train_test_drive -f

# 数据诊断
uv run python scripts/diagnose_data.py ./data/202603.tsv
```

详细命令和参数说明见 `docs/TRAINING.md`。

---

## 入口架构

```
run.py (一级入口)
├── train <task>     → train_model.py (二级入口)
│   ├── arrive       → train_arrive.py
│   ├── test_drive   → train_test_drive.py
│   ├── ohab         → train_ohab.py
│   ├── ensemble     → train_test_drive_ensemble.py  # 试驾预测（三模型）
│   └── order_after_drive → train_order_after_drive.py  # 下订预测（三模型）
└── validate         → validate_model.py
```

---

## 统一数据源方案

```bash
# 生成统一分割数据
uv run python scripts/pipeline/06_split_unified.py \
    --input ./data/线索宽表_合并_补充试驾.parquet \
    --output ./data/unified_split \
    --time-column 线索创建时间 \
    --cutoff 2026-03-01

# 输出：
# - train.parquet / test.parquet → 试驾预测
# - train_driven.parquet / test_driven.parquet → 下订预测
```

---

## 数据管道架构

```
scripts/pipeline/
├── 01_merge.py        # 数据合并
├── 02_profile.py      # 数据探查
├── 03_clean.py        # 数据清洗
├── 04_desensitize.py  # 数据脱敏
├── 05_split.py        # 数据拆分
├── 06_split_unified.py # 统一数据拆分
└── run_pipeline.py    # 统一运行器
```

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/run.py` | 一级入口：统一调度器 |
| `scripts/pipeline/06_split_unified.py` | 统一数据拆分 |
| `config/config.py` | 配置管理：ID列、泄漏字段、标签分组、泄漏验证函数 |
| `src/data/adapter.py` | 数据格式适配 |
| `src/data/loader.py` | 数据加载：特征工程 |
| `src/models/predictor.py` | 模型封装 |

---

## 模型选择配置

通过 `--included-model-types` 指定模型类型：

| 模型类型 | 说明 | 训练时间 |
|----------|------|---------|
| `CAT` | CatBoost（推荐） | ~60s |
| `GBM` | LightGBM | ~30s |
| `XGB` | XGBoost | ~25s |
| `CAT,GBM` | 多模型 | ~90s |

---

## 预测模式

`scripts/predict.py` 支持三种模式：

| 模式 | 描述 | 业务匹配度 |
|------|------|-----------|
| `simple` | 单模型 | 部分 |
| `medium` | 三模型集成（推荐） | 完整匹配试驾前阶段 |
| `advanced` | 试驾+下订双阶段 | 完整匹配全部规则 |

详细说明见 `docs/TRAINING.md`。

---

## 数据质量警告

### ⚠️ 特征泄漏风险（已修复）

**历史问题**：训练某时间窗口模型时，其他时间窗口标签列导致性能虚高（ROC-AUC 99.96%）。

**修复方案**：标签分组系统自动排除同组兄弟标签，详见 `docs/CONFIGURATION.md`。

**仍需排除的泄漏特征**（`config/config.py` 的 `leakage_columns`）：
- **直接编码目标**：`是否到店`、`是否试驾`、`试驾天数差`
- **后验评级**：`线索评级结果`、`线索评级_试驾后`
- **JSON 提取**：`提及试驾`、`提及到店`

---

## AutoML 预处理

**自动处理**（无需手动）：
- 类别编码、缺失值填充、异常值变换

**需手动处理**：
- 时间特征提取（在 `loader.py` 中实现）

---

## 配置优先级

```
命令行参数 > .env 环境变量 > config/config.py 默认值
```

详细配置见 `docs/CONFIGURATION.md`。

---

## 项目约定

### 数据处理优化
- **DuckDB 优先**：管道脚本优先使用 DuckDB SQL 向量化处理

### 文档规范
- **简洁一致**：文档描述简洁，与代码实现保持同步
- **CHANGELOG.md**：仅记录关键变更节点

---

## 故障排查

```bash
# KeyError: '线索创建时间' → 列映射错误
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 服务器代码未同步
git pull && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

详细故障排查见 `docs/TROUBLESHOOTING.md`。
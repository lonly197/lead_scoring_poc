# 快速开始指南

本文档提供销售线索智能评级 POC 项目的快速入门指南。

---

## 环境准备

```bash
# 使用 uv 安装依赖
uv sync

# 复制环境变量配置
cp .env.example .env

# 编辑 .env 配置数据路径
```

---

## 数据诊断

在训练前，可使用诊断脚本检查数据格式：

```bash
# 检查数据格式和关键列
uv run python scripts/diagnose_data.py ./data/202603.tsv
```

**诊断功能**：

- 检测分隔符和列数
- 验证关键列是否存在（线索创建时间、线索唯一ID 等）
- 显示 OHAB 评级分布、是否仍保留 O 级
- 检查 `is_final_ordered` 是否存在及其分布
- 输出 `线索创建时间` 范围，便于判断是否适合显式启用 OOT
- 列出所有列名和样本值（方便调试映射问题）

---

## 运行训练

### 原数据格式训练（train_arrive, train_ohab）

适用于原格式数据（如 `20260308-v2.csv`，逗号分隔，有表头，目标变量预计算）。

```bash
# 方式 1：前台运行（推荐用于调试）
uv run python scripts/train_arrive.py

# 方式 2：后台运行（推荐用于长时间训练）
uv run python scripts/run.py train_arrive --daemon

# 方式 3：指定参数后台运行
uv run python scripts/run.py train_arrive --daemon \
    --preset high_quality \
    --time-limit 3600 \
    --data-path ./data/20260308-v2.csv

# 可选：试驾预测训练
uv run python scripts/run.py train_test_drive --daemon --preset high_quality

# 可选：OHAB 评级训练
uv run python scripts/run.py train_ohab --daemon --preset high_quality
```

### HAB 评级训练与验证（推荐 POC 主流程）

适用于新格式数据（如 `202602~03.tsv`，跨度覆盖 2 月和 3 月）。当前推荐将 `train_ohab.py` 用作 **H/A/B 智能评级主入口**：

- 默认 `--label-mode hab`，仅训练 `H/A/B`
- `O` 视为已成交/已锁单状态，不进入常规评级桶
- 可选 `--enable-model-comparison`，保留一个内部子模型作为基线（推荐 `gbm`）
- 训练完成后会在模型目录输出：
  - `feature_metadata.json`（含 `artifact_status`，用于校验训练是否完整）
  - `feature_importance.*`
  - `business_dimension_contribution.*`
  - `predictions_test.csv`
  - `hab_bucket_summary.*`
  - `evaluation_summary.json`
  - `model_comparison_config.json`

**服务器端最简单闭环**

如果服务器已经按最新 [.env.example](../.env.example) 部署好 `DATA_PATH` 和 `OHAB_*` 环境变量，那么训练、验证、报告生成三步都可以直接使用默认参数，完成默认两阶段 HAB 流水线闭环。

```bash
# 1. 训练：后台启动 HAB 模型（默认使用 .env 中的 DATA_PATH + server_16g_compare）
uv run python scripts/run.py train_ohab --daemon

# 2. 查看训练状态 / 跟日志
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_ohab -f

# 3. 验证：生成两阶段 HAB 评估结果
uv run python scripts/validate_model.py

# 4. 生成客户版 Markdown 报告
uv run python scripts/generate_business_report.py
```

这套最短命令默认依赖 `.env` 中的以下关键配置：

- `DATA_PATH=./data/202602~03.tsv`
- `OHAB_TRAINING_PROFILE=server_16g_compare`
- `OHAB_PIPELINE_MODE=two_stage`
- `OHAB_SPLIT_MODE=random`
- `OHAB_SPLIT_GROUP_MODE=phone_or_lead`

执行完成后，重点检查：

- `outputs/validation/predictions_best.csv`
- `outputs/reports/hab_poc_report.md`

补充说明：

- 默认 `server_16g_compare` 会训练两阶段流水线，并输出统一的 `predictions_best.csv`、`hab_bucket_summary_best.csv`、`evaluation_summary.json`；
- 如需 `baseline vs best` 技术对比，需要显式切回单阶段模式，并启用模型对比；
- 客户版 `hab_poc_report.md` 默认读取验证目录中的主评估结果，不再依赖旧的单阶段对比文件。

**推荐切分策略**：

当前默认推荐随机分组切分，因为现有数据集更适合先保证“同客不泄漏”和 HAB 分层稳定性。只有在你明确确认数据已是评分时点快照时，再显式切到 OOT。

```bash
# 推荐：16GB 服务器 HAB 评级训练（默认两阶段 + 随机分组切分）
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare
```

**为什么推荐 `server_16g_compare`**：

- 适配 16GB 内存服务器，默认使用 `good_quality + 3 folds`
- 默认训练阶段按 `balanced_accuracy` 选模
- 默认使用 `two_stage` 流水线
- 默认使用 `split_mode=random`
- 启动前会自动探测当前 CPU 和可用内存，并自动收敛内存上限
- 在 16GB 服务器上，默认固定使用 `sequential + num_folds_parallel=1`，避免模型级并行或多折并行把内存打满
- 默认排除 `RF/XT/KNN/FASTAI/NN_TORCH` 等高内存模型
- 保留切回单阶段后做技术对比的能力

**手动覆盖只留给高级场景**：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare \
    --memory-limit-gb 8.5 \
    --num-folds-parallel 1
```

**如果只想验证 `NN_TORCH` 是否有增益**：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_probe_nn_torch
```

这个实验档只恢复 `NN_TORCH`，继续排除 `FASTAI/RF/XT`，并关闭 bagging，适合做单变量对比；未验证出显著收益前，不建议把它升级为正式默认档。

**如果想验证业务指标导向的训练口径**：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare_balanced
```

这个档位只把训练阶段的 `eval_metric` 改为 `balanced_accuracy`，适合和默认档 `server_16g_compare` 做受控对比。

**如果在更大机器上追求更高精度**：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile lab_full_quality
```

**训练完成后查看状态**：

```bash
uv run python scripts/monitor.py status
uv run python scripts/monitor.py log train_ohab -f
```

### 验证 HAB 结果

训练结束后，直接对同一份数据跑验证脚本。验证脚本会自动读取模型元数据里的：

- `split_info`
- `label_policy`
- `label_mode`
- `decision_policy`

因此不需要手动再指定阈值。

```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv
```

验证目录默认输出到 `outputs/validation/`，重点关注：

- `evaluation_report.txt`
- `hab_bucket_summary.csv`
- `lead_actions.csv`
- `monotonicity_check.json`
- `predictions_best.csv`

其中：

- `hab_bucket_summary.csv`：看 `H/A/B` 三桶的 14 天到店率、试驾率是否形成单调分层
- `lead_actions.csv`：可直接给业务看，包含 `预测HAB + 建议SOP + 原因1-3`
- `evaluation_summary.json`：看 `balanced_accuracy / macro_f1 / accuracy` 以及两阶段决策策略

### 生成客户版 Markdown 报告

如果要把训练和验证结果整理成面向客户的汇报文档，继续执行：

```bash
uv run python scripts/generate_business_report.py \
    --model-dir ./outputs/models/ohab_model \
    --validation-dir ./outputs/validation \
    --output-path ./outputs/reports/hab_poc_report.md
```

### 服务器最短闭环

在服务器上完成一次最短闭环，按下面顺序执行即可：

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 训练 HAB 模型
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --training-profile server_16g_compare

# 3. 验证
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv

# 4. 生成客户版报告
uv run python scripts/generate_business_report.py \
    --model-dir ./outputs/models/ohab_model \
    --validation-dir ./outputs/validation \
    --output-path ./outputs/reports/hab_poc_report.md
```

如果服务器 `.env` 已经配置完成，也可以直接使用上一节的“服务器端最简单闭环”，直接使用默认参数即可。

---

## 监控后台任务

```bash
# 查看运行中的任务
uv run python scripts/monitor.py status

# 查看任务详情
uv run python scripts/monitor.py detail train_arrive

# 查看任务日志（显示最新任务的日志）
uv run python scripts/monitor.py log train_arrive

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive -f

# 停止指定任务
uv run python scripts/monitor.py stop train_arrive

# 一键停止所有运行中的任务
uv run python scripts/monitor.py stop --all

# 列出所有任务（包括已完成）
uv run python scripts/monitor.py list
```

---

## 生成 Top-K 名单

```bash
# 到店二分类任务：沿用原有用法
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/arrive_model \
    --data-path ./data/20260308-v2.csv \
    --k 100 500 1000

# OHAB 多分类任务：必须显式指定目标类别，推荐按 P(H) 排序
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv \
    --target-class H \
    --k 100 500 1000
```

说明：
- 二分类任务默认按正类概率排序。
- 多分类任务不会再默认使用“最大类别概率”排序；必须显式传 `--target-class`。
- OHAB 推荐使用 `--target-class H`，表示按 `P(H)` 生成优先跟进名单。

---

## 模型推理

### 使用已训练模型预测

```python
from autogluon.tabular import TabularPredictor

# 加载模型
predictor = TabularPredictor.load("outputs/models/arrive_model")

# 预测新数据
import pandas as pd
new_data = pd.read_csv("new_leads.csv")

# 预测类别
predictions = predictor.predict(new_data)

# 预测概率
probabilities = predictor.predict_proba(new_data)
```

### 命令行快速查看

```bash
# 查看模型信息
uv run python -c "
from autogluon.tabular import TabularPredictor
p = TabularPredictor.load('outputs/models/arrive_model')
print('最佳模型:', p.model_best)
print('评估指标:', p.eval_metric)
"
```

### 验证脚本

```bash
# 使用验证脚本（默认读取 ./outputs/models/ohab_model）
uv run python scripts/validate_model.py --daemon --data-path ./data/202602~03.tsv

# 显式指定模型路径，避免误用旧的 ohab_oot 目录
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv
```

**验证行为说明**：
- 若模型元数据中存在 `split_info`，脚本会自动只评估训练时记录的测试集；随机分组切分与显式 OOT 都适用。
- 若验证数据中存在 `is_final_ordered`，脚本会额外输出“AI 评级 vs 最终下定”的业务转化统计，但该列不会参与模型特征。
- 验证脚本只接受由统一入口 `train_ohab.py` / `scripts/run.py train_ohab` 生成且 `feature_metadata.json -> artifact_status.training_complete=true` 的模型目录。
- 若模型目录缺少 `artifact_status`，或来自旧的 `ohab_oot` 流程，`validate_model.py` 会直接报错并要求重新训练。
- 若训练仅缺少 `feature_importance` / `business_dimension_contribution` / `topk` 等补充产物，仍可继续验证，但客户报告中的特征贡献与 Top 特征章节会为空。
- `train_ohab` 默认只生成 `feature_importance.csv/json` 与 `business_dimension_contribution.json`，不会默认生成 PNG 图表；客户报告不依赖这些图片。
- 若确实需要汇报配图，可在训练时显式开启 `OHAB_GENERATE_PLOTS=true` 或传 `--generate-plots`。

---

## 更多文档

- [FAQ 常见问题](FAQ.md)
- [训练脚本说明](TRAINING.md)
- [配置说明](CONFIGURATION.md)
- [架构说明](ARCHITECTURE.md)

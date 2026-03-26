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
uv run python scripts/run.py train_ohab --daemon

# 2. 查看训练日志
uv run python scripts/monitor.py log train_ohab -f

# 3. 验证模型
uv run python scripts/validate_model.py --model-type ohab

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

## 常用命令

### 监控任务

```bash
uv run python scripts/monitor.py status      # 查看运行状态
uv run python scripts/monitor.py log train_ohab -f   # 跟踪日志
uv run python scripts/monitor.py stop --all   # 停止所有任务
```

### 数据诊断

```bash
uv run python scripts/diagnose_data.py ./data/202602~03.tsv
```

### 生成 Top-K 名单

```bash
uv run python scripts/generate_topk.py \
    --model-path ./outputs/models/ohab_model \
    --target-class H \
    --k 100 500 1000
```

### 验证其他模型

```bash
# 到店模型
uv run python scripts/validate_model.py --model-type arrive

# 试驾模型
uv run python scripts/validate_model.py --model-type test_drive
```

---

## 高级参数

如需覆盖 `.env` 配置，可使用命令行参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | 数据文件路径 | `.env` 中的 `DATA_PATH` |
| `--training-profile` | 训练档位 | `server_16g_compare` |
| `--time-limit` | 训练时间限制（秒） | 5400 |
| `--preset` | 模型预设 | `good_quality` |
| `--memory-limit-gb` | 内存软限制 | 自动探测 |
| `--generate-plots` | 生成 PNG 图表 | false |

**示例**：

```bash
# 指定数据路径
uv run python scripts/run.py train_ohab --daemon --data-path ./data/custom.tsv

# 调整训练时间
uv run python scripts/run.py train_ohab --daemon --time-limit 3600

# 生成汇报配图
uv run python scripts/run.py train_ohab --daemon --generate-plots
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

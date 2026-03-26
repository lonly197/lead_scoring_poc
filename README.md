# 销售线索智能评级 POC 项目

基于 AutoML 技术的销售线索智能评级预测系统。

## 项目目标

- **主流程**：HAB 智能评级训练与验证
- **默认目标变量**：`线索评级结果`
- **业务目标**：输出可解释的 `H/A/B` 分层结果，支持销售资源优先分配与 SOP 差异化跟进。

## 核心特性

- **🚀 默认随机分组切分**：`train_ohab` 默认按手机号优先、线索 ID 回退的分组键做随机 `70/15/15` 切分，避免同客泄漏。
- **🛡️ 全链路防泄漏**：训练阶段记录测试集分组键，`validate_model.py` 会自动只评估训练时定义的测试集。
- **🚿 特征脱水技术**：内置严格的后验特征排除机制，彻底杜绝模型“偷看答案”的行为。
- **🧭 两阶段 HAB 流水线**：默认先做 `H vs 非H`，再做 `A vs B`，并在验证集调 `H` 阈值，优先兼顾分类稳定性与业务单调性。
- **⚖️ 可选单阶段对比**：如需保留 `baseline vs best` 技术对比，可切换回单阶段训练并启用模型对比。
- **🧠 资源自适应训练**：`train_ohab` 启动前会自动探测当前 CPU 数和可用内存，自动收敛 `memory_limit_gb` 与 `num_folds_parallel`；显式命令行参数仍可覆盖。

## 快速开始

详细指南请参阅 **[快速开始指南](docs/QUICKSTART.md)**。

```bash
# 安装依赖
uv sync && cp .env.example .env

# HAB 评级训练（16GB 服务器推荐档，默认两阶段 + 随机分组切分）
uv run python scripts/run.py train_ohab --daemon \
  --data-path ./data/202602~03.tsv \
  --training-profile server_16g_compare

# 验证模型（自动按训练时记录的测试集做防泄漏评估）
uv run python scripts/validate_model.py \
  --model-path ./outputs/models/ohab_model \
  --data-path ./data/202602~03.tsv

# 生成客户版报告（首页展示基线 vs 最优模型）
uv run python scripts/generate_business_report.py \
  --model-dir ./outputs/models/ohab_model \
  --validation-dir ./outputs/validation \
  --output-path ./outputs/reports/hab_poc_report.md
```

### 服务端运行注意事项

在服务器上运行训练脚本时，确保虚拟环境配置正确：

```bash
# 推荐方式：干净 shell，直接运行（不要预先 source .venv/bin/activate）
cd /opt/lead_scoring_poc
uv run python scripts/train_arrive.py

# 如果已激活 venv，使用 --active 参数
source .venv/bin/activate
uv run --active python scripts/train_arrive.py

# 如果遇到环境问题，清除变量后重试
unset VIRTUAL_ENV
uv run python scripts/train_arrive.py
```

**问题现象**：日志中出现大量 `workers have not registered within the timeout`，且每个 worker 都在重复下载依赖包（torch、catboost 等）。

**原因**：`uv run` 检测到 `VIRTUAL_ENV` 环境变量与项目 `.venv` 路径不匹配（如预先激活了 venv），导致 Ray worker 无法复用已有环境，为每个 worker 重新创建虚拟环境和下载依赖。

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 环境准备、运行训练、监控任务 |
| [训练脚本](docs/TRAINING.md) | 脚本说明、命令行参数、评估指标 |
| [配置说明](docs/CONFIGURATION.md) | 环境变量、特征配置、数据字段 |
| [架构说明](docs/ARCHITECTURE.md) | 技术栈、项目结构、核心功能 |
| [常见问题](docs/FAQ.md) | 数据处理、训练相关、故障排查 |

## 环境要求

- Python 3.9-3.12（推荐 3.11）
- 内存：16GB+（16GB 服务器建议使用 `server_16g_compare`，更大机器再用 `lab_full_quality`）
- 磁盘：2-10G（根据 preset）

## 许可证

MIT

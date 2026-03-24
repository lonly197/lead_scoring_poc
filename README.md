# 销售线索智能评级 POC 项目

基于 AutoGluon TabularPredictor 的销售线索智能评级预测系统。

## 项目目标

- **主指标**：到店预测（Top-K 高优线索到店率）
- **目标变量**：`到店标签_14天`
- **业务目标**：输出线索评分排名，支持销售资源优先分配。

## 核心特性

- **🚀 智能自适应切分**：脚本自动探查数据时间跨度。跨度充足时自动执行 OOT（Out-of-Time）时间切分；跨度不足时（如单日快照）自动退化为分层随机切分。
- **🛡️ 全链路防泄漏**：在随机切分降级模式下，脚本自动提取“防泄漏指纹”（测试集 ID），强制 `validate_model.py` 仅评估从未见过的数据，确保指标真实。
- **🚿 特征脱水技术**：内置严格的后验特征排除机制，彻底杜绝模型“偷看答案”的行为。

## 快速开始

详细指南请参阅 **[快速开始指南](docs/QUICKSTART.md)**。

```bash
# 安装依赖
uv sync && cp .env.example .env

# 到店预测训练 (统一智能自适应版)
uv run python scripts/train_arrive.py --data-path ./data/202602_03.csv

# OHAB 评级训练 (统一智能自适应版)
uv run python scripts/train_ohab.py --data-path ./data/202602_03.csv

# 验证模型 (防泄漏指纹自动识别)
uv run python scripts/validate_model.py --model-path ./outputs/models/ohab_model
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
- 内存：16GB+
- 磁盘：2-10G（根据 preset）

## 许可证

MIT
# 销售线索智能评级 POC 项目

基于 AutoGluon TabularPredictor 的销售线索智能评级预测系统。

## 项目目标

- **主指标**：到店预测（Top-K 高优线索到店率）
- **目标变量**：`到店标签_14天`（14 天内到店标签）
- **业务目标**：输出线索评分排名，支持销售资源优先分配

> **选择说明**：POC 方案的核心业务规则是 OHAB 评级，但当前数据中 OHAB 评级 85% 为 Unknown，成交标签无正样本，试驾标签仅 1.4%。综合考虑数据可用性，选择到店预测作为 POC 主指标，后续迭代再扩展到 OHAB 评级预测。

## 快速开始

详细指南请参阅 **[快速开始指南](docs/QUICKSTART.md)**。

```bash
# 安装依赖
uv sync && cp .env.example .env

# 到店预测训练（核心任务）
uv run python scripts/train_arrive.py

# 后台运行长时间训练
uv run python scripts/run.py train_arrive --daemon

# 验证模型（OOT 测试集，避免数据泄露）
uv run python scripts/validate_model.py \
    --data-path ./data/202603.tsv \
    --oot-test \
    --valid-end "2026-03-16"
```

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
#!/bin/bash
# 销售线索评级模型快速启动脚本
# 清除可能导致环境冲突的变量，确保 Ray worker 正确复用 .venv

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 清除可能存在的错误环境变量（解决 uv run 路径匹配问题）
unset VIRTUAL_ENV

cd "$PROJECT_DIR"

# 如果没有参数，显示帮助
if [ $# -eq 0 ]; then
    echo "用法: $0 <命令> [参数...]"
    echo ""
    echo "常用命令:"
    echo "  train_arrive      训练到店预测模型（核心任务）"
    echo "  train_ohab        训练 OHAB 评级模型"
    echo "  validate_model    验证模型"
    echo ""
    echo "示例:"
    echo "  $0 train_arrive"
    echo "  $0 train_arrive --data-path ./data/202603.tsv"
    exit 1
fi

# 运行命令
uv run python "scripts/${1}.py" "${@:2}"
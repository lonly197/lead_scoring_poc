#!/usr/bin/env python3
"""
数据探查脚本

功能：
- 复用 diagnose_data.py 的 DataProfiler 类
- 生成数据概览报告
- 输出清洗建议

用法：
    uv run python scripts/pipeline/02_profile.py \\
        --input ./data/merged.parquet \\
        --target 线索评级结果 \\
        --output ./reports/profile.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import print_step


def profile_data(
    input_path: Path,
    output_path: Optional[Path] = None,
    target_column: Optional[str] = None,
) -> dict:
    """
    探查数据并生成报告

    Args:
        input_path: 输入数据文件路径
        output_path: 报告输出路径
        target_column: 目标变量列名

    Returns:
        探查结果字典
    """
    from scripts.diagnose_data import DataProfiler

    print("=" * 60)
    print("数据探查脚本")
    print("=" * 60)

    # 创建探查器
    profiler = DataProfiler(str(input_path), target_column)

    # 执行探查
    print_step("加载数据", "running")
    if not profiler.load():
        print_step("加载数据", "error", "文件加载失败")
        return {}
    print_step("加载数据", "success")

    # 基础探查
    print_step("基础信息探查", "running")
    profiler.profile_basic()
    print_step("基础信息探查", "success")

    # 缺失值统计
    print_step("缺失值统计", "running")
    profiler.profile_missing()
    print_step("缺失值统计", "success")

    # 列类型统计
    print_step("列类型统计", "running")
    profiler.profile_columns()
    print_step("列类型统计", "success")

    # 目标变量检查
    print_step("目标变量检查", "running")
    profiler.profile_target()
    print_step("目标变量检查", "success")

    # 生成清洗建议
    print_step("生成清洗建议", "running")
    profiler.generate_cleaning_suggestions()
    print_step("生成清洗建议", "success")

    # 打印建议
    profiler.print_suggestions()

    # 生成报告
    if output_path:
        print_step("生成报告", "running", str(output_path))
        profiler.generate_report(str(output_path))
        print_step("生成报告", "success")

    # 打印摘要
    print("\n" + "=" * 60)
    print("探查完成")
    print("=" * 60)

    if output_path:
        print(f"报告文件: {output_path}")

    # 打印关键指标
    basic = profiler.profile.get("basic", {})
    missing = profiler.profile.get("missing", {})
    target = profiler.profile.get("target", {})

    print(f"数据量: {basic.get('rows', 0):,} 行 × {basic.get('columns', 0)} 列")
    print(f"缺失率: {missing.get('overall_missing_ratio', 0)}%")
    print(f"高缺失列: {missing.get('high_missing_columns', 0)} 个")

    if target.get("found"):
        print(f"目标变量: {target.get('column')}")
        print(f"不平衡比: {target.get('imbalance_ratio', 'N/A')}")

    print("=" * 60)

    return profiler.profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据探查脚本 - 生成数据概览报告和清洗建议",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    uv run python scripts/pipeline/02_profile.py \\
        --input ./data/merged.parquet

    # 指定目标变量
    uv run python scripts/pipeline/02_profile.py \\
        --input ./data/merged.parquet \\
        --target 线索评级结果

    # 输出报告
    uv run python scripts/pipeline/02_profile.py \\
        --input ./data/merged.parquet \\
        --output ./reports/profile.md
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--target", "-t",
        default=None,
        help="目标变量列名（可选）"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="报告输出路径（Markdown 格式）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1

    # 默认报告路径
    output_path = Path(args.output) if args.output else None

    try:
        profile_data(
            input_path=input_path,
            output_path=output_path,
            target_column=args.target,
        )
        return 0

    except Exception as e:
        print(f"❌ 探查失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
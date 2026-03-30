#!/usr/bin/env python3
"""
数据管道统一运行器

功能：
- 一键执行完整管道
- 支持跳过特定步骤
- 支持单独运行某个步骤

用法：
    # 完整管道
    uv run python scripts/pipeline/run_pipeline.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/final

    # 跳过脱敏步骤
    uv run python scripts/pipeline/run_pipeline.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --skip desensitize

    # 仅运行特定步骤
    uv run python scripts/pipeline/run_pipeline.py \\
        --step clean \\
        --input ./data/merged.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import print_step, print_summary


# 管道步骤定义
PIPELINE_STEPS = [
    {
        "name": "merge",
        "script": "01_merge.py",
        "description": "数据合并",
        "required_args": ["excel", "dmp"],
        "output_arg": "output",
    },
    {
        "name": "profile",
        "script": "02_profile.py",
        "description": "数据探查",
        "required_args": ["input"],
        "output_arg": "output",
        "optional": True,
    },
    {
        "name": "clean",
        "script": "03_clean.py",
        "description": "数据清洗",
        "required_args": ["input"],
        "output_arg": "output",
    },
    {
        "name": "desensitize",
        "script": "04_desensitize.py",
        "description": "数据脱敏",
        "required_args": ["input"],
        "output_arg": "output",
        "optional": True,
    },
    {
        "name": "split",
        "script": "05_split.py",
        "description": "数据拆分",
        "required_args": ["input"],
        "output_arg": "output",
    },
]


def get_script_path(script_name: str) -> Path:
    """获取脚本完整路径"""
    return Path(__file__).parent / script_name


def run_step(
    step: dict,
    args: dict,
    extra_args: List[str] = None,
) -> dict:
    """
    运行单个管道步骤

    Args:
        step: 步骤定义
        args: 参数字典
        extra_args: 额外的命令行参数

    Returns:
        执行结果字典
    """
    import polars as pl

    script_path = get_script_path(step["script"])
    cmd = [sys.executable, str(script_path)]

    # 构建命令行参数
    for arg_name in step["required_args"]:
        if arg_name in args and args[arg_name]:
            cmd.extend([f"--{arg_name}", str(args[arg_name])])

    if step["output_arg"] in args and args[step["output_arg"]]:
        cmd.extend([f"--{step['output_arg']}", str(args[step["output_arg"]])])

    # 添加额外参数
    if extra_args:
        cmd.extend(extra_args)

    print_step(step["description"], "running")
    print(f"  命令: {' '.join(cmd[:4])}...")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print_step(step["description"], "success", f"{elapsed:.1f}s")
            return {
                "status": "success",
                "time": elapsed,
                "output": str(args.get(step["output_arg"], "")),
            }
        else:
            print_step(step["description"], "error", result.stderr[:200])
            return {
                "status": "error",
                "time": elapsed,
                "error": result.stderr,
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print_step(step["description"], "error", str(e))
        return {
            "status": "error",
            "time": elapsed,
            "error": str(e),
        }


def run_full_pipeline(
    args: dict,
    skip_steps: List[str] = None,
    extra_args: Dict[str, List[str]] = None,
) -> dict:
    """
    执行完整管道

    Args:
        args: 参数字典
        skip_steps: 要跳过的步骤列表
        extra_args: 每个步骤的额外参数

    Returns:
        执行结果字典
    """
    skip_steps = skip_steps or []
    extra_args = extra_args or {}

    results = {}
    total_start = time.time()

    # 确定中间文件路径
    output_prefix = Path(args.get("output", "./data/final"))
    intermediate_dir = output_prefix.parent

    # 构建中间文件路径
    current_input = None
    current_output = None

    for i, step in enumerate(PIPELINE_STEPS):
        step_name = step["name"]

        # 检查是否跳过
        if step_name in skip_steps:
            print_step(step["description"], "skip")
            results[step_name] = {"status": "skip", "time": 0}
            continue

        # 确定输入输出
        if step_name == "merge":
            # 第一步：使用原始输入
            current_input = None
            current_output = intermediate_dir / "merged.parquet"
            step_args = {
                "excel": args.get("excel"),
                "dmp": args.get("dmp"),
                "output": current_output,
            }
        elif step_name == "profile":
            # 探查步骤：使用上一步的输出
            current_input = intermediate_dir / "merged.parquet"
            step_args = {
                "input": current_input,
                "output": intermediate_dir.parent / "reports" / "profile.md",
            }
        elif step_name == "clean":
            current_input = intermediate_dir / "merged.parquet"
            current_output = intermediate_dir / "cleaned.parquet"
            step_args = {
                "input": current_input,
                "output": current_output,
            }
        elif step_name == "desensitize":
            current_input = intermediate_dir / "cleaned.parquet"
            current_output = intermediate_dir / "desensitized.parquet"
            step_args = {
                "input": current_input,
                "output": current_output,
            }
        elif step_name == "split":
            current_input = intermediate_dir / "desensitized.parquet"
            # 检查是否跳过了脱敏
            if "desensitize" in skip_steps:
                current_input = intermediate_dir / "cleaned.parquet"
            step_args = {
                "input": current_input,
                "output": output_prefix,
            }

        # 执行步骤
        step_extra = extra_args.get(step_name, [])
        results[step_name] = run_step(step, step_args, step_extra)

        # 如果失败，停止管道
        if results[step_name]["status"] == "error":
            break

    total_time = time.time() - total_start

    # 打印摘要
    print_summary(results, total_time)

    return {
        "steps": results,
        "total_time": total_time,
        "success": all(
            r["status"] in ["success", "skip"]
            for r in results.values()
        ),
    }


def run_single_step(
    step_name: str,
    args: dict,
    extra_args: List[str] = None,
) -> dict:
    """
    运行单个步骤

    Args:
        step_name: 步骤名称
        args: 参数字典
        extra_args: 额外的命令行参数

    Returns:
        执行结果
    """
    # 查找步骤定义
    step = None
    for s in PIPELINE_STEPS:
        if s["name"] == step_name:
            step = s
            break

    if not step:
        print(f"❌ 未知的步骤: {step_name}")
        return {"status": "error", "error": f"未知的步骤: {step_name}"}

    return run_step(step, args, extra_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据管道统一运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整管道
    uv run python scripts/pipeline/run_pipeline.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/final

    # 跳过脱敏步骤
    uv run python scripts/pipeline/run_pipeline.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/final \\
        --skip desensitize

    # 仅运行清洗步骤
    uv run python scripts/pipeline/run_pipeline.py \\
        --step clean \\
        --input ./data/merged.parquet \\
        --output ./data/cleaned.parquet

    # 传递额外参数
    uv run python scripts/pipeline/run_pipeline.py \\
        --step split \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        -- --mode oot --time-column 线索创建时间
        """,
    )

    # 管道级参数
    parser.add_argument(
        "--excel", "-e",
        help="Excel 文件路径（线索宽表）"
    )
    parser.add_argument(
        "--dmp", "-d",
        help="DMP 行为数据文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/final",
        help="输出文件前缀（默认: data/final）"
    )

    # 单步执行参数
    parser.add_argument(
        "--step", "-s",
        choices=[s["name"] for s in PIPELINE_STEPS],
        help="仅运行指定步骤"
    )
    parser.add_argument(
        "--input", "-i",
        help="输入文件路径（单步执行时使用）"
    )

    # 跳过步骤
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[s["name"] for s in PIPELINE_STEPS if s.get("optional")],
        help="跳过指定步骤"
    )

    # 解析参数
    args, extra = parser.parse_known_args()

    print("=" * 60)
    print("数据管道统一运行器")
    print("=" * 60)

    try:
        if args.step:
            # 单步执行
            if not args.input:
                print("❌ 单步执行需要 --input 参数")
                return 1

            step_args = {
                "input": args.input,
                "output": args.output,
            }

            # 对于 merge 步骤，需要额外参数
            if args.step == "merge":
                if not args.excel or not args.dmp:
                    print("❌ merge 步骤需要 --excel 和 --dmp 参数")
                    return 1
                step_args["excel"] = args.excel
                step_args["dmp"] = args.dmp

            result = run_single_step(args.step, step_args, extra)

        else:
            # 完整管道
            if not args.excel or not args.dmp:
                print("❌ 完整管道需要 --excel 和 --dmp 参数")
                return 1

            pipeline_args = {
                "excel": args.excel,
                "dmp": args.dmp,
                "output": args.output,
            }

            result = run_full_pipeline(pipeline_args, args.skip)

        # 返回状态
        if args.step:
            return 0 if result["status"] == "success" else 1
        else:
            return 0 if result.get("success") else 1

    except Exception as e:
        print(f"❌ 管道执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
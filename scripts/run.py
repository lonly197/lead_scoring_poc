#!/usr/bin/env python3
"""
训练任务启动脚本

支持前台运行和后台运行模式。
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_timestamp


TASK_SCRIPT_ALIASES = {
    "train_arrive_oot": "train_arrive",
    "train_ohab_oot": "train_ohab",
}


def run_background(script_path: str, args: list[str], log_dir: str = "./outputs/logs") -> int:
    """
    后台运行训练脚本

    Args:
        script_path: 脚本路径
        args: 命令行参数
        log_dir: 日志目录

    Returns:
        进程 ID
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    timestamp = get_timestamp()
    task_name = Path(script_path).stem
    log_file = log_dir / f"{task_name}_{timestamp}.log"

    # 构建命令（传递日志文件路径给子进程）
    cmd = [sys.executable, script_path] + args + ["--log-file", str(log_file)]
    cmd_str = " ".join(cmd)

    print(f"启动后台任务: {task_name}")
    print(f"日志文件: {log_file}")
    print(f"命令: {cmd_str}")

    # 启动进程
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"启动时间: {datetime.now().isoformat()}\n")
        f.write(f"命令: {cmd_str}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print(f"进程 ID: {process.pid}")
    print(f"\n查看状态: uv run python scripts/monitor.py status")
    print(f"查看日志: uv run python scripts/monitor.py log {task_name}")
    print(f"持续跟踪: tail -f {log_file}")

    return process.pid


def run_foreground(script_path: str, args: list[str]) -> int:
    """
    前台运行训练脚本

    Args:
        script_path: 脚本路径
        args: 命令行参数

    Returns:
        退出码
    """
    cmd = [sys.executable, script_path] + args
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(
        description="训练任务启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 前台运行到店预测训练
  uv run python scripts/run.py train_arrive

  # 后台运行到店预测训练
  uv run python scripts/run.py train_arrive --daemon

  # 指定参数后台运行
  uv run python scripts/run.py train_arrive --daemon --preset best_quality --time-limit 7200

  # 传递交叉验证折数
  uv run python scripts/run.py train_ohab --daemon --num-bag-folds 5

  # 运行试驾预测
  uv run python scripts/run.py train_test_drive --daemon

可用任务:
  train_arrive       - 到店预测训练（核心任务）
  train_test_drive   - 试驾预测训练
  train_ohab         - OHAB 评级训练
  train_arrive_oot   - 兼容旧名称，内部转发到 train_arrive
  train_ohab_oot     - 兼容旧名称，内部转发到 train_ohab
        """,
    )

    parser.add_argument(
        "task",
        choices=["train_arrive", "train_test_drive", "train_ohab",
                  "train_arrive_oot", "train_ohab_oot"],
        help="任务名称",
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="后台运行模式",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="数据文件路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="目标变量名",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon 预设",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        help="训练时间限制（秒）",
    )
    parser.add_argument(
        "--num-bag-folds",
        type=int,
        help="交叉验证折数（0 表示禁用）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="测试集比例",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        help="训练集截止日期（OOT验证专用）",
    )
    parser.add_argument(
        "--valid-end",
        type=str,
        help="验证集截止日期（OOT验证专用）",
    )

    args = parser.parse_args()

    resolved_task = TASK_SCRIPT_ALIASES.get(args.task, args.task)
    if resolved_task != args.task:
        print(f"提示: 任务 {args.task} 已统一到 {resolved_task}，自动转发。")

    # 确定脚本路径
    script_path = Path(__file__).parent / f"{resolved_task}.py"
    if not script_path.exists():
        print(f"错误: 脚本不存在: {script_path}")
        sys.exit(1)

    # 构建参数
    pass_args = []
    if args.data_path:
        pass_args.extend(["--data-path", args.data_path])
    if args.target:
        pass_args.extend(["--target", args.target])
    if args.preset:
        pass_args.extend(["--preset", args.preset])
    if args.time_limit:
        pass_args.extend(["--time-limit", str(args.time_limit)])
    if args.num_bag_folds is not None:
        pass_args.extend(["--num-bag-folds", str(args.num_bag_folds)])
    if args.test_size:
        pass_args.extend(["--test-size", str(args.test_size)])
    if args.output_dir:
        pass_args.extend(["--output-dir", args.output_dir])
    if args.train_end:
        pass_args.extend(["--train-end", args.train_end])
    if args.valid_end:
        pass_args.extend(["--valid-end", args.valid_end])

    # 运行
    if args.daemon:
        pid = run_background(str(script_path), pass_args)
        print(f"\n✅ 后台任务已启动 (PID: {pid})")
    else:
        exit_code = run_foreground(str(script_path), pass_args)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

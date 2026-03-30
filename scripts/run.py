#!/usr/bin/env python3
"""
训练与评估统一入口

作为一级入口，支持：
1. train - 模型训练（转发到 train_model.py）
2. validate - 模型验证（转发到 validate_model.py）
3. monitor - 任务监控（转发到 monitor.py）

支持：
- 前台/后台运行模式
- 参数透传给底层脚本
- 子命令模式
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_timestamp, get_local_now, format_timestamp


def run_background(script_path: str, args: list[str], log_dir: str = "./outputs/logs") -> int:
    """
    后台运行脚本

    Args:
        script_path: 脚本路径
        args: 命令行参数
        log_dir: 日志目录

    Returns:
        进程 ID
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()
    task_name = Path(script_path).stem
    log_file = log_dir / f"{task_name}_{timestamp}.log"

    cmd = [sys.executable, script_path] + args + ["--log-file", str(log_file)]
    cmd_str = " ".join(cmd)

    print(f"启动后台任务: {task_name}")
    print(f"日志文件: {log_file}")
    print(f"命令: {cmd_str}")

    start_time = get_local_now()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"启动时间: {format_timestamp(start_time)}\n")
        f.write(f"命令: {cmd_str}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "LEAD_SCORING_DISABLE_CONSOLE_LOG": "1"},
        )

    print(f"进程 ID: {process.pid}")
    print(f"\n查看状态: uv run python scripts/run.py monitor status")
    print(f"查看日志: uv run python scripts/run.py monitor log {task_name}")
    print(f"持续跟踪: tail -f {log_file}")

    return process.pid


def run_foreground(script_path: str, args: list[str]) -> int:
    """
    前台运行脚本

    Args:
        script_path: 脚本路径
        args: 命令行参数

    Returns:
        退出码
    """
    cmd = [sys.executable, script_path] + args
    return subprocess.run(cmd).returncode


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="训练与评估统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  train <task>     - 模型训练
  validate         - 模型验证
  monitor          - 任务监控

训练任务 (train 子命令):
  arrive      - 到店预测训练
  test_drive  - 试驾预测训练
  ohab        - OHAB 评级训练
  ensemble    - 三模型集成训练

监控命令 (monitor 子命令):
  status      - 查看运行中的任务
  list        - 列出所有任务（包括已完成）
  log <task>  - 查看任务日志
  detail <task> - 查看任务详情
  stop <task> - 停止任务

示例:
  # 训练试驾预测模型（前台）
  uv run python scripts/run.py train test_drive

  # 训练试驾预测模型（后台）
  uv run python scripts/run.py train test_drive --daemon

  # 仅训练 CatBoost（推荐）
  uv run python scripts/run.py train test_drive --daemon \\
      --included-model-types CAT

  # 训练三模型集成
  uv run python scripts/run.py train ensemble --daemon

  # 验证模型
  uv run python scripts/run.py validate \\
      --model-path outputs/models/test_drive_model

  # 查看运行状态
  uv run python scripts/run.py monitor status

  # 查看任务日志
  uv run python scripts/run.py monitor log train_test_drive

  # 持续跟踪日志
  uv run python scripts/run.py monitor log train_test_drive -f

  # 停止所有任务
  uv run python scripts/run.py monitor stop --all

旧版兼容:
  # 直接指定任务名（自动识别为 train 子命令）
  uv run python scripts/run.py train_test_drive --daemon

  # 内存优化模式
  uv run python scripts/run.py train ohab --daemon \\
      --preset good_quality \\
      --exclude-memory-heavy-models \\
      --max-memory-ratio 0.7
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # train 子命令
    train_parser = subparsers.add_parser("train", help="模型训练")
    train_parser.add_argument(
        "task",
        nargs="?",
        default="test_drive",
        help="训练任务: arrive, test_drive, ohab, ensemble",
    )
    train_parser.add_argument("--daemon", "-d", action="store_true", help="后台运行")
    train_parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行训练三个模型（仅 ensemble 任务，需大内存服务器 32GB+）",
    )
    train_parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="并行训练的最大进程数（仅 ensemble 任务，默认 3）",
    )
    train_parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="训练集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    train_parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )

    # validate 子命令
    validate_parser = subparsers.add_parser("validate", help="模型验证")
    validate_parser.add_argument("--daemon", "-d", action="store_true", help="后台运行")
    validate_parser.add_argument(
        "--model-type",
        choices=["ohab", "arrive", "test_drive"],
        default=None,
        help="显式指定模型类型",
    )
    validate_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径",
    )
    validate_parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )

    # monitor 子命令
    monitor_parser = subparsers.add_parser("monitor", help="任务监控")
    monitor_parser.add_argument(
        "monitor_cmd",
        nargs="?",
        default="status",
        help="监控命令: status, list, log, detail, stop",
    )
    monitor_parser.add_argument("monitor_arg", nargs="?", help="任务名称（log/detail/stop 命令）")
    monitor_parser.add_argument("--pid", type=int, help="进程 ID")
    monitor_parser.add_argument("--lines", type=int, default=50, help="日志显示行数")
    monitor_parser.add_argument("--follow", "-f", action="store_true", help="持续跟踪日志")
    monitor_parser.add_argument("--all", "-a", action="store_true", help="停止所有任务")

    # 旧版兼容：直接任务名
    parser.add_argument(
        "legacy_task",
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,  # 隐藏帮助
    )
    parser.add_argument("--daemon", "-d", action="store_true", dest="legacy_daemon", help=argparse.SUPPRESS)

    return parser


def main() -> int:
    parser = build_parser()
    args, pass_args = parser.parse_known_args()

    scripts_dir = Path(__file__).parent

    # 处理命令
    if args.command == "train":
        # train 子命令
        train_script = scripts_dir / "train_model.py"
        task = getattr(args, "task", "test_drive")
        daemon = args.daemon

        # 过滤参数
        forwarded_args = [a for a in pass_args if a not in {"--daemon", "-d", "--parallel", "--max-workers", "--train-path", "--test-path"}]
        forwarded_args = [task] + forwarded_args
        if daemon:
            forwarded_args.append("--daemon")

        # 传递并行训练参数（仅 ensemble 任务）
        if getattr(args, "parallel", False):
            forwarded_args.append("--parallel")
        if getattr(args, "max_workers", None):
            forwarded_args.extend(["--max-workers", str(args.max_workers)])

        # 传递数据路径参数
        if getattr(args, "train_path", None):
            forwarded_args.extend(["--train-path", args.train_path])
        if getattr(args, "test_path", None):
            forwarded_args.extend(["--test-path", args.test_path])

        return run_foreground(str(train_script), forwarded_args)

    elif args.command == "validate":
        # validate 子命令
        validate_script = scripts_dir / "validate_model.py"

        # 构建参数
        forwarded_args = list(pass_args)
        if getattr(args, "model_type"):
            forwarded_args.extend(["--model-type", args.model_type])
        if getattr(args, "model_path"):
            forwarded_args.extend(["--model-path", args.model_path])
        if getattr(args, "test_path"):
            forwarded_args.extend(["--test-path", args.test_path])
        if args.daemon:
            forwarded_args.append("--daemon")

        return run_foreground(str(validate_script), forwarded_args)

    elif args.command == "monitor":
        # monitor 子命令
        monitor_script = scripts_dir / "monitor.py"

        monitor_cmd = getattr(args, "monitor_cmd", "status")
        forwarded_args = [monitor_cmd]

        # 处理各监控命令的参数
        if monitor_cmd == "log":
            if getattr(args, "monitor_arg"):
                forwarded_args.append(args.monitor_arg)
            if getattr(args, "lines"):
                forwarded_args.extend(["--lines", str(args.lines)])
            if getattr(args, "follow"):
                forwarded_args.append("-f")

        elif monitor_cmd == "detail":
            if getattr(args, "monitor_arg"):
                forwarded_args.append(args.monitor_arg)
            if getattr(args, "pid"):
                forwarded_args.extend(["--pid", str(args.pid)])

        elif monitor_cmd == "stop":
            if getattr(args, "all"):
                forwarded_args.append("--all")
            elif getattr(args, "monitor_arg"):
                forwarded_args.append(args.monitor_arg)
            if getattr(args, "pid"):
                forwarded_args.extend(["--pid", str(args.pid)])

        return run_foreground(str(monitor_script), forwarded_args)

    elif args.legacy_task:
        # 旧版兼容：直接任务名
        legacy_tasks = {
            "train_arrive": "arrive",
            "train_test_drive": "test_drive",
            "train_ohab": "ohab",
            "train_arrive_oot": "arrive",
            "train_ohab_oot": "ohab",
            "train_ensemble": "ensemble",
        }

        if args.legacy_task in legacy_tasks:
            train_script = scripts_dir / "train_model.py"
            forwarded_args = [legacy_tasks[args.legacy_task]] + list(pass_args)
            if args.legacy_daemon:
                forwarded_args.append("--daemon")
            print(f"[兼容模式] 任务 {args.legacy_task} -> train {legacy_tasks[args.legacy_task]}")
            return run_foreground(str(train_script), forwarded_args)
        else:
            print(f"错误: 未知任务 '{args.legacy_task}'")
            print("可用任务: train_arrive, train_test_drive, train_ohab, train_ensemble")
            return 1

    else:
        # 默认显示帮助
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
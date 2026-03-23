#!/usr/bin/env python3
"""
训练任务监控脚本

查看训练任务的运行状态、日志和进程信息。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    format_duration,
    get_process_info,
    list_running_processes,
    load_json,
    stop_process,
)


def print_status():
    """打印所有运行中的任务状态"""
    processes = list_running_processes()

    if not processes:
        print("当前没有运行中的训练任务")
        return

    print(f"\n{'=' * 80}")
    print(f"运行中的训练任务 ({len(processes)} 个)")
    print("=" * 80)

    for i, info in enumerate(processes, 1):
        duration = format_duration(info["start_time"])
        print(f"\n[{i}] {info['task_name']}")
        print(f"    进程 ID: {info['pid']}")
        print(f"    运行时间: {duration}")
        print(f"    目标变量: {info.get('target', 'N/A')}")
        print(f"    预设: {info.get('preset', 'N/A')}")
        print(f"    日志文件: {info['log_file']}")
        print(f"    启动时间: {info['start_time']}")


def print_detail(task_name: str, pid: int | None = None):
    """打印任务详情"""
    info = get_process_info(task_name, pid)

    if not info:
        print(f"未找到任务: {task_name}")
        return

    print(f"\n{'=' * 80}")
    print(f"任务详情: {info['task_name']}")
    print("=" * 80)

    status_emoji = {
        "running": "🟢",
        "completed": "✅",
        "failed": "❌",
        "stopped": "⏹️",
        "terminated": "⚠️",
    }.get(info.get("status", "unknown"), "❓")

    print(f"\n状态: {status_emoji} {info.get('status', 'unknown')}")

    if info.get("start_time"):
        duration = format_duration(
            info["start_time"], info.get("end_time")
        )
        print(f"运行时间: {duration}")

    print(f"\n进程 ID: {info['pid']}")
    print(f"启动时间: {info['start_time']}")
    if info.get("end_time"):
        print(f"结束时间: {info['end_time']}")

    print(f"\n命令:")
    print(f"  {info['command']}")

    print(f"\n配置:")
    print(f"  数据路径: {info.get('data_path', 'N/A')}")
    print(f"  目标变量: {info.get('target', 'N/A')}")
    print(f"  预设: {info.get('preset', 'N/A')}")
    print(f"  输出目录: {info.get('output_dir', 'N/A')}")

    print(f"\n日志文件: {info['log_file']}")

    if info.get("error"):
        print(f"\n错误信息: {info['error']}")


def tail_log(log_file: str, lines: int = 50):
    """查看日志尾部"""
    path = Path(log_file)
    if not path.exists():
        print(f"日志文件不存在: {log_file}")
        print("\n可能原因:")
        print("  1. 任务尚未开始写入日志")
        print("  2. 日志文件已被清理 (如执行了 rm -rf outputs/)")
        print("\n建议:")
        print("  1. 查看所有任务: uv run python scripts/monitor.py list")
        print("  2. 重新启动训练任务")
        return

    print(f"\n{'=' * 80}")
    print(f"日志内容 (最后 {lines} 行): {log_file}")
    print("=" * 80 + "\n")

    with open(path, encoding="utf-8") as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())


def main():
    parser = argparse.ArgumentParser(description="训练任务监控")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # status 命令
    subparsers.add_parser("status", help="查看运行中的任务")

    # detail 命令
    detail_parser = subparsers.add_parser("detail", help="查看任务详情")
    detail_parser.add_argument("task_name", help="任务名称")
    detail_parser.add_argument("--pid", type=int, help="进程 ID")

    # log 命令
    log_parser = subparsers.add_parser("log", help="查看任务日志")
    log_parser.add_argument("task_name", help="任务名称")
    log_parser.add_argument("--pid", type=int, help="进程 ID")
    log_parser.add_argument("--lines", type=int, default=50, help="显示行数")
    log_parser.add_argument("--follow", "-f", action="store_true", help="持续跟踪日志")

    # stop 命令
    stop_parser = subparsers.add_parser("stop", help="停止任务")
    stop_parser.add_argument("task_name", nargs="?", help="任务名称")
    stop_parser.add_argument("--pid", type=int, help="进程 ID")
    stop_parser.add_argument("--all", "-a", action="store_true", help="停止所有运行中的任务")

    # list 命令
    subparsers.add_parser("list", help="列出所有任务（包括已完成）")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        print_status()

    elif args.command == "detail":
        print_detail(args.task_name, args.pid)

    elif args.command == "log":
        info = get_process_info(args.task_name, args.pid)
        if info:
            if args.follow:
                import subprocess

                subprocess.run(["tail", "-f", info["log_file"]])
            else:
                tail_log(info["log_file"], args.lines)
        else:
            print(f"未找到任务: {args.task_name}")

    elif args.command == "stop":
        if args.all:
            # 停止所有运行中的任务
            processes = list_running_processes()
            if not processes:
                print("没有运行中的任务")
            else:
                stopped_count = 0
                for info in processes:
                    if stop_process(info["task_name"], info["pid"]):
                        print(f"已停止: {info['task_name']} (PID: {info['pid']})")
                        stopped_count += 1
                print(f"\n共停止 {stopped_count} 个任务")
        elif args.task_name:
            if stop_process(args.task_name, args.pid):
                print(f"已发送停止信号: {args.task_name}")
            else:
                print(f"停止失败或任务不存在: {args.task_name}")
        else:
            print("请指定任务名称或使用 --all 停止所有任务")

    elif args.command == "list":
        # 列出所有任务
        process_dir = Path("./outputs/.process")
        if not process_dir.exists():
            print("没有找到任何任务记录")
            return

        files = sorted(process_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not files:
            print("没有找到任何任务记录")
            return

        print(f"\n{'=' * 80}")
        print(f"所有任务记录 ({len(files)} 个)")
        print("=" * 80)

        for f in files[:20]:  # 只显示最近 20 个
            info = load_json(str(f))
            status_emoji = {
                "running": "🟢",
                "completed": "✅",
                "failed": "❌",
                "stopped": "⏹️",
                "terminated": "⚠️",
            }.get(info.get("status", "unknown"), "❓")

            duration = format_duration(
                info["start_time"], info.get("end_time")
            )

            print(f"\n{status_emoji} {info['task_name']} (PID: {info['pid']})")
            print(f"   状态: {info.get('status', 'unknown')}, 运行时间: {duration}")


if __name__ == "__main__":
    main()
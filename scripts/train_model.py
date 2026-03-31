#!/usr/bin/env python3
"""
模型训练路由入口

负责识别训练任务类型并转发到专属训练脚本：
- train_arrive.py - 到店预测训练
- train_test_drive.py - 试驾预测训练
- train_ohab.py - OHAB 评级训练
- train_test_drive_ensemble.py - 三模型集成训练

支持：
1. 前台/后台运行模式
2. 参数透传给底层脚本
3. 自动识别模型类型或显式指定
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import format_timestamp, get_local_now, get_timestamp


# 训练脚本映射
TRAIN_SCRIPTS = {
    "arrive": "train_arrive.py",
    "test_drive": "train_test_drive.py",
    "ohab": "train_ohab.py",
    "ensemble": "train_test_drive_ensemble.py",
    "order_after_drive": "train_order_after_drive.py",
}

# 任务别名
TASK_ALIASES = {
    "train_arrive": "arrive",
    "train_test_drive": "test_drive",
    "train_ohab": "ohab",
    "train_ensemble": "ensemble",
    "train_order_after_drive": "order_after_drive",
    "arrive_oot": "arrive",
    "ohab_oot": "ohab",
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
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()
    task_name = Path(script_path).stem
    log_file = log_dir_path / f"{task_name}_{timestamp}.log"

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
    print("\n查看状态: uv run python scripts/monitor.py status")
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


def resolve_script(task: str) -> str:
    """
    解析任务名称到脚本路径

    Args:
        task: 任务名称

    Returns:
        脚本路径
    """
    # 处理别名
    resolved_task = TASK_ALIASES.get(task, task)

    script_name = TRAIN_SCRIPTS.get(resolved_task)
    if script_name is None:
        raise ValueError(f"不支持的训练任务: {task}")

    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"训练脚本不存在: {script_path}")

    return str(script_path)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="模型训练路由入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用任务:
  arrive            - 到店预测训练
  test_drive        - 试驾预测训练
  ohab              - OHAB 评级训练
  ensemble          - 三模型集成训练（7/14/21天试驾预测）
  order_after_drive - 三模型下订预测训练（试驾后下订商谈阶段）

别名支持:
  train_arrive, arrive_oot -> arrive
  train_test_drive -> test_drive
  train_ohab, ohab_oot -> ohab
  train_ensemble -> ensemble
  train_order_after_drive -> order_after_drive

示例:
  # 前台运行试驾预测训练
  uv run python scripts/train_model.py test_drive

  # 后台运行试驾预测训练
  uv run python scripts/train_model.py test_drive --daemon

  # 指定参数训练
  uv run python scripts/train_model.py test_drive --daemon \\
      --preset high_quality --time-limit 7200

  # 仅训练 CatBoost（推荐，性能最佳）
  uv run python scripts/train_model.py test_drive --daemon \\
      --included-model-types CAT

  # 三模型集成训练
  uv run python scripts/train_model.py ensemble --daemon

  # 内存优化模式
  uv run python scripts/train_model.py ohab --daemon \\
      --preset good_quality \\
      --exclude-memory-heavy-models \\
      --max-memory-ratio 0.7 \\
      --num-folds-parallel 2
        """,
    )

    parser.add_argument(
        "task",
        choices=list(TRAIN_SCRIPTS.keys()) + list(TASK_ALIASES.keys()),
        help="训练任务名称",
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="后台运行模式",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行训练三个模型（仅 ensemble 任务，需大内存服务器 32GB+）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="并行训练的最大进程数（仅 ensemble 任务，默认 3）",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="训练集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args, pass_args = parser.parse_known_args()

    # 过滤 daemon、并行训练和数据路径参数
    forwarded_args = [arg for arg in pass_args if arg not in {"--daemon", "-d", "--parallel", "--max-workers", "--train-path", "--test-path"}]

    # 传递并行训练参数（仅 ensemble 任务有效）
    if getattr(args, "parallel", False):
        forwarded_args.append("--parallel")
    if getattr(args, "max_workers", None) and args.max_workers != 3:
        forwarded_args.extend(["--max-workers", str(args.max_workers)])

    # 传递数据路径参数
    if getattr(args, "train_path", None):
        forwarded_args.extend(["--train-path", args.train_path])
    if getattr(args, "test_path", None):
        forwarded_args.extend(["--test-path", args.test_path])

    # 解析脚本路径
    try:
        script_path = resolve_script(args.task)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        return 1

    task_name = Path(script_path).stem
    print(f"训练任务: {task_name}")

    # 运行
    if args.daemon:
        pid = run_background(script_path, forwarded_args)
        print(f"\n✅ 后台任务已启动 (PID: {pid})")
        return 0
    else:
        return run_foreground(script_path, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
模型验证统一入口

负责识别模型类型并转发到专属验证脚本：
- validate_ohab_model.py
- validate_arrive_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import format_timestamp, get_local_now, get_timestamp


VALIDATOR_SCRIPTS = {
    "ohab": "validate_ohab_model.py",
    "arrive": "validate_arrive_model.py",
}


def load_feature_metadata(model_path: Path) -> dict:
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def infer_model_type_from_path(model_path: Path) -> str | None:
    model_name = model_path.name.lower()
    if "arrive" in model_name:
        return "arrive"
    if "ohab" in model_name or "hab" in model_name:
        return "ohab"
    return None


def infer_model_type_from_metadata(model_path: Path) -> str | None:
    metadata = load_feature_metadata(model_path)
    pipeline_metadata = metadata.get("pipeline_metadata", {})
    if pipeline_metadata.get("pipeline_mode"):
        return "ohab"

    label = str(metadata.get("label", ""))
    if label == "到店标签_14天":
        return "arrive"
    if label in {"线索评级结果", "线索评级_试驾前"}:
        return "ohab"
    return None


def resolve_validator_script(args: argparse.Namespace) -> str:
    model_type = getattr(args, "model_type", None)
    if model_type:
        resolved_type = model_type
    else:
        model_path = Path(getattr(args, "model_path", "outputs/models/ohab_model"))
        resolved_type = infer_model_type_from_path(model_path)
        if resolved_type is None and model_path.exists():
            resolved_type = infer_model_type_from_metadata(model_path)
        if resolved_type is None:
            resolved_type = "ohab"

    script_name = VALIDATOR_SCRIPTS.get(resolved_type)
    if script_name is None:
        raise ValueError(f"不支持的模型类型: {resolved_type}")

    script_path = Path(__file__).parent / script_name
    return str(script_path)


def run_background(script_path: str, args: list[str], log_dir: str = "./outputs/logs") -> int:
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
    cmd = [sys.executable, script_path] + args
    return subprocess.run(cmd).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型验证统一入口")
    parser.add_argument("--daemon", "-d", action="store_true", help="后台运行模式")
    parser.add_argument(
        "--model-type",
        choices=["ohab", "arrive"],
        default=None,
        help="显式指定模型类型；不传则尝试自动识别",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/ohab_model",
        help="模型路径，用于自动识别模型类型",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args, pass_args = parser.parse_known_args()
    raw_args = sys.argv[1:]

    # 过滤掉统一入口的专用参数，不传递给子脚本
    forwarded_args = []
    skip_next = False
    for arg in raw_args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--daemon", "-d"}:
            continue
        if arg == "--model-type":
            skip_next = True  # 跳过 --model-type 的值
            continue
        forwarded_args.append(arg)

    script_path = resolve_validator_script(args)
    if args.daemon:
        pid = run_background(script_path, forwarded_args)
        print(f"\n✅ 后台任务已启动 (PID: {pid})")
        return 0

    return run_foreground(script_path, forwarded_args if forwarded_args else pass_args)


if __name__ == "__main__":
    raise SystemExit(main())

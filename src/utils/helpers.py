"""
工具函数模块
"""

import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# 进程信息文件目录
PROCESS_DIR = Path("./outputs/.process")


def setup_logging(
    log_file: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志

    Args:
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的 Logger
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # 清除已有处理器（避免重复添加）
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    保存 JSON 文件

    Args:
        data: 数据字典
        path: 文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """
    加载 JSON 文件

    Args:
        path: 文件路径

    Returns:
        数据字典
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    获取当前时间戳字符串

    Returns:
        时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_local_now() -> datetime:
    """
    获取当前时区感知的本地时间

    Returns:
        带时区信息的 datetime 对象
    """
    return datetime.now().astimezone()


def format_timestamp(dt: datetime) -> str:
    """
    格式化带时区的时间戳

    Args:
        dt: datetime 对象

    Returns:
        格式化的时间字符串 (YYYY-MM-DD HH:MM:SS+TZ)
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")


def format_training_duration(seconds: float) -> str:
    """
    格式化训练耗时为易读的中文格式

    Args:
        seconds: 秒数

    Returns:
        格式化的耗时字符串 (如 "1小时23分45秒")
    """
    if seconds < 0:
        return "0秒"

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分")
    if secs > 0 or not parts:
        parts.append(f"{secs}秒")

    return "".join(parts)


def print_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """
    打印分隔线

    Args:
        title: 标题
        char: 分隔字符
        width: 宽度
    """
    if title:
        print(f"\n{char * 10} {title} {char * (width - len(title) - 12)}")
    else:
        print(char * width)


def format_number(num: float, decimals: int = 2) -> str:
    """
    格式化数字

    Args:
        num: 数字
        decimals: 小数位数

    Returns:
        格式化后的字符串
    """
    if isinstance(num, (int, float)):
        if abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.{decimals}f}M"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.{decimals}f}K"
        else:
            return f"{num:.{decimals}f}"
    return str(num)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查数据质量

    Args:
        df: 数据 DataFrame

    Returns:
        数据质量报告
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_values": {},
        "duplicates": df.duplicated().sum(),
        "constant_columns": [],
    }

    # 缺失值统计
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    for col in df.columns:
        if missing[col] > 0:
            report["missing_values"][col] = {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col]),
            }

    # 常量列检测
    for col in df.columns:
        if df[col].nunique() <= 1:
            report["constant_columns"].append(col)

    return report


# ==================== 进程管理功能 ====================


def save_process_info(
    task_name: str,
    pid: int,
    command: str,
    log_file: str,
    **kwargs: Any,
) -> Path:
    """
    保存进程信息到文件

    Args:
        task_name: 任务名称
        pid: 进程 ID
        command: 执行命令
        log_file: 日志文件路径
        **kwargs: 其他元数据

    Returns:
        进程信息文件路径
    """
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = get_local_now()
    info = {
        "task_name": task_name,
        "pid": pid,
        "command": command,
        "log_file": log_file,
        "start_time": start_time.isoformat(),
        "start_time_readable": format_timestamp(start_time),
        "status": "running",
        **kwargs,
    }

    info_path = PROCESS_DIR / f"{task_name}_{pid}.json"
    save_json(info, str(info_path))

    return info_path


def update_process_status(task_name: str, pid: int, status: str, **kwargs: Any) -> None:
    """
    更新进程状态

    Args:
        task_name: 任务名称
        pid: 进程 ID
        status: 新状态
        **kwargs: 其他要更新的字段
    """
    info_path = PROCESS_DIR / f"{task_name}_{pid}.json"
    if info_path.exists():
        info = load_json(str(info_path))
        info["status"] = status
        end_time = get_local_now()
        info["end_time"] = end_time.isoformat()
        info["end_time_readable"] = format_timestamp(end_time)
        info.update(kwargs)
        save_json(info, str(info_path))


def complete_process_if_running(task_name: str, pid: int, **kwargs: Any) -> None:
    """
    仅在进程仍处于 running 状态时，将其标记为 completed。

    用于 atexit 收口，避免失败/停止状态被错误覆盖。

    Args:
        task_name: 任务名称
        pid: 进程 ID
        **kwargs: 其他要更新的字段
    """
    info_path = PROCESS_DIR / f"{task_name}_{pid}.json"
    if not info_path.exists():
        return

    try:
        info = load_json(str(info_path))
    except Exception:
        return

    if info.get("status") == "running":
        update_process_status(task_name, pid, "completed", **kwargs)


def list_running_processes() -> list[Dict[str, Any]]:
    """
    列出所有运行中的进程

    Returns:
        进程信息列表
    """
    processes = []
    if not PROCESS_DIR.exists():
        return processes

    for info_file in PROCESS_DIR.glob("*.json"):
        try:
            info = load_json(str(info_file))
            # 检查进程是否仍在运行
            if info.get("status") == "running":
                try:
                    os.kill(info["pid"], 0)  # 检查进程是否存在
                    processes.append(info)
                except OSError:
                    # 进程已结束，更新状态
                    info["status"] = "terminated"
                    end_time = get_local_now()
                    info["end_time"] = end_time.isoformat()
                    info["end_time_readable"] = format_timestamp(end_time)
                    save_json(info, str(info_file))
        except Exception:
            continue

    return processes


def get_process_info(task_name: str, pid: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    获取指定任务/进程的信息

    Args:
        task_name: 任务名称
        pid: 进程 ID（可选）

    Returns:
        进程信息，不存在则返回 None（返回最新的记录）
    """
    if not PROCESS_DIR.exists():
        return None

    # 获取所有匹配的文件，按修改时间排序（最新的在前）
    matching_files = sorted(
        PROCESS_DIR.glob(f"{task_name}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for info_file in matching_files:
        try:
            info = load_json(str(info_file))
            if pid is None or info.get("pid") == pid:
                return info
        except Exception:
            continue

    return None


def stop_process(task_name: str, pid: Optional[int] = None) -> bool:
    """
    停止运行中的进程

    Args:
        task_name: 任务名称
        pid: 进程 ID（可选，不指定则停止该任务所有进程）

    Returns:
        是否成功停止
    """
    import time

    processes = list_running_processes()
    stopped = False

    for info in processes:
        if info["task_name"] == task_name:
            if pid is None or info["pid"] == pid:
                try:
                    os.kill(info["pid"], signal.SIGTERM)
                    stopped = True

                    # 等待进程终止
                    for _ in range(10):  # 最多等待 5 秒
                        try:
                            os.kill(info["pid"], 0)
                            time.sleep(0.5)
                        except OSError:
                            break

                    # 更新状态为 stopped（覆盖 atexit 设置的 completed）
                    update_process_status(info["task_name"], info["pid"], "stopped")

                except OSError:
                    pass

    return stopped


def format_duration(start_time: str, end_time: Optional[str] = None) -> str:
    """
    格式化持续时间

    Args:
        start_time: 开始时间（ISO 格式）
        end_time: 结束时间（可选）

    Returns:
        格式化的持续时间字符串
    """
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time) if end_time else datetime.now()
    delta = end - start

    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# ==================== 磁盘空间管理 ====================

# AutoGluon preset 磁盘需求估算（GB）
PRESET_DISK_REQUIREMENTS = {
    "medium_quality": 1.0,
    "good_quality": 2.0,
    "high_quality": 4.0,
    "best_quality": 8.0,
}


def check_disk_space(path: str, required_gb: float = 5.0) -> Dict[str, Any]:
    """
    检查磁盘空间是否充足

    Args:
        path: 检查路径
        required_gb: 所需空间（GB）

    Returns:
        磁盘空间信息字典
    """
    import shutil

    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)

    return {
        "path": path,
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "sufficient": free_gb >= required_gb,
        "utilization_pct": round(used_gb / total_gb * 100, 1),
    }


def get_preset_disk_requirement(preset: str) -> float:
    """
    获取指定 preset 的磁盘需求估算

    Args:
        preset: AutoGluon preset 名称

    Returns:
        估算的磁盘需求（GB）
    """
    return PRESET_DISK_REQUIREMENTS.get(preset, 3.0)


def suggest_preset_by_disk(available_gb: float) -> str:
    """
    根据可用磁盘空间推荐 preset

    Args:
        available_gb: 可用磁盘空间（GB）

    Returns:
        推荐的 preset 名称
    """
    if available_gb >= 6.0:
        return "high_quality"
    elif available_gb >= 3.0:
        return "good_quality"
    else:
        return "medium_quality"

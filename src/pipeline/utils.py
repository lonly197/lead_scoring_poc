"""
数据管道共享工具函数

提供数据加载、保存、状态打印等共享功能。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    import polars as pl


def load_data(
    path: Union[str, Path],
    engine: str = "polars",
    **kwargs,
) -> "pl.DataFrame":
    """
    自动检测格式加载数据文件

    支持格式：.parquet, .csv, .tsv, .xlsx

    Args:
        path: 数据文件路径
        engine: 加载引擎 ('polars' 或 'pandas')
        **kwargs: 传递给加载函数的额外参数

    Returns:
        DataFrame 对象
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()

    if engine == "polars":
        import polars as pl

        if suffix == ".parquet":
            return pl.read_parquet(path, **kwargs)
        elif suffix == ".csv":
            return pl.read_csv(path, **kwargs)
        elif suffix == ".tsv":
            return pl.read_csv(path, separator="\t", **kwargs)
        elif suffix in (".xlsx", ".xls"):
            return pl.read_excel(path, engine="calamine", **kwargs)
        else:
            # 尝试自动检测
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
            sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
            return pl.read_csv(path, separator=sep, **kwargs)

    else:  # pandas
        import pandas as pd

        if suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)
        elif suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif suffix == ".tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, **kwargs)
        else:
            return pd.read_csv(path, **kwargs)


def save_data(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    path: Union[str, Path],
    **kwargs,
) -> Path:
    """
    自动检测格式保存数据文件

    Args:
        df: DataFrame 对象
        path: 输出文件路径
        **kwargs: 传递给保存函数的额外参数

    Returns:
        保存的文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    # 检测 DataFrame 类型
    df_type = type(df).__module__

    if "polars" in df_type:
        import polars as pl

        if suffix == ".parquet":
            df.write_parquet(path, **kwargs)
        elif suffix == ".csv":
            df.write_csv(path, **kwargs)
        else:
            # 默认 parquet
            path = path.with_suffix(".parquet")
            df.write_parquet(path, **kwargs)

    elif "pandas" in df_type:
        if suffix == ".parquet":
            df.to_parquet(path, **kwargs)
        elif suffix == ".csv":
            df.to_csv(path, index=False, **kwargs)
        else:
            path = path.with_suffix(".parquet")
            df.to_parquet(path, **kwargs)

    else:
        raise TypeError(f"不支持的 DataFrame 类型: {type(df)}")

    return path


def print_step(
    name: str,
    status: str = "running",
    message: Optional[str] = None,
) -> None:
    """
    打印管道步骤状态

    Args:
        name: 步骤名称
        status: 状态 ('running', 'success', 'error', 'skip')
        message: 额外消息
    """
    status_icons = {
        "running": "⏳",
        "success": "✅",
        "error": "❌",
        "skip": "⏭️",
    }

    icon = status_icons.get(status, "•")
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"[{timestamp}] {icon} {name}", end="")

    if message:
        print(f": {message}")
    else:
        print()


def print_summary(
    steps: Dict[str, Dict[str, Any]],
    total_time: Optional[float] = None,
) -> None:
    """
    打印管道执行摘要

    Args:
        steps: 步骤执行结果字典
        total_time: 总执行时间（秒）
    """
    print("\n" + "=" * 60)
    print("管道执行摘要")
    print("=" * 60)

    for name, info in steps.items():
        status = info.get("status", "unknown")
        time_s = info.get("time", 0)
        output = info.get("output", "")

        status_icons = {
            "success": "✅",
            "error": "❌",
            "skip": "⏭️",
        }
        icon = status_icons.get(status, "•")

        print(f"{icon} {name}: {status} ({time_s:.1f}s)")
        if output:
            print(f"   → {output}")

    if total_time:
        print(f"\n总耗时: {total_time:.1f}s")

    print("=" * 60)


def get_default_output_path(
    step: str,
    input_path: Union[str, Path],
    suffix: str = ".parquet",
) -> Path:
    """
    根据步骤名称生成默认输出路径

    Args:
        step: 步骤名称 (merge, clean, desensitize)
        input_path: 输入文件路径
        suffix: 输出文件后缀

    Returns:
        默认输出路径
    """
    input_path = Path(input_path)

    # 输出到同目录
    parent = input_path.parent

    # 根据步骤生成文件名
    stem = input_path.stem
    if step == "merge":
        name = f"{stem}_merged"
    elif step == "clean":
        name = f"{stem}_cleaned"
    elif step == "desensitize":
        name = f"{stem}_desensitized"
    elif step == "split":
        name = f"{stem}"  # 会加 _train/_test 后缀
    else:
        name = stem

    return parent / f"{name}{suffix}"


def format_size(path: Union[str, Path]) -> str:
    """
    格式化文件大小

    Args:
        path: 文件路径

    Returns:
        格式化的大小字符串
    """
    path = Path(path)
    if not path.exists():
        return "N/A"

    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def check_file_exists(path: Union[str, Path], overwrite: bool = False) -> bool:
    """
    检查文件是否存在

    Args:
        path: 文件路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        是否可以继续操作
    """
    path = Path(path)
    if path.exists():
        if overwrite:
            print(f"⚠️ 文件已存在，将被覆盖: {path}")
            return True
        else:
            print(f"❌ 文件已存在: {path}")
            print("   使用 --overwrite 参数覆盖")
            return False
    return True
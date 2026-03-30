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

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    path = Path(path).resolve()  # 规范化路径，防止路径遍历

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

    Raises:
        TypeError: 不支持的 DataFrame 类型
    """
    path = Path(path).resolve()  # 规范化路径
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


def get_or_create_parquet_cache(
    source_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    engine: str = "openpyxl",
    force_refresh: bool = False,
    sheet_name: Optional[str] = None,
) -> Path:
    """
    获取或创建 Excel 的 Parquet 缓存

    对于大 Excel 文件，首次运行时转换为 Parquet 格式缓存。
    后续运行直接读取 Parquet 缓存，大幅降低内存占用和读取时间。

    Args:
        source_path: Excel 文件路径
        cache_dir: 缓存目录（默认 ./data/cache）
        engine: Excel 读取引擎（openpyxl/fastexcel/calamine）
        force_refresh: 是否强制刷新缓存
        sheet_name: 指定读取的 Sheet 名称（用于多 Sheet 缓存）

    Returns:
        Parquet 缓存文件路径

    Example:
        >>> cache_path = get_or_create_parquet_cache("./data/large.xlsx")
        >>> df = pl.read_parquet(cache_path)  # 内存友好
    """
    import hashlib

    source_path = Path(source_path).resolve()

    if cache_dir is None:
        cache_dir = Path("./data/cache")
    else:
        cache_dir = Path(cache_dir)
    cache_dir = cache_dir.resolve()

    # 计算缓存 key（基于文件路径 + 修改时间 + 大小 + Sheet名）
    file_stat = source_path.stat()
    key_parts = f"{source_path}:{file_stat.st_mtime}:{file_stat.st_size}"
    if sheet_name:
        key_parts += f":{sheet_name}"
    cache_key = hashlib.md5(key_parts.encode()).hexdigest()[:12]

    # 缓存文件名
    stem = source_path.stem
    if sheet_name:
        cache_name = f"{stem}_{sheet_name}_{cache_key}.parquet"
    else:
        cache_name = f"{stem}_{cache_key}.parquet"
    cache_path = cache_dir / cache_name

    # 检查缓存是否有效
    if cache_path.exists() and not force_refresh:
        cache_stat = cache_path.stat()
        # 缓存必须比源文件新
        if cache_stat.st_mtime > file_stat.st_mtime:
            print(f"  使用缓存: {cache_path} ({format_size(cache_path)})")
            return cache_path

    # 转换 Excel → Parquet
    print(f"  转换 Excel → Parquet 缓存...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    import polars as pl

    # 读取 Excel
    if sheet_name:
        df = pl.read_excel(str(source_path), sheet_name=sheet_name, engine=engine)
    else:
        df = pl.read_excel(str(source_path), engine=engine)

    # 写入 Parquet（使用 zstd 压缩，压缩率和速度平衡）
    df.write_parquet(cache_path, compression="zstd")

    # 释放内存
    del df

    print(f"  缓存已保存: {cache_path} ({format_size(cache_path)})")

    return cache_path


def clear_cache(
    cache_dir: Union[str, Path] = None,
    older_than_days: Optional[int] = None,
) -> int:
    """
    清理缓存目录

    Args:
        cache_dir: 缓存目录（默认 ./data/cache）
        older_than_days: 清理多少天前的缓存（None 表示全部清理）

    Returns:
        删除的文件数量
    """
    import time

    if cache_dir is None:
        cache_dir = Path("./data/cache")
    else:
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return 0

    deleted = 0
    cutoff_time = time.time() - (older_than_days * 86400) if older_than_days else 0

    for cache_file in cache_dir.glob("*.parquet"):
        if older_than_days is None or cache_file.stat().st_mtime < cutoff_time:
            cache_file.unlink()
            deleted += 1

    if deleted > 0:
        print(f"  已清理 {deleted} 个缓存文件")

    return deleted
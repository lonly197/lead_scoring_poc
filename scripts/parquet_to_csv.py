#!/usr/bin/env python3
"""
Parquet 转 CSV 脚本（DuckDB 优化版）

使用 DuckDB 将 Parquet 文件转换为 CSV 格式，性能优异。

用法:
  # 转换单个文件
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet

  # 指定输出路径
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet --output ./output/test.csv

  # 批量转换目录
  uv run python scripts/parquet_to_csv.py ./data --batch

  # 自定义分隔符（默认逗号）
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet --sep tab

  # 仅加载指定列
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet --columns "线索唯一ID,线索创建时间,线索评级结果"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def format_size(mb: float) -> str:
    """格式化文件大小"""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"


def convert_parquet_to_csv(
    input_path: Path,
    output_path: Optional[Path] = None,
    sep: str = ",",
    columns: Optional[List[str]] = None,
    memory_limit: str = "4GB",
    threads: int = 4,
    dry_run: bool = False,
) -> dict:
    """
    使用 DuckDB 将 Parquet 转换为 CSV

    Args:
        input_path: 输入 Parquet 文件路径
        output_path: 输出 CSV 文件路径（默认同名 .csv）
        sep: 分隔符（comma/tab/semicolon 或直接指定字符）
        columns: 仅导出指定列（可选）
        memory_limit: DuckDB 内存限制
        threads: 线程数
        dry_run: 是否仅预览

    Returns:
        统计信息字典
    """
    import duckdb

    stats = {
        "input_path": str(input_path),
        "output_path": None,
        "input_size_mb": input_path.stat().st_size / 1024 / 1024,
        "output_size_mb": 0,
        "rows": 0,
        "columns": 0,
        "time_s": 0,
        "error": None,
    }

    # 确定输出路径
    if output_path is None:
        output_path = input_path.with_suffix(".csv")
    stats["output_path"] = str(output_path)

    # 解析分隔符
    sep_map = {
        "comma": ",",
        "tab": "\t",
        "semicolon": ";",
        "pipe": "|",
    }
    if sep.lower() in sep_map:
        delimiter = sep_map[sep.lower()]
    else:
        delimiter = sep

    log(f"输入文件: {input_path}")
    log(f"输出文件: {output_path}")
    log(f"输入大小: {format_size(stats['input_size_mb'])}")
    log(f"分隔符: {'Tab' if delimiter == chr(9) else repr(delimiter)}")

    if dry_run:
        log("[DRY-RUN] 仅预览，不实际执行")
        return stats

    start_time = time.time()

    try:
        # 创建 DuckDB 连接
        con = duckdb.connect(":memory:")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")

        # 创建源视图
        con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

        # 获取基本信息
        stats["rows"] = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        cols_info = con.execute("DESCRIBE source").fetchall()
        stats["columns"] = len(cols_info)

        log(f"数据量: {stats['rows']:,} 行, {stats['columns']} 列")

        # 构建 SELECT 表达式
        if columns:
            # 仅导出指定列
            col_list = [f'"{c.strip()}"' for c in columns]
            select_expr = ", ".join(col_list)
            log(f"仅导出列: {len(col_list)} 个")
        else:
            select_expr = "*"

        # 执行转换（DuckDB 直接输出 CSV）
        log("执行转换...")

        # 转义分隔符用于 COPY 命令
        delim_escaped = delimiter.replace("\t", "\\t")

        con.execute(f"""
            COPY (
                SELECT {select_expr} FROM source
            ) TO '{output_path}' (
                FORMAT CSV,
                DELIMITER '{delim_escaped}',
                HEADER true,
                QUOTE '"',
                ESCAPE '"'
            )
        """)

        con.close()

        # 统计结果
        stats["time_s"] = time.time() - start_time
        stats["output_size_mb"] = output_path.stat().st_size / 1024 / 1024

        log(f"✅ 转换完成")
        log(f"   输出大小: {format_size(stats['output_size_mb'])}")
        log(f"   耗时: {stats['time_s']:.2f}s")

        return stats

    except Exception as e:
        stats["error"] = str(e)
        log(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return stats


def find_parquet_files(directory: Path) -> List[Path]:
    """查找目录下的 Parquet 文件"""
    return sorted(directory.glob("*.parquet"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parquet 转 CSV 脚本（DuckDB 优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet

  # 批量转换目录
  uv run python scripts/parquet_to_csv.py ./data --batch

  # 使用 Tab 分隔符
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet --sep tab

  # 仅导出指定列
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet \
      --columns "线索唯一ID,线索创建时间,线索评级结果"

  # 预览模式
  uv run python scripts/parquet_to_csv.py ./data/final_test.parquet --dry-run
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="输入 Parquet 文件或目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 CSV 文件路径（仅单文件模式）",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量转换目录下所有 Parquet 文件",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="comma",
        help="分隔符: comma/tab/semicolon/pipe 或直接指定字符（默认: comma）",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="仅导出指定列（逗号分隔）",
    )
    parser.add_argument(
        "--memory-limit",
        default="4GB",
        help="DuckDB 内存限制（默认: 4GB）",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="线程数（默认: 4）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际执行转换",
    )

    args = parser.parse_args()
    input_path = Path(args.path)

    # 解析列名
    columns_arg = None
    if args.columns:
        columns_arg = [c.strip() for c in args.columns.split(",") if c.strip()]

    # 检查 DuckDB 是否安装
    try:
        import duckdb  # noqa: F401
    except ImportError:
        log("❌ 缺少 duckdb 依赖，请运行: uv add duckdb")
        return 1

    # 统计信息
    total_stats = {
        "total_files": 0,
        "success_count": 0,
        "failed_count": 0,
        "total_input_mb": 0,
        "total_output_mb": 0,
        "total_time_s": 0,
    }

    if args.batch:
        # 批量转换模式
        if not input_path.is_dir():
            log(f"❌ 批量模式需要指定目录: {input_path}")
            return 1

        files = find_parquet_files(input_path)
        if not files:
            log(f"⚠️ 未找到 Parquet 文件: {input_path}")
            return 0

        log(f"找到 {len(files)} 个文件待转换")
        total_stats["total_files"] = len(files)

        for file_path in files:
            log("=" * 60)
            stats = convert_parquet_to_csv(
                input_path=file_path,
                sep=args.sep,
                columns=columns_arg,
                memory_limit=args.memory_limit,
                threads=args.threads,
                dry_run=args.dry_run,
            )

            if stats.get("error") is None:
                total_stats["success_count"] += 1
                total_stats["total_input_mb"] += stats.get("input_size_mb", 0)
                total_stats["total_output_mb"] += stats.get("output_size_mb", 0)
                total_stats["total_time_s"] += stats.get("time_s", 0)
            else:
                total_stats["failed_count"] += 1

    else:
        # 单文件转换模式
        if not input_path.exists():
            log(f"❌ 文件不存在: {input_path}")
            return 1

        output_path = Path(args.output) if args.output else None

        stats = convert_parquet_to_csv(
            input_path=input_path,
            output_path=output_path,
            sep=args.sep,
            columns=columns_arg,
            memory_limit=args.memory_limit,
            threads=args.threads,
            dry_run=args.dry_run,
        )

        total_stats["total_files"] = 1
        if stats.get("error") is None:
            total_stats["success_count"] = 1
            total_stats["total_input_mb"] = stats.get("input_size_mb", 0)
            total_stats["total_output_mb"] = stats.get("output_size_mb", 0)
            total_stats["total_time_s"] = stats.get("time_s", 0)
        else:
            total_stats["failed_count"] = 1

    # 输出总结
    log("")
    log("=" * 60)
    log("转换完成!")
    log(f"  成功: {total_stats['success_count']}/{total_stats['total_files']}")
    if total_stats["failed_count"] > 0:
        log(f"  失败: {total_stats['failed_count']}")
    log(f"  输入总大小: {format_size(total_stats['total_input_mb'])}")
    log(f"  输出总大小: {format_size(total_stats['total_output_mb'])}")
    log(f"  总耗时: {total_stats['total_time_s']:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
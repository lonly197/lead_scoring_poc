#!/usr/bin/env python3
"""
数据格式转换脚本

将 CSV/TSV 数据文件转换为 Parquet 格式，提升加载速度和减少存储空间。

用法:
  # 转换单个文件
  uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv

  # 批量转换目录
  uv run python scripts/convert_to_parquet.py ./data --batch

  # 指定压缩算法
  uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv --compression gzip

  # 查看转换预览（不实际执行）
  uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 支持的输入格式
SUPPORTED_INPUT_FORMATS = {".csv", ".tsv", ".txt"}

# 默认压缩算法
DEFAULT_COMPRESSION = "snappy"

# 压缩算法选项
COMPRESSION_OPTIONS = ["snappy", "gzip", "brotli", "lz4", "zstd", "none"]


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）"""
    return path.stat().st_size / (1024 * 1024)


def format_size(mb: float) -> str:
    """格式化文件大小"""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"


def infer_separator(file_path: Path) -> str:
    """推断文件分隔符"""
    suffix = file_path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    elif suffix == ".csv":
        return ","
    else:
        # 尝试自动检测
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            if "\t" in first_line:
                return "\t"
            elif "," in first_line:
                return ","
            else:
                return "\t"  # 默认 Tab


def convert_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    compression: str = DEFAULT_COMPRESSION,
    dry_run: bool = False,
    sep: Optional[str] = None,
    header: Union[int, str, None] = "infer",
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> Tuple[bool, dict]:
    """
    转换单个文件为 Parquet 格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（默认同名 .parquet）
        compression: 压缩算法
        dry_run: 是否仅预览
        sep: 分隔符（默认自动推断）
        header: 表头行号，None 表示无表头，"infer" 表示自动推断
        columns: 列名列表（当 header=None 时使用）
        chunksize: 分块大小（大文件流式处理）

    Returns:
        (是否成功, 统计信息字典)
    """
    stats = {
        "input_path": str(input_path),
        "output_path": None,
        "input_size_mb": 0,
        "output_size_mb": 0,
        "compression_ratio": 0,
        "rows": 0,
        "columns": 0,
        "load_time_s": 0,
        "save_time_s": 0,
        "error": None,
    }

    # 检查输入文件
    if not input_path.exists():
        stats["error"] = f"输入文件不存在: {input_path}"
        return False, stats

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_FORMATS:
        stats["error"] = f"不支持的文件格式: {suffix}"
        return False, stats

    # 确定输出路径
    if output_path is None:
        output_path = input_path.with_suffix(".parquet")

    stats["output_path"] = str(output_path)
    stats["input_size_mb"] = get_file_size_mb(input_path)

    if dry_run:
        logger.info(f"[DRY-RUN] 将转换: {input_path} -> {output_path}")
        logger.info(f"[DRY-RUN] 输入大小: {format_size(stats['input_size_mb'])}")
        return True, stats

    try:
        # 加载数据
        logger.info(f"加载文件: {input_path}")
        # 确定分隔符
        if sep is None:
            sep = infer_separator(input_path)
        logger.info(f"使用分隔符: {'Tab' if sep == '\\t' else sep}")

        start_time = time.time()

        # 确定表头设置
        header_arg = header
        names_arg = columns

        if chunksize:
            # 流式处理大文件
            logger.info(f"流式处理模式，分块大小: {chunksize:,}")
            chunks = []
            for i, chunk in enumerate(
                pd.read_csv(
                    input_path,
                    sep=sep,
                    header=header_arg,
                    names=names_arg,
                    chunksize=chunksize,
                    low_memory=False,
                    encoding="utf-8",
                    on_bad_lines="warn",
                )
            ):
                chunks.append(chunk)
                if i % 10 == 0:
                    logger.info(f"已处理 {(i + 1) * chunksize:,} 行...")
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                input_path,
                sep=sep,
                header=header_arg,
                names=names_arg,
                low_memory=False,
                encoding="utf-8",
                on_bad_lines="warn",
            )

        stats["load_time_s"] = time.time() - start_time
        stats["rows"] = len(df)
        stats["columns"] = len(df.columns)

        logger.info(f"加载完成: {stats['rows']:,} 行, {stats['columns']} 列")
        logger.info(f"加载耗时: {stats['load_time_s']:.2f}s")

        # 处理压缩参数
        compression_arg = None if compression == "none" else compression

        # 保存为 Parquet
        logger.info(f"保存为 Parquet: {output_path}")
        start_time = time.time()

        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression=compression_arg,
            index=False,
        )

        stats["save_time_s"] = time.time() - start_time
        stats["output_size_mb"] = get_file_size_mb(output_path)

        # 计算压缩比
        if stats["output_size_mb"] > 0:
            stats["compression_ratio"] = stats["input_size_mb"] / stats["output_size_mb"]

        logger.info(f"保存完成: {format_size(stats['output_size_mb'])}")
        logger.info(f"保存耗时: {stats['save_time_s']:.2f}s")
        logger.info(
            f"压缩比: {stats['compression_ratio']:.2f}x "
            f"({stats['input_size_mb']:.1f}MB -> {stats['output_size_mb']:.1f}MB)"
        )

        return True, stats

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"转换失败: {e}")
        return False, stats


def find_convertible_files(directory: Path) -> List[Path]:
    """查找目录下可转换的文件"""
    files = []
    for ext in SUPPORTED_INPUT_FORMATS:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="将 CSV/TSV 数据文件转换为 Parquet 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv

  # 批量转换 data 目录下所有 CSV/TSV 文件
  uv run python scripts/convert_to_parquet.py ./data --batch

  # 使用 gzip 压缩（压缩率更高，速度较慢）
  uv run python scripts/convert_to_parquet.py ./data/202602~03.tsv --compression gzip

  # 预览转换（不实际执行）
  uv run python scripts/convert_to_parquet.py ./data --batch --dry-run

压缩算法对比:
  snappy  - 默认，速度快，压缩率中等（推荐）
  gzip    - 压缩率高，速度慢
  lz4     - 速度最快，压缩率较低
  zstd    - 平衡速度与压缩率
  none    - 不压缩
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="输入文件或目录路径",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量转换目录下所有 CSV/TSV 文件",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=DEFAULT_COMPRESSION,
        choices=COMPRESSION_OPTIONS,
        help=f"压缩算法（默认: {DEFAULT_COMPRESSION}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（仅单文件模式）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际执行转换",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=True,
        help="保留原始文件（默认保留）",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="转换成功后删除原始文件",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help="分隔符（默认自动推断，如 '\\t' 表示 Tab）",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="文件无表头，第一行即为数据",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="列名列表，逗号分隔或文件路径（每行一个列名）",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="流式处理分块大小（大文件推荐 100000）",
    )

    args = parser.parse_args()
    input_path = Path(args.path)

    # 处理分隔符参数
    sep_arg = args.sep
    if sep_arg == "\\t":
        sep_arg = "\t"

    # 处理表头参数
    header_arg = None if args.no_header else "infer"

    # 处理列名参数
    columns_arg = None
    if args.columns:
        if Path(args.columns).exists():
            # 从文件读取列名
            with open(args.columns, "r", encoding="utf-8") as f:
                columns_arg = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件加载列名: {len(columns_arg)} 个")
        else:
            # 直接解析逗号分隔的列名
            columns_arg = [c.strip() for c in args.columns.split(",") if c.strip()]
            logger.info(f"使用命令行列名: {len(columns_arg)} 个")

    # 检查 pyarrow 是否安装
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        logger.error("缺少 pyarrow 依赖，请运行: uv add pyarrow")
        sys.exit(1)

    # 统计信息
    total_stats = {
        "total_files": 0,
        "success_count": 0,
        "failed_count": 0,
        "total_input_mb": 0,
        "total_output_mb": 0,
        "total_load_time_s": 0,
        "total_save_time_s": 0,
    }

    if args.batch:
        # 批量转换模式
        if not input_path.is_dir():
            logger.error(f"批量模式需要指定目录: {input_path}")
            sys.exit(1)

        files = find_convertible_files(input_path)
        if not files:
            logger.warning(f"未找到可转换的文件: {input_path}")
            sys.exit(0)

        logger.info(f"找到 {len(files)} 个文件待转换")
        total_stats["total_files"] = len(files)

        for file_path in files:
            logger.info("=" * 60)
            success, stats = convert_file(
                input_path=file_path,
                compression=args.compression,
                dry_run=args.dry_run,
                sep=sep_arg,
                header=header_arg,
                columns=columns_arg,
                chunksize=args.chunksize,
            )

            if success:
                total_stats["success_count"] += 1
                total_stats["total_input_mb"] += stats["input_size_mb"]
                total_stats["total_output_mb"] += stats["output_size_mb"]
                total_stats["total_load_time_s"] += stats["load_time_s"]
                total_stats["total_save_time_s"] += stats["save_time_s"]

                # 删除原始文件
                if args.delete_original and not args.dry_run:
                    file_path.unlink()
                    logger.info(f"已删除原始文件: {file_path}")
            else:
                total_stats["failed_count"] += 1

    else:
        # 单文件转换模式
        output_path = Path(args.output) if args.output else None

        success, stats = convert_file(
            input_path=input_path,
            output_path=output_path,
            compression=args.compression,
            dry_run=args.dry_run,
            sep=sep_arg,
            header=header_arg,
            columns=columns_arg,
            chunksize=args.chunksize,
        )

        total_stats["total_files"] = 1
        if success:
            total_stats["success_count"] = 1
            total_stats["total_input_mb"] = stats["input_size_mb"]
            total_stats["total_output_mb"] = stats["output_size_mb"]
            total_stats["total_load_time_s"] = stats["load_time_s"]
            total_stats["total_save_time_s"] = stats["save_time_s"]

            # 删除原始文件
            if args.delete_original and not args.dry_run:
                input_path.unlink()
                logger.info(f"已删除原始文件: {input_path}")
        else:
            total_stats["failed_count"] = 1

    # 输出总结
    logger.info("=" * 60)
    logger.info("转换完成!")
    logger.info(f"  成功: {total_stats['success_count']}/{total_stats['total_files']}")
    if total_stats["failed_count"] > 0:
        logger.warning(f"  失败: {total_stats['failed_count']}")
    logger.info(f"  输入总大小: {format_size(total_stats['total_input_mb'])}")
    logger.info(f"  输出总大小: {format_size(total_stats['total_output_mb'])}")
    if total_stats["total_output_mb"] > 0:
        ratio = total_stats["total_input_mb"] / total_stats["total_output_mb"]
        saved_mb = total_stats["total_input_mb"] - total_stats["total_output_mb"]
        logger.info(f"  压缩比: {ratio:.2f}x，节省 {format_size(saved_mb)}")
    logger.info(f"  总耗时: {total_stats['total_load_time_s'] + total_stats['total_save_time_s']:.2f}s")


if __name__ == "__main__":
    main()
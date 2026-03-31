#!/usr/bin/env python3
"""
数据清洗脚本（DuckDB 优化版）

功能：
- 删除高缺失列（保留关键时间字段）
- 删除常量列
- 内存可控，适合大文件

用法：
    uv run python scripts/pipeline/03_clean_duckdb.py \
        --input ./data/线索宽表_合并.parquet \
        --output ./data/线索宽表_清洗后_v2.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import format_size
from config.column_mapping import normalize_column_names


# 必须保留的列（即使缺失率很高）
PRESERVE_COLUMNS: Set[str] = {
    "线索唯一ID",
    "到店时间",
    "试驾时间",
    "线索创建时间",
    "手机号_脱敏",  # 规范化后的名称
    "手机号（脱敏）",  # 兼容旧名称
}


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def clean_with_duckdb(
    input_path: Path,
    output_path: Path,
    high_missing_threshold: float = 0.5,
    memory_limit: str = "4GB",
    threads: int = 4,
    preserve_columns: Set[str] = None,
) -> Path:
    """
    使用 DuckDB 进行数据清洗

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        high_missing_threshold: 高缺失阈值（默认 0.5）
        memory_limit: DuckDB 内存限制
        threads: 线程数
        preserve_columns: 必须保留的列名集合

    Returns:
        输出文件路径
    """
    import duckdb

    start_time = time.time()
    preserve = preserve_columns or PRESERVE_COLUMNS

    log("=" * 60)
    log("数据清洗（DuckDB 优化版）")
    log("=" * 60)
    log(f"输入文件: {input_path}")
    log(f"输出文件: {output_path}")
    log(f"内存限制: {memory_limit}")
    log(f"线程数: {threads}")
    log(f"高缺失阈值: {high_missing_threshold * 100}%")
    log(f"保留列: {', '.join(sorted(preserve))}")

    # 创建 DuckDB 连接
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")

    try:
        # 1. 读取数据
        log("\n读取数据...")
        con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

        # 获取列信息
        cols_info = con.execute("DESCRIBE source").fetchall()
        total_cols = len(cols_info)
        col_names = [col[0] for col in cols_info]

        # 字段名规范化（PostgreSQL 不合法字符）
        rename_map = normalize_column_names(col_names)
        if rename_map:
            log(f"  字段名规范化: {len(rename_map)} 个字段需要重命名")
            for old_name, new_name in rename_map.items():
                log(f"    \"{old_name}\" → \"{new_name}\"")

        # 获取总行数
        total_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        log(f"  数据量: {total_rows:,} 行, {total_cols} 列")

        # 2. 计算每列缺失率
        log("\n计算缺失率...")
        missing_stats = []

        for col_name in col_names:
            null_count = con.execute(f'SELECT COUNT(*) FROM source WHERE "{col_name}" IS NULL').fetchone()[0]
            missing_ratio = null_count / total_rows if total_rows > 0 else 0
            missing_stats.append({
                "name": col_name,
                "null_count": null_count,
                "missing_ratio": missing_ratio,
            })

        # 3. 确定要删除的列
        dropped_cols = []
        kept_cols = []

        for stat in missing_stats:
            col_name = stat["name"]
            missing_ratio = stat["missing_ratio"]

            # 保留列不删除
            if col_name in preserve:
                kept_cols.append(col_name)
                continue

            # 高缺失列删除
            if missing_ratio > high_missing_threshold:
                dropped_cols.append({
                    "name": col_name,
                    "reason": "high_missing",
                    "missing_ratio": missing_ratio,
                })
                continue

            kept_cols.append(col_name)

        log(f"  高缺失列: {len(dropped_cols)} 个")

        # 4. 检查常量列
        log("\n检查常量列...")
        constant_cols = []

        for col_name in kept_cols:
            if col_name in preserve:
                continue

            unique_count = con.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM source').fetchone()[0]
            if unique_count <= 1:
                constant_cols.append(col_name)
                dropped_cols.append({
                    "name": col_name,
                    "reason": "constant",
                    "unique_count": unique_count,
                })

        # 从 kept_cols 中移除常量列
        kept_cols = [c for c in kept_cols if c not in constant_cols]

        log(f"  常量列: {len(constant_cols)} 个")

        # 5. 构建并执行输出 SQL
        log("\n执行清洗...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 构建字段选择表达式（包含字段名规范化）
        select_exprs = []
        for col in kept_cols:
            new_name = rename_map.get(col, col)
            if new_name != col:
                select_exprs.append(f'"{col}" AS "{new_name}"')
            else:
                select_exprs.append(f'"{col}"')

        select_sql = ", ".join(select_exprs)

        con.execute(f"""
            COPY (
                SELECT {select_sql}
                FROM source
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 6. 统计
        elapsed = time.time() - start_time
        output_size_mb = output_path.stat().st_size / 1024 / 1024

        log("")
        log("=" * 60)
        log("清洗完成")
        log("=" * 60)
        log(f"输出文件: {output_path}")
        log(f"文件大小: {output_size_mb:.1f} MB")
        log(f"数据量: {total_rows:,} 行, {len(kept_cols)} 列")
        log(f"删除列数: {len(dropped_cols)}")
        log(f"  - 高缺失列: {len([c for c in dropped_cols if c['reason'] == 'high_missing'])}")
        log(f"  - 常量列: {len([c for c in dropped_cols if c['reason'] == 'constant'])}")
        log(f"保留时间字段: 到店时间, 试驾时间, 线索创建时间")
        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return output_path

    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据清洗脚本 - 删除高缺失列和常量列，保留时间字段（DuckDB 优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
保留字段（即使缺失率 > 50%）：
  - 线索唯一ID
  - 到店时间
  - 试驾时间
  - 线索创建时间
  - 手机号（脱敏）

示例:
    uv run python scripts/pipeline/03_clean_duckdb.py \\
        --input ./data/线索宽表_合并.parquet \\
        --output ./data/线索宽表_清洗后_v2.parquet

    # 自定义内存限制
    uv run python scripts/pipeline/03_clean_duckdb.py \\
        --input ./data/input.parquet \\
        --output ./data/output.parquet \\
        --memory-limit 2GB \\
        --high-missing-threshold 0.3
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--high-missing-threshold",
        type=float,
        default=0.5,
        help="高缺失阈值（默认: 0.5，即 50%%）"
    )
    parser.add_argument(
        "--memory-limit",
        default="4GB",
        help="DuckDB 内存限制（默认: 4GB）"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="线程数（默认: 4）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 检查输入文件
    if not input_path.exists():
        log(f"❌ 输入文件不存在: {input_path}")
        return 1

    try:
        clean_with_duckdb(
            input_path=input_path,
            output_path=output_path,
            high_missing_threshold=args.high_missing_threshold,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        return 0

    except Exception as e:
        log(f"❌ 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
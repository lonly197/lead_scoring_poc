#!/usr/bin/env python3
"""
时间窗口标签计算脚本（DuckDB 优化版）

功能：
- 计算试驾天数差（试驾时间 - 线索创建时间）
- 生成时间窗口二分类标签：7天内、14天内、21天内试驾
- 生成 OHAB 级别标签

用法：
    uv run python scripts/pipeline/06_compute_labels.py \
        --input ./data/线索宽表_清洗后_v2.parquet \
        --output ./data/线索宽表_带标签_v2.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import format_size


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def compute_labels_with_duckdb(
    input_path: Path,
    output_path: Path,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> Path:
    """
    使用 DuckDB SQL 计算时间窗口标签

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        memory_limit: DuckDB 内存限制
        threads: 线程数

    Returns:
        输出文件路径
    """
    import duckdb

    start_time = time.time()

    log("=" * 60)
    log("时间窗口标签计算（DuckDB 优化版）")
    log("=" * 60)
    log(f"输入文件: {input_path}")
    log(f"输出文件: {output_path}")
    log(f"内存限制: {memory_limit}")
    log(f"线程数: {threads}")

    # 创建 DuckDB 连接
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")

    try:
        # 1. 读取数据
        log("\n读取数据...")
        con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

        # 获取统计
        total_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        test_drive_rows = con.execute('SELECT COUNT(*) FROM source WHERE "试驾时间" IS NOT NULL').fetchone()[0]
        log(f"  总数据量: {total_rows:,} 行")
        log(f"  有试驾时间: {test_drive_rows:,} 行 ({test_drive_rows/total_rows*100:.1f}%)")

        # 2. 计算标签
        log("\n计算时间窗口标签...")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 CTE 计算天数差，然后基于天数差计算标签
        con.execute(f"""
            COPY (
                WITH parsed AS (
                    SELECT
                        *,
                        -- 解析时间并计算天数差
                        CASE
                            WHEN "试驾时间" IS NULL THEN NULL
                            ELSE CAST(
                                strptime("试驾时间", '%Y-%m-%d')::DATE -
                                strptime("线索创建时间", '%Y-%m-%d %H:%M:%S')::DATE
                                AS INTEGER
                            )
                        END AS 试驾天数差
                    FROM source
                )
                SELECT
                    *,
                    -- 时间窗口二分类标签
                    CASE
                        WHEN "试驾时间" IS NULL THEN NULL
                        WHEN 试驾天数差 <= 7 THEN 1
                        ELSE 0
                    END AS label_7天内试驾,

                    CASE
                        WHEN "试驾时间" IS NULL THEN NULL
                        WHEN 试驾天数差 <= 14 THEN 1
                        ELSE 0
                    END AS label_14天内试驾,

                    CASE
                        WHEN "试驾时间" IS NULL THEN NULL
                        WHEN 试驾天数差 <= 21 THEN 1
                        ELSE 0
                    END AS label_21天内试驾,

                    -- OHAB 级别（字符串）
                    CASE
                        WHEN "试驾时间" IS NULL THEN NULL
                        WHEN 试驾天数差 <= 7 THEN 'H'
                        WHEN 试驾天数差 <= 14 THEN 'A'
                        WHEN 试驾天数差 <= 21 THEN 'B'
                        ELSE 'O'
                    END AS label_OHAB

                FROM parsed
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 3. 统计标签分布
        log("\n标签分布统计:")

        # OHAB 分布
        ohab_dist = con.execute(f"""
            SELECT label_OHAB, COUNT(*) as cnt
            FROM read_parquet('{output_path}')
            WHERE label_OHAB IS NOT NULL
            GROUP BY label_OHAB
            ORDER BY label_OHAB
        """).fetchall()

        log("  OHAB 级别分布:")
        total_labeled = sum(row[1] for row in ohab_dist)
        for level, cnt in ohab_dist:
            pct = cnt / total_labeled * 100 if total_labeled > 0 else 0
            log(f"    {level}: {cnt:,} ({pct:.1f}%)")

        # 7天内试驾分布
        label_7_dist = con.execute(f"""
            SELECT label_7天内试驾, COUNT(*) as cnt
            FROM read_parquet('{output_path}')
            WHERE label_7天内试驾 IS NOT NULL
            GROUP BY label_7天内试驾
        """).fetchall()
        log(f"  7天内试驾: {dict(label_7_dist)}")

        # 14天内试驾分布
        label_14_dist = con.execute(f"""
            SELECT label_14天内试驾, COUNT(*) as cnt
            FROM read_parquet('{output_path}')
            WHERE label_14天内试驾 IS NOT NULL
            GROUP BY label_14天内试驾
        """).fetchall()
        log(f"  14天内试驾: {dict(label_14_dist)}")

        # 21天内试驾分布
        label_21_dist = con.execute(f"""
            SELECT label_21天内试驾, COUNT(*) as cnt
            FROM read_parquet('{output_path}')
            WHERE label_21天内试驾 IS NOT NULL
            GROUP BY label_21天内试驾
        """).fetchall()
        log(f"  21天内试驾: {dict(label_21_dist)}")

        # 4. 输出摘要
        elapsed = time.time() - start_time
        output_size_mb = output_path.stat().st_size / 1024 / 1024

        log("")
        log("=" * 60)
        log("标签计算完成")
        log("=" * 60)
        log(f"输出文件: {output_path}")
        log(f"文件大小: {output_size_mb:.1f} MB")
        log(f"数据量: {total_rows:,} 行")
        log(f"新增列: 试驾天数差, label_7天内试驾, label_14天内试驾, label_21天内试驾, label_OHAB")
        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return output_path

    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="时间窗口标签计算脚本 - 计算7/14/21天试驾标签和OHAB级别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
标签说明：
  label_7天内试驾  - 1: 7天内试驾（H级）, 0: 超过7天
  label_14天内试驾 - 1: 14天内试驾（H+A级）, 0: 超过14天
  label_21天内试驾 - 1: 21天内试驾（H+A+B级）, 0: 超过21天
  label_OHAB       - H: 7天内, A: 14天内, B: 21天内, O: 超过21天

业务规则：
  H级: 7天内试驾
  A级: 14天内试驾
  B级: 21天内试驾

示例:
    uv run python scripts/pipeline/06_compute_labels.py \\
        --input ./data/线索宽表_清洗后_v2.parquet \\
        --output ./data/线索宽表_带标签_v2.parquet

    # 自定义内存限制
    uv run python scripts/pipeline/06_compute_labels.py \\
        --input ./data/input.parquet \\
        --output ./data/output.parquet \\
        --memory-limit 2GB
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
        compute_labels_with_duckdb(
            input_path=input_path,
            output_path=output_path,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        return 0

    except Exception as e:
        log(f"❌ 标签计算失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
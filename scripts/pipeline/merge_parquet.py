#!/usr/bin/env python3
"""
Parquet 数据合并脚本

直接合并 Parquet 格式的线索数据和 DMP 数据，无需 Excel 转换。

用法：
    uv run python scripts/pipeline/merge_parquet.py \
        --clue ./data/线索数据.parquet \
        --dmp ./data/DMP数据.parquet \
        --output ./data/merged.parquet \
        --phone-column "手机号（脱敏）"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def merge_parquet(
    clue_path: Path,
    dmp_path: Path,
    output_path: Path,
    phone_column: str = "手机号（脱敏）",
    memory_limit: str = "4GB",
    threads: int = 4,
) -> Path:
    """
    使用 DuckDB 合并 Parquet 数据

    Args:
        clue_path: 线索数据 Parquet 文件
        dmp_path: DMP 数据 Parquet 文件
        output_path: 输出文件路径
        phone_column: 线索数据中的手机号列名
        memory_limit: DuckDB 内存限制
        threads: 线程数

    Returns:
        输出文件路径
    """
    start_time = time.time()

    log("=" * 60)
    log("Parquet 数据合并（DuckDB）")
    log("=" * 60)
    log(f"线索数据: {clue_path}")
    log(f"DMP 数据: {dmp_path}")
    log(f"输出文件: {output_path}")
    log(f"内存限制: {memory_limit}")
    log(f"线程数: {threads}")
    log("")

    # 创建 DuckDB 连接
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")

    try:
        # 1. 读取线索数据
        log("读取线索数据...")
        con.execute(f"""
            CREATE VIEW clue AS
            SELECT * FROM read_parquet('{clue_path}')
        """)

        clue_rows = con.execute("SELECT COUNT(*) FROM clue").fetchone()[0]
        clue_cols = len(con.execute("SELECT * FROM clue LIMIT 0").description)
        log(f"  线索数据: {clue_rows:,} 行, {clue_cols} 列")

        # 2. 读取并聚合 DMP 数据
        log("读取并聚合 DMP 数据...")
        con.execute(f"""
            CREATE VIEW dmp_agg AS
            SELECT
                手机号 as dmp_手机号,
                COUNT(*) as dmp_行为次数,
                MAX(行为时间) as dmp_最近行为时间,
                COUNT(*) FILTER (
                    WHERE 行为事件 LIKE '%testDrive%'
                       OR 行为事件 LIKE '%试驾%'
                ) as dmp_试驾相关次数,
                COUNT(*) FILTER (
                    WHERE 行为事件 LIKE '%submitOrder%'
                       OR 行为事件 LIKE '%payOrder%'
                       OR 行为事件 LIKE '%下单%'
                       OR 行为事件 LIKE '%支付%'
                ) as dmp_下单支付次数,
                COUNT(DISTINCT 行为事件) as dmp_事件类型数
            FROM read_parquet('{dmp_path}')
            GROUP BY 手机号
        """)

        dmp_users = con.execute("SELECT COUNT(*) FROM dmp_agg").fetchone()[0]
        log(f"  DMP 聚合后: {dmp_users:,} 用户")

        # 3. Left Join 关联
        log("关联数据...")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 执行关联并输出到 Parquet
        con.execute(f"""
            COPY (
                SELECT
                    c.*,
                    d.dmp_行为次数,
                    d.dmp_最近行为时间,
                    d.dmp_试驾相关次数,
                    d.dmp_下单支付次数,
                    d.dmp_事件类型数
                FROM clue c
                LEFT JOIN dmp_agg d ON c."{phone_column}" = d.dmp_手机号
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 4. 统计匹配率
        stats = con.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT(dmp_行为次数) as matched
            FROM read_parquet('{output_path}')
        """).fetchone()

        total, matched = stats
        match_rate = matched / total * 100 if total > 0 else 0

        # 5. 输出摘要
        elapsed = time.time() - start_time
        output_size_mb = output_path.stat().st_size / 1024 / 1024

        log("")
        log("=" * 60)
        log("合并完成")
        log("=" * 60)
        log(f"输出文件: {output_path}")
        log(f"文件大小: {output_size_mb:.1f} MB")
        log(f"数据量: {total:,} 行")
        log(f"DMP 匹配: {matched:,}/{total:,} ({match_rate:.1f}%)")
        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return output_path

    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parquet 数据合并脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--clue", "-c",
        required=True,
        help="线索数据 Parquet 文件路径"
    )
    parser.add_argument(
        "--dmp", "-d",
        required=True,
        help="DMP 数据 Parquet 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--phone-column",
        default="手机号（脱敏）",
        help="线索数据中的手机号列名（默认: 手机号（脱敏））"
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

    clue_path = Path(args.clue)
    dmp_path = Path(args.dmp)
    output_path = Path(args.output)

    # 检查输入文件
    if not clue_path.exists():
        log(f"❌ 线索数据文件不存在: {clue_path}")
        return 1
    if not dmp_path.exists():
        log(f"❌ DMP 数据文件不存在: {dmp_path}")
        return 1

    try:
        merge_parquet(
            clue_path=clue_path,
            dmp_path=dmp_path,
            output_path=output_path,
            phone_column=args.phone_column,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        return 0

    except Exception as e:
        log(f"❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
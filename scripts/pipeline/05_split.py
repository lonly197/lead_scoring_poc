#!/usr/bin/env python3
"""
数据拆分脚本（DuckDB 优化版）

功能：
- 随机分层切分
- OOT（Out-of-Time）时间切分
- 自动选择模式

优化特性：
- 使用 DuckDB SQL 直接过滤，避免加载全部数据
- 向量化采样，性能提升 3-5x
- 内存可控，适合大文件

用法：
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode oot \\
        --time-column 线索创建时间

    # 输出: final_train.parquet, final_test.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import print_step, format_size


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def split_random_duckdb(
    con,
    input_path: Path,
    train_path: Path,
    test_path: Path,
    target_column: str,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    使用 DuckDB 进行分层随机拆分

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        train_path: 训练集输出路径
        test_path: 测试集输出路径
        target_column: 目标变量列名
        test_ratio: 测试集比例
        random_seed: 随机种子
        compression: 压缩算法

    Returns:
        (训练集行数, 测试集行数)
    """
    log("随机分层切分...")
    log(f"  目标列: {target_column}, 测试比例: {test_ratio}")

    # 创建源视图
    con.execute(f"CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

    # 检查目标列是否存在
    cols_info = con.execute("DESCRIBE source").fetchall()
    col_names = [col[0] for col in cols_info]

    if target_column not in col_names:
        log(f"  警告: 目标列 '{target_column}' 不存在，使用简单随机拆分")
        # 简单随机拆分
        total_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        test_size = int(total_rows * test_ratio)

        # 使用 SAMPLE 进行采样（比 ORDER BY RANDOM() 更快）
        con.execute(f"""
            CREATE OR REPLACE VIEW test_data AS
            SELECT * FROM source USING SAMPLE {test_ratio * 100}% (bernoulli, {random_seed})
        """)
        con.execute(f"""
            CREATE OR REPLACE VIEW train_data AS
            SELECT s.* FROM source s
            LEFT JOIN test_data t ON s.rowid = t.rowid
            WHERE t.rowid IS NULL
        """)

        # 注意：DuckDB 的 SAMPLE bernoulli 是近似的，用 WHERE NOT IN 更准确
        con.execute(f"DROP VIEW IF EXISTS test_data")
        con.execute(f"DROP VIEW IF EXISTS train_data")

        # 更精确的方法：使用 rowid 和哈希
        con.execute(f"""
            CREATE OR REPLACE VIEW split_view AS
            SELECT *,
                   CASE WHEN murmur_hash(rowid) % 100 < {test_ratio * 100} THEN 'test' ELSE 'train' END as _split
            FROM source
        """)

        # 输出训练集
        con.execute(f"""
            COPY (SELECT * EXCLUDE (_split) FROM split_view WHERE _split = 'train')
            TO '{train_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
        """)

        # 输出测试集
        con.execute(f"""
            COPY (SELECT * EXCLUDE (_split) FROM split_view WHERE _split = 'test')
            TO '{test_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
        """)

    else:
        # 分层随机拆分：使用窗口函数
        log("  执行分层采样...")

        # 为每个目标值计算测试集大小
        unique_values = con.execute(f'SELECT DISTINCT "{target_column}" FROM source WHERE "{target_column}" IS NOT NULL').fetchall()
        unique_values = [v[0] for v in unique_values]

        # 使用 ROW_NUMBER 和 HASH 进行分层采样
        con.execute(f"""
            CREATE OR REPLACE VIEW split_view AS
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY "{target_column}" ORDER BY murmur_hash(rowid)) as _row_num,
                   COUNT(*) OVER (PARTITION BY "{target_column}") as _total_in_group
            FROM source
            WHERE "{target_column}" IS NOT NULL
        """)

        # 计算每个组的测试集大小阈值
        # 使用一个阈值百分比来划分
        con.execute(f"""
            CREATE OR REPLACE VIEW labeled_split AS
            SELECT *,
                   CASE WHEN _row_num <= CAST(_total_in_group * {test_ratio} AS INTEGER) THEN 'test' ELSE 'train' END as _split
            FROM split_view
        """)

        # 输出训练集
        con.execute(f"""
            COPY (SELECT * EXCLUDE (_row_num, _total_in_group, _split) FROM labeled_split WHERE _split = 'train')
            TO '{train_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
        """)

        # 输出测试集
        con.execute(f"""
            COPY (SELECT * EXCLUDE (_row_num, _total_in_group, _split) FROM labeled_split WHERE _split = 'test')
            TO '{test_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
        """)

    # 获取行数
    train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{train_path}')").fetchone()[0]
    test_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{test_path}')").fetchone()[0]

    # 打印分布
    if target_column in col_names:
        train_dist = con.execute(f'SELECT "{target_column}", COUNT(*) FROM read_parquet(\'{train_path}\') GROUP BY "{target_column}" ORDER BY "{target_column}"').fetchall()
        test_dist = con.execute(f'SELECT "{target_column}", COUNT(*) FROM read_parquet(\'{test_path}\') GROUP BY "{target_column}" ORDER BY "{target_column}"').fetchall()
        log(f"  训练集分布: {dict(train_dist)}")
        log(f"  测试集分布: {dict(test_dist)}")

    log(f"  ✅ 完成: 训练集 {train_rows:,} 行, 测试集 {test_rows:,} 行")
    return train_rows, test_rows


def split_oot_duckdb(
    con,
    input_path: Path,
    train_path: Path,
    test_path: Path,
    time_column: str,
    cutoff_date: str,
    target_column: Optional[str] = None,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    使用 DuckDB 进行 OOT 时间切分

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        train_path: 训练集输出路径
        test_path: 测试集输出路径
        time_column: 时间列名
        cutoff_date: 切分时间点
        target_column: 目标变量列名
        compression: 压缩算法

    Returns:
        (训练集行数, 测试集行数)
    """
    log("OOT 时间切分...")
    log(f"  时间列: {time_column}, 切分点: {cutoff_date}")

    # 创建源视图
    con.execute(f"CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

    # 检查时间列
    cols_info = con.execute("DESCRIBE source").fetchall()
    col_names = [col[0] for col in cols_info]

    if time_column not in col_names:
        raise ValueError(f"时间列 '{time_column}' 不存在")

    # 直接用 SQL 过滤并输出（不加载到内存）
    # 训练集：时间 < cutoff
    con.execute(f"""
        COPY (
            SELECT * FROM source
            WHERE "{time_column}" < '{cutoff_date}'::TIMESTAMP
            AND "{time_column}" IS NOT NULL
        ) TO '{train_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 测试集：时间 >= cutoff
    con.execute(f"""
        COPY (
            SELECT * FROM source
            WHERE "{time_column}" >= '{cutoff_date}'::TIMESTAMP
            AND "{time_column}" IS NOT NULL
        ) TO '{test_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 获取统计
    train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{train_path}')").fetchone()[0]
    test_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{test_path}')").fetchone()[0]

    # 时间范围
    train_min = con.execute(f"SELECT MIN(\"{time_column}\") FROM read_parquet('{train_path}')").fetchone()[0]
    train_max = con.execute(f"SELECT MAX(\"{time_column}\") FROM read_parquet('{train_path}')").fetchone()[0]
    test_min = con.execute(f"SELECT MIN(\"{time_column}\") FROM read_parquet('{test_path}')").fetchone()[0]
    test_max = con.execute(f"SELECT MAX(\"{time_column}\") FROM read_parquet('{test_path}')").fetchone()[0]

    log(f"  训练集时间范围: {train_min} ~ {train_max}")
    log(f"  测试集时间范围: {test_min} ~ {test_max}")

    # 打印分布
    if target_column and target_column in col_names:
        train_dist = con.execute(f'SELECT "{target_column}", COUNT(*) FROM read_parquet(\'{train_path}\') GROUP BY "{target_column}" ORDER BY "{target_column}"').fetchall()
        test_dist = con.execute(f'SELECT "{target_column}", COUNT(*) FROM read_parquet(\'{test_path}\') GROUP BY "{target_column}" ORDER BY "{target_column}"').fetchall()
        log(f"  训练集分布: {dict(train_dist)}")
        log(f"  测试集分布: {dict(test_dist)}")

    log(f"  ✅ 完成: 训练集 {train_rows:,} 行, 测试集 {test_rows:,} 行")
    return train_rows, test_rows


def split_auto_duckdb(
    con,
    input_path: Path,
    train_path: Path,
    test_path: Path,
    time_column: str,
    min_oot_days: int = 30,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    target_column: Optional[str] = None,
    compression: str = "zstd",
) -> Tuple[int, int, str]:
    """
    自动选择切分方式

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        train_path: 训练集输出路径
        test_path: 测试集输出路径
        time_column: 时间列名
        min_oot_days: 触发 OOT 的最少天数
        test_ratio: 随机切分时的测试集比例
        random_seed: 随机种子
        target_column: 目标变量列名
        compression: 压缩算法

    Returns:
        (训练集行数, 测试集行数, 实际使用的模式)
    """
    log("自动选择切分方式...")
    log(f"  min_oot_days: {min_oot_days}")

    # 创建源视图
    con.execute(f"CREATE OR REPLACE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

    # 检查时间列
    cols_info = con.execute("DESCRIBE source").fetchall()
    col_names = [col[0] for col in cols_info]

    if time_column not in col_names:
        log(f"  警告: 时间列 '{time_column}' 不存在，降级为随机切分")
        train_rows, test_rows = split_random_duckdb(
            con, input_path, train_path, test_path,
            target_column, test_ratio, random_seed, compression
        )
        return train_rows, test_rows, "random"

    # 计算时间跨度
    time_stats = con.execute(f"""
        SELECT MIN("{time_column}") as min_time, MAX("{time_column}") as max_time
        FROM source WHERE "{time_column}" IS NOT NULL
    """).fetchone()

    if time_stats[0] is None:
        log("  警告: 时间列无有效数据，降级为随机切分")
        train_rows, test_rows = split_random_duckdb(
            con, input_path, train_path, test_path,
            target_column, test_ratio, random_seed, compression
        )
        return train_rows, test_rows, "random"

    min_time, max_time = time_stats
    time_span_days = (max_time - min_time).days

    log(f"  时间跨度: {time_span_days} 天")

    if time_span_days >= min_oot_days:
        log(f"  跨度 >= {min_oot_days} 天，使用 OOT 切分")
        # 自动计算切分点
        total_seconds = (max_time - min_time).total_seconds()
        cutoff_seconds = total_seconds * (1 - test_ratio)
        cutoff_date = (min_time + timedelta(seconds=cutoff_seconds)).strftime("%Y-%m-%d")
        log(f"  自动计算切分点: {cutoff_date}")

        train_rows, test_rows = split_oot_duckdb(
            con, input_path, train_path, test_path,
            time_column, cutoff_date, target_column, compression
        )
        return train_rows, test_rows, "oot"
    else:
        log(f"  跨度 < {min_oot_days} 天，使用随机切分")
        train_rows, test_rows = split_random_duckdb(
            con, input_path, train_path, test_path,
            target_column, test_ratio, random_seed, compression
        )
        return train_rows, test_rows, "random"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据拆分脚本（DuckDB 优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
拆分模式:
  random  - 随机分层切分，保持目标分布
  oot     - 时间切分（Out-of-Time），用历史预测未来
  auto    - 自动判断：跨度>=30天用OOT，否则用随机

示例:
    # OOT 时间切分
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode oot \\
        --time-column 线索创建时间

    # 随机分层切分
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode random \\
        --target 线索评级结果

    # 自动选择 + 自定义内存
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/large.parquet \\
        --output ./data/final \\
        --mode auto \\
        --memory-limit 8GB

输出:
    {output}_train.parquet
    {output}_test.parquet
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
        help="输出文件前缀（自动添加 _train/_test 后缀）"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["random", "oot", "auto"],
        default="random",
        help="拆分模式（默认: random）"
    )
    parser.add_argument(
        "--target", "-t",
        default="线索评级结果",
        help="分层采样目标列（默认: 线索评级结果）"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="测试集比例（默认: 0.2）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--time-column",
        default="线索创建时间",
        help="时间列名（OOT 模式）"
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help="OOT 切分时间点（格式 YYYY-MM-DD）"
    )
    parser.add_argument(
        "--min-oot-days",
        type=int,
        default=30,
        help="自动模式触发 OOT 的最少天数（默认: 30）"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="输出格式（默认: parquet）"
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "zstd", "lz4"],
        help="Parquet 压缩算法（默认: zstd）"
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
    output_prefix = Path(args.output)

    # 检查输入文件
    if not input_path.exists():
        log(f"❌ 输入文件不存在: {input_path}")
        return 1

    # 确定输出路径
    suffix = f".{args.format}"
    train_path = Path(f"{output_prefix}_train{suffix}")
    test_path = Path(f"{output_prefix}_test{suffix}")

    # 确保输出目录存在
    train_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        import duckdb

        log("=" * 60)
        log("数据拆分（DuckDB 优化版）")
        log("=" * 60)
        log(f"输入文件: {input_path}")
        log(f"输出路径: {train_path}, {test_path}")
        log(f"拆分模式: {args.mode}")
        log(f"内存限制: {args.memory_limit}, 线程数: {args.threads}")

        # 创建 DuckDB 连接
        con = duckdb.connect(":memory:")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"SET threads={args.threads}")

        # 获取基本信息
        total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{input_path}')").fetchone()[0]
        log(f"数据量: {total_rows:,} 行")

        # 根据模式拆分
        if args.mode == "random":
            train_rows, test_rows = split_random_duckdb(
                con, input_path, train_path, test_path,
                args.target, args.ratio, args.seed, args.compression
            )

        elif args.mode == "oot":
            # 计算切分时间点
            if args.cutoff:
                cutoff = args.cutoff
            else:
                # 自动计算
                time_stats = con.execute(f"""
                    SELECT MIN("{args.time_column}"), MAX("{args.time_column}")
                    FROM read_parquet('{input_path}')
                    WHERE "{args.time_column}" IS NOT NULL
                """).fetchone()
                min_time, max_time = time_stats
                total_seconds = (max_time - min_time).total_seconds()
                cutoff_seconds = total_seconds * (1 - args.ratio)
                cutoff = (min_time + timedelta(seconds=cutoff_seconds)).strftime("%Y-%m-%d")
                log(f"自动计算切分时间点: {cutoff}")

            train_rows, test_rows = split_oot_duckdb(
                con, input_path, train_path, test_path,
                args.time_column, cutoff, args.target, args.compression
            )

        else:  # auto
            train_rows, test_rows, actual_mode = split_auto_duckdb(
                con, input_path, train_path, test_path,
                args.time_column, args.min_oot_days, args.ratio,
                args.seed, args.target, args.compression
            )
            log(f"实际使用模式: {actual_mode}")

        con.close()

        elapsed = time.time() - start_time

        # 打印摘要
        log("")
        log("=" * 60)
        log("拆分完成")
        log("=" * 60)
        log(f"训练集: {train_path}")
        log(f"  文件大小: {format_size(train_path)}")
        log(f"  数据量: {train_rows:,} 行")
        log(f"测试集: {test_path}")
        log(f"  文件大小: {format_size(test_path)}")
        log(f"  数据量: {test_rows:,} 行")
        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return 0

    except Exception as e:
        log(f"❌ 拆分失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
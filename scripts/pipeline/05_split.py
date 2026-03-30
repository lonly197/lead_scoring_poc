#!/usr/bin/env python3
"""
数据拆分脚本

功能：
- 随机分层切分
- OOT（Out-of-Time）时间切分
- 自动选择模式

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
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import load_data, save_data, print_step, format_size


# ==================== 拆分函数 ====================

def split_random(
    pl,
    df: "pl.DataFrame",
    target_column: str,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple["pl.DataFrame", "pl.DataFrame"]:
    """
    分层随机拆分

    Args:
        pl: polars 模块
        df: 数据 DataFrame
        target_column: 目标变量列名（用于分层）
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        (训练集, 测试集)
    """
    print_step("随机分层切分", "running", f"目标列={target_column}, 测试比例={test_ratio}")

    # 检查目标列是否存在
    if target_column not in df.columns:
        print(f"  警告: 目标列 '{target_column}' 不存在，使用简单随机拆分")
        n = len(df)
        test_size = int(n * test_ratio)
        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=random_seed)
        test_df = df_shuffled.head(test_size)
        train_df = df_shuffled.tail(n - test_size)
        return train_df, test_df

    # 过滤目标列为空的行
    df_valid = df.filter(pl.col(target_column).is_not_null())
    null_count = len(df) - len(df_valid)
    if null_count > 0:
        print(f"  过滤目标列为空的行: {null_count}")

    # 分层采样
    unique_values = df_valid.select(target_column).unique().to_series().to_list()

    train_parts = []
    test_parts = []

    for val in unique_values:
        subset = df_valid.filter(pl.col(target_column) == val)
        n = len(subset)
        test_size = int(n * test_ratio)

        # 随机打乱
        subset_shuffled = subset.sample(fraction=1.0, shuffle=True, seed=random_seed)

        test_parts.append(subset_shuffled.head(test_size))
        train_parts.append(subset_shuffled.tail(n - test_size))

    train_df = pl.concat(train_parts)
    test_df = pl.concat(test_parts)

    # 最终打乱
    train_df = train_df.sample(fraction=1.0, shuffle=True, seed=random_seed)
    test_df = test_df.sample(fraction=1.0, shuffle=True, seed=random_seed)

    print_step("随机分层切分", "success", f"训练集 {len(train_df):,} 行, 测试集 {len(test_df):,} 行")

    # 打印目标分布
    train_dist = train_df.group_by(target_column).len().sort(target_column)
    test_dist = test_df.group_by(target_column).len().sort(target_column)
    print(f"  训练集分布: {train_dist.to_dict(as_series=False)}")
    print(f"  测试集分布: {test_dist.to_dict(as_series=False)}")

    return train_df, test_df


def split_oot(
    pl,
    df: "pl.DataFrame",
    time_column: str,
    cutoff_date: str,
    target_column: Optional[str] = None,
) -> Tuple["pl.DataFrame", "pl.DataFrame"]:
    """
    OOT（Out-of-Time）时间切分

    使用历史数据训练，预测未来数据。

    Args:
        pl: polars 模块
        df: 数据 DataFrame
        time_column: 时间列名
        cutoff_date: 切分时间点（该日期及之后为测试集）
        target_column: 目标变量列名（可选）

    Returns:
        (训练集, 测试集)
    """
    print_step("OOT 时间切分", "running", f"time_column={time_column}, cutoff={cutoff_date}")

    # 检查时间列是否存在
    if time_column not in df.columns:
        raise ValueError(f"时间列 '{time_column}' 不存在")

    # 转换时间列
    df = df.with_columns(
        pl.col(time_column).cast(pl.Datetime).alias("_time_col")
    )

    # 过滤无效时间
    df_valid = df.filter(pl.col("_time_col").is_not_null())
    null_count = len(df) - len(df_valid)
    if null_count > 0:
        print(f"  过滤时间列为空的行: {null_count}")

    # 时间切分
    cutoff = pl.lit(cutoff_date).str.to_datetime("%Y-%m-%d")
    train_df = df_valid.filter(pl.col("_time_col") < cutoff).drop("_time_col")
    test_df = df_valid.filter(pl.col("_time_col") >= cutoff).drop("_time_col")

    print_step("OOT 时间切分", "success", f"训练集 {len(train_df):,} 行, 测试集 {len(test_df):,} 行")

    # 打印时间范围
    train_min = train_df.select(pl.col(time_column).min()).item()
    train_max = train_df.select(pl.col(time_column).max()).item()
    test_min = test_df.select(pl.col(time_column).min()).item()
    test_max = test_df.select(pl.col(time_column).max()).item()
    print(f"  训练集时间范围: {train_min} ~ {train_max}")
    print(f"  测试集时间范围: {test_min} ~ {test_max}")

    # 打印目标分布
    if target_column and target_column in train_df.columns:
        train_dist = train_df.group_by(target_column).len().sort(target_column)
        test_dist = test_df.group_by(target_column).len().sort(target_column)
        print(f"  训练集分布: {train_dist.to_dict(as_series=False)}")
        print(f"  测试集分布: {test_dist.to_dict(as_series=False)}")

    return train_df, test_df


def split_auto(
    pl,
    df: "pl.DataFrame",
    time_column: str,
    min_oot_days: int = 30,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    target_column: Optional[str] = None,
) -> Tuple["pl.DataFrame", "pl.DataFrame", str]:
    """
    自动选择切分方式

    检测时间跨度：
    - 跨度 >= min_oot_days：使用 OOT 切分
    - 跨度 < min_oot_days：使用随机切分

    Args:
        pl: polars 模块
        df: 数据 DataFrame
        time_column: 时间列名
        min_oot_days: 触发 OOT 的最少天数
        test_ratio: 随机切分时的测试集比例
        random_seed: 随机种子
        target_column: 目标变量列名

    Returns:
        (训练集, 测试集, 实际使用的模式)
    """
    print_step("自动选择切分方式", "running", f"min_oot_days={min_oot_days}")

    # 检查时间列
    if time_column not in df.columns:
        print(f"  警告: 时间列 '{time_column}' 不存在，降级为随机切分")
        train_df, test_df = split_random(pl, df, target_column, test_ratio, random_seed)
        return train_df, test_df, "random"

    # 计算时间跨度
    df_time = df.with_columns(pl.col(time_column).cast(pl.Datetime).alias("_time"))
    df_valid = df_time.filter(pl.col("_time").is_not_null())

    if len(df_valid) == 0:
        print(f"  警告: 时间列无有效数据，降级为随机切分")
        train_df, test_df = split_random(pl, df, target_column, test_ratio, random_seed)
        return train_df, test_df, "random"

    min_time = df_valid.select(pl.col("_time").min()).item()
    max_time = df_valid.select(pl.col("_time").max()).item()
    time_span_days = (max_time - min_time).days

    print(f"  时间跨度: {min_time} ~ {max_time} ({time_span_days} 天)")

    if time_span_days >= min_oot_days:
        # 使用 OOT 切分：按 80/20 时间比例
        total_seconds = (max_time - min_time).total_seconds()
        cutoff_seconds = total_seconds * (1 - test_ratio)
        cutoff_date = (min_time + timedelta(seconds=cutoff_seconds)).strftime("%Y-%m-%d")

        print(f"  触发 OOT 切分 (跨度 >= {min_oot_days} 天)")
        train_df, test_df = split_oot(pl, df, time_column, cutoff_date, target_column)
        return train_df, test_df, "oot"
    else:
        # 使用随机切分
        print(f"  降级为随机切分 (跨度 < {min_oot_days} 天)")
        train_df, test_df = split_random(pl, df, target_column, test_ratio, random_seed)
        return train_df, test_df, "random"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据拆分脚本 - 训练集/测试集切分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
拆分模式：
  random: 随机分层切分，保持目标分布一致（默认）
  oot:    时间切分，用历史数据预测未来
  auto:   自动判断，时间跨度>=30天用OOT，否则用随机

示例:
    # 随机切分
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode random

    # OOT 时间切分
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode oot \\
        --time-column 线索创建时间 \\
        --cutoff 2026-03-01

    # 自动选择
    uv run python scripts/pipeline/05_split.py \\
        --input ./data/desensitized.parquet \\
        --output ./data/final \\
        --mode auto \\
        --time-column 线索创建时间

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

    args = parser.parse_args()

    input_path = Path(args.input)
    output_prefix = Path(args.output)

    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1

    try:
        import polars as pl

        print("=" * 60)
        print("数据拆分脚本")
        print("=" * 60)

        # 加载数据
        print_step("加载数据", "running", str(input_path))
        df = load_data(input_path, engine="polars")
        print_step("加载数据", "success", f"{len(df):,} 行, {len(df.columns)} 列")

        # 根据模式拆分
        if args.mode == "random":
            train_df, test_df = split_random(
                pl=pl,
                df=df,
                target_column=args.target,
                test_ratio=args.ratio,
                random_seed=args.seed,
            )

        elif args.mode == "oot":
            # 计算切分时间点
            if args.cutoff:
                cutoff = args.cutoff
            else:
                # 自动计算 80/20 时间比例
                df_time = df.with_columns(
                    pl.col(args.time_column).cast(pl.Datetime).alias("_time")
                )
                df_valid = df_time.filter(pl.col("_time").is_not_null())
                min_time = df_valid.select(pl.col("_time").min()).item()
                max_time = df_valid.select(pl.col("_time").max()).item()
                total_seconds = (max_time - min_time).total_seconds()
                cutoff_seconds = total_seconds * (1 - args.ratio)
                cutoff = (min_time + timedelta(seconds=cutoff_seconds)).strftime("%Y-%m-%d")
                print(f"  自动计算切分时间点: {cutoff}")

            train_df, test_df = split_oot(
                pl=pl,
                df=df,
                time_column=args.time_column,
                cutoff_date=cutoff,
                target_column=args.target,
            )

        else:  # auto
            train_df, test_df, actual_mode = split_auto(
                pl=pl,
                df=df,
                time_column=args.time_column,
                min_oot_days=args.min_oot_days,
                test_ratio=args.ratio,
                random_seed=args.seed,
                target_column=args.target,
            )
            print(f"  实际使用模式: {actual_mode}")

        # 确定输出路径
        suffix = f".{args.format}"
        train_path = Path(f"{output_prefix}_train{suffix}")
        test_path = Path(f"{output_prefix}_test{suffix}")

        # 确保输出目录存在
        train_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存训练集
        print_step("保存训练集", "running", str(train_path))
        save_data(train_df, train_path)
        print_step("保存训练集", "success", format_size(train_path))

        # 保存测试集
        print_step("保存测试集", "running", str(test_path))
        save_data(test_df, test_path)
        print_step("保存测试集", "success", format_size(test_path))

        # 打印摘要
        print("\n" + "=" * 60)
        print("拆分完成")
        print("=" * 60)
        print(f"训练集: {train_path}")
        print(f"  文件大小: {format_size(train_path)}")
        print(f"  数据量: {len(train_df):,} 行")
        print(f"\n测试集: {test_path}")
        print(f"  文件大小: {format_size(test_path)}")
        print(f"  数据量: {len(test_df):,} 行")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"❌ 拆分失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
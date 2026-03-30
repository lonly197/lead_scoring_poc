#!/usr/bin/env python3
"""
数据合并脚本

功能：
- 读取 Excel 多 Sheet 并合并
- 读取 DMP 行为数据
- 按手机号关联，聚合 DMP 特征
- 输出合并后的 parquet 文件

用法：
    uv run python scripts/pipeline/01_merge.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/merged.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import load_data, save_data, print_step, format_size, get_or_create_parquet_cache
from src.pipeline.config import DMP_COLUMNS, default_config


def find_phone_column(columns: list, patterns: list = None) -> Optional[str]:
    """
    查找手机号列

    Args:
        columns: 列名列表
        patterns: 匹配模式列表

    Returns:
        手机号列名，未找到返回 None
    """
    if patterns is None:
        patterns = ["手机", "phone"]

    for col in columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern in col_lower:
                return col
    return None


def aggregate_dmp_features(pl, dmp_df: "pl.DataFrame") -> "pl.DataFrame":
    """
    聚合 DMP 行为特征

    Args:
        pl: polars 模块
        dmp_df: DMP 原始数据

    Returns:
        聚合后的 DataFrame
    """
    print_step("聚合 DMP 特征", "running")

    dmp_agg = dmp_df.group_by("dmp_手机号").agg([
        pl.len().alias("dmp_行为次数"),
        pl.col("dmp_事件时间").max().alias("dmp_最近行为时间"),
        pl.col("dmp_事件名称").filter(
            pl.col("dmp_事件名称").str.contains("testDrive|试驾")
        ).len().alias("dmp_试驾相关次数"),
        pl.col("dmp_事件名称").filter(
            pl.col("dmp_事件名称").str.contains("submitOrder|payOrder|下单|支付")
        ).len().alias("dmp_下单支付次数"),
        pl.col("dmp_事件名称").n_unique().alias("dmp_事件类型数"),
    ])

    print_step("聚合 DMP 特征", "success", f"{len(dmp_agg):,} 用户")
    return dmp_agg


def merge_data(
    excel_path: Path,
    dmp_path: Path,
    output_path: Path,
    output_format: str = "parquet",
    sheet_limit: int = 2,
    use_cache: bool = True,
    force_refresh_cache: bool = False,
) -> Path:
    """
    合并 Excel 和 DMP 数据

    Args:
        excel_path: Excel 文件路径
        dmp_path: DMP 数据文件路径
        output_path: 输出文件路径
        output_format: 输出格式 (parquet/csv)
        sheet_limit: 最多读取的 Sheet 数量
        use_cache: 是否使用 Parquet 缓存
        force_refresh_cache: 是否强制刷新缓存

    Returns:
        输出文件路径
    """
    import polars as pl

    print("=" * 60)
    print("数据合并脚本")
    print("=" * 60)

    # 1. 读取 Excel（支持缓存）
    if use_cache:
        print_step("准备 Excel 缓存", "running", str(excel_path))

        # 获取 Sheet 列表
        from openpyxl import load_workbook
        wb = load_workbook(str(excel_path), read_only=True)
        all_sheets = wb.sheetnames
        wb.close()

        print(f"  Sheet 列表: {all_sheets}")

        # 读取数据 sheet（排除 Query）
        data_sheets = [s for s in all_sheets if 'query' not in s.lower()][:sheet_limit]
        print(f"  待合并 Sheet: {data_sheets}")

        # 使用缓存读取每个 Sheet
        cache_dir = default_config.cache_dir
        dfs = []

        for sheet in data_sheets:
            print(f"    处理 {sheet}...")

            # 获取或创建该 Sheet 的 Parquet 缓存
            cache_path = get_or_create_parquet_cache(
                source_path=excel_path,
                cache_dir=cache_dir,
                engine="openpyxl",
                force_refresh=force_refresh_cache,
                sheet_name=sheet,
            )

            # 从 Parquet 缓存读取（内存高效）
            df = pl.read_parquet(str(cache_path))
            print(f"      {len(df):,} 行, {len(df.columns)} 列")
            dfs.append(df)

        # 合并
        if len(dfs) > 1:
            clue_df = pl.concat(dfs)
        else:
            clue_df = dfs[0]

        # 释放中间数据
        dfs.clear()
        del dfs

        print_step("准备 Excel 缓存", "success", f"{len(clue_df):,} 行, {len(clue_df.columns)} 列")

    else:
        # 不使用缓存，直接读取 Excel（原逻辑）
        print_step("读取 Excel", "running", str(excel_path))

        from openpyxl import load_workbook
        wb = load_workbook(str(excel_path), read_only=True)
        all_sheets = wb.sheetnames
        wb.close()

        print(f"  Sheet 列表: {all_sheets}")

        # 读取数据 sheet（排除 Query）
        data_sheets = [s for s in all_sheets if 'query' not in s.lower()][:sheet_limit]
        print(f"  待合并 Sheet: {data_sheets}")

        dfs = []
        for sheet in data_sheets:
            print(f"    读取 {sheet}...")
            # 优先使用 openpyxl（更稳定），fastexcel 作为备选
            df = None
            last_error = None
            for engine in ["openpyxl", "fastexcel"]:
                try:
                    df = pl.read_excel(str(excel_path), sheet_name=sheet, engine=engine)
                    print(f"      使用 {engine} 引擎")
                    break
                except FileNotFoundError:
                    # 文件不存在，直接抛出，不需要尝试其他引擎
                    raise
                except PermissionError:
                    # 权限问题，直接抛出
                    raise
                except Exception as e:
                    # 格式错误、Sheet 不存在等，尝试其他引擎
                    last_error = e
                    continue
            if df is None:
                raise RuntimeError(f"无法读取 Sheet: {sheet}, 最后错误: {last_error}")
            print(f"      {len(df):,} 行, {len(df.columns)} 列")
            dfs.append(df)

        # 合并
        if len(dfs) > 1:
            clue_df = pl.concat(dfs)
        else:
            clue_df = dfs[0]

        # 释放中间数据
        dfs.clear()
        del dfs

        print_step("读取 Excel", "success", f"{len(clue_df):,} 行, {len(clue_df.columns)} 列")

    # 2. 读取 DMP 数据
    print_step("读取 DMP", "running", str(dmp_path))

    dmp_df = pl.read_csv(
        str(dmp_path),
        separator="\t",
        has_header=False,
        new_columns=DMP_COLUMNS,
        dtypes={"dmp_手机号": pl.Utf8}
    )

    print_step("读取 DMP", "success", f"{len(dmp_df):,} 行")

    # 3. 聚合 DMP 特征
    dmp_agg = aggregate_dmp_features(pl, dmp_df)

    # 释放原始 DMP 数据
    del dmp_df

    # 4. 关联数据
    print_step("关联数据", "running")

    phone_col = find_phone_column(clue_df.columns)
    if phone_col is None:
        phone_col = "手机号（脱敏）"
        print(f"  警告: 未找到手机号列，使用默认: {phone_col}")
    else:
        print(f"  手机号列: {phone_col}")

    clue_df = clue_df.with_columns(pl.col(phone_col).cast(pl.Utf8).alias("_key"))
    dmp_agg = dmp_agg.with_columns(pl.col("dmp_手机号").cast(pl.Utf8).alias("_key"))

    result = clue_df.join(dmp_agg, on="_key", how="left").drop("_key")

    matched = result.filter(pl.col("dmp_行为次数").is_not_null()).height
    match_rate = matched / len(clue_df) * 100

    print_step("关联数据", "success", f"匹配 {matched:,}/{len(clue_df):,} ({match_rate:.1f}%)")

    # 5. 保存结果
    print_step("保存结果", "running", str(output_path))

    if output_format == "parquet" and output_path.suffix != ".parquet":
        output_path = output_path.with_suffix(".parquet")

    save_data(result, output_path)

    print_step("保存结果", "success", f"{format_size(output_path)}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("合并完成")
    print("=" * 60)
    print(f"输出文件: {output_path}")
    print(f"文件大小: {format_size(output_path)}")
    print(f"数据量: {len(result):,} 行, {len(result.columns)} 列")
    print(f"DMP 匹配率: {match_rate:.1f}%")
    print("=" * 60)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据合并脚本 - 合并 Excel 线索宽表和 DMP 行为数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    uv run python scripts/pipeline/01_merge.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/merged.parquet

    # 指定输出格式
    uv run python scripts/pipeline/01_merge.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/merged.csv \\
        --format csv

    # 强制刷新 Excel 缓存
    uv run python scripts/pipeline/01_merge.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/merged.parquet \\
        --force-refresh-cache

    # 禁用缓存，直接读取 Excel
    uv run python scripts/pipeline/01_merge.py \\
        --excel ./data/线索宽表.xlsx \\
        --dmp ./data/DMP行为数据.csv \\
        --output ./data/merged.parquet \\
        --no-cache
        """,
    )

    parser.add_argument(
        "--excel", "-e",
        required=True,
        help="Excel 文件路径（线索宽表）"
    )
    parser.add_argument(
        "--dmp", "-d",
        required=True,
        help="DMP 行为数据文件路径（CSV/TSV）"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/merged.parquet",
        help="输出文件路径（默认: data/merged.parquet）"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="输出格式（默认: parquet）"
    )
    parser.add_argument(
        "--sheet-limit",
        type=int,
        default=2,
        help="最多读取的 Sheet 数量（默认: 2）"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用 Parquet 缓存，直接读取 Excel"
    )
    parser.add_argument(
        "--force-refresh-cache",
        action="store_true",
        help="强制刷新 Excel 的 Parquet 缓存"
    )

    args = parser.parse_args()

    excel_path = Path(args.excel)
    dmp_path = Path(args.dmp)
    output_path = Path(args.output)

    # 检查输入文件
    if not excel_path.exists():
        print(f"❌ Excel 文件不存在: {excel_path}")
        return 1
    if not dmp_path.exists():
        print(f"❌ DMP 文件不存在: {dmp_path}")
        return 1

    try:
        merge_data(
            excel_path=excel_path,
            dmp_path=dmp_path,
            output_path=output_path,
            output_format=args.format,
            sheet_limit=args.sheet_limit,
            use_cache=not args.no_cache,
            force_refresh_cache=args.force_refresh_cache,
        )
        return 0

    except Exception as e:
        print(f"❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
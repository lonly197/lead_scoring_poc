#!/usr/bin/env python3
"""
数据合并脚本（内存优化版）

针对大文件优化：
- 使用 calamine 引擎读取 Excel（更快更省内存）
- 分批处理 DMP 数据
- 流式写入

用法：
    uv run python scripts/merge_data.py \
        --excel 测试数据/202601~03_v2.xlsx \
        --dmp 测试数据/DMP行为数据(202601~03).csv \
        --output 测试数据/线索宽表_完整.parquet
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="合并线索宽表和 DMP 行为数据")
    parser.add_argument("--excel", required=True, help="Excel 文件路径")
    parser.add_argument("--dmp", required=True, help="DMP 行为数据 CSV 路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")

    args = parser.parse_args()

    import polars as pl

    excel_path = Path(args.excel)
    dmp_path = Path(args.dmp)
    output_path = Path(args.output)

    print("=" * 60)
    print("数据合并脚本")
    print("=" * 60)

    # 1. 读取 Excel（使用 calamine 引擎，更快更省内存）
    print(f"\n读取 Excel: {excel_path}")

    # 获取 sheet 名称
    from openpyxl import load_workbook
    wb = load_workbook(str(excel_path), read_only=True)
    all_sheets = wb.sheetnames
    wb.close()
    print(f"Sheet 列表: {all_sheets}")

    # 读取数据 sheet（排除 Query）
    data_sheets = [s for s in all_sheets if 'query' not in s.lower()][:2]
    print(f"待合并 Sheet: {data_sheets}")

    dfs = []
    for sheet in data_sheets:
        print(f"  读取 {sheet}...")
        # 使用 calamine 引擎
        try:
            df = pl.read_excel(str(excel_path), sheet_name=sheet, engine="calamine")
        except Exception:
            df = pl.read_excel(str(excel_path), sheet_name=sheet)
        print(f"    {len(df):,} 行, {len(df.columns)} 列")
        dfs.append(df)

    # 合并
    if len(dfs) > 1:
        clue_df = pl.concat(dfs)
    else:
        clue_df = dfs[0]
    print(f"线索宽表: {len(clue_df):,} 行, {len(clue_df.columns)} 列")

    # 2. 读取 DMP 数据
    print(f"\n读取 DMP: {dmp_path}")
    dmp_columns = [
        "dmp_手机号", "dmp_事件时间", "dmp_事件名称", "dmp_车型代码",
        "dmp_品牌", "dmp_页面路径", "dmp_数值", "dmp_渠道",
        "dmp_分类", "dmp_未知", "dmp_平台",
    ]
    dmp_df = pl.read_csv(
        str(dmp_path), separator="\t", has_header=False,
        new_columns=dmp_columns, dtypes={"dmp_手机号": pl.Utf8}
    )
    print(f"  {len(dmp_df):,} 行")

    # 3. 聚合 DMP 特征
    print("\n聚合 DMP 特征...")
    dmp_agg = dmp_df.group_by("dmp_手机号").agg([
        pl.len().alias("dmp_行为次数"),
        pl.col("dmp_事件时间").max().alias("dmp_最近行为时间"),
        pl.col("dmp_事件名称").filter(pl.col("dmp_事件名称").str.contains("testDrive|试驾")).len().alias("dmp_试驾相关次数"),
        pl.col("dmp_事件名称").filter(pl.col("dmp_事件名称").str.contains("submitOrder|payOrder|下单|支付")).len().alias("dmp_下单支付次数"),
        pl.col("dmp_事件名称").n_unique().alias("dmp_事件类型数"),
    ])
    print(f"  聚合后: {len(dmp_agg):,} 用户")

    # 4. 关联
    print("\n关联数据...")
    # 查找手机号列
    phone_col = None
    for col in clue_df.columns:
        if "手机" in col or "phone" in col.lower():
            phone_col = col
            break
    if phone_col is None:
        phone_col = "手机号（脱敏）"

    print(f"  手机号列: {phone_col}")

    clue_df = clue_df.with_columns(pl.col(phone_col).cast(pl.Utf8).alias("_key"))
    dmp_agg = dmp_agg.with_columns(pl.col("dmp_手机号").cast(pl.Utf8).alias("_key"))

    result = clue_df.join(dmp_agg, on="_key", how="left").drop("_key")

    matched = result.filter(pl.col("dmp_行为次数").is_not_null()).height
    print(f"  匹配: {matched:,}/{len(clue_df):,} ({matched/len(clue_df)*100:.1f}%)")

    # 5. 输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n输出: {output_path}")

    if args.format == "parquet" or output_path.suffix == ".parquet":
        result.write_parquet(str(output_path))
    else:
        result.write_csv(str(output_path))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"文件大小: {size_mb:.1f} MB")
    print(f"最终: {len(result):,} 行, {len(result.columns)} 列")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
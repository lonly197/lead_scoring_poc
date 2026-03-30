#!/usr/bin/env python3
"""
数据脱敏脚本

功能：
- 品牌关键词替换
- ID 字段掩码
- 手机号/身份证脱敏

用法：
    uv run python scripts/pipeline/04_desensitize.py \\
        --input ./data/cleaned.parquet \\
        --output ./data/desensitized.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import load_data, save_data, print_step, format_size
from src.pipeline.config import BRAND_MAPPING, ID_MASK_COLUMNS, BRAND_TEXT_COLUMNS


# ==================== 脱敏函数 ====================

def mask_id(value: str) -> str:
    """ID 脱敏：保留前2后2位"""
    if not value or len(str(value)) <= 4:
        return "****"
    value = str(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def mask_phone(value: str) -> str:
    """手机号脱敏：保留前3后4位"""
    if not value:
        return value
    # 匹配11位手机号
    phone_pattern = re.compile(r"1[3-9]\d{9}")
    return phone_pattern.sub(
        lambda m: f"{m.group()[:3]}****{m.group()[-4:]}",
        str(value)
    )


def mask_id_card(value: str) -> str:
    """身份证脱敏：保留前6后4位"""
    if not value:
        return value
    # 匹配15或18位身份证
    id_pattern = re.compile(r"\d{15}[\dXx]?[\dXx]?[\dXx]?")
    return id_pattern.sub(
        lambda m: f"{m.group()[:6]}********{m.group()[-4:]}",
        str(value)
    )


def replace_brand_keywords(text: str, brand_mapping: dict = None) -> str:
    """替换品牌关键词"""
    if not text:
        return text
    result = str(text)
    mapping = brand_mapping or BRAND_MAPPING
    for keyword, replacement in mapping.items():
        result = result.replace(keyword, replacement)
    return result


def desensitize_column(
    pl,
    series: "pl.Series",
    col_name: str,
    brand_mapping: dict = None,
) -> "pl.Series":
    """
    对单列进行脱敏处理

    Args:
        pl: polars 模块
        series: polars Series
        col_name: 列名
        brand_mapping: 品牌关键词映射

    Returns:
        脱敏后的 Series
    """
    mapping = brand_mapping or BRAND_MAPPING

    # 品牌关键词替换
    for keyword, replacement in mapping.items():
        series = series.str.replace_all(keyword, replacement)

    # ID 字段掩码
    if "客户ID" in col_name:
        series = series.cast(pl.Utf8).map_elements(mask_id, return_dtype=pl.Utf8)

    # 手机号和身份证脱敏（跟进记录等文本中可能包含）
    if "跟进" in col_name or "战败" in col_name or "记录" in col_name:
        series = series.map_elements(mask_phone, return_dtype=pl.Utf8)
        series = series.map_elements(mask_id_card, return_dtype=pl.Utf8)

    return series


def desensitize_data(
    pl,
    df: "pl.DataFrame",
    columns: Optional[List[str]] = None,
    brand_mapping: dict = None,
) -> "pl.DataFrame":
    """
    数据脱敏处理

    Args:
        pl: polars 模块
        df: 原始 DataFrame
        columns: 指定脱敏列（可选，默认自动检测）
        brand_mapping: 品牌关键词映射

    Returns:
        脱敏后的 DataFrame
    """
    print_step("执行脱敏", "running")

    # 自动检测需要脱敏的列
    if columns is None:
        columns = []
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                # 品牌关键词替换
                if any(kw in col for kw in ["车型", "渠道", "跟进", "战败", "品牌", "dmp_"]):
                    columns.append(col)
                # ID 掩码
                elif "客户ID" in col:
                    columns.append(col)

    print(f"  待脱敏列数: {len(columns)}")

    # 批量处理
    for col in columns:
        df = df.with_columns(
            desensitize_column(pl, df[col], col, brand_mapping).alias(col)
        )

    print_step("执行脱敏", "success", f"处理 {len(columns)} 列")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据脱敏脚本 - 品牌关键词替换 + ID/手机号掩码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
脱敏规则：
  品牌关键词:
    - 广汽丰田/广丰 → 品牌A
    - 广汽 → 集团A
    - GTMC → 代号G
    - 广汽本地 → 区域A

  ID 字段: 保留前2后2位（如 AB****XY）
  手机号: 保留前3后4位（如 138****1234）
  身份证: 保留前6后4位

示例:
    uv run python scripts/pipeline/04_desensitize.py \\
        --input ./data/cleaned.parquet \\
        --output ./data/desensitized.parquet

    # 指定脱敏列
    uv run python scripts/pipeline/04_desensitize.py \\
        --input ./data/cleaned.parquet \\
        --output ./data/desensitized.parquet \\
        --columns 客户ID 首触跟进记录
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
        "--columns", "-c",
        nargs="+",
        default=None,
        help="指定脱敏列（可选，默认自动检测）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1

    try:
        import polars as pl

        print("=" * 60)
        print("数据脱敏脚本")
        print("=" * 60)

        # 加载数据
        print_step("加载数据", "running", str(input_path))
        df = load_data(input_path, engine="polars")
        print_step("加载数据", "success", f"{len(df):,} 行, {len(df.columns)} 列")

        # 执行脱敏
        df_desensitized = desensitize_data(
            pl=pl,
            df=df,
            columns=args.columns,
        )

        # 保存结果
        print_step("保存结果", "running", str(output_path))
        save_data(df_desensitized, output_path)
        print_step("保存结果", "success", format_size(output_path))

        # 打印摘要
        print("\n" + "=" * 60)
        print("脱敏完成")
        print("=" * 60)
        print(f"输出文件: {output_path}")
        print(f"文件大小: {format_size(output_path)}")
        print(f"数据量: {len(df_desensitized):,} 行, {len(df_desensitized.columns)} 列")
        print("=" * 60)

        # 释放内存
        del df, df_desensitized

        return 0

    except Exception as e:
        print(f"❌ 脱敏失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
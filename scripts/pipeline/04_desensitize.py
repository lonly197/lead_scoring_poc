#!/usr/bin/env python3
"""
数据脱敏脚本（DuckDB 优化版）

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
import sys
import time
from pathlib import Path
from typing import List, Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import print_step, format_size
from src.pipeline.config import (
    BRAND_MAPPING,
    CAR_MODEL_MAPPING,
    BRAND_TEXT_COLUMNS,
    CAR_MODEL_COLUMNS,
    ID_MASK_COLUMNS,
    JSON_SENSITIVE_PATTERNS,
)


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def desensitize_with_duckdb(
    input_path: Path,
    output_path: Path,
    brand_mapping: dict = None,
    car_model_mapping: dict = None,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> Path:
    """
    使用 DuckDB 进行数据脱敏

    优势：
    - 向量化正则处理，比逐行快 10-50 倍
    - 内存可控
    - SQL 表达简洁

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        brand_mapping: 品牌关键词映射
        car_model_mapping: 车型名称映射
        memory_limit: DuckDB 内存限制
        threads: 线程数

    Returns:
        输出文件路径
    """
    import duckdb

    start_time = time.time()
    mapping = brand_mapping or BRAND_MAPPING
    car_mapping = car_model_mapping or CAR_MODEL_MAPPING

    log("=" * 60)
    log("数据脱敏（DuckDB 优化版）")
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

        # 获取列信息（DESCRIBE 返回 6 列：列名、类型、nullable 等）
        cols_info = con.execute("DESCRIBE source").fetchall()
        total_cols = len(cols_info)
        log(f"  列数: {total_cols}")

        # 2. 构建 SQL SELECT 表达式
        log("\n构建脱敏表达式...")

        select_exprs = []
        desensitized_cols = []

        for col_info in cols_info:
            col_name = col_info[0]
            col_type = col_info[1]
            expr = f'"{col_name}"'

            # === 处理整数类型的手机号列 ===
            if col_type in ['BIGINT', 'INTEGER', 'INT'] and ('手机' in col_name or 'phone' in col_name.lower()):
                # 保留前3后4位：189****1234
                expr = f"""
                    CASE
                        WHEN LENGTH(CAST("{col_name}" AS VARCHAR)) = 11
                        THEN CONCAT(
                            LEFT(CAST("{col_name}" AS VARCHAR), 3),
                            '****',
                            RIGHT(CAST("{col_name}" AS VARCHAR), 4)
                        )
                        ELSE CAST("{col_name}" AS VARCHAR)
                    END
                """
                if col_name not in desensitized_cols:
                    desensitized_cols.append(col_name)

            # 字符串列脱敏
            if col_type == 'VARCHAR' or col_type == 'TEXT':
                # === 1. 品牌关键词替换 ===
                # 优先使用配置中的列名列表，再使用关键词匹配作为补充
                needs_brand_replace = (
                    col_name in BRAND_TEXT_COLUMNS or
                    any(kw in col_name for kw in ["车型", "渠道", "跟进", "战败", "品牌", "dmp_", "标签", "JSON", "json", "销售", "经销", "店"])
                )
                if needs_brand_replace:
                    for keyword, replacement in mapping.items():
                        expr = f"regexp_replace({expr}, '{keyword}', '{replacement}', 'g')"
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

                # === 2. 车型名称替换 ===
                needs_car_replace = (
                    col_name in CAR_MODEL_COLUMNS or
                    "车型" in col_name or
                    "意向" in col_name or
                    "json" in col_name.lower()
                )
                if needs_car_replace:
                    for keyword, replacement in car_mapping.items():
                        expr = f"regexp_replace({expr}, '{keyword}', '{replacement}', 'g')"
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

                # === 3. ID 字段掩码 ===
                if col_name in ID_MASK_COLUMNS or "客户ID" in col_name:
                    # 保留前2后2位
                    expr = f"""
                        CASE
                            WHEN LENGTH("{col_name}") <= 4 THEN '****'
                            ELSE CONCAT(
                                LEFT("{col_name}", 2),
                                REPEAT('*', LENGTH("{col_name}") - 4),
                                RIGHT("{col_name}", 2)
                            )
                        END
                    """
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

                # === 4. 手机号脱敏 ===
                # 列名包含"手机"的字段，整体脱敏
                if "手机" in col_name or "phone" in col_name.lower():
                    # 保留前3后4位：189****1234
                    expr = f"""
                        CASE
                            WHEN LENGTH(CAST("{col_name}" AS VARCHAR)) = 11
                            THEN CONCAT(
                                LEFT(CAST("{col_name}" AS VARCHAR), 3),
                                '****',
                                RIGHT(CAST("{col_name}" AS VARCHAR), 4)
                            )
                            ELSE CAST("{col_name}" AS VARCHAR)
                        END
                    """
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

                # === 5. 文本中的手机号正则替换 ===
                # 跟进记录、战败等文本中的手机号用简单掩码
                if "跟进" in col_name or "战败" in col_name or "记录" in col_name:
                    # 将11位手机号替换为星号掩码（前3后4）
                    # DuckDB 不支持复杂替换，使用分段匹配
                    expr = f"regexp_replace({expr}, '(1[3-9][0-9])([0-9]{{4}})([0-9]{{4}})', '\\1****\\3', 'g')"
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

                # === 6. 身份证号脱敏 ===
                if "身份证" in col_name or "证件" in col_name:
                    # 18位身份证保留前4后4位
                    expr = f"""
                        CASE
                            WHEN LENGTH(CAST("{col_name}" AS VARCHAR)) >= 15
                            THEN CONCAT(
                                LEFT(CAST("{col_name}" AS VARCHAR), 4),
                                '**********',
                                RIGHT(CAST("{col_name}" AS VARCHAR), 4)
                            )
                            ELSE CAST("{col_name}" AS VARCHAR)
                        END
                    """
                    if col_name not in desensitized_cols:
                        desensitized_cols.append(col_name)

            select_exprs.append(f"{expr} AS \"{col_name}\"")

        log(f"  待脱敏列: {len(desensitized_cols)} 个")

        # 3. 执行脱敏并输出
        log("\n执行脱敏...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        select_sql = ", ".join(select_exprs)
        con.execute(f"""
            COPY (
                SELECT {select_sql}
                FROM source
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # 4. 统计
        row_count = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
        elapsed = time.time() - start_time
        output_size_mb = output_path.stat().st_size / 1024 / 1024

        log("")
        log("=" * 60)
        log("脱敏完成")
        log("=" * 60)
        log(f"输出文件: {output_path}")
        log(f"文件大小: {output_size_mb:.1f} MB")
        log(f"数据量: {row_count:,} 行, {total_cols} 列")
        log(f"脱敏列数: {len(desensitized_cols)}")
        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return output_path

    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据脱敏脚本 - 品牌关键词/车型替换 + ID/手机号/身份证掩码（DuckDB 优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
脱敏规则：
  品牌关键词:
    - 广汽丰田/广丰 → 品牌A
    - 丰田 → 品牌B
    - 广汽 → 集团A
    - GTMC → 代号G
    - 广汽本地 → 区域A

  车型名称:
    - 钂智3X/钂智4X → 车型A/B
    - 凯美瑞 → 车型D
    - 汉兰达 → 车型E
    - 其他车型按配置映射

  ID 字段: 保留前2后2位（如 AB****XY）
  手机号: 保留前3后4位（如 138****1234）
  身份证: 保留前4后4位

示例:
    uv run python scripts/pipeline/04_desensitize.py \\
        --input ./data/线索宽表_带标签.parquet \\
        --output ./data/线索宽表_脱敏.parquet

    # 自定义内存限制
    uv run python scripts/pipeline/04_desensitize.py \\
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
        desensitize_with_duckdb(
            input_path=input_path,
            output_path=output_path,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        return 0

    except Exception as e:
        log(f"❌ 脱敏失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
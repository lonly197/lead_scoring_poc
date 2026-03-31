#!/usr/bin/env python3
"""
试驾后下订数据集生成脚本

功能：
- 从原始线索宽表筛选已试驾样本
- 计算试驾后下订天数差（下订时间 - 试驾时间）
- 生成时间窗口二分类标签：7天内、14天内、21天内下订
- 删除泄漏字段
- 支持 OOT 时间切分

业务规则（参照 O:H:A:B定级业务规则）：
- H级：7天内下订，下次联络最晚时间 < 2天
- A级：14天内下订，下次联络最晚时间 < 5天
- B级：21天内下订，下次联络最晚时间 < 7天

用法：
    # 完整流程：筛选 + 计算标签 + 拆分
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/order_after_drive

    # 仅筛选已试驾样本
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/driven_only.parquet \\
        --step filter

    # OOT 时间切分
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/order_after_drive \\
        --split-mode oot \\
        --time-column 试驾时间
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import format_size
from src.pipeline.config import (
    BRAND_MAPPING,
    CAR_MODEL_MAPPING,
    BRAND_TEXT_COLUMNS,
    CAR_MODEL_COLUMNS,
    CITY_COLUMNS,
    PROVINCE_COLUMNS,
    DEALER_CODE_COLUMNS,
    JSON_SENSITIVE_PATTERNS,
)


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# 试驾后下订阶段的泄漏字段（比试驾前阶段少，因为下订时间是目标来源）
LEAKAGE_COLUMNS_ORDER_STAGE = [
    # 直接编码目标（致命泄漏）
    "下定状态",
    "意向金支付状态",
    "战败原因",
    "战败次数",
    # 后验信息
    "线索当前意向级别",
    "线索评级结果（最新）",
    "线索评级变化时间",
    "意向级别",
    "购车阶段",
    "线索评级结果",
    "线索评级_试驾前",
    "线索评级_试驾后",
]

# ID 列（始终删除）
ID_COLUMNS = [
    "线索唯一ID",
    "客户ID",
    "客户ID(店端)",
    "店端潜客编号",
    "手机号码",
    "手机号（脱敏）",
    "主键",
]


def filter_driven_samples(
    con,
    input_path: Path,
    output_path: Path,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    筛选已试驾样本

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        output_path: 输出文件路径
        compression: 压缩算法

    Returns:
        (原始行数, 筛选后行数)
    """
    log("筛选已试驾样本...")

    # 读取原始数据
    con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{input_path}')")

    total_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]

    # 筛选已试驾样本（试驾时间不为空）
    con.execute(f"""
        COPY (
            SELECT * FROM source
            WHERE "试驾时间" IS NOT NULL
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    driven_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]

    log(f"  原始数据: {total_rows:,} 行")
    log(f"  已试驾样本: {driven_rows:,} 行 ({driven_rows/total_rows*100:.2f}%)")

    return total_rows, driven_rows


def compute_order_labels(
    con,
    input_path: Path,
    output_path: Path,
    compression: str = "zstd",
) -> Tuple[int, dict]:
    """
    计算试驾后下订标签

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径（已试驾样本）
        output_path: 输出文件路径
        compression: 压缩算法

    Returns:
        (总行数, 标签分布字典)
    """
    log("计算试驾后下订标签...")

    con.execute(f"CREATE VIEW driven AS SELECT * FROM read_parquet('{input_path}')")

    total_rows = con.execute("SELECT COUNT(*) FROM driven").fetchone()[0]

    # 计算试驾后下订天数差并生成标签
    #
    # 标签定义：
    # - 下订标签_7天 = 1: 试驾后 0-7 天内下订（试驾后下订天数差 >= 0 且 <= 7）
    # - 下订标签_7天 = 0: 未下订或试驾前下订或超过 7 天才下订
    #
    # 注意：试驾后下订天数差 < 0 表示试驾前已下订，应视为负样本
    #
    # 使用 DuckDB 的 DATE_DIFF 函数计算天数差
    #
    con.execute(f"""
        COPY (
            SELECT
                *,
                -- 试驾后下订天数差
                CASE
                    WHEN "下订时间" IS NULL THEN NULL
                    ELSE DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    )
                END AS 试驾后下订天数差,

                -- 时间窗口二分类标签（必须是试驾后下订，天数差 >= 0）
                CASE
                    WHEN "下订时间" IS NULL THEN 0
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) < 0 THEN 0  -- 试驾前下订，视为负样本
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) <= 7 THEN 1
                    ELSE 0
                END AS 下订标签_7天,

                CASE
                    WHEN "下订时间" IS NULL THEN 0
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) < 0 THEN 0  -- 试驾前下订，视为负样本
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) <= 14 THEN 1
                    ELSE 0
                END AS 下订标签_14天,

                CASE
                    WHEN "下订时间" IS NULL THEN 0
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) < 0 THEN 0  -- 试驾前下订，视为负样本
                    WHEN DATE_DIFF('day',
                        CAST("试驾时间" AS DATE),
                        CAST("下订时间" AS DATE)
                    ) <= 21 THEN 1
                    ELSE 0
                END AS 下订标签_21天

            FROM driven
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 统计标签分布
    label_7_dist = con.execute(f"""
        SELECT 下订标签_7天, COUNT(*) as cnt
        FROM read_parquet('{output_path}')
        GROUP BY 下订标签_7天
    """).fetchall()

    label_14_dist = con.execute(f"""
        SELECT 下订标签_14天, COUNT(*) as cnt
        FROM read_parquet('{output_path}')
        GROUP BY 下订标签_14天
    """).fetchall()

    label_21_dist = con.execute(f"""
        SELECT 下订标签_21天, COUNT(*) as cnt
        FROM read_parquet('{output_path}')
        GROUP BY 下订标签_21天
    """).fetchall()

    log(f"  7天内下订: {dict(label_7_dist)}")
    log(f"  14天内下订: {dict(label_14_dist)}")
    log(f"  21天内下订: {dict(label_21_dist)}")

    return total_rows, {
        "label_7": dict(label_7_dist),
        "label_14": dict(label_14_dist),
        "label_21": dict(label_21_dist),
    }


def remove_leakage_columns(
    con,
    input_path: Path,
    output_path: Path,
    keep_columns: List[str] = None,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    删除泄漏字段

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        output_path: 输出文件路径
        keep_columns: 保留的列名列表
        compression: 压缩算法

    Returns:
        (原始列数, 删除后列数)
    """
    log("删除泄漏字段...")

    keep_columns = keep_columns or []

    con.execute(f"CREATE VIEW labeled AS SELECT * FROM read_parquet('{input_path}')")

    # 获取列信息
    cols_info = con.execute("DESCRIBE labeled").fetchall()
    col_names = [col[0] for col in cols_info]
    original_cols = len(col_names)

    # 合并要删除的列
    cols_to_remove = set(LEAKAGE_COLUMNS_ORDER_STAGE + ID_COLUMNS)

    # 移除保留列
    for col in keep_columns:
        cols_to_remove.discard(col)

    # 只删除存在的列
    existing_to_remove = [c for c in cols_to_remove if c in col_names]

    log(f"  删除 {len(existing_to_remove)} 个泄漏字段")

    # 构建保留列列表
    keep_cols = [c for c in col_names if c not in cols_to_remove]

    # 输出
    con.execute(f"""
        COPY (
            SELECT {', '.join(f'"{c}"' for c in keep_cols)}
            FROM labeled
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 获取新列数
    new_cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{output_path}')").fetchall()
    new_cols = len(new_cols_info)

    log(f"  列数: {original_cols} → {new_cols}")

    return original_cols, new_cols


def desensitize_data(
    con,
    input_path: Path,
    output_path: Path,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    数据脱敏处理

    脱敏内容：
    1. 品牌关键词替换：广汽丰田→品牌A, 广丰→品牌A, 广汽→集团A, GTMC→代号G, 丰田→品牌B
    2. 车型名称替换：铂智3X→车型A 等
    3. 城市名称脱敏：动态编号为城市A、城市B 等
    4. 省份名称脱敏：动态编号为省份A、省份B 等
    5. 经销店代码掩码：保留前2后2位
    6. JSON 中敏感信息：车牌号、手机号等

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        output_path: 输出文件路径
        compression: 压缩算法

    Returns:
        (处理前敏感字段数, 处理后敏感字段数)
    """
    log("数据脱敏处理...")

    con.execute(f"CREATE VIEW to_desensitize AS SELECT * FROM read_parquet('{input_path}')")

    # 获取列信息
    cols_info = con.execute("DESCRIBE to_desensitize").fetchall()
    col_names = [col[0] for col in cols_info]
    col_types = {col[0]: col[1] for col in cols_info}

    # 1. 收集城市和省份的唯一值，生成映射
    city_mapping = {}
    province_mapping = {}

    for col in CITY_COLUMNS:
        if col in col_names:
            cities = con.execute(f'SELECT DISTINCT "{col}" FROM to_desensitize WHERE "{col}" IS NOT NULL').fetchall()
            for city_tuple in cities:
                city = city_tuple[0]  # 从 tuple 中提取值
                if city and city not in city_mapping:
                    idx = len(city_mapping)
                    # 生成城市A、城市B、...、城市Z、城市AA、城市AB 等
                    if idx < 26:
                        city_mapping[city] = f"城市{chr(65 + idx)}"
                    else:
                        city_mapping[city] = f"城市{chr(65 + idx // 26 - 1)}{chr(65 + idx % 26)}"

    for col in PROVINCE_COLUMNS:
        if col in col_names:
            provinces = con.execute(f'SELECT DISTINCT "{col}" FROM to_desensitize WHERE "{col}" IS NOT NULL').fetchall()
            for province_tuple in provinces:
                province = province_tuple[0]  # 从 tuple 中提取值
                if province and province not in province_mapping:
                    idx = len(province_mapping)
                    if idx < 26:
                        province_mapping[province] = f"省份{chr(65 + idx)}"
                    else:
                        province_mapping[province] = f"省份{chr(65 + idx // 26 - 1)}{chr(65 + idx % 26)}"

    log(f"  城市映射: {len(city_mapping)} 个")
    log(f"  省份映射: {len(province_mapping)} 个")

    # 2. 构建 SELECT 表达式
    select_exprs = []
    desensitized_count = 0

    for col_name in col_names:
        col_type = col_types[col_name]
        expr = f'"{col_name}"'

        # 只处理字符串类型
        if col_type not in ('VARCHAR', 'TEXT'):
            select_exprs.append(expr)
            continue

        # 品牌关键词替换（使用配置列表 BRAND_TEXT_COLUMNS）
        if col_name in BRAND_TEXT_COLUMNS:
            # 按长度降序排序，确保长的关键词先被替换
            sorted_brands = sorted(BRAND_MAPPING.keys(), key=len, reverse=True)
            for brand in sorted_brands:
                replacement = BRAND_MAPPING[brand]
                expr = f"regexp_replace({expr}, '{brand}', '{replacement}', 'g')"
            desensitized_count += 1

        # 车型名称替换（使用配置列表 CAR_MODEL_COLUMNS）
        if col_name in CAR_MODEL_COLUMNS:
            sorted_models = sorted(CAR_MODEL_MAPPING.keys(), key=len, reverse=True)
            for model in sorted_models:
                replacement = CAR_MODEL_MAPPING[model]
                expr = f"regexp_replace({expr}, '{model}', '{replacement}', 'g')"
            desensitized_count += 1

        # 城市名称替换
        if col_name in CITY_COLUMNS:
            for city, code in sorted(city_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                expr = f"regexp_replace({expr}, '{city}', '{code}', 'g')"
            desensitized_count += 1

        # 省份名称替换
        if col_name in PROVINCE_COLUMNS:
            for province, code in sorted(province_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                expr = f"regexp_replace({expr}, '{province}', '{code}', 'g')"
            desensitized_count += 1

        # 经销店代码掩码
        if col_name in DEALER_CODE_COLUMNS:
            expr = f"""
                CASE
                    WHEN LENGTH("{col_name}") <= 4 THEN '****'
                    ELSE CONCAT(LEFT("{col_name}", 2), '****', RIGHT("{col_name}", 2))
                END
            """
            desensitized_count += 1

        # JSON 字段中的敏感信息（车牌号、手机号等）
        if "跟进" in col_name or "记录" in col_name or "json" in col_name.lower():
            # 车牌号脱敏：赣ED89157 → 车牌***157
            expr = f"regexp_replace({expr}, '[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{{5,6}}', '车牌***', 'g')"
            # 手机号脱敏：13812345678 → 138****5678
            expr = f"regexp_replace({expr}, '1[3-9][0-9]{{9}}', '手机***********', 'g')"
            desensitized_count += 1

        select_exprs.append(f"{expr} AS \"{col_name}\"")

    log(f"  脱敏字段数: {desensitized_count}")

    # 3. 执行脱敏并输出
    select_sql = ", ".join(select_exprs)
    con.execute(f"""
        COPY (
            SELECT {select_sql}
            FROM to_desensitize
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 获取行数
    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    log(f"  输出数据: {row_count:,} 行")

    return desensitized_count, len(col_names)


def split_data_oot(
    con,
    input_path: Path,
    train_path: Path,
    test_path: Path,
    time_column: str,
    cutoff_date: str,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    OOT 时间切分

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        train_path: 训练集输出路径
        test_path: 测试集输出路径
        time_column: 时间列名
        cutoff_date: 切分时间点
        compression: 压缩算法

    Returns:
        (训练集行数, 测试集行数)
    """
    log(f"OOT 时间切分...")
    log(f"  时间列: {time_column}, 切分点: {cutoff_date}")

    con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{input_path}')")

    # 训练集
    con.execute(f"""
        COPY (
            SELECT * FROM data
            WHERE "{time_column}" < '{cutoff_date}'::TIMESTAMP
        ) TO '{train_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 测试集
    con.execute(f"""
        COPY (
            SELECT * FROM data
            WHERE "{time_column}" >= '{cutoff_date}'::TIMESTAMP
        ) TO '{test_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{train_path}')").fetchone()[0]
    test_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{test_path}')").fetchone()[0]

    log(f"  训练集: {train_rows:,} 行")
    log(f"  测试集: {test_rows:,} 行")

    return train_rows, test_rows


def split_data_random(
    con,
    input_path: Path,
    train_path: Path,
    test_path: Path,
    target_column: str,
    test_ratio: float = 0.2,
    compression: str = "zstd",
) -> Tuple[int, int]:
    """
    随机分层切分

    Args:
        con: DuckDB 连接
        input_path: 输入文件路径
        train_path: 训练集输出路径
        test_path: 测试集输出路径
        target_column: 目标变量列名
        test_ratio: 测试集比例
        compression: 压缩算法

    Returns:
        (训练集行数, 测试集行数)
    """
    log(f"随机分层切分...")
    log(f"  目标列: {target_column}, 测试比例: {test_ratio}")

    con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{input_path}')")

    # 分层采样
    con.execute(f"""
        CREATE VIEW split_view AS
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY "{target_column}" ORDER BY RANDOM()) as _row_num,
               COUNT(*) OVER (PARTITION BY "{target_column}") as _total_in_group
        FROM data
        WHERE "{target_column}" IS NOT NULL
    """)

    # 训练集
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (_row_num, _total_in_group)
            FROM split_view
            WHERE _row_num > CAST(_total_in_group * {test_ratio} AS INTEGER)
        ) TO '{train_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    # 测试集
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (_row_num, _total_in_group)
            FROM split_view
            WHERE _row_num <= CAST(_total_in_group * {test_ratio} AS INTEGER)
        ) TO '{test_path}' (FORMAT PARQUET, COMPRESSION {compression.upper()})
    """)

    train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{train_path}')").fetchone()[0]
    test_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{test_path}')").fetchone()[0]

    log(f"  训练集: {train_rows:,} 行")
    log(f"  测试集: {test_rows:,} 行")

    return train_rows, test_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="试驾后下订数据集生成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
业务规则（试驾后下订商谈阶段）：
  H级: 7天内下订，下次联络最晚时间 < 2天
  A级: 14天内下订，下次联络最晚时间 < 5天
  B级: 21天内下订，下次联络最晚时间 < 7天

示例:
    # 完整流程
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/order_after_drive

    # OOT 时间切分
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/order_after_drive \\
        --split-mode oot \\
        --time-column 试驾时间 \\
        --cutoff 2026-03-15

    # 随机切分（默认）
    uv run python scripts/pipeline/07_order_after_drive.py \\
        --input ./data/线索宽表_合并_补充试驾.parquet \\
        --output ./data/order_after_drive \\
        --split-mode random \\
        --target 下订标签_7天

输出:
    {output}_train.parquet
    {output}_test.parquet
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据文件路径（原始线索宽表）"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件前缀"
    )
    parser.add_argument(
        "--split-mode",
        choices=["oot", "random", "none"],
        default="random",
        help="拆分模式（默认: random）"
    )
    parser.add_argument(
        "--time-column",
        default="试驾时间",
        help="OOT 切分时间列（默认: 试驾时间）"
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help="OOT 切分时间点（格式 YYYY-MM-DD，默认自动计算）"
    )
    parser.add_argument(
        "--target",
        default="下订标签_7天",
        help="分层采样目标列（默认: 下订标签_7天）"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="测试集比例（默认: 0.2）"
    )
    parser.add_argument(
        "--step",
        choices=["all", "filter", "labels", "clean", "desensitize", "split"],
        default="all",
        help="执行步骤（默认: all）"
    )
    parser.add_argument(
        "--keep-order-time",
        action="store_true",
        help="保留下订时间列（用于后续分析，训练时需手动排除）"
    )
    parser.add_argument(
        "--no-desensitize",
        action="store_true",
        help="跳过脱敏步骤（默认执行脱敏）"
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
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "zstd", "lz4"],
        help="Parquet 压缩算法（默认: zstd）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_prefix = Path(args.output)

    # 检查输入文件
    if not input_path.exists():
        log(f"❌ 输入文件不存在: {input_path}")
        return 1

    # 确保输出目录存在
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        import duckdb

        log("=" * 60)
        log("试驾后下订数据集生成")
        log("=" * 60)
        log(f"输入文件: {input_path}")
        log(f"输出前缀: {output_prefix}")
        log(f"拆分模式: {args.split_mode}")

        # 创建 DuckDB 连接
        con = duckdb.connect(":memory:")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"SET threads={args.threads}")

        # 定义中间文件路径
        temp_dir = output_prefix.parent / ".temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        driven_path = temp_dir / "driven.parquet"
        labeled_path = temp_dir / "labeled.parquet"
        cleaned_path = temp_dir / "cleaned.parquet"
        train_path = Path(f"{output_prefix}_train.parquet")
        test_path = Path(f"{output_prefix}_test.parquet")

        # 1. 筛选已试驾样本
        if args.step in ["all", "filter"]:
            total_rows, driven_rows = filter_driven_samples(
                con, input_path, driven_path, args.compression
            )

        # 2. 计算下订标签
        if args.step in ["all", "labels"]:
            if not driven_path.exists():
                log(f"❌ 中间文件不存在，请先执行 filter 步骤")
                return 1
            driven_rows, label_dist = compute_order_labels(
                con, driven_path, labeled_path, args.compression
            )

        # 3. 删除泄漏字段
        if args.step in ["all", "clean"]:
            if not labeled_path.exists():
                log(f"❌ 中间文件不存在，请先执行 labels 步骤")
                return 1

            # 保留下订时间（用于后续分析，但训练时需排除）
            keep_cols = []
            if args.keep_order_time:
                keep_cols.append("下订时间")

            original_cols, new_cols = remove_leakage_columns(
                con, labeled_path, cleaned_path, keep_cols, args.compression
            )

        # 3.5. 数据脱敏
        desensitized_path = temp_dir / "desensitized.parquet"
        if args.step in ["all", "desensitize"]:
            if not cleaned_path.exists():
                log(f"❌ 中间文件不存在，请先执行 clean 步骤")
                return 1

            # 检查是否需要脱敏
            if not args.no_desensitize:
                desensitized_count, total_cols = desensitize_data(
                    con, cleaned_path, desensitized_path, args.compression
                )
                # 后续使用脱敏后的数据
                final_data_path = desensitized_path
            else:
                log("跳过脱敏步骤 (--no-desensitize)")
                final_data_path = cleaned_path
        else:
            final_data_path = cleaned_path

        # 4. 数据拆分
        if args.step in ["all", "split"]:
            if not final_data_path.exists():
                log(f"❌ 中间文件不存在，请先执行 clean/desensitize 步骤")
                return 1

            if args.split_mode == "oot":
                # 自动计算切分点
                if args.cutoff:
                    cutoff = args.cutoff
                else:
                    time_stats = con.execute(f"""
                        SELECT MIN("{args.time_column}"), MAX("{args.time_column}")
                        FROM read_parquet('{final_data_path}')
                        WHERE "{args.time_column}" IS NOT NULL
                    """).fetchone()
                    from datetime import timedelta
                    import pandas as pd
                    min_time_str, max_time_str = time_stats
                    # 解析字符串为 datetime
                    min_time = pd.to_datetime(min_time_str)
                    max_time = pd.to_datetime(max_time_str)
                    total_seconds = (max_time - min_time).total_seconds()
                    cutoff_seconds = total_seconds * (1 - args.ratio)
                    cutoff = (min_time + timedelta(seconds=cutoff_seconds)).strftime("%Y-%m-%d")
                    log(f"自动计算切分点: {cutoff}")

                train_rows, test_rows = split_data_oot(
                    con, final_data_path, train_path, test_path,
                    args.time_column, cutoff, args.compression
                )

            elif args.split_mode == "random":
                train_rows, test_rows = split_data_random(
                    con, final_data_path, train_path, test_path,
                    args.target, args.ratio, args.compression
                )

            else:  # none
                # 仅复制文件
                import shutil
                shutil.copy(final_data_path, train_path)
                train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{train_path}')").fetchone()[0]
                test_rows = 0

        # 清理临时文件
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            log("清理临时文件完成")

        con.close()

        elapsed = time.time() - start_time

        # 打印摘要
        log("")
        log("=" * 60)
        log("数据集生成完成")
        log("=" * 60)

        if args.split_mode != "none":
            log(f"训练集: {train_path}")
            log(f"  文件大小: {format_size(train_path)}")
            log(f"  数据量: {train_rows:,} 行")
            log(f"测试集: {test_path}")
            log(f"  文件大小: {format_size(test_path)}")
            log(f"  数据量: {test_rows:,} 行")
        else:
            log(f"输出文件: {train_path}")
            log(f"  文件大小: {format_size(train_path)}")
            log(f"  数据量: {train_rows:,} 行")

        log(f"总耗时: {elapsed:.1f} 秒")
        log("=" * 60)

        return 0

    except Exception as e:
        log(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
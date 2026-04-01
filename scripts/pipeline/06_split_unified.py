#!/usr/bin/env python3
"""
统一数据拆分脚本

从统一数据源生成训练集和测试集，供试驾预测和下订预测模型共同使用。

拆分策略：
1. OOT 时间切分（Out-of-Time）：按线索创建时间切分
2. 输出两套数据：
   - 全量线索（试驾预测用）：train.parquet / test.parquet
   - 已试驾线索子集（下订预测用）：train_driven.parquet / test_driven.parquet

使用方法：
    uv run python scripts/pipeline/06_split_unified.py \
        --input ./data/线索宽表_合并_补充试驾.parquet \
        --output ./data/unified_split \
        --time-column 线索创建时间 \
        --cutoff 2026-03-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader, FeatureEngineer
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="统一数据拆分")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="线索创建时间",
        help="时间列名（用于 OOT 切分）",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="切分时间点（YYYY-MM-DD），不指定则自动计算 80/20 分割",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径",
    )

    return parser.parse_args()


def split_by_time(
    df: pd.DataFrame,
    time_column: str,
    cutoff: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    OOT 时间切分

    Args:
        df: 数据框
        time_column: 时间列名
        cutoff: 切分时间点

    Returns:
        (train_df, test_df)
    """
    time_col = pd.to_datetime(df[time_column], errors='coerce')

    if cutoff is None:
        # 自动计算 80/20 分割点
        cutoff_date = time_col.quantile(0.8)
        logger.info(f"自动计算切分点: {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        cutoff_date = pd.to_datetime(cutoff)

    train_mask = time_col < cutoff_date
    test_mask = time_col >= cutoff_date

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"训练集: {len(train_df):,} 行 ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"测试集: {len(test_df):,} 行 ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, test_df


def main():
    args = parse_args()

    # 日志配置
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = Path(args.output) / "split.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=str(log_file), level=logging.INFO)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("统一数据拆分")
    logger.info("=" * 60)
    logger.info(f"输入: {input_path}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"时间列: {args.time_column}")
    logger.info(f"切分点: {args.cutoff or '自动'}")

    try:
        # 1. 加载数据（带 auto_adapt）
        logger.info("步骤 1/4: 加载数据")
        loader = DataLoader(str(input_path), auto_adapt=True)
        df = loader.load()
        logger.info(f"数据量: {len(df):,} 行, {len(df.columns)} 列")

        # 2. 时间切分
        logger.info("步骤 2/4: 时间切分")
        train_df, test_df = split_by_time(
            df,
            time_column=args.time_column,
            cutoff=args.cutoff,
        )

        # 3. 保存全量数据（试驾预测用）
        logger.info("步骤 3/4: 保存全量数据（试驾预测用）")
        train_path = output_dir / "train.parquet"
        test_path = output_dir / "test.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        logger.info(f"训练集: {train_path}")
        logger.info(f"测试集: {test_path}")

        # 4. 保存已试驾子集（下订预测用）
        logger.info("步骤 4/4: 保存已试驾子集（下订预测用）")

        # 检查试驾时间列
        if "试驾时间" in train_df.columns:
            train_driven = train_df[train_df["试驾时间"].notna()].copy()
            test_driven = test_df[test_df["试驾时间"].notna()].copy()

            train_driven_path = output_dir / "train_driven.parquet"
            test_driven_path = output_dir / "test_driven.parquet"
            train_driven.to_parquet(train_driven_path, index=False)
            test_driven.to_parquet(test_driven_path, index=False)

            logger.info(f"已试驾训练集: {len(train_driven):,} 行 ({len(train_driven)/len(train_df)*100:.1f}%)")
            logger.info(f"已试驾测试集: {len(test_driven):,} 行 ({len(test_driven)/len(test_df)*100:.1f}%)")
            logger.info(f"已试驾训练集: {train_driven_path}")
            logger.info(f"已试驾测试集: {test_driven_path}")
        else:
            logger.warning("未找到试驾时间列，跳过已试驾子集生成")

        # 5. 保存拆分元数据
        split_info = {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "time_column": args.time_column,
            "cutoff": args.cutoff or "auto",
            "split_time": datetime.now().isoformat(),
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
        with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info("拆分完成!")
        logger.info(f"输出目录: {output_dir}")

        print("\n" + "=" * 60)
        print("统一数据拆分完成")
        print("=" * 60)
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
        print(f"\n试驾预测用: train.parquet / test.parquet")
        print(f"下订预测用: train_driven.parquet / test_driven.parquet")

    except Exception as e:
        logger.error(f"拆分失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
模型预测脚本

对输入数据进行预测，将预测结果追加到 DataFrame 中返回。
支持 OHAB 评级推导：O 级（已成交）/ H 级 / A 级 / B 级 / N 级。

使用方法：
    # 基本用法
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/final_v4_test.parquet \
        --output ./predictions.csv

    # 包含 OHAB 评级
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/final_v4_test.parquet \
        --output ./predictions.csv \
        --include-ohab

    # 返回完整 DataFrame（含原始列 + 预测列）
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/final_v4_test.parquet \
        --output ./predictions.csv \
        --include-original \
        --include-ohab
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer
from src.models.predictor import LeadScoringPredictor

logger = logging.getLogger(__name__)


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def detect_ordered_status(df: pd.DataFrame) -> np.ndarray:
    """
    检测已成交状态（O 级）

    根据以下字段判断线索是否已下定/成交：
    - 下订时间不为空
    - 成交标签 = 1
    - is_final_ordered = 1
    - 下定状态/订单状态包含已下定/已成交关键词
    - 意向金支付状态为已支付
    - 成交日期/结算日期不为空
    - 订单号不为空

    Args:
        df: 输入数据 DataFrame

    Returns:
        布尔数组，True 表示已成交
    """
    is_ordered = np.zeros(len(df), dtype=bool)

    # 1. 检查下订时间（时间字段，有值表示已下订）
    if "下订时间" in df.columns:
        is_ordered |= df["下订时间"].notna()

    # 2. 检查成交标签
    if "成交标签" in df.columns:
        is_ordered |= (df["成交标签"] == 1)

    # 3. 检查 is_final_ordered
    if "is_final_ordered" in df.columns:
        is_ordered |= (df["is_final_ordered"] == 1)

    # 4. 检查下定状态
    if "下定状态" in df.columns:
        ordered_keywords = ["已下定", "已成交", "已订车", "成交", "订车", "已下单"]
        for kw in ordered_keywords:
            is_ordered |= df["下定状态"].astype(str).str.contains(kw, na=False)

    # 5. 检查订单状态
    if "订单状态" in df.columns:
        ordered_keywords = ["已下定", "已成交", "已订车", "成交", "订车", "已完成", "已下单"]
        for kw in ordered_keywords:
            is_ordered |= df["订单状态"].astype(str).str.contains(kw, na=False)

    # 6. 检查意向金支付状态
    if "意向金支付状态" in df.columns:
        paid_keywords = ["已支付", "已付", "支付成功"]
        for kw in paid_keywords:
            is_ordered |= df["意向金支付状态"].astype(str).str.contains(kw, na=False)

    # 7. 检查成交日期
    if "成交日期" in df.columns:
        is_ordered |= df["成交日期"].notna()

    # 8. 检查结算日期
    if "结算日期" in df.columns:
        is_ordered |= df["结算日期"].notna()

    # 9. 检查订单号（有订单号表示已下单）
    for col in ["订单号", "customer_order_no"]:
        if col in df.columns:
            is_ordered |= df[col].notna()

    return is_ordered


def derive_ohab_rating(
    y_proba: np.ndarray,
    is_ordered: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    推导 OHAB 评级

    业务规则（来自《O/H/A/B定级业务规则》）：
    - O 级：已订车、已成交
    - H 级：客户计划 7 天内试驾
    - A 级：客户计划 14 天内试驾
    - B 级：客户计划 21 天内试驾
    - N 级：无意向

    推导逻辑（基于 14 天试驾概率）：
    - O 级：已成交状态（优先级最高）
    - H 级：P(14天试驾) >= threshold 且预测标签 = 1（高置信度正例）
    - A 级：P(14天试驾) >= threshold（中等置信度）
    - B 级：threshold/2 <= P(14天试驾) < threshold（低置信度）
    - N 级：P(14天试驾) < threshold/2（无意向）

    Args:
        y_proba: 预测概率数组（14 天试驾概率）
        is_ordered: 已成交状态数组
        threshold: 评级判定阈值

    Returns:
        OHAB 评级数组
    """
    ratings = np.full(len(y_proba), "N", dtype=object)

    # 1. O 级：已成交（最高优先级）
    ratings[is_ordered] = "O"

    # 2. 非成交样本：根据概率判断 H/A/B/N
    not_ordered = ~is_ordered

    # H 级：高置信度正例（概率 >= threshold 且预测标签 = 1）
    h_mask = not_ordered & (y_proba >= threshold)
    ratings[h_mask] = "H"

    # A 级：中等置信度（threshold * 0.7 <= 概率 < threshold）
    a_mask = not_ordered & (y_proba >= threshold * 0.7) & (y_proba < threshold)
    ratings[a_mask] = "A"

    # B 级：低置信度（threshold * 0.3 <= 概率 < threshold * 0.7）
    b_mask = not_ordered & (y_proba >= threshold * 0.3) & (y_proba < threshold * 0.7)
    ratings[b_mask] = "B"

    # N 级：无意向（概率 < threshold * 0.3）- 已默认

    return ratings


def predict(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None,
    include_original: bool = False,
    include_ohab: bool = False,
    ohab_threshold: float = 0.5,
    id_column: str = "线索唯一ID",
) -> pd.DataFrame:
    """
    对数据进行预测

    Args:
        model_path: 模型路径
        data_path: 数据文件路径
        output_path: 输出文件路径（可选）
        include_original: 是否包含原始列
        include_ohab: 是否包含 OHAB 评级
        ohab_threshold: OHAB 评级判定阈值
        id_column: ID 列名（用于标识记录）

    Returns:
        追加预测结果的 DataFrame
    """
    model_path = Path(model_path)
    data_path = Path(data_path)

    # 1. 加载模型元数据
    logger.info("加载模型元数据...")
    metadata = load_feature_metadata(model_path)
    interaction_context = metadata.get("interaction_context", {})

    # 2. 加载模型
    logger.info(f"加载模型: {model_path}")
    predictor = LeadScoringPredictor.load(str(model_path))
    target = predictor.label
    logger.info(f"目标变量: {target}")

    # 3. 加载数据（自动适配）
    logger.info(f"加载数据: {data_path}")
    loader = DataLoader(str(data_path), auto_adapt=True)
    df = loader.load()
    logger.info(f"数据量: {len(df)} 行, {len(df.columns)} 列")

    # 4. 特征工程（与训练时一致）
    logger.info("执行特征工程...")
    from config.config import config

    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        numeric_columns=config.feature.numeric_features,
        interaction_context=interaction_context,
    )
    df_processed, _ = feature_engineer.transform(df, interaction_context=interaction_context)

    # 5. 删除排除列
    excluded_columns = get_excluded_columns(target)
    cols_to_drop = [col for col in excluded_columns if col in df_processed.columns and col != target]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
        logger.info(f"删除 {len(cols_to_drop)} 个排除列")

    # 6. 预测
    logger.info("执行预测...")
    y_proba = predictor.get_positive_proba(df_processed)
    y_pred = predictor.predict(df_processed)

    # 7. 构建结果 DataFrame
    if include_original:
        # 返回原始数据 + 预测结果
        result_df = df.copy()
    else:
        # 仅返回 ID + 预测结果
        result_df = pd.DataFrame()
        if id_column in df.columns:
            result_df[id_column] = df[id_column]

    result_df["预测概率"] = y_proba
    result_df["预测标签"] = y_pred

    # 8. OHAB 评级推导
    if include_ohab:
        logger.info("推导 OHAB 评级...")
        is_ordered = detect_ordered_status(df)
        ohab_ratings = derive_ohab_rating(y_proba, is_ordered, threshold=ohab_threshold)
        result_df["OHAB评级"] = ohab_ratings

        # 统计 OHAB 分布
        ohab_dist = pd.Series(ohab_ratings).value_counts()
        logger.info(f"OHAB 分布: {ohab_dist.to_dict()}")

    logger.info(f"预测完成: {len(result_df)} 条记录")

    # 9. 保存输出
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"结果已保存: {output_path}")

    return result_df


def parse_args():
    parser = argparse.ArgumentParser(description="模型预测脚本")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="数据文件路径（支持 .parquet/.csv/.tsv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认不保存）",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="包含原始数据列（默认仅输出 ID + 预测结果）",
    )
    parser.add_argument(
        "--include-ohab",
        action="store_true",
        help="包含 OHAB 评级（O/H/A/B/N）",
    )
    parser.add_argument(
        "--ohab-threshold",
        type=float,
        default=0.5,
        help="OHAB 评级判定阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="线索唯一ID",
        help="ID 列名（默认: 线索唯一ID）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result_df = predict(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output,
        include_original=args.include_original,
        include_ohab=args.include_ohab,
        ohab_threshold=args.ohab_threshold,
        id_column=args.id_column,
    )

    # 输出统计
    print("\n" + "=" * 60)
    print("预测完成")
    print("=" * 60)
    print(f"数据量: {len(result_df)} 条")
    print(f"预测概率分布:")
    print(f"  均值: {result_df['预测概率'].mean():.4f}")
    print(f"  中位数: {result_df['预测概率'].median():.4f}")
    print(f"  最大值: {result_df['预测概率'].max():.4f}")
    print(f"  最小值: {result_df['预测概率'].min():.4f}")
    print(f"\n预测标签分布:")
    print(result_df["预测标签"].value_counts())

    if "OHAB评级" in result_df.columns:
        print(f"\nOHAB 评级分布:")
        for rating in ["O", "H", "A", "B", "N"]:
            count = (result_df["OHAB评级"] == rating).sum()
            pct = count / len(result_df) * 100
            print(f"  {rating} 级: {count} ({pct:.2f}%)")

    if args.output:
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
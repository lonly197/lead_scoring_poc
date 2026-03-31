#!/usr/bin/env python3
"""
模型预测脚本

对输入数据进行预测，将预测结果追加到 DataFrame 中返回。

使用方法：
    # 基本用法
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/final_v4_test.parquet \
        --output ./predictions.csv

    # 返回完整 DataFrame（含原始列 + 预测列）
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/final_v4_test.parquet \
        --output ./predictions.csv \
        --include-original
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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


def predict(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None,
    include_original: bool = False,
    id_column: str = "线索唯一ID",
) -> pd.DataFrame:
    """
    对数据进行预测

    Args:
        model_path: 模型路径
        data_path: 数据文件路径
        output_path: 输出文件路径（可选）
        include_original: 是否包含原始列
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

    logger.info(f"预测完成: {len(result_df)} 条记录")

    # 8. 保存输出
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

    if args.output:
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
"""
Top-K 名单生成脚本

从已训练的模型生成 Top-K 高优线索名单。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config
from src.data.label_policy import apply_ohab_label_policy
from src.data.loader import DataLoader, FeatureEngineer
from src.evaluation.metrics import ModelReport
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import print_separator, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成 Top-K 名单")

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
        help="数据文件路径",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="K 值列表",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="线索唯一ID",
        help="ID 列名",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    setup_logging(level=logging.INFO)

    print_separator("Top-K 名单生成")

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir or model_path / "topk_lists")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    print_separator("加载模型")
    predictor = LeadScoringPredictor.load(str(model_path))
    logger.info(f"模型信息: {predictor.get_model_info()}")
    feature_metadata = {}
    feature_metadata_path = model_path / "feature_metadata.json"
    if feature_metadata_path.exists():
        with open(feature_metadata_path, encoding="utf-8") as f:
            feature_metadata = json.load(f)

    # 2. 加载数据
    print_separator("加载数据")
    loader = DataLoader(str(data_path), auto_adapt=True)
    df = loader.load()
    logger.info(f"数据量: {len(df)} 行")
    target_label = predictor.label
    if target_label in df.columns and feature_metadata.get("label_policy"):
        df = apply_ohab_label_policy(df, target_label, feature_metadata.get("label_policy", {}))

    # 3. 特征工程
    print_separator("特征工程")
    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        numeric_columns=config.feature.numeric_features,
        interaction_context=feature_metadata.get("interaction_context", {}),
    )
    df_processed, _ = feature_engineer.transform(
        df,
        interaction_context=feature_metadata.get("interaction_context", {}),
    )
    logger.info(f"特征工程完成: {len(df_processed.columns)} 列")

    # 4. 预测
    print_separator("预测")
    y_proba = predictor.get_positive_proba(df_processed)

    # 4. 生成 Top-K 名单
    print_separator("生成名单")

    # 获取 ID
    if args.id_column in df.columns:
        ids = df[args.id_column].values
    else:
        ids = np.arange(len(df))

    # 目标变量
    y_true = df[target_label].values if target_label in df.columns else None

    report_generator = ModelReport(output_dir)

    for k in args.k:
        if k > len(df):
            logger.warning(f"k={k} 超过样本数 {len(df)}，跳过")
            continue

        topk_list = report_generator.generate_topk_list(
            ids=ids,
            y_proba=y_proba,
            y_true=y_true,
            k=k,
            model_name="topk",
            id_column=args.id_column,
        )

        # 打印统计
        if y_true is not None:
            hit_count = topk_list["actual_label"].sum()
            hit_rate = hit_count / k
            logger.info(f"Top-{k}: 命中 {hit_count}/{k} ({hit_rate:.2%})")

    print_separator("完成")
    logger.info(f"Top-K 名单已保存到: {output_dir}")


if __name__ == "__main__":
    main()

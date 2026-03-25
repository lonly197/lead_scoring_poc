"""
Top-K 名单生成脚本

从已训练的模型生成 Top-K 高优线索名单。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

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
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="多分类 Top-K 排序时使用的目标类别，如 OHAB 任务传 H",
    )

    return parser.parse_args()


def _resolve_topk_scores(
    predictor: LeadScoringPredictor,
    data: pd.DataFrame,
    target_class: str | None = None,
    model: str | None = None,
) -> np.ndarray:
    if target_class is not None:
        return predictor.get_class_proba(data, target_class=target_class, model=model)

    proba = predictor.predict_proba(data, model=model)
    if proba.shape[1] > 2:
        raise ValueError("多分类 Top-K 必须显式指定 --target-class；OHAB 推荐使用 --target-class H")
    return predictor.get_positive_proba(data, model=model)


def _resolve_hit_count(topk_df: pd.DataFrame, target_class: str | None) -> int | None:
    if "actual_label" not in topk_df.columns:
        return None

    if target_class is not None:
        return int((topk_df["actual_label"].astype(str) == str(target_class)).sum())

    numeric_actual = pd.to_numeric(topk_df["actual_label"], errors="coerce")
    if numeric_actual.notna().all():
        return int(numeric_actual.sum())
    return None


def generate_topk_from_predictor(
    predictor: LeadScoringPredictor,
    data: pd.DataFrame,
    output_path: str,
    target_class: str | None = None,
    id_column: str = "线索唯一ID",
    k_values: Sequence[int] = (100, 500, 1000),
    model: str | None = None,
) -> list[Path]:
    """
    基于已加载的 predictor 生成 Top-K 名单。

    Args:
        predictor: 已加载或已训练的 LeadScoringPredictor
        data: 已完成特征工程、可直接打分的数据
        output_path: 输出前缀路径；最终文件名为 `<prefix>_top{k}.csv`
        target_class: 多分类时用于排序的目标类别
        id_column: ID 列名
        k_values: 需要输出的 Top-K 阈值列表
        model: 指定模型名（可选）

    Returns:
        已生成的文件路径列表
    """
    output_prefix = Path(output_path)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = output_prefix.stem if output_prefix.suffix else output_prefix.name

    y_proba = _resolve_topk_scores(
        predictor=predictor,
        data=data,
        target_class=target_class,
        model=model,
    )

    if id_column in data.columns:
        ids = data[id_column].values
    else:
        ids = np.arange(len(data))

    y_true = data[predictor.label].values if predictor.label in data.columns else None
    report_generator = ModelReport(output_dir)
    generated_files: list[Path] = []

    for k in k_values:
        if k > len(data):
            logger.warning("k=%s 超过样本数 %s，跳过", k, len(data))
            continue

        topk_df = report_generator.generate_topk_list(
            ids=ids,
            y_proba=y_proba,
            y_true=y_true,
            k=k,
            model_name=model_name,
            id_column=id_column,
        )
        output_file = output_dir / f"{model_name}_top{k}.csv"
        generated_files.append(output_file)

        hit_count = _resolve_hit_count(topk_df, target_class=target_class)
        if hit_count is not None:
            logger.info("Top-%s: 命中 %s/%s (%.2f%%)", k, hit_count, k, hit_count / k * 100)

    if not generated_files:
        raise ValueError("未生成任何 Top-K 文件，请检查 k 值与输入数据量")

    return generated_files


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

    # 4. 生成 Top-K 名单
    print_separator("生成名单")
    generated_files = generate_topk_from_predictor(
        predictor=predictor,
        data=df_processed,
        output_path=str(output_dir / "topk"),
        target_class=args.target_class,
        id_column=args.id_column,
        k_values=tuple(args.k),
    )

    print_separator("完成")
    logger.info("Top-K 名单已保存到: %s", ", ".join(str(path) for path in generated_files))


if __name__ == "__main__":
    main()

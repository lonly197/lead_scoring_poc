#!/usr/bin/env python3
"""
模型预测脚本

对输入数据进行预测，将预测结果追加到 DataFrame 中返回。
支持 OHABCN 评级推导：
- O 级：已成交（100分）
- H 级：7天内试驾/下订（80-99分，高意向）
- A 级：14天内试驾/下订（60-79分，中意向）
- B 级：21天内试驾/下订（40-59分，低意向）
- C 级：有意向但超过21天（20-39分，超长尾意向）
- N 级：无效线索（0分，无电话、已购买竞品、明确拒绝等）

三种预测模式：
- simple: 简单模式，使用单模型（14天试驾概率）推断评级
- medium: 中等模式，使用三模型集成（7/14/21天试驾概率）推断评级
- advanced: 高等模式，分阶段预测（试驾前+试驾后），完全符合业务规则

使用方法：
    # 简单模式（默认）
    uv run python scripts/predict.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/unified_split/test.parquet \
        --output ./predictions.csv \
        --mode simple

    # 中等模式（推荐）
    uv run python scripts/predict.py \
        --ensemble-path ./outputs/models/test_drive_ensemble \
        --data-path ./data/unified_split/test.parquet \
        --output ./predictions.csv \
        --mode medium \
        --include-ohab

    # 高等模式
    uv run python scripts/predict.py \
        --drive-ensemble-path ./outputs/models/test_drive_ensemble \
        --order-ensemble-path ./outputs/models/order_after_drive_ensemble \
        --data-path ./data/unified_split/test.parquet \
        --output ./predictions.csv \
        --mode advanced \
        --include-ohab
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer
from src.models.predictor import LeadScoringPredictor
from src.models.ohab_rater import OHABRater, PredictionMode

logger = logging.getLogger(__name__)


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_ensemble_metadata(ensemble_path: Path) -> dict:
    """加载集成模型元数据"""
    metadata_path = ensemble_path / "ensemble_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_ensemble_models(
    ensemble_path: Path,
    time_windows: List[str] = ["7天", "14天", "21天"],
    label_prefix: str = "试驾标签",
) -> Dict[str, LeadScoringPredictor]:
    """
    加载集成模型的所有子模型

    Args:
        ensemble_path: 集成模型目录
        time_windows: 时间窗口列表
        label_prefix: 标签前缀（"试驾标签" 或 "下订标签"）

    Returns:
        {标签名: Predictor} 字典
    """
    models = {}
    for window in time_windows:
        label = f"{label_prefix}_{window}"
        model_path = ensemble_path / label
        if model_path.exists():
            logger.info(f"加载模型: {label}")
            models[label] = LeadScoringPredictor.load(str(model_path))
        else:
            logger.warning(f"模型不存在: {model_path}")
    return models


def prepare_data_for_prediction(
    df: pd.DataFrame,
    target: str,
    interaction_context: dict,
    excluded_columns: List[str],
    keep_label_columns: bool = True,
) -> pd.DataFrame:
    """
    为预测准备数据

    Args:
        df: 原始数据
        target: 目标变量
        interaction_context: 特征工程上下文
        excluded_columns: 排除列
        keep_label_columns: 是否保留标签列（若训练时存在标签泄漏，预测时需保留以匹配特征）

    Returns:
        处理后的数据
    """
    from config.config import config

    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        numeric_columns=config.feature.numeric_features,
        interaction_context=interaction_context,
    )
    df_processed, _ = feature_engineer.transform(df, interaction_context=interaction_context)

    # 删除排除列
    # 注意：如果训练时模型将标签列作为特征学习（标签泄漏），
    # 预测时必须保留相同的列结构，否则会导致特征不匹配而预测失败
    if keep_label_columns:
        label_columns = [col for col in df_processed.columns if '标签' in col or '评级' in col]
        cols_to_drop = [col for col in excluded_columns if col in df_processed.columns and col not in label_columns]
    else:
        cols_to_drop = [col for col in excluded_columns if col in df_processed.columns]

    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)

    return df_processed


def predict_simple(
    df: pd.DataFrame,
    model_path: Path,
    id_column: str,
    include_original: bool,
    ohab_threshold: float,
) -> pd.DataFrame:
    """
    简单模式预测

    使用单个模型（14天试驾概率）推断 OHAB 评级。

    Args:
        df: 输入数据
        model_path: 模型路径
        id_column: ID 列名
        include_original: 是否包含原始列
        ohab_threshold: OHAB 阈值

    Returns:
        预测结果 DataFrame
    """
    logger.info("=== 简单模式预测 ===")

    # 加载模型
    predictor = LeadScoringPredictor.load(str(model_path))
    target = predictor.label
    logger.info(f"目标变量: {target}")

    # 加载特征元数据
    metadata = load_feature_metadata(model_path)
    interaction_context = metadata.get("interaction_context", {})

    # 准备数据
    excluded_columns = get_excluded_columns(target)
    df_processed = prepare_data_for_prediction(df, target, interaction_context, excluded_columns)

    # 预测
    y_proba = predictor.get_positive_proba(df_processed)
    y_pred = predictor.predict(df_processed)

    # 推导 OHAB 评级
    rater = OHABRater(mode=PredictionMode.SIMPLE, thresholds={"H": ohab_threshold})
    result = rater.derive(df, proba_14d=y_proba)

    # 构建结果
    if include_original:
        result_df = df.copy()
    else:
        result_df = pd.DataFrame()
        if id_column in df.columns:
            result_df[id_column] = df[id_column]

    result_df["预测概率"] = y_proba
    result_df["预测标签"] = y_pred
    result_df = rater.add_to_dataframe(result_df, result, include_proba=True)

    return result_df


def predict_medium(
    df: pd.DataFrame,
    ensemble_path: Path,
    id_column: str,
    include_original: bool,
    thresholds: Dict[str, float],
    label_prefix: str = "试驾标签",
) -> pd.DataFrame:
    """
    中等模式预测

    使用三模型集成（7/14/21天试驾概率）推断 OHAB 评级。

    Args:
        df: 输入数据
        ensemble_path: 集成模型目录
        id_column: ID 列名
        include_original: 是否包含原始列
        thresholds: 各级别阈值
        label_prefix: 标签前缀（"试驾标签" 或 "下订标签"）

    Returns:
        预测结果 DataFrame
    """
    logger.info("=== 中等模式预测 ===")

    # 加载模型
    models = load_ensemble_models(ensemble_path, label_prefix=label_prefix)

    # 加载特征元数据（使用第一个模型）
    first_model_path = ensemble_path / f"{label_prefix}_7天"
    metadata = load_feature_metadata(first_model_path)
    interaction_context = metadata.get("interaction_context", {})

    # 预测各时间窗口概率
    probas = {}
    for window in ["7天", "14天", "21天"]:
        label = f"{label_prefix}_{window}"
        if label not in models:
            logger.warning(f"模型缺失: {label}，使用默认概率 0")
            probas[window] = np.zeros(len(df))
            continue

        predictor = models[label]
        excluded_columns = get_excluded_columns(label)
        df_processed = prepare_data_for_prediction(df, label, interaction_context, excluded_columns)
        probas[window] = predictor.get_positive_proba(df_processed)
        logger.info(f"{label} 预测完成")

    # 推导 OHAB 评级
    rater = OHABRater(mode=PredictionMode.MEDIUM, thresholds=thresholds)
    result = rater.derive(
        df,
        drive_proba_7d=probas["7天"],
        drive_proba_14d=probas["14天"],
        drive_proba_21d=probas["21天"],
    )

    # 构建结果
    if include_original:
        result_df = df.copy()
    else:
        result_df = pd.DataFrame()
        if id_column in df.columns:
            result_df[id_column] = df[id_column]

    result_df = rater.add_to_dataframe(result_df, result, include_proba=True)

    return result_df


def predict_advanced(
    df: pd.DataFrame,
    drive_ensemble_path: Path,
    order_ensemble_path: Path,
    id_column: str,
    include_original: bool,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    高等模式预测

    分阶段预测（试驾前+试驾后），完全符合业务规则：
    1. 检测已试驾状态
    2. 已试驾客户 → 只调用下订模型
    3. 未试驾客户 → 只调用试驾模型

    Args:
        df: 输入数据
        drive_ensemble_path: 试驾集成模型目录
        order_ensemble_path: 下订集成模型目录
        id_column: ID 列名
        include_original: 是否包含原始列
        thresholds: 各级别阈值

    Returns:
        预测结果 DataFrame
    """
    logger.info("=== 高等模式预测 ===")
    n_samples = len(df)

    # 检测已试驾状态
    from src.models.ohab_rater import detect_driven_status
    is_driven = detect_driven_status(df)
    driven_count = is_driven.sum()
    not_driven_count = n_samples - driven_count
    logger.info(f"已试驾客户: {driven_count} ({driven_count/n_samples*100:.1f}%)")
    logger.info(f"未试驾客户: {not_driven_count} ({not_driven_count/n_samples*100:.1f}%)")

    # 初始化概率数组
    drive_probas = {"7天": np.zeros(n_samples), "14天": np.zeros(n_samples), "21天": np.zeros(n_samples)}
    order_probas = {"7天": np.zeros(n_samples), "14天": np.zeros(n_samples), "21天": np.zeros(n_samples)}

    # 加载试驾模型
    drive_models = load_ensemble_models(drive_ensemble_path, label_prefix="试驾标签")
    logger.info(f"试驾模型: {list(drive_models.keys())}")

    # 加载下订模型
    order_models = load_ensemble_models(order_ensemble_path, label_prefix="下订标签")
    logger.info(f"下订模型: {list(order_models.keys())}")

    # 加载特征元数据
    drive_metadata = load_feature_metadata(drive_ensemble_path / "试驾标签_7天")
    drive_interaction_context = drive_metadata.get("interaction_context", {})

    order_metadata = load_feature_metadata(order_ensemble_path / "下订标签_7天")
    order_interaction_context = order_metadata.get("interaction_context", {})

    # 对【未试驾客户】预测试驾概率
    if not_driven_count > 0:
        not_driven_mask = ~is_driven
        df_not_driven = df[not_driven_mask].copy()
        logger.info(f"对 {not_driven_count} 个未试驾客户预测试驾概率...")

        for window, label in [("7天", "试驾标签_7天"), ("14天", "试驾标签_14天"), ("21天", "试驾标签_21天")]:
            if label not in drive_models:
                logger.warning(f"试驾模型缺失: {label}")
                continue

            predictor = drive_models[label]
            excluded_columns = get_excluded_columns(label)
            df_processed = prepare_data_for_prediction(df_not_driven, label, drive_interaction_context, excluded_columns)
            proba = predictor.get_positive_proba(df_processed)
            drive_probas[window][not_driven_mask] = proba
            logger.info(f"{label} 预测完成")
    else:
        logger.info("无未试驾客户，跳过试驾概率预测")

    # 对【已试驾客户】预测下订概率
    if driven_count > 0:
        driven_mask = is_driven
        df_driven = df[driven_mask].copy()
        logger.info(f"对 {driven_count} 个已试驾客户预测下订概率...")

        for window, label in [("7天", "下订标签_7天"), ("14天", "下订标签_14天"), ("21天", "下订标签_21天")]:
            if label not in order_models:
                logger.warning(f"下订模型缺失: {label}")
                continue

            predictor = order_models[label]
            excluded_columns = get_excluded_columns(label)

            try:
                df_processed = prepare_data_for_prediction(df_driven, label, order_interaction_context, excluded_columns)
                proba = predictor.get_positive_proba(df_processed)
                order_probas[window][driven_mask] = proba
                logger.info(f"{label} 预测完成")
            except ValueError as e:
                # 数据缺少下订模型所需特征，使用默认概率
                logger.warning(f"{label} 预测失败（数据缺少特征）: {e}")
                logger.warning(f"使用默认概率 0.5 作为已试驾客户的下订概率")
                order_probas[window][driven_mask] = 0.5
    else:
        logger.info("无已试驾客户，跳过下订概率预测")

    # 推导 OHABCN 评级
    rater = OHABRater(mode=PredictionMode.ADVANCED, thresholds=thresholds)
    result = rater.derive(
        df,
        drive_proba_7d=drive_probas["7天"],
        drive_proba_14d=drive_probas["14天"],
        drive_proba_21d=drive_probas["21天"],
        order_proba_7d=order_probas["7天"],
        order_proba_14d=order_probas["14天"],
        order_proba_21d=order_probas["21天"],
    )

    # 构建结果
    if include_original:
        result_df = df.copy()
    else:
        result_df = pd.DataFrame()
        if id_column in df.columns:
            result_df[id_column] = df[id_column]

    result_df = rater.add_to_dataframe(result_df, result, include_proba=True)

    return result_df


def predict(
    data_path: str,
    output_path: Optional[str] = None,
    mode: str = "simple",
    # 简单模式参数
    model_path: Optional[str] = None,
    # 中等模式参数
    ensemble_path: Optional[str] = None,
    # 高等模式参数
    drive_ensemble_path: Optional[str] = None,
    order_ensemble_path: Optional[str] = None,
    # 通用参数
    include_original: bool = False,
    thresholds: Optional[Dict[str, float]] = None,
    id_column: str = "线索唯一ID",
    skip_adapter: bool = False,
    label_prefix: str = "试驾标签",
) -> pd.DataFrame:
    """
    对数据进行预测

    Args:
        data_path: 数据文件路径
        output_path: 输出文件路径
        mode: 预测模式（simple/medium/advanced）
        model_path: 单模型路径（简单模式）
        ensemble_path: 试驾集成模型目录（中等模式）
        drive_ensemble_path: 试驾集成模型目录（高等模式）
        order_ensemble_path: 下订集成模型目录（高等模式）
        include_original: 是否包含原始列
        thresholds: OHAB 评级阈值
        id_column: ID 列名
        skip_adapter: 是否跳过数据适配器
        label_prefix: 标签前缀（"试驾标签" 或 "下订标签"）

    Returns:
        预测结果 DataFrame
    """
    thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
    data_path = Path(data_path)

    # 加载数据
    logger.info(f"加载数据: {data_path}")
    if skip_adapter and str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
        logger.info(f"直接加载 Parquet（跳过适配器）: {len(df)} 行")
    else:
        loader = DataLoader(str(data_path), auto_adapt=True)
        df = loader.load()
    logger.info(f"数据量: {len(df)} 行, {len(df.columns)} 列")

    # 根据模式执行预测
    if mode == "simple":
        if not model_path:
            raise ValueError("简单模式需要 --model-path 参数")
        result_df = predict_simple(
            df=df,
            model_path=Path(model_path),
            id_column=id_column,
            include_original=include_original,
            ohab_threshold=thresholds.get("H", 0.5),
        )

    elif mode == "medium":
        if not ensemble_path:
            raise ValueError("中等模式需要 --ensemble-path 参数")
        result_df = predict_medium(
            df=df,
            ensemble_path=Path(ensemble_path),
            id_column=id_column,
            include_original=include_original,
            thresholds=thresholds,
            label_prefix=label_prefix,
        )

    elif mode == "advanced":
        if not drive_ensemble_path or not order_ensemble_path:
            raise ValueError("高等模式需要 --drive-ensemble-path 和 --order-ensemble-path 参数")
        result_df = predict_advanced(
            df=df,
            drive_ensemble_path=Path(drive_ensemble_path),
            order_ensemble_path=Path(order_ensemble_path),
            id_column=id_column,
            include_original=include_original,
            thresholds=thresholds,
        )

    else:
        raise ValueError(f"未知的预测模式: {mode}")

    # 保存输出
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"结果已保存: {output_path}")

    return result_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="模型预测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预测模式说明:
  simple   - 简单模式：使用单模型（14天试驾概率）推断 OHAB
  medium   - 中等模式：使用三模型集成（7/14/21天试驾概率）推断 OHAB
  advanced - 高等模式：分阶段预测（试驾前+试驾后），完全符合业务规则

示例:
  # 简单模式
  uv run python scripts/predict.py \\
      --mode simple \\
      --model-path ./outputs/models/test_drive_model \\
      --data-path ./data/final_v4_test.parquet \\
      --output ./predictions.csv

  # 中等模式
  uv run python scripts/predict.py \\
      --mode medium \\
      --ensemble-path ./outputs/models/test_drive_ensemble \\
      --data-path ./data/final_v4_test.parquet \\
      --output ./predictions.csv

  # 高等模式
  uv run python scripts/predict.py \\
      --mode advanced \\
      --drive-ensemble-path ./outputs/models/test_drive_ensemble \\
      --order-ensemble-path ./outputs/models/order_after_drive_ensemble \\
      --data-path ./data/final_v4_test.parquet \\
      --output ./predictions.csv
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "medium", "advanced"],
        help="预测模式（默认: simple）",
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

    # 简单模式参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="单模型路径（简单模式必需）",
    )

    # 中等模式参数
    parser.add_argument(
        "--ensemble-path",
        type=str,
        default=None,
        help="试驾集成模型目录（中等模式必需）",
    )

    # 高等模式参数
    parser.add_argument(
        "--drive-ensemble-path",
        type=str,
        default=None,
        help="试驾集成模型目录（高等模式必需）",
    )
    parser.add_argument(
        "--order-ensemble-path",
        type=str,
        default=None,
        help="下订集成模型目录（高等模式必需）",
    )

    # 通用参数
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="包含原始数据列",
    )
    parser.add_argument(
        "--threshold-h",
        type=float,
        default=0.5,
        help="H 级阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--threshold-a",
        type=float,
        default=0.5,
        help="A 级阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--threshold-b",
        type=float,
        default=0.5,
        help="B 级阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="线索唯一ID",
        help="ID 列名（默认: 线索唯一ID）",
    )
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="跳过数据适配器，直接加载 Parquet 文件",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="试驾标签",
        help="标签前缀（试驾标签 或 下订标签，默认: 试驾标签）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 构建阈值字典
    thresholds = {
        "H": args.threshold_h,
        "A": args.threshold_a,
        "B": args.threshold_b,
    }

    result_df = predict(
        data_path=args.data_path,
        output_path=args.output,
        mode=args.mode,
        model_path=args.model_path,
        ensemble_path=args.ensemble_path,
        drive_ensemble_path=args.drive_ensemble_path,
        order_ensemble_path=args.order_ensemble_path,
        include_original=args.include_original,
        thresholds=thresholds,
        id_column=args.id_column,
        skip_adapter=args.skip_adapter,
        label_prefix=args.label_prefix,
    )

    # 输出统计
    print("\n" + "=" * 60)
    print(f"预测完成（模式: {args.mode}）")
    print("=" * 60)
    print(f"数据量: {len(result_df)} 条")

    # OHAB 分布
    if "OHAB评级" in result_df.columns:
        print(f"\nOHAB 评级分布:")
        for rating in ["O", "H", "A", "B", "N"]:
            count = (result_df["OHAB评级"] == rating).sum()
            pct = count / len(result_df) * 100
            print(f"  {rating} 级: {count} ({pct:.2f}%)")

    # 评级阶段分布
    if "评级阶段" in result_df.columns:
        print(f"\n评级阶段分布:")
        for stage in ["O", "试驾前", "试驾后"]:
            count = (result_df["评级阶段"] == stage).sum()
            pct = count / len(result_df) * 100
            print(f"  {stage}: {count} ({pct:.2f}%)")

    if args.output:
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
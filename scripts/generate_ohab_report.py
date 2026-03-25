#!/usr/bin/env python3
"""
OHAB评级报告生成脚本

功能：
1. 加载模型预测用户行为概率（试驾/到店）
2. 将概率映射为OHAB等级
3. 计算SHAP特征归因
4. 生成业务可解释报告

使用示例：
    uv run python scripts/generate_ohab_report.py \
        --model-path ./outputs/models/test_drive_model \
        --data-path ./data/202602~03.tsv \
        --output-dir ./outputs/ohab_reports
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

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer
from src.evaluation.explainability import (
    OHABResult,
    probability_to_ohab,
    generate_single_report,
    generate_batch_report,
    get_action_suggestion,
    get_risk_alert,
)
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="生成OHAB评级报告")
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
        "--target",
        type=str,
        default="试驾标签_14天",
        choices=["试驾标签_14天", "到店标签_14天", "试驾标签_7天", "到店标签_7天"],
        help="预测目标变量",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/ohab_reports",
        help="报告输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="输出TOP-K高意向线索报告",
    )

    return parser.parse_args()


def load_model(model_path: str):
    """加载AutoGluon模型"""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(
        model_path,
        require_version_match=False,
        require_py_version_match=False,
    )
    return predictor


def predict_probabilities(predictor, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    预测概率并计算多时间窗口概率

    Args:
        predictor: AutoGluon预测器
        df: 数据框
        target: 目标变量

    Returns:
        包含概率的数据框
    """
    # 获取预测概率
    proba = predictor.predict_proba(df)

    # 根据目标变量确定时间窗口
    if "14天" in target:
        prob_14day = proba.iloc[:, 1].values  # 正类概率
        # 模拟其他时间窗口（实际应训练多时间窗口模型）
        prob_7day = prob_14day * 0.7  # 简化假设
        prob_21day = prob_14day * 1.2  # 简化假设
        prob_21day = np.clip(prob_21day, 0, 1)
    elif "7天" in target:
        prob_7day = proba.iloc[:, 1].values
        prob_14day = prob_7day * 1.3
        prob_14day = np.clip(prob_14day, 0, 1)
        prob_21day = prob_14day * 1.1
        prob_21day = np.clip(prob_21day, 0, 1)
    else:
        # 默认处理
        prob_7day = proba.iloc[:, 1].values
        prob_14day = prob_7day
        prob_21day = prob_7day

    result = pd.DataFrame({
        "prob_7day": prob_7day,
        "prob_14day": prob_14day,
        "prob_21day": prob_21day,
    })

    return result


def compute_feature_importance(predictor, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    计算特征重要性（使用AutoGluon内置方法）

    Args:
        predictor: AutoGluon预测器
        df: 数据框
        target: 目标变量

    Returns:
        特征重要性数据框
    """
    try:
        importance = predictor.feature_importance(df)
        return importance
    except Exception as e:
        logger.warning(f"特征重要性计算失败: {e}")
        return pd.DataFrame()


def compute_shap_contributions(predictor, df: pd.DataFrame, importance_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    计算SHAP风格的特征贡献（简化版，基于特征重要性和特征值）

    注：完整SHAP计算需要shap库，这里使用简化方法

    Args:
        predictor: 预测器
        df: 数据框
        importance_df: 特征重要性

    Returns:
        每个样本的特征贡献字典
    """
    # 获取特征重要性排序
    if importance_df.empty:
        return {}

    feature_weights = importance_df["importance"].to_dict()
    total_importance = sum(feature_weights.values())

    # 归一化权重
    normalized_weights = {k: v / total_importance for k, v in feature_weights.items()}

    # 对每个样本计算贡献（简化：基于特征值与均值的偏差）
    contributions = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for idx in df.index:
        sample_contrib = {}
        for feat in normalized_weights:
            if feat in numeric_cols and feat in df.columns:
                val = df.loc[idx, feat]
                mean_val = df[feat].mean()
                std_val = df[feat].std()

                if pd.notna(val) and pd.notna(mean_val) and std_val > 0:
                    # 标准化偏差
                    deviation = (val - mean_val) / std_val
                    # 贡献值 = 权重 × 偏差
                    sample_contrib[feat] = normalized_weights[feat] * deviation
                else:
                    sample_contrib[feat] = 0.0

        contributions[str(idx)] = sample_contrib

    return contributions


def generate_ohab_results(
    df: pd.DataFrame,
    probabilities: pd.DataFrame,
    contributions: Dict[str, Dict[str, float]],
    lead_id_col: str = "线索唯一ID"
) -> List[OHABResult]:
    """
    生成OHAB评级结果列表

    Args:
        df: 原始数据框
        probabilities: 预测概率数据框
        contributions: 特征贡献字典
        lead_id_col: 线索ID列名

    Returns:
        OHABResult列表
    """
    results = []

    for i, idx in enumerate(df.index):
        prob_7day = probabilities.iloc[i]["prob_7day"]
        prob_14day = probabilities.iloc[i]["prob_14day"]
        prob_21day = probabilities.iloc[i]["prob_21day"]

        # 映射OHAB等级
        level, confidence = probability_to_ohab(prob_7day, prob_14day, prob_21day)

        # 获取特征贡献
        sample_contrib = contributions.get(str(idx), {})

        # 排序获取TOP正向和负向特征
        sorted_contrib = sorted(sample_contrib.items(), key=lambda x: x[1], reverse=True)
        top_positive = [(f, v) for f, v in sorted_contrib if v > 0][:5]
        top_negative = [(f, v) for f, v in sorted_contrib if v < 0][:3]

        # 生成建议和风险提示
        action = get_action_suggestion(level, top_positive)
        risk = get_risk_alert(sample_contrib)

        result = OHABResult(
            level=level,
            prob_7day=prob_7day,
            prob_14day=prob_14day,
            prob_21day=prob_21day,
            confidence=confidence,
            feature_contributions=sample_contrib,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            action_suggestion=action,
            risk_alert=risk,
        )
        results.append(result)

    return results


def main():
    args = parse_args()
    setup_logging(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("OHAB评级报告生成")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info(f"加载数据: {args.data_path}")
    loader = DataLoader(args.data_path, auto_adapt=True)
    df = loader.load()

    # 过滤有效线索（排除已转化为目标的）
    target = args.target
    if target in df.columns:
        # 对于报告生成，我们可以包含所有线索
        logger.info(f"数据总量: {len(df)} 条")

    # 2. 特征工程
    logger.info("执行特征工程...")
    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        numeric_columns=config.feature.numeric_features,
    )
    df_processed, _ = feature_engineer.process(df)

    # 保存线索ID
    lead_ids = df_processed["线索唯一ID"].tolist() if "线索唯一ID" in df_processed.columns else [f"LX{i:06d}" for i in range(len(df_processed))]

    # 3. 加载模型
    logger.info(f"加载模型: {args.model_path}")
    predictor = load_model(args.model_path)
    logger.info(f"模型标签: {predictor.label}")

    # 4. 预测概率
    logger.info("执行预测...")
    probabilities = predict_probabilities(predictor, df_processed, target)

    # 5. 计算特征重要性和贡献
    logger.info("计算特征贡献...")
    importance_df = compute_feature_importance(predictor, df_processed, target)
    contributions = compute_shap_contributions(predictor, df_processed, importance_df)

    # 6. 生成OHAB结果
    logger.info("生成OHAB评级...")
    ohab_results = generate_ohab_results(df_processed, probabilities, contributions)

    # 7. 生成报告
    logger.info("生成报告...")
    summary = generate_batch_report(ohab_results, lead_ids, str(output_dir))

    # 8. 输出TOP-K高意向线索
    logger.info(f"输出TOP-{args.top_k} 高意向线索...")
    ranked_results = sorted(
        zip(lead_ids, ohab_results),
        key=lambda x: x[1].confidence,
        reverse=True
    )[:args.top_k]

    topk_path = output_dir / f"top_{args.top_k}_leads.md"
    with open(topk_path, "w", encoding="utf-8") as f:
        f.write(f"# TOP-{args.top_k} 高意向线索\n\n")
        f.write("| 排名 | 线索ID | 等级 | 置信度 | 7天概率 | 14天概率 | 跟进建议 |\n")
        f.write("|------|--------|------|--------|---------|----------|----------|\n")
        for rank, (lead_id, result) in enumerate(ranked_results, 1):
            f.write(f"| {rank} | {lead_id} | {result.level} | {result.confidence:.1%} | {result.prob_7day:.1%} | {result.prob_14day:.1%} | {result.action_suggestion[:30]}... |\n")

    # 9. 打印汇总
    print("\n" + "=" * 60)
    print("OHAB评级汇总")
    print("=" * 60)
    print(f"总线索数: {summary['total_leads']}")
    print(f"\n等级分布:")
    for level, count in summary['level_distribution'].items():
        pct = summary['level_percentage'][level]
        print(f"  {level}级: {count} ({pct:.1f}%)")

    print(f"\n概率统计:")
    for window, stats in summary['probability_statistics'].items():
        print(f"  {window}: 均值={stats['mean']:.1%}, 中位数={stats['median']:.1%}")

    print(f"\n报告已保存至: {output_dir}")
    print(f"  - 汇总报告: batch_summary.json")
    print(f"  - 单线索报告: individual_reports/")
    print(f"  - TOP-{args.top_k}: top_{args.top_k}_leads.md")


if __name__ == "__main__":
    main()
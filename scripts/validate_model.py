#!/usr/bin/env python3
"""
模型验证脚本

验证训练好的 OHAB 模型效果，包括：
1. 加载模型和新数据
2. 预测并评估
3. 生成详细报告
"""

import argparse
import logging
import pickle  # noqa: S403 - 加载 AutoGluon 模型需要
import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer
from src.utils.helpers import setup_logging


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="验证 OHAB 模型")

    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/ohab_model",
        help="模型路径",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="测试数据路径（默认使用训练数据）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="线索评级_试驾前",
        help="目标变量名",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation",
        help="输出目录",
    )

    return parser.parse_args()


def load_model(model_path: Path):
    """加载 AutoGluon 模型，处理版本兼容性问题"""
    # 方式1：尝试使用 require_py_version_match 参数
    try:
        predictor = TabularPredictor.load(
            str(model_path),
            require_version_match=False,
            require_py_version_match=False,
        )
        return predictor
    except TypeError:
        # 旧版本可能不支持 require_py_version_match
        pass
    except Exception as e:
        logger.warning(f"标准加载失败: {e}")

    # 方式2：直接加载
    try:
        predictor = TabularPredictor.load(str(model_path))
        return predictor
    except Exception as e:
        logger.warning(f"直接加载失败: {e}")

    # 方式3：使用 pickle 手动加载
    predictor_path = model_path / "predictor.pkl"
    if predictor_path.exists():
        logger.info("使用 pickle 手动加载...")
        with open(predictor_path, "rb") as f:
            predictor = pickle.load(f)
        return predictor

    raise RuntimeError(f"无法加载模型: {model_path}")


def main():
    args = parse_args()
    setup_logging(level=logging.INFO)

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    logger.info("=" * 60)
    logger.info("加载模型")
    logger.info("=" * 60)

    predictor = load_model(model_path)
    logger.info(f"模型加载完成: {model_path}")
    logger.info(f"标签: {predictor.label}")
    logger.info(f"评估指标: {predictor.eval_metric}")
    logger.info(f"问题类型: {predictor.problem_type}")
    logger.info(f"最佳模型: {predictor.model_best}")

    # 2. 加载测试数据
    logger.info("\n" + "=" * 60)
    logger.info("加载测试数据")
    logger.info("=" * 60)

    data_path = args.data_path or "./data/20260308-v2.csv"
    loader = DataLoader(data_path)
    df = loader.load()

    # 过滤 Unknown
    target = predictor.label
    if target in df.columns:
        df = df[df[target] != "Unknown"].copy()
        logger.info(f"过滤 Unknown 后: {len(df)} 行")

    # 排除不需要的列（但保留目标列用于评估）
    excluded_columns = get_excluded_columns(target)
    cols_to_drop = [col for col in excluded_columns if col in df.columns and col != target]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"排除 {len(cols_to_drop)} 列: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")

    # 加载特征工程元数据
    feature_metadata = load_feature_metadata(model_path)
    has_category_mappings = bool(feature_metadata.get("category_mappings"))

    # 特征工程（与训练时相同的处理）
    logger.info("执行特征工程...")
    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        categorical_columns=config.feature.categorical_features,
        numeric_columns=config.feature.numeric_features,
    )

    if has_category_mappings:
        # 使用训练时保存的编码映射
        logger.info("使用训练时的类别编码映射")
        df_processed, _ = feature_engineer.process(
            df,
            fit=False,
            category_mappings=feature_metadata["category_mappings"],
        )
    else:
        # 兼容旧模型：重新拟合（验证数据与训练数据相同时可用）
        logger.warning("未找到特征元数据，使用 fit=True 模式")
        df_processed, _ = feature_engineer.process(df, fit=True)

    # 保存目标值用于评估
    y_true = df_processed[target].values

    # 3. 预测
    logger.info("\n" + "=" * 60)
    logger.info("执行预测")
    logger.info("=" * 60)

    y_pred = predictor.predict(df_processed)
    y_proba = predictor.predict_proba(df_processed)

    logger.info(f"预测完成: {len(y_pred)} 个样本")

    # 4. 评估
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        matthews_corrcoef,
    )

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"MCC: {mcc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n混淆矩阵:\n{cm}")

    report = classification_report(y_true, y_pred)
    logger.info(f"\n分类报告:\n{report}")

    # 5. 按类别分析
    logger.info("\n" + "=" * 60)
    logger.info("各类别详细分析")
    logger.info("=" * 60)

    classes = df_processed[target].unique()
    for cls in sorted(classes):
        mask = y_true == cls
        correct = (y_pred[mask] == cls).sum()
        total = mask.sum()
        acc = correct / total if total > 0 else 0
        logger.info(f"类别 {cls}: {correct}/{total} 正确 ({acc:.1%})")

    # 6. 保存结果
    logger.info("\n" + "=" * 60)
    logger.info("保存结果")
    logger.info("=" * 60)

    # 预测结果
    results_df = pd.DataFrame({
        "真实标签": y_true,
        "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred,
    })

    # 添加概率
    for col in y_proba.columns:
        results_df[f"概率_{col}"] = y_proba[col].values

    results_path = output_dir / "predictions.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info(f"预测结果已保存: {results_path}")

    # 评估报告
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("OHAB 模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"样本数量: {len(y_true)}\n")
        f.write(f"最佳模型: {predictor.model_best}\n\n")
        f.write("评估指标\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n\n")
        f.write("混淆矩阵\n")
        f.write("-" * 40 + "\n")
        f.write(f"{cm}\n\n")
        f.write("分类报告\n")
        f.write("-" * 40 + "\n")
        f.write(report)

    logger.info(f"评估报告已保存: {report_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
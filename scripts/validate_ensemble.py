"""
三模型验证脚本

验证三模型试驾预测的效果和 HAB 评级推导的合理性。

验证内容：
1. 各模型单独性能（ROC-AUC、Lift、召回率）
2. HAB 推导结果与实际试驾率的对应关系
3. 分层效果验证（H 类试驾率 > A 类 > B 类）

使用方法：
    uv run python scripts/validate_ensemble.py \
        --model-dir ./outputs/models/test_drive_ensemble \
        --data-path ./data/202602~03.tsv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config
from src.data.loader import DataLoader, FeatureEngineer
from src.evaluation.metrics import TopKEvaluator
from src.inference.hab_deriver import HABDeriver, HABRating, get_hab_distribution_summary
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

TIME_WINDOWS = ["7天", "14天", "21天"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="三模型验证")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="模型目录路径",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径（动态拆分模式）",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="HAB 评级判定阈值",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径",
    )

    return parser.parse_args()


def load_models(model_dir: Path) -> Dict[str, LeadScoringPredictor]:
    """
    加载三个时间窗口的模型

    Args:
        model_dir: 模型目录

    Returns:
        Dict[str, LeadScoringPredictor] 模型字典
    """
    models = {}

    for window in TIME_WINDOWS:
        target = f"试驾标签_{window}"
        model_path = model_dir / target

        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")

        logger.info(f"加载模型: {target}")
        predictor = LeadScoringPredictor.load(str(model_path))
        models[target] = predictor

    return models


def evaluate_single_model(
    predictor: LeadScoringPredictor,
    test_df: pd.DataFrame,
    target: str,
) -> Dict[str, Any]:
    """
    评估单个模型

    Args:
        predictor: 预测器
        test_df: 测试数据
        target: 目标变量名

    Returns:
        Dict[str, Any] 评估结果
    """
    metrics = predictor.evaluate(test_df)
    y_proba = predictor.get_positive_proba(test_df)
    y_true = test_df[target].values

    # Top-K 评估
    topk_evaluator = TopKEvaluator(y_true, y_proba)
    topk_metrics = topk_evaluator.compute_topk_metrics()

    return {
        "target": target,
        "roc_auc": metrics.get("roc_auc", None),
        "accuracy": metrics.get("accuracy", None),
        "topk_metrics": topk_metrics,
        "y_proba": y_proba,
        "y_true": y_true,
    }


def validate_hab_derivation(
    results: List[Dict[str, Any]],
    threshold: float,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    验证 HAB 推导效果

    Args:
        results: 各模型评估结果
        threshold: 判定阈值
        test_df: 测试数据

    Returns:
        Dict[str, Any] 验证结果
    """
    # 提取概率
    prob_7d = results[0]["y_proba"]
    prob_14d = results[1]["y_proba"]
    prob_21d = results[2]["y_proba"]

    # 真实标签
    actual_7d = results[0]["y_true"]
    actual_14d = results[1]["y_true"]
    actual_21d = results[2]["y_true"]

    # 推导 HAB 评级
    deriver = HABDeriver(threshold=threshold)
    hab_results = deriver.derive_batch(
        prob_7d.tolist(),
        prob_14d.tolist(),
        prob_21d.tolist(),
    )

    # 统计分布
    distribution = get_hab_distribution_summary(hab_results)

    # 计算各评级的实际试驾率
    actual_drive_rates = {}
    for rating in [HABRating.H, HABRating.A, HABRating.B, HABRating.N]:
        mask = np.array([r.rating == rating for r in hab_results])
        if mask.sum() > 0:
            # 计算该评级在各时间窗口的实际试驾率
            rate_7d = actual_7d[mask].mean()
            rate_14d = actual_14d[mask].mean()
            rate_21d = actual_21d[mask].mean()
            actual_drive_rates[rating.value] = {
                "count": int(mask.sum()),
                "rate_7d": float(rate_7d),
                "rate_14d": float(rate_14d),
                "rate_21d": float(rate_21d),
            }
        else:
            actual_drive_rates[rating.value] = {
                "count": 0,
                "rate_7d": 0.0,
                "rate_14d": 0.0,
                "rate_21d": 0.0,
            }

    # 验证分层效果
    is_hierarchical = True
    issues = []

    h_rate = actual_drive_rates.get("H", {}).get("rate_14d", 0)
    a_rate = actual_drive_rates.get("A", {}).get("rate_14d", 0)
    b_rate = actual_drive_rates.get("B", {}).get("rate_14d", 0)

    if h_rate > 0 and a_rate > 0 and h_rate < a_rate:
        is_hierarchical = False
        issues.append(f"H 级 14 天试驾率 ({h_rate:.2%}) 低于 A 级 ({a_rate:.2%})")

    if a_rate > 0 and b_rate > 0 and a_rate < b_rate:
        is_hierarchical = False
        issues.append(f"A 级 14 天试驾率 ({a_rate:.2%}) 低于 B 级 ({b_rate:.2%})")

    return {
        "distribution": distribution,
        "actual_drive_rates": actual_drive_rates,
        "is_hierarchical": is_hierarchical,
        "issues": issues,
        "threshold": threshold,
    }


def generate_validation_report(
    model_results: List[Dict[str, Any]],
    hab_validation: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    生成验证报告

    Args:
        model_results: 各模型评估结果
        hab_validation: HAB 验证结果
        output_path: 输出路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# 三模型验证报告",
        "",
        f"**验证时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**判定阈值**: {hab_validation['threshold']:.0%}",
        "",
        "## 一、各模型性能",
        "",
        "| 模型 | ROC-AUC | 准确率 | Top-100 命中率 | Top-100 Lift |",
        "|------|---------|--------|----------------|--------------|",
    ]

    for r in model_results:
        roc_auc = f"{r['roc_auc']:.4f}" if r['roc_auc'] else "N/A"
        accuracy = f"{r['accuracy']:.2%}" if r['accuracy'] else "N/A"
        top100 = r['topk_metrics'].get("top_100", {})
        hit_rate = f"{top100.get('hit_rate', 0):.2%}"
        lift = f"{top100.get('lift', 0):.1f}x"
        report_lines.append(
            f"| {r['target']} | {roc_auc} | {accuracy} | {hit_rate} | {lift} |"
        )

    report_lines.extend([
        "",
        "## 二、HAB 评级分布",
        "",
        "| 评级 | 数量 | 占比 |",
        "|------|------|------|",
    ])

    total = sum(hab_validation["distribution"].values())
    for rating in ["H", "A", "B", "N"]:
        count = hab_validation["distribution"][rating]
        pct = f"{count / total:.1%}" if total > 0 else "0%"
        report_lines.append(f"| {rating} 级 | {count} | {pct} |")

    report_lines.extend([
        "",
        "## 三、各评级实际试驾率",
        "",
        "| 评级 | 样本数 | 7天试驾率 | 14天试驾率 | 21天试驾率 |",
        "|------|--------|-----------|------------|------------|",
    ])

    for rating in ["H", "A", "B", "N"]:
        rates = hab_validation["actual_drive_rates"][rating]
        count = rates["count"]
        rate_7d = f"{rates['rate_7d']:.2%}"
        rate_14d = f"{rates['rate_14d']:.2%}"
        rate_21d = f"{rates['rate_21d']:.2%}"
        report_lines.append(
            f"| {rating} 级 | {count} | {rate_7d} | {rate_14d} | {rate_21d} |"
        )

    report_lines.extend([
        "",
        "## 四、分层效果验证",
        "",
    ])

    if hab_validation["is_hierarchical"]:
        report_lines.append("✅ **通过**：H 级试驾率 > A 级 > B 级，分层效果符合业务预期。")
    else:
        report_lines.append("❌ **未通过**：分层效果存在问题。")
        for issue in hab_validation["issues"]:
            report_lines.append(f"- {issue}")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"验证报告已保存: {output_path}")


def main():
    """主函数"""
    args = parse_args()

    # 配置路径
    model_dir = Path(args.model_dir)

    # 判断数据加载模式：优先使用提前拆分的测试集
    test_path = args.test_path or config.data.test_data_path
    data_path = args.data_path or config.data.data_path
    actual_data_path = test_path if test_path else data_path

    output_dir = Path(args.output_dir or model_dir / "validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_file = args.log_file or (output_dir / "validation.log")
    setup_logging(log_file=str(log_file), level=logging.INFO)

    logger.info("=" * 60)
    logger.info("三模型验证")
    logger.info("=" * 60)
    logger.info(f"模型目录: {model_dir}")
    if test_path:
        logger.info(f"数据加载模式: 提前拆分的测试集")
        logger.info(f"测试集路径: {test_path}")
    else:
        logger.info(f"数据加载模式: 动态拆分")
        logger.info(f"数据路径: {data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"判定阈值: {args.threshold:.0%}")

    try:
        # 1. 加载数据
        logger.info("步骤 1/5: 加载数据")
        loader = DataLoader(actual_data_path, auto_adapt=True)
        df = loader.load()
        logger.info(f"数据: {len(df)} 行")

        # 2. 特征工程
        logger.info("步骤 2/5: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        df_processed, _ = feature_engineer.process(df)

        # 3. 加载模型
        logger.info("步骤 3/5: 加载模型")
        models = load_models(model_dir)

        # 4. 评估各模型
        logger.info("步骤 4/5: 评估各模型")
        model_results = []
        for window in TIME_WINDOWS:
            target = f"试驾标签_{window}"
            logger.info(f"  评估: {target}")
            result = evaluate_single_model(models[target], df_processed, target)
            model_results.append(result)
            logger.info(f"    ROC-AUC: {result['roc_auc']:.4f if result['roc_auc'] else 'N/A'}")

        # 5. 验证 HAB 推导
        logger.info("步骤 5/5: 验证 HAB 推导")
        hab_validation = validate_hab_derivation(
            model_results,
            args.threshold,
            df_processed,
        )

        # 输出分布
        logger.info("HAB 分布:")
        for rating, count in hab_validation["distribution"].items():
            logger.info(f"  {rating} 级: {count}")

        # 输出试驾率
        logger.info("各评级实际试驾率:")
        for rating, rates in hab_validation["actual_drive_rates"].items():
            logger.info(
                f"  {rating} 级: 7天={rates['rate_7d']:.2%}, "
                f"14天={rates['rate_14d']:.2%}, 21天={rates['rate_21d']:.2%}"
            )

        # 分层效果
        if hab_validation["is_hierarchical"]:
            logger.info("✅ 分层效果符合业务预期")
        else:
            logger.warning("❌ 分层效果存在问题:")
            for issue in hab_validation["issues"]:
                logger.warning(f"  - {issue}")

        # 生成报告
        report_path = output_dir / "validation_report.md"
        generate_validation_report(model_results, hab_validation, report_path)

        # 保存详细结果
        validation_data = {
            "model_results": [
                {
                    "target": r["target"],
                    "roc_auc": r["roc_auc"],
                    "accuracy": r["accuracy"],
                    "topk_metrics": {
                        k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                            for kk, vv in v.items()}
                        for k, v in r["topk_metrics"].items()
                    },
                }
                for r in model_results
            ],
            "hab_validation": hab_validation,
        }

        with open(output_dir / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("验证完成!")
        logger.info(f"报告路径: {report_path}")

        print("\n" + "=" * 60)
        print("验证完成")
        print("=" * 60)
        print("\n各模型 ROC-AUC:")
        for r in model_results:
            print(f"  {r['target']}: {r['roc_auc']:.4f if r['roc_auc'] else 'N/A'}")
        print(f"\n分层效果: {'✅ 通过' if hab_validation['is_hierarchical'] else '❌ 未通过'}")
        print(f"\n报告路径: {report_path}")

    except Exception as e:
        logger.error(f"验证失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
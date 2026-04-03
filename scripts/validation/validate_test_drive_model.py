#!/usr/bin/env python3
"""
试驾模型验证脚本
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer
from src.evaluation.metrics import TopKEvaluator, plot_feature_importance, plot_lift_chart
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    complete_process_if_running,
    format_training_duration,
    format_timestamp,
    get_local_now,
    get_timestamp,
    save_process_info,
    setup_logging,
    update_process_status,
)

logger = logging.getLogger(__name__)
TASK_NAME = "validate_test_drive_model"


def load_feature_metadata(model_path: Path) -> dict:
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def infer_model_type(model_path: Path) -> str | None:
    model_name = model_path.name.lower()
    if "test_drive" in model_name or "drive" in model_name:
        return "test_drive"
    metadata = load_feature_metadata(model_path)
    if str(metadata.get("label")) == "试驾标签_14天":
        return "test_drive"
    return None


def validate_test_drive_model_artifacts(metadata: dict, model_path: Path) -> None:
    pipeline_metadata = metadata.get("pipeline_metadata", {})
    if pipeline_metadata.get("pipeline_mode") == "two_stage":
        raise RuntimeError(f"检测到 OHAB 两阶段产物，当前路径不是试驾模型: {model_path}")


def find_available_model(default_path: Path) -> Path:
    if default_path.exists():
        return default_path

    models_dir = Path("outputs/models")
    if models_dir.exists():
        for name in ["test_drive_model", "test_drive_oot"]:
            candidate = models_dir / name
            if candidate.exists() and (candidate / "predictor.pkl").exists():
                return candidate
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and infer_model_type(model_dir) == "test_drive":
                return model_dir
    return default_path


def dump_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="验证试驾预测模型")
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--model-path", type=str, default="outputs/models/test_drive_model", help="模型路径")
    parser.add_argument("--data-path", type=str, default=None, help="测试数据路径（动态拆分模式）")
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument("--target", type=str, default="试驾标签_14天", help="目标变量名")
    parser.add_argument("--output-dir", type=str, default="outputs/validation/test_drive_validation", help="输出目录")
    parser.add_argument("--report-topk", type=str, default="100,500,1000", help="Top-K 列表")
    parser.add_argument(
        "--generate-plots",
        dest="generate_plots",
        action="store_true",
        default=False,
        help="生成 PNG 图表（默认关闭，服务器 CLI 环境建议保持关闭）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 判断数据加载模式：优先使用提前拆分的测试集
    test_path = args.test_path or config.data.test_data_path
    data_path = args.data_path or config.data.data_path

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_dir = Path("./outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{TASK_NAME}_{get_timestamp()}.log"

    setup_logging(log_file=str(log_file), level=logging.INFO)
    save_process_info(
        task_name=TASK_NAME,
        pid=os.getpid(),
        command=" ".join(sys.argv),
        log_file=str(log_file),
        data_path=data_path,
        target=args.target,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )
    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())
    evaluation_start_time = get_local_now()

    # 确定实际使用的数据路径
    actual_data_path = test_path if test_path else data_path

    logger.info("=" * 60)
    logger.info("试驾模型验证")
    logger.info("=" * 60)
    logger.info(f"评估开始时间: {format_timestamp(evaluation_start_time)}")
    logger.info(f"模型路径: {args.model_path}")
    if test_path:
        logger.info(f"数据加载模式: 提前拆分的测试集")
        logger.info(f"测试集路径: {test_path}")
    else:
        logger.info(f"数据加载模式: 动态拆分")
        logger.info(f"数据路径: {data_path}")

    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            model_path = find_available_model(model_path)
        if not model_path.exists():
            raise RuntimeError(f"未找到试驾模型目录: {args.model_path}")

        metadata = load_feature_metadata(model_path)
        validate_test_drive_model_artifacts(metadata, model_path)
        interaction_context = metadata.get("interaction_context", {})
        label_policy = metadata.get("label_policy", {})

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictor = LeadScoringPredictor.load(str(model_path))
        target = predictor.label if predictor.label else args.target

        loader = DataLoader(actual_data_path, auto_adapt=True)
        df = loader.load()

        if label_policy:
            from src.data.label_policy import apply_ohab_label_policy
            df = apply_ohab_label_policy(df, target, label_policy)

        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
            interaction_context=interaction_context,
        )
        df_processed, _ = feature_engineer.transform(df, interaction_context=interaction_context)

        excluded_columns = get_excluded_columns(target)
        cols_to_drop = [col for col in excluded_columns if col in df_processed.columns and col != target]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)

        metrics = predictor.evaluate(df_processed)
        y_true = df_processed[target].values
        y_pred = predictor.predict(df_processed)
        y_proba = predictor.get_positive_proba(df_processed)
        topk_values = [int(item.strip()) for item in args.report_topk.split(",") if item.strip()]
        topk_evaluator = TopKEvaluator(y_true, y_proba)
        topk_metrics = topk_evaluator.compute_topk_metrics(k_values=topk_values)
        decile_lift = topk_evaluator.compute_lift_by_decile()

        results_df = df_processed.copy()
        results_df["预测标签"] = y_pred
        results_df["预测概率"] = y_proba
        results_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        dump_json(output_dir / "metrics.json", metrics)
        dump_json(output_dir / "topk_metrics.json", topk_metrics)
        decile_lift.to_csv(output_dir / "lift_deciles.csv", index=False, encoding="utf-8-sig")

        try:
            importance = predictor.get_feature_importance(df_processed)
            importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
            if args.generate_plots:
                plot_feature_importance(importance, output_path=str(output_dir / "feature_importance.png"))
        except Exception as exc:
            logger.warning("特征重要性计算失败: %s", exc)

        if args.generate_plots:
            try:
                plot_lift_chart(decile_lift, output_path=str(output_dir / "lift_chart.png"))
            except Exception as exc:
                logger.warning("Lift 图生成失败: %s", exc)

        with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("试驾模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"数据路径: {data_path}\n")
            f.write(f"样本数量: {len(df_processed)}\n")
            f.write(f"目标变量: {target}\n")
            f.write(f"最佳模型: {predictor._predictor.model_best}\n\n")
            f.write("指标摘要\n")
            f.write("-" * 40 + "\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    f.write(f"{metric_name}: {metric_value:.6f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")
            f.write("\nTop-K 指标\n")
            f.write("-" * 40 + "\n")
            for key, row in topk_metrics.items():
                f.write(f"{key}: hit_rate={row['hit_rate']:.4f}, lift={row['lift']:.4f}\n")

        evaluation_end_time = get_local_now()
        duration_seconds = (evaluation_end_time - evaluation_start_time).total_seconds()
        dump_json(
            output_dir / "evaluation_summary.json",
            {
                "model_type": "test_drive",
                "model_path": str(model_path),
                "target": target,
                "metrics": metrics,
                "topk_metrics": topk_metrics,
                "duration_seconds": duration_seconds,
            },
        )
        logger.info("验证完成，耗时: %s", format_training_duration(duration_seconds))
    except Exception as e:
        logger.error("验证失败: %s", e, exc_info=True)
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

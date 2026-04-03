"""
到店预测训练脚本 (统一智能版)

功能说明：
1. 自动探查数据时间跨度，智能选择 OOT 切分或随机切分。
2. 支持手动指定时间切分点 (--train-end, --valid-end)。
3. 自动开启防泄漏指纹记录 (test_ids)，确保评估结果真实。
4. 针对二分类任务优化，输出 Top-K 和 Lift 指标。
"""

import argparse
import atexit
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer, smart_split_data, split_data_oot_three_way
from src.evaluation.metrics import (
    ModelReport,
    TopKEvaluator,
    plot_feature_importance,
    plot_lift_chart,
)
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_disk_space,
    complete_process_if_running,
    format_training_duration,
    get_local_now,
    get_preset_disk_requirement,
    get_timestamp,
    save_process_info,
    setup_logging,
    update_process_status,
)

logger = logging.getLogger(__name__)

TASK_NAME = "train_arrive"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="到店预测模型训练 (统一智能版)")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="到店标签_14天",
        help="目标变量名",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="high_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="模型预设",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=3600,
        help="训练时间限制（秒）",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="训练集截止日期 (YYYY-MM-DD)。不提供则自动计算",
    )
    parser.add_argument(
        "--valid-end",
        type=str,
        default=None,
        help="验证集截止日期 (YYYY-MM-DD)。不提供则自动计算",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--num-bag-folds",
        type=int,
        default=5,
        help="交叉验证折数（0 表示禁用）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径",
    )
    # 内存控制参数
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=0.8,
        help="最大内存使用比例（默认 0.8，低内存机器建议 0.6-0.8）",
    )
    parser.add_argument(
        "--exclude-memory-heavy-models",
        action="store_true",
        help="排除内存密集型模型（KNN, RF, XT）以降低内存使用",
    )
    parser.add_argument(
        "--num-folds-parallel",
        type=int,
        default=None,
        help="并行训练的 fold 数量（默认自动，低内存机器建议 2-3）",
    )
    parser.add_argument(
        "--generate-plots",
        dest="generate_plots",
        action="store_true",
        default=False,
        help="生成 PNG 图表（默认关闭，服务器 CLI 环境建议保持关闭）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置路径
    data_path = args.data_path or config.data.data_path
    target_label = args.target
    output_dir = Path(args.output_dir or config.output.models_dir) / "arrive_model"
    
    # 日志设置
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_dir = Path("./outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
        log_file = log_dir / f"{TASK_NAME}_{timestamp}.log"
    setup_logging(log_file=str(log_file), level=logging.INFO)

    # 记录进程信息
    save_process_info(
        task_name=TASK_NAME,
        pid=os.getpid(),
        command=" ".join(sys.argv),
        log_file=str(log_file),
        data_path=data_path,
        target=target_label,
        output_dir=str(output_dir),
    )

    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())

    # 记录训练开始时间
    train_start_time = get_local_now()

    logger.info("=" * 60)
    logger.info("到店预测模型训练 (统一自适应版)")
    logger.info("=" * 60)
    logger.info(f"训练开始时间: {train_start_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")

    # 1. 加载数据
    try:
        logger.info("步骤 1/8: 加载数据")
        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()

        # 2. 特征工程
        logger.info("步骤 2/8: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        df_processed, feature_metadata = feature_engineer.process(df)

        # 3. 智能数据切分
        logger.info("步骤 3/8: 执行智能数据切分")
        if args.train_end and args.valid_end:
            logger.info(f"采用手动 OOT 切分: {args.train_end} / {args.valid_end}")
            train_df, valid_df, test_df = split_data_oot_three_way(
                df_processed, target_label, "线索创建时间", args.train_end, args.valid_end
            )
            split_info = {"mode": "oot_manual", "train_end": args.train_end, "valid_end": args.valid_end}
        else:
            train_df, valid_df, test_df, split_info = smart_split_data(df_processed, target_label)
        
        feature_metadata["split_info"] = split_info
        
        # 4. 训练模型
        logger.info("步骤 4/8: 模型训练")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存防泄漏标记
        with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)

        excluded_columns = get_excluded_columns(target_label)

        # 内存密集型模型排除
        excluded_model_types = None
        if args.exclude_memory_heavy_models:
            excluded_model_types = ["KNN", "RF", "XT"]
            logger.info("排除内存密集型模型: KNN, RF, XT")

        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="roc_auc",
            problem_type="binary",
            sample_weight="balance_weight",
            max_memory_usage_ratio=args.max_memory_ratio,
            excluded_model_types=excluded_model_types,
            num_folds_parallel=args.num_folds_parallel,
        )

        train_kwargs = {
            "train_data": train_df,
            "presets": args.preset,
            "time_limit": args.time_limit,
            "excluded_columns": excluded_columns,
            "num_bag_folds": args.num_bag_folds if args.num_bag_folds > 0 else None,
        }
        if len(valid_df) > 0:
            train_kwargs["tuning_data"] = valid_df

        predictor.train(**train_kwargs)

        # 5. 模型评估
        logger.info("步骤 5/8: 模型评估")
        metrics = predictor.evaluate(test_df)
        logger.info(f"测试集 ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

        # 6. Top-K 评估
        logger.info("步骤 6/8: Top-K 评估")
        y_proba = predictor.get_positive_proba(test_df)
        y_true = test_df[target_label].values
        topk_evaluator = TopKEvaluator(y_true, y_proba)
        topk_metrics = topk_evaluator.compute_topk_metrics(k_values=[100, 500, 1000])
        for k, v in topk_metrics.items():
            logger.info(f"  {k}: 命中率 {v['hit_rate']:.2%}, Lift {v['lift']:.2f}x")

        # 7. 生成报告与图表
        logger.info("步骤 7/8: 生成报告与可视化")
        importance = predictor.get_feature_importance(test_df)
        importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        decile_lift = topk_evaluator.compute_lift_by_decile()
        decile_lift.to_csv(output_dir / "lift_deciles.csv", index=False, encoding="utf-8-sig")

        if args.generate_plots:
            plot_feature_importance(importance, output_path=str(output_dir / "feature_importance.png"))
            plot_lift_chart(decile_lift, output_path=str(output_dir / "lift_chart.png"))

        # 8. 保存与清理
        logger.info("步骤 8/8: 保存与清理")
        predictor.save()
        predictor.cleanup(keep_best_only=True)

        # 计算并输出训练耗时
        train_end_time = get_local_now()
        duration_seconds = (train_end_time - train_start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"训练总耗时: {format_training_duration(duration_seconds)}")
        logger.info(f"训练结束时间: {train_end_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
        logger.info(f"训练完成! 模型已保存至: {output_dir}")

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))
        raise

if __name__ == "__main__":
    main()

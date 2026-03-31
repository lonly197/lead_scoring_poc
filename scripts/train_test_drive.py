"""
试驾预测训练脚本

训练预测线索 14 天内试驾概率的模型。
支持前台运行和后台运行模式。
"""

import argparse
import atexit
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer, split_data
from src.evaluation.metrics import (
    ModelReport,
    TopKEvaluator,
    plot_feature_importance,
    plot_lift_chart,
)
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_data_quality,
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

TASK_NAME = "train_test_drive"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="试驾预测模型训练")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径（动态拆分模式）",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="训练集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="试驾标签_14天",
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
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./outputs/logs",
        help="日志目录",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径（由 run.py 传入）",
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
    parser.add_argument(
        "--included-model-types",
        type=str,
        default=None,
        help="指定训练的模型类型（逗号分隔），如 CAT,GBM。空值使用默认预设所有模型",
    )
    parser.add_argument(
        "--num-bag-folds",
        type=int,
        default=None,
        help="Bagging folds 数量（None=使用预设默认值，1=禁用 bagging）",
    )
    parser.add_argument(
        "--num-stack-levels",
        type=int,
        default=None,
        help="Stacking 层数（None=使用预设默认值，0=禁用 stacking）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 判断数据加载模式
    train_path = args.train_path or config.data.train_data_path
    test_path = args.test_path or config.data.test_data_path
    use_split_data = train_path and test_path
    data_path = args.data_path or config.data.data_path

    target_label = args.target
    output_dir = Path(args.output_dir or config.output.models_dir) / "test_drive_model"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件（优先使用 run.py 传入的路径）
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        timestamp = get_timestamp()
        log_file = log_dir / f"{TASK_NAME}_{timestamp}.log"
    setup_logging(log_file=str(log_file), level=logging.INFO)

    # 记录进程信息
    command = " ".join(sys.argv)
    process_info_path = save_process_info(
        task_name=TASK_NAME,
        pid=os.getpid(),
        command=command,
        log_file=str(log_file),
        data_path=data_path,
        target=target_label,
        preset=args.preset,
        output_dir=str(output_dir),
    )

    # 注册退出处理
    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())

    # 记录训练开始时间
    train_start_time = get_local_now()

    # 启动信息
    logger.info("=" * 60)
    logger.info("试驾预测模型训练")
    logger.info("=" * 60)
    logger.info(f"训练开始时间: {train_start_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
    logger.info(f"进程 ID: {os.getpid()}")
    logger.info(f"日志文件: {log_file}")
    if use_split_data:
        logger.info(f"数据加载模式: 提前拆分")
        logger.info(f"训练集路径: {train_path}")
        logger.info(f"测试集路径: {test_path}")
    else:
        logger.info(f"数据加载模式: 动态拆分")
        logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")
    logger.info(f"输出目录: {output_dir}")
    if args.included_model_types:
        logger.info(f"指定模型类型: {args.included_model_types}")
    if args.num_bag_folds is not None:
        logger.info(f"Bagging folds: {args.num_bag_folds}")
    if args.num_stack_levels is not None:
        logger.info(f"Stacking层数: {args.num_stack_levels}")

    # 磁盘空间检查
    required_gb = get_preset_disk_requirement(args.preset)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G ({args.preset})")

    if not disk_info["sufficient"]:
        logger.warning(f"磁盘空间不足！建议使用 medium_quality preset 或清理磁盘空间")

    try:
        # 1. 加载数据
        logger.info("步骤 1/6: 加载数据")

        # 判断数据加载模式：优先使用提前拆分的数据文件
        train_path = args.train_path or config.data.train_data_path
        test_path = args.test_path or config.data.test_data_path
        use_split_data = train_path and test_path

        if use_split_data:
            # 模式一：提前拆分的数据文件
            logger.info(f"使用提前拆分的数据文件（训练集: {train_path}, 测试集: {test_path}）")
            train_loader = DataLoader(train_path, auto_adapt=True)
            test_loader = DataLoader(test_path, auto_adapt=True)
            df_train = train_loader.load()
            df_test = test_loader.load()

            # 特征工程（训练集 fit_transform，测试集 transform）
            logger.info("步骤 2/6: 特征工程")
            feature_engineer = FeatureEngineer(
                time_columns=config.feature.time_columns,
                numeric_columns=config.feature.numeric_features,
            )
            train_df, feature_metadata = feature_engineer.fit_transform(df_train)
            test_df, _ = feature_engineer.transform(df_test, interaction_context=feature_metadata.get("interaction_context"))

            quality_train = check_data_quality(train_df)
            quality_test = check_data_quality(test_df)
            logger.info(f"训练集: {quality_train['total_rows']} 行, {quality_train['total_columns']} 列")
            logger.info(f"测试集: {quality_test['total_rows']} 行, {quality_test['total_columns']} 列")
        else:
            # 模式二：动态拆分
            data_path = args.data_path or config.data.data_path
            logger.info(f"使用单一数据文件，动态拆分: {data_path}")
            loader = DataLoader(data_path, auto_adapt=True)
            df = loader.load()
            quality = check_data_quality(df)
            logger.info(f"数据: {quality['total_rows']} 行, {quality['total_columns']} 列")

            # 特征工程
            logger.info("步骤 2/6: 特征工程")
            feature_engineer = FeatureEngineer(
                time_columns=config.feature.time_columns,
                numeric_columns=config.feature.numeric_features,
            )
            df_processed, feature_metadata = feature_engineer.process(df)

            # 数据划分
            logger.info("步骤 3/6: 数据划分")
            train_df, test_df = split_data(
                df_processed,
                target_label=target_label,
                test_size=args.test_size,
                stratify=True,
            )
            logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

        # 保存特征工程元数据（时间特征信息）
        import json
        feature_metadata_path = output_dir / "feature_metadata.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"特征元数据已保存: {feature_metadata_path}")

        # 4. 训练模型
        logger.info("步骤 4/6: 模型训练")
        excluded_columns = get_excluded_columns(target_label)

        # 内存密集型模型排除
        excluded_model_types = None
        if args.exclude_memory_heavy_models:
            excluded_model_types = ["KNN", "RF", "XT"]
            logger.info("排除内存密集型模型: KNN, RF, XT")

        # 指定训练的模型类型（白名单模式）
        included_model_types = None
        if args.included_model_types:
            included_model_types = [t.strip() for t in args.included_model_types.split(",")]
            logger.info(f"指定训练模型类型: {included_model_types}")

        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="roc_auc",
            problem_type="binary",
            sample_weight="balance_weight",  # 自动平衡类别权重
            # weight_evaluation 设为 False 以兼容 AutoGluon 动态堆叠
            # 样本权重已在训练时生效，评估时无需额外权重
            weight_evaluation=False,
            max_memory_usage_ratio=args.max_memory_ratio,
            excluded_model_types=excluded_model_types,
            num_folds_parallel=args.num_folds_parallel,
            included_model_types=included_model_types,
            num_bag_folds=args.num_bag_folds,
            num_stack_levels=args.num_stack_levels,
        )
        logger.info("启用类别权重自动平衡 (sample_weight='balance_weight')")

        predictor.train(
            train_data=train_df,
            presets=args.preset,
            time_limit=args.time_limit,
            excluded_columns=excluded_columns,
        )

        # 5. 评估
        logger.info("步骤 5/6: 模型评估")
        metrics = predictor.evaluate(test_df)
        logger.info(f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

        y_proba = predictor.get_positive_proba(test_df)
        y_true = test_df[target_label].values
        topk_evaluator = TopKEvaluator(y_true, y_proba)
        topk_metrics = topk_evaluator.compute_topk_metrics()

        for k, v in topk_metrics.items():
            logger.info(f"  {k}: 命中率 {v['hit_rate']:.2%}, Lift {v['lift']:.2f}x")

        try:
            importance = predictor.get_feature_importance(test_df)
            importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
            if args.generate_plots:
                plot_feature_importance(
                    importance, output_path=str(output_dir / "feature_importance.png")
                )
        except Exception as e:
            logger.warning(f"特征重要性计算失败: {e}")
            importance = None

        # 6. 保存
        logger.info("步骤 6/6: 保存模型")
        predictor.save()

        # 报告
        report_generator = ModelReport(output_dir / "reports")
        report_generator.generate(
            model_metrics=metrics,
            topk_metrics=topk_metrics,
            feature_importance=importance,
            model_name="test_drive_model",
        )

        # 计算并输出训练耗时
        train_end_time = get_local_now()
        duration_seconds = (train_end_time - train_start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"训练总耗时: {format_training_duration(duration_seconds)}")
        logger.info(f"训练结束时间: {train_end_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
        logger.info("训练完成!")
        logger.info(f"模型路径: {output_dir}")
        logger.info(f"日志路径: {log_file}")

        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"训练耗时: {format_training_duration(duration_seconds)}")
        print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        for k, v in topk_metrics.items():
            print(f"{k}: {v['hit_rate']:.2%}, Lift: {v['lift']:.2f}x")
        print(f"\n日志文件: {log_file}")

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))

        # 清理失败的模型目录
        if output_dir.exists():
            import shutil
            logger.info(f"清理失败的模型目录: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                logger.info("清理完成")
            except Exception as cleanup_error:
                logger.warning(f"清理失败: {cleanup_error}")

        # 关闭 Ray 运行时
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray 运行时已关闭")
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()

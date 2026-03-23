"""
到店预测训练脚本

训练预测线索 14 天内到店概率的模型。
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

from config.config import config, get_excluded_columns, get_feature_columns
from src.data.loader import (
    DataLoader,
    FeatureEngineer,
    prepare_features,
    split_data,
)
from src.evaluation.metrics import (
    ModelReport,
    TopKEvaluator,
    compute_all_metrics,
    plot_feature_importance,
    plot_lift_chart,
)
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_data_quality,
    check_disk_space,
    get_preset_disk_requirement,
    get_timestamp,
    print_separator,
    save_process_info,
    setup_logging,
    update_process_status,
)

logger = logging.getLogger(__name__)

# 任务名称
TASK_NAME = "train_arrive"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="到店预测模型训练")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径（默认使用配置中的路径）",
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
        help="AutoGluon 预设",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=3600,
        help="训练时间限制（秒）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="后台运行模式",
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

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置路径
    data_path = args.data_path or config.data.data_path
    target_label = args.target
    output_dir = Path(args.output_dir or config.output.models_dir) / "arrive_model"
    log_dir = Path(args.log_dir)

    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件（优先使用 run.py 传入的路径）
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        timestamp = get_timestamp()
        log_file = log_dir / f"{TASK_NAME}_{timestamp}.log"

    # 设置日志（同时输出到控制台和文件）
    setup_logging(log_file=str(log_file), level=logging.INFO)

    # 记录进程信息（用于监控）
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

    # 注册退出时的清理函数
    def cleanup(status: str = "completed"):
        update_process_status(TASK_NAME, os.getpid(), status)

    atexit.register(cleanup, "completed")

    # 打印启动信息
    logger.info("=" * 60)
    logger.info("到店预测模型训练")
    logger.info("=" * 60)
    logger.info(f"进程 ID: {os.getpid()}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"进程信息: {process_info_path}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"预设: {args.preset}")
    logger.info(f"时间限制: {args.time_limit}秒")
    logger.info("=" * 60)

    # 磁盘空间检查
    required_gb = get_preset_disk_requirement(args.preset)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G ({args.preset})")

    if not disk_info["sufficient"]:
        logger.warning(f"磁盘空间不足！建议使用 medium_quality preset 或清理磁盘空间")

    try:
        # 1. 加载数据
        logger.info("步骤 1/10: 加载数据")
        loader = DataLoader(data_path, random_seed=args.random_seed)
        df = loader.load()

        quality_report = check_data_quality(df)
        logger.info(f"数据质量: {quality_report['total_rows']} 行, {quality_report['total_columns']} 列")

        # 2. 特征工程
        logger.info("步骤 2/10: 特征工程")
        excluded_columns = get_excluded_columns(target_label)

        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )

        df_processed, feature_metadata = feature_engineer.process(df)
        logger.info(f"特征工程完成: {len(df_processed.columns)} 列")

        # 保存特征工程元数据（时间特征信息）
        import json
        feature_metadata_path = output_dir / "feature_metadata.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"特征元数据已保存: {feature_metadata_path}")

        # 3. 数据划分
        logger.info("步骤 3/10: 数据划分")
        train_df, test_df = split_data(
            df_processed,
            target_label=target_label,
            test_size=args.test_size,
            random_seed=args.random_seed,
            stratify=True,
        )
        logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

        # 4. 训练模型
        logger.info("步骤 4/10: 模型训练")
        logger.info(f"开始训练 (预设: {args.preset}, 时间限制: {args.time_limit}秒)")

        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="roc_auc",
            problem_type="binary",
            sample_weight="balance_weight",  # 自动平衡类别权重
            weight_evaluation=True,
        )
        logger.info("启用类别权重自动平衡 (sample_weight='balance_weight')")

        predictor.train(
            train_data=train_df,
            presets=args.preset,
            time_limit=args.time_limit,
            excluded_columns=excluded_columns,
        )

        # 5. 模型评估
        logger.info("步骤 5/10: 模型评估")
        metrics = predictor.evaluate(test_df)
        logger.info(f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

        # 6. Top-K 评估
        logger.info("步骤 6/10: Top-K 评估")
        y_proba = predictor.get_positive_proba(test_df)
        y_true = test_df[target_label].values

        topk_evaluator = TopKEvaluator(y_true, y_proba)
        topk_metrics = topk_evaluator.compute_topk_metrics(k_values=[100, 500, 1000, 2000])

        for k, v in topk_metrics.items():
            logger.info(f"  {k}: 命中率 {v['hit_rate']:.2%}, Lift {v['lift']:.2f}x")

        # 十分位 Lift
        decile_lift = topk_evaluator.compute_lift_by_decile()

        # 7. 特征重要性
        logger.info("步骤 7/10: 特征重要性分析")
        try:
            importance = predictor.get_feature_importance(test_df)
            logger.info(f"Top 5 特征: {importance.head(5).index.tolist()}")

            importance_path = output_dir / "feature_importance.png"
            plot_feature_importance(importance, top_n=20, output_path=str(importance_path))
        except Exception as e:
            logger.warning(f"特征重要性计算失败: {e}")
            importance = None

        # 8. 生成报告
        logger.info("步骤 8/10: 生成报告")
        report_generator = ModelReport(output_dir / "reports")
        report = report_generator.generate(
            model_metrics=metrics,
            topk_metrics=topk_metrics,
            feature_importance=importance,
            stratified_metrics=decile_lift,
            model_name="arrive_model",
        )

        lift_path = output_dir / "lift_chart.png"
        plot_lift_chart(decile_lift, output_path=str(lift_path))

        # 9. 生成 Top-K 名单
        logger.info("步骤 9/10: 生成 Top-K 名单")
        if "线索唯一ID" in test_df.columns:
            ids = test_df["线索唯一ID"].values
        else:
            ids = list(range(len(test_df)))

        topk_list = report_generator.generate_topk_list(
            ids=ids,
            y_proba=y_proba,
            y_true=y_true,
            k=1000,
            model_name="arrive_model",
            id_column="线索唯一ID",
        )

        # 10. 保存模型
        logger.info("步骤 10/10: 保存模型")
        predictor.save()

        # 完成
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)
        logger.info(f"模型路径: {output_dir}")
        logger.info(f"报告路径: {output_dir / 'reports'}")
        logger.info(f"日志路径: {log_file}")

        # 打印关键指标
        print("\n" + "=" * 60)
        print("关键指标总结")
        print("=" * 60)
        print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print("\nTop-K 命中率:")
        for k, v in topk_metrics.items():
            print(f"  {k}: {v['hit_rate']:.2%} ({v['hit_count']}/{v['k']}), Lift: {v['lift']:.2f}x")
        print("=" * 60)
        print(f"\n日志文件: {log_file}")

        return predictor

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
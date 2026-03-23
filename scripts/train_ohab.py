"""
OHAB 评级预测训练脚本

训练预测线索 OHAB 评级的多分类模型。
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
from src.evaluation.metrics import ModelReport, plot_feature_importance
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_data_quality,
    check_disk_space,
    get_preset_disk_requirement,
    get_timestamp,
    save_process_info,
    setup_logging,
    suggest_preset_by_disk,
    update_process_status,
)

logger = logging.getLogger(__name__)

TASK_NAME = "train_ohab"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OHAB 评级模型训练")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="线索评级_试驾前",
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
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
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
    output_dir = Path(args.output_dir or config.output.models_dir) / "ohab_model"
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
    atexit.register(update_process_status, TASK_NAME, os.getpid(), "completed")

    # 启动信息
    logger.info("=" * 60)
    logger.info("OHAB 评级模型训练")
    logger.info("=" * 60)
    logger.info(f"进程 ID: {os.getpid()}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")
    logger.info(f"输出目录: {output_dir}")

    # 磁盘空间检查
    required_gb = get_preset_disk_requirement(args.preset)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G ({args.preset})")

    if not disk_info["sufficient"]:
        suggested_preset = suggest_preset_by_disk(disk_info["free_gb"])
        logger.warning(f"磁盘空间不足！建议使用 {suggested_preset} preset")
        logger.warning(f"或手动清理磁盘空间后重试")

    try:
        # 1. 加载数据
        logger.info("步骤 1/6: 加载数据")
        loader = DataLoader(data_path)
        df = loader.load()

        # 过滤掉 Unknown 类别
        if target_label in df.columns:
            df = df[df[target_label] != "Unknown"].copy()
            logger.info(f"过滤 Unknown 后: {len(df)} 行")
            logger.info(f"类别分布:\n{df[target_label].value_counts()}")

        quality = check_data_quality(df)
        logger.info(f"数据: {quality['total_rows']} 行, {quality['total_columns']} 列")

        # 2. 特征工程
        logger.info("步骤 2/6: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        df_processed, feature_metadata = feature_engineer.process(df)

        # 保存特征工程元数据（时间特征信息）
        import json
        feature_metadata_path = output_dir / "feature_metadata.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"特征元数据已保存: {feature_metadata_path}")

        # 3. 数据划分
        logger.info("步骤 3/6: 数据划分")
        train_df, test_df = split_data(
            df_processed, target_label=target_label, test_size=0.2, stratify=True
        )
        logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

        # 4. 训练模型
        logger.info("步骤 4/6: 模型训练")
        excluded_columns = get_excluded_columns(target_label)

        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="accuracy",
            problem_type="multiclass",
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

        # 5. 评估
        logger.info("步骤 5/6: 模型评估")
        metrics = predictor.evaluate(test_df)
        logger.info(f"评估指标: {metrics}")

        # 混淆矩阵
        from sklearn.metrics import classification_report, confusion_matrix

        y_true = test_df[target_label].values
        y_pred = predictor.predict(test_df)

        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"混淆矩阵:\n{cm}")

        report = classification_report(y_true, y_pred)
        logger.info(f"分类报告:\n{report}")

        try:
            importance = predictor.get_feature_importance(test_df)
            plot_feature_importance(
                importance, output_path=str(output_dir / "feature_importance.png")
            )
        except Exception as e:
            logger.warning(f"特征重要性计算失败: {e}")
            importance = None

        # 6. 保存
        logger.info("步骤 6/6: 保存模型")
        predictor.save()

        # 清理非最佳模型释放空间
        disk_usage = predictor.get_disk_usage()
        logger.info(f"模型目录大小: {disk_usage.get('total_size_mb', 0):.1f} MB")

        cleanup_result = predictor.cleanup(keep_best_only=True)
        if cleanup_result.get("status") == "success":
            logger.info(f"清理完成: 释放 {cleanup_result.get('freed_mb', 0):.1f} MB")

        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"模型路径: {output_dir}")
        logger.info(f"日志路径: {log_file}")

        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"\n日志文件: {log_file}")

        # 关闭 Ray 运行时
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

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
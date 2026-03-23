"""
OHAB 评级训练脚本（OOT 验证版本）

使用新数据格式，预测 OHAB 线索评级，采用三层 OOT 时间切分。

支持前台运行和后台运行模式。
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer, split_data_oot_three_way
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_disk_space,
    get_preset_disk_requirement,
    get_timestamp,
    save_process_info,
    setup_logging,
    update_process_status,
)

logger = logging.getLogger(__name__)

TASK_NAME = "train_ohab_oot"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OHAB 评级训练（OOT验证）")

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
        help="目标变量",
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
        "--train-end",
        type=str,
        default="2026-03-11",
        help="训练集截止日期",
    )
    parser.add_argument(
        "--valid-end",
        type=str,
        default="2026-03-16",
        help="验证集截止日期",
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
    output_dir = Path(args.output_dir or "./outputs/models/ohab_oot")
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
    logger.info("OHAB 评级训练（OOT验证）")
    logger.info("=" * 60)
    logger.info(f"进程 ID: {os.getpid()}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")
    logger.info(f"OOT切分: 训练<{args.train_end}, 验证<{args.valid_end}, 测试>={args.valid_end}")
    logger.info(f"输出目录: {output_dir}")

    # 磁盘空间检查
    required_gb = get_preset_disk_requirement(args.preset)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G ({args.preset})")

    if not disk_info["sufficient"]:
        logger.warning(f"磁盘空间不足！建议使用 medium_quality preset 或手动清理磁盘空间")

    try:
        # 1. 加载数据
        logger.info("步骤 1/6: 加载数据")
        # 注意：auto_adapt=True 启用数据适配器处理新格式数据
        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()

        data_format = loader.get_data_format()
        logger.info(f"数据格式: {data_format}")
        logger.info(f"数据量: {len(df):,} 行")

        # 过滤掉 Unknown 评级
        if target_label in df.columns:
            df = df[df[target_label] != "Unknown"].copy()
            logger.info(f"过滤 Unknown 后: {len(df):,} 行")
            logger.info(f"OHAB 分布:\n{df[target_label].value_counts()}")

        # 2. 特征工程
        logger.info("步骤 2/6: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        df_processed, _ = feature_engineer.process(df)

        # 3. OOT 三层时间切分
        logger.info("步骤 3/6: OOT 三层时间切分")
        train_df, valid_df, test_df = split_data_oot_three_way(
            df=df_processed,
            target_label=target_label,
            time_column="线索创建时间",
            train_end=args.train_end,
            valid_end=args.valid_end,
        )

        # 4. 训练模型
        logger.info("步骤 4/6: 模型训练")
        output_dir.mkdir(parents=True, exist_ok=True)

        excluded_columns = get_excluded_columns(target_label)

        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="log_loss",
            problem_type="multiclass",
            preset=args.preset,
            sample_weight="balance_weight",
            weight_evaluation=True,
        )
        logger.info("启用类别权重自动平衡 (sample_weight='balance_weight')")

        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

        predictor.fit(
            train_data=train_valid_df,
            excluded_columns=excluded_columns,
            time_limit=args.time_limit,
        )

        # 5. 评估
        logger.info("步骤 5/6: 模型评估")

        from sklearn.metrics import classification_report, confusion_matrix

        def evaluate_multiclass(df, name):
            y_true = df[target_label].values
            y_pred = predictor.predict(df)

            logger.info(f"【{name} 分类报告】")
            report = classification_report(y_true, y_pred, output_dict=True)
            logger.info(classification_report(y_true, y_pred))

            labels = sorted(df[target_label].unique())
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            logger.info(f"混淆矩阵 (行=真实, 列=预测):")
            logger.info(f"标签: {labels}")
            logger.info(cm)

            return report

        valid_report = evaluate_multiclass(valid_df, "验证集")
        test_report = evaluate_multiclass(test_df, "测试集")

        # 6. 保存
        logger.info("步骤 6/6: 保存模型和报告")

        report = {
            "timestamp": datetime.now().isoformat(),
            "data_info": {
                "data_path": str(data_path),
                "data_format": data_format,
                "total_rows": len(df),
                "target": target_label,
            },
            "oot_split": {
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "test_rows": len(test_df),
            },
            "model_info": {
                "preset": args.preset,
                "best_model": predictor.get_model_name(),
            },
            "valid_report": valid_report,
            "test_report": test_report,
        }

        report_path = output_dir / "reports" / "ohab_oot_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {report_path}")

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
            except Exception as cleanup_error:
                logger.warning(f"清理失败: {cleanup_error}")

        # 关闭 Ray 运行时
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

        raise


if __name__ == "__main__":
    import pandas as pd
    main()
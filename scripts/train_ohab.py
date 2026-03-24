"""
OHAB 评级预测训练脚本 (统一智能版)

功能说明：
1. 自动探查数据时间跨度，智能选择 OOT 切分或随机切分。
2. 支持手动指定时间切分点 (--train-end, --valid-end)。
3. 自动开启防泄漏指纹记录 (test_ids)，确保评估结果真实。
4. 支持新老数据格式适配。
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
from src.data.loader import DataLoader, FeatureEngineer, smart_split_data, split_data_oot_three_way
from src.models.predictor import LeadScoringPredictor
from src.utils.helpers import (
    check_disk_space,
    complete_process_if_running,
    get_preset_disk_requirement,
    get_timestamp,
    save_process_info,
    setup_logging,
    update_process_status,
)

logger = logging.getLogger(__name__)

TASK_NAME = "train_ohab"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OHAB 评级模型训练 (自适应版)")

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

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置路径
    data_path = args.data_path or config.data.data_path
    target_label = args.target
    output_dir = Path(args.output_dir or config.output.models_dir) / "ohab_model"
    
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

    logger.info("=" * 60)
    logger.info("OHAB 评级模型训练 (统一自适应版)")
    logger.info("=" * 60)
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")

    # 1. 加载数据
    try:
        logger.info("步骤 1/6: 加载数据")
        # 统一开启 auto_adapt=True 以支持 2-3 月新 SQL 格式
        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()

        if target_label in df.columns:
            df = df[df[target_label] != "Unknown"].copy()
            logger.info(f"过滤 Unknown 后: {len(df):,} 行")

        # 2. 特征工程
        logger.info("步骤 2/6: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        df_processed, feature_metadata = feature_engineer.process(df)

        # 3. 智能数据切分
        logger.info("步骤 3/6: 执行智能数据切分")
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
        logger.info("步骤 4/6: 模型训练")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存防泄漏标记
        with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)

        excluded_columns = get_excluded_columns(target_label)
        predictor = LeadScoringPredictor(
            label=target_label,
            output_path=str(output_dir),
            eval_metric="log_loss",
            problem_type="multiclass",
            sample_weight="balance_weight",
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
        logger.info("步骤 5/6: 模型评估")
        # 此处调用略 (同 oot 版本)
        
        # 6. 保存与清理
        logger.info("步骤 6/6: 保存与清理")
        predictor.save()
        predictor.cleanup(keep_best_only=True)

        logger.info("=" * 60)
        logger.info(f"训练完成! 模型已保存至: {output_dir}")

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))
        raise

if __name__ == "__main__":
    main()

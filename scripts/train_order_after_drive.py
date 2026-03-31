"""
三模型下订预测训练脚本（试驾后下订商谈阶段）

训练预测线索 7/14/21 天内下订概率的三个模型。
用于后续基于业务规则推导 HAB 评级。

业务规则（试驾后下订商谈阶段）：
- H级：7天内下订，下次联络最晚时间 < 2天
- A级：14天内下订，下次联络最晚时间 < 5天
- B级：21天内下订，下次联络最晚时间 < 7天

使用方法：
    uv run python scripts/train_order_after_drive.py \\
        --train-path ./data/order_after_drive_v2_train.parquet \\
        --test-path ./data/order_after_drive_v2_test.parquet \\
        --preset high_quality \\
        --included-model-types CAT
"""

# === Ray 配置（必须在导入 Ray 之前设置）===
import os
from pathlib import Path

# 设置 Ray 临时目录（避免使用 /tmp 导致磁盘空间警告）
_project_root = Path(__file__).parent.parent
_ray_temp_dir = os.getenv("RAY_TEMP_DIR", str(_project_root / ".ray_tmp"))
os.environ["RAY_TEMP_DIR"] = _ray_temp_dir
Path(_ray_temp_dir).mkdir(parents=True, exist_ok=True)

import argparse
import atexit
import json
import logging
import multiprocessing as mp
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer, split_data
from src.evaluation.metrics import ModelReport, TopKEvaluator
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

TASK_NAME = "train_order_after_drive"

# 时间窗口定义（对应 H/A/B 业务规则）
TIME_WINDOWS = ["7天", "14天", "21天"]

# 目标变量列名（训练时保留，不能删除）
# 试驾后下订阶段的标签
TARGET_COLUMNS = [
    "下订标签_7天", "下订标签_14天", "下订标签_21天",
    "试驾后下订天数差",
]


def remove_leakage_columns(df, target_label: str = None):
    """
    删除泄漏字段（保留目标变量）

    Args:
        df: 数据框
        target_label: 当前训练的目标变量（如果指定则保留）

    Returns:
        清理后的数据框
    """
    # 获取所有泄漏字段
    leakage_cols = config.feature.leakage_columns
    id_cols = config.feature.id_columns

    # 合并要删除的列
    cols_to_remove = leakage_cols + id_cols

    # 保留目标变量
    for target in TARGET_COLUMNS:
        if target in cols_to_remove:
            cols_to_remove.remove(target)

    # 如果指定了特定目标变量，确保保留
    if target_label and target_label in cols_to_remove:
        cols_to_remove.remove(target_label)

    # 只删除数据中实际存在的列
    existing_cols = [c for c in cols_to_remove if c in df.columns]

    if existing_cols:
        logger.info(f"删除泄漏字段: {len(existing_cols)} 个")
        for c in existing_cols[:10]:  # 只显示前 10 个
            logger.info(f"  - {c}")
        if len(existing_cols) > 10:
            logger.info(f"  ... 及其他 {len(existing_cols) - 10} 个字段")

        df = df.drop(columns=existing_cols)

    return df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="三模型下订预测训练（试驾后下订商谈阶段）")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径（动态拆分模式）",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="./data/order_after_drive_v2_train.parquet",
        help="训练集文件路径（提前拆分模式，优先级高于 --data-path）",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./data/order_after_drive_v2_test.parquet",
        help="测试集文件路径（提前拆分模式，优先级高于 --data-path）",
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
        help="单个模型训练时间限制（秒）",
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
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=None,
        help="最大内存使用比例（默认从 config 读取，16G 服务器建议 0.9）",
    )
    parser.add_argument(
        "--exclude-memory-heavy-models",
        action="store_true",
        help="排除内存密集型模型（KNN, RF, XT）",
    )
    parser.add_argument(
        "--num-folds-parallel",
        type=int,
        default=None,
        help="并行训练的 fold 数量",
    )
    parser.add_argument(
        "--time-windows",
        type=str,
        nargs="+",
        default=TIME_WINDOWS,
        choices=TIME_WINDOWS,
        help="训练的时间窗口列表",
    )
    parser.add_argument(
        "--included-model-types",
        type=str,
        default=None,
        help="指定训练的模型类型（逗号分隔），如 CAT,GBM。空值使用默认预设所有模型",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行训练三个模型（需大内存服务器 32GB+）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="并行训练的最大进程数（默认 3）",
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
    parser.add_argument(
        "--dynamic-stacking",
        type=lambda x: x.lower() in ["true", "1", "yes"] if isinstance(x, str) else bool(x),
        default=None,
        help="是否启用动态堆叠检测（None=使用预设默认值）",
    )

    return parser.parse_args()


def train_single_model(
    target_label: str,
    train_df,
    test_df,
    output_dir: Path,
    args,
) -> dict:
    """
    训练单个时间窗口模型

    Args:
        target_label: 目标变量名（如 "下订标签_7天"）
        train_df: 训练数据
        test_df: 测试数据
        output_dir: 模型输出目录
        args: 命令行参数

    Returns:
        训练结果字典
    """
    model_dir = output_dir / target_label
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  开始训练: {target_label}")
    logger.info(f"  模型目录: {model_dir}")

    excluded_columns = get_excluded_columns(target_label)

    excluded_model_types = None
    if args.exclude_memory_heavy_models:
        excluded_model_types = ["KNN", "RF", "XT"]

    # 指定训练的模型类型（白名单模式）
    included_model_types = None
    if args.included_model_types:
        included_model_types = [t.strip() for t in args.included_model_types.split(",")]

    # 当指定模型类型时，默认禁用集成模型
    fit_weighted_ensemble = False if included_model_types else None

    predictor = LeadScoringPredictor(
        label=target_label,
        output_path=str(model_dir),
        eval_metric="roc_auc",
        problem_type="binary",
        sample_weight="balance_weight",
        weight_evaluation=False,
        max_memory_usage_ratio=args.max_memory_ratio if args.max_memory_ratio is not None else config.model.max_memory_ratio,
        excluded_model_types=excluded_model_types,
        num_folds_parallel=args.num_folds_parallel,
        included_model_types=included_model_types,
        fit_weighted_ensemble=fit_weighted_ensemble,
        num_bag_folds=args.num_bag_folds if args.num_bag_folds is not None else config.model.num_bag_folds,
        num_stack_levels=args.num_stack_levels if args.num_stack_levels is not None else config.model.num_stack_levels,
        dynamic_stacking=getattr(args, "dynamic_stacking", None),
    )

    predictor.train(
        train_data=train_df,
        presets=args.preset,
        time_limit=args.time_limit,
        excluded_columns=excluded_columns,
    )

    # 评估
    metrics = predictor.evaluate(test_df)
    y_proba = predictor.get_positive_proba(test_df)
    y_true = test_df[target_label].values

    topk_evaluator = TopKEvaluator(y_true, y_proba)
    topk_metrics = topk_evaluator.compute_topk_metrics()

    # 特征重要性
    try:
        importance = predictor.get_feature_importance(test_df)
        importance.to_csv(model_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"特征重要性计算失败: {e}")
        importance = None

    # 保存模型
    predictor.save()

    # 生成报告
    report_generator = ModelReport(model_dir / "reports")
    report_generator.generate(
        model_metrics=metrics,
        topk_metrics=topk_metrics,
        feature_importance=importance,
        model_name=target_label,
    )

    logger.info(f"  {target_label} ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

    return {
        "target": target_label,
        "roc_auc": metrics.get("roc_auc", None),
        "topk_metrics": topk_metrics,
        "model_path": str(model_dir),
    }


def prepare_parallel_cache(train_df, test_df, cache_dir: Path) -> tuple:
    """为并行训练准备数据缓存"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "train_df.parquet"
    test_path = cache_dir / "test_df.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return str(train_path), str(test_path)


def train_single_model_spawn(
    target_label: str,
    train_path: str,
    test_path: str,
    output_dir: str,
    args_dict: dict,
) -> dict:
    """并行训练的安全包装函数"""
    import pandas as pd

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    class ArgsWrapper:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = ArgsWrapper(args_dict)

    model_dir = Path(output_dir) / target_label
    model_dir.mkdir(parents=True, exist_ok=True)

    excluded_columns = get_excluded_columns(target_label)

    excluded_model_types = None
    if getattr(args, "exclude_memory_heavy_models", False):
        excluded_model_types = ["KNN", "RF", "XT"]

    included_model_types = getattr(args, "included_model_types", None)

    predictor = LeadScoringPredictor(
        label=target_label,
        output_path=str(model_dir),
        eval_metric="roc_auc",
        problem_type="binary",
        sample_weight="balance_weight",
        weight_evaluation=False,
        max_memory_usage_ratio=args.max_memory_ratio if args.max_memory_ratio is not None else config.model.max_memory_ratio,
        excluded_model_types=excluded_model_types,
        num_folds_parallel=getattr(args, "num_folds_parallel", None),
        included_model_types=included_model_types,
        fit_weighted_ensemble=getattr(args, "fit_weighted_ensemble", None),
        num_bag_folds=args.num_bag_folds if args.num_bag_folds is not None else config.model.num_bag_folds,
        num_stack_levels=args.num_stack_levels if args.num_stack_levels is not None else config.model.num_stack_levels,
        dynamic_stacking=getattr(args, "dynamic_stacking", None),
    )

    predictor.train(
        train_data=train_df,
        presets=args.preset,
        time_limit=args.time_limit,
        excluded_columns=excluded_columns,
    )

    metrics = predictor.evaluate(test_df)
    predictor.save()

    return {
        "target": target_label,
        "roc_auc": metrics.get("roc_auc", None),
        "model_path": str(model_dir),
    }


def train_models_parallel(
    targets: list,
    train_df,
    test_df,
    output_dir: Path,
    args,
    max_workers: int = 3,
    cache_dir: Path = None,
) -> list:
    """并行训练多个模型"""
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="ensemble_cache_"))

    train_path, test_path = prepare_parallel_cache(train_df, test_df, cache_dir)
    logger.info(f"数据缓存: {cache_dir}")

    included_model_types_list = None
    if args.included_model_types:
        included_model_types_list = [t.strip() for t in args.included_model_types.split(",")]

    fit_weighted_ensemble = False if included_model_types_list else None

    args_dict = {
        "preset": args.preset,
        "time_limit": args.time_limit,
        "max_memory_ratio": args.max_memory_ratio,
        "exclude_memory_heavy_models": getattr(args, "exclude_memory_heavy_models", False),
        "num_folds_parallel": getattr(args, "num_folds_parallel", None),
        "included_model_types": included_model_types_list,
        "fit_weighted_ensemble": fit_weighted_ensemble,
        "num_bag_folds": getattr(args, "num_bag_folds", None),
        "num_stack_levels": getattr(args, "num_stack_levels", None),
        "dynamic_stacking": getattr(args, "dynamic_stacking", None),
    }

    task_args = [
        (target, train_path, test_path, str(output_dir), args_dict)
        for target in targets
    ]

    results = []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(train_single_model_spawn, task_arg): task_arg[0]
            for task_arg in task_args
        }

        for future in as_completed(futures):
            target = futures[future]
            try:
                result = future.result()
                results.append(result)
                roc_auc_display = f"{result['roc_auc']:.4f}" if result.get('roc_auc') is not None else 'N/A'
                logger.info(f"完成: {target} (ROC-AUC: {roc_auc_display})")
            except Exception as e:
                logger.error(f"训练失败: {target}, 错误: {e}")
                shutil.rmtree(cache_dir, ignore_errors=True)
                raise

    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info(f"清理缓存: {cache_dir}")

    return results


def main():
    """主函数"""
    args = parse_args()

    train_path = args.train_path
    test_path = args.test_path
    use_split_data = train_path and test_path
    data_path = args.data_path

    output_dir = Path(args.output_dir or config.output.models_dir) / "order_after_drive_ensemble"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        timestamp = get_timestamp()
        log_file = log_dir / f"{TASK_NAME}_{timestamp}.log"
    setup_logging(log_file=str(log_file), level=logging.INFO)

    command = " ".join(sys.argv)
    process_info_path = save_process_info(
        task_name=TASK_NAME,
        pid=os.getpid(),
        command=command,
        log_file=str(log_file),
        data_path=data_path or train_path,
        preset=args.preset,
        output_dir=str(output_dir),
    )

    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())

    train_start_time = get_local_now()

    logger.info("=" * 60)
    logger.info("三模型下订预测训练（试驾后下订商谈阶段）")
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
    logger.info(f"时间窗口: {args.time_windows}")
    logger.info(f"输出目录: {output_dir}")

    required_gb = get_preset_disk_requirement(args.preset) * len(args.time_windows)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G")

    if not disk_info["sufficient"]:
        logger.warning("磁盘空间不足！建议使用 medium_quality preset 或清理磁盘空间")

    all_results = []

    try:
        logger.info("步骤 1/4: 加载数据")

        if use_split_data:
            logger.info("使用提前拆分的数据文件")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 直接使用 pandas 加载 Parquet（管道脚本已处理数据，无需 DataLoader 验证）
            import pandas as pd
            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)

            logger.info(f"训练集列数: {len(df_train.columns)}, 测试集列数: {len(df_test.columns)}")

            logger.info("步骤 2/4: 特征工程")
            feature_engineer = FeatureEngineer(
                time_columns=config.feature.time_columns,
                numeric_columns=config.feature.numeric_features,
            )
            train_df, feature_metadata = feature_engineer.fit_transform(df_train)
            test_df, _ = feature_engineer.transform(df_test, interaction_context=feature_metadata.get("interaction_context"))

            quality_train = check_data_quality(train_df)
            quality_test = check_data_quality(df_test)
            logger.info(f"训练集: {quality_train['total_rows']} 行, {quality_train['total_columns']} 列")
            logger.info(f"测试集: {quality_test['total_rows']} 行, {quality_test['total_columns']} 列")

            split_info = {
                "mode": "pre_split",
                "train_path": train_path,
                "test_path": test_path,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "split_time": datetime.now().isoformat(),
            }
            with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)

        else:
            output_dir.mkdir(parents=True, exist_ok=True)

            loader = DataLoader(data_path, auto_adapt=True)
            df = loader.load()

            logger.info("删除泄漏字段...")
            df = remove_leakage_columns(df)

            quality = check_data_quality(df)
            logger.info(f"数据: {quality['total_rows']} 行, {quality['total_columns']} 列")

            logger.info("步骤 2/4: 特征工程")
            feature_engineer = FeatureEngineer(
                time_columns=config.feature.time_columns,
                numeric_columns=config.feature.numeric_features,
            )
            df_processed, feature_metadata = feature_engineer.process(df)

            first_target = f"下订标签_{args.time_windows[0]}"
            logger.info(f"步骤 3/4: 数据划分（分层目标: {first_target}）")
            train_df, test_df = split_data(
                df_processed,
                target_label=first_target,
                test_size=args.test_size,
                stratify=True,
            )
            logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

            split_info = {
                "mode": "dynamic_split",
                "data_path": data_path,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "test_ratio": args.test_size,
                "stratify_target": first_target,
                "split_time": datetime.now().isoformat(),
            }
            with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)

        output_dir.mkdir(parents=True, exist_ok=True)
        feature_metadata_path = output_dir / "feature_metadata.json"
        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"特征元数据已保存: {feature_metadata_path}")

        logger.info("步骤 4/4: 训练模型")

        parallel_mode = args.parallel or config.ensemble.parallel_training
        max_workers = args.max_workers or config.ensemble.max_workers

        if parallel_mode:
            logger.info(f"并行训练模式（max_workers={max_workers}）")
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                if available_gb < 24:
                    logger.warning(f"可用内存仅 {available_gb:.1f}GB，并行训练可能触发 OOM，建议使用顺序训练")
            except ImportError:
                logger.warning("未安装 psutil，无法检查内存状态")

            all_results = train_models_parallel(
                targets=[f"下订标签_{w}" for w in args.time_windows],
                train_df=train_df,
                test_df=test_df,
                output_dir=output_dir,
                args=args,
                max_workers=max_workers,
            )
        else:
            logger.info("顺序训练模式")
            for window in args.time_windows:
                target_label = f"下订标签_{window}"
                result = train_single_model(
                    target_label=target_label,
                    train_df=train_df,
                    test_df=test_df,
                    output_dir=output_dir,
                    args=args,
                )
                all_results.append(result)

        ensemble_metadata = {
            "time_windows": args.time_windows,
            "preset": args.preset,
            "time_limit_per_model": args.time_limit,
            "results": all_results,
            "training_start": train_start_time.isoformat(),
            "training_end": get_local_now().isoformat(),
        }
        with open(output_dir / "ensemble_metadata.json", "w", encoding="utf-8") as f:
            json.dump(ensemble_metadata, f, ensure_ascii=False, indent=2)

        train_end_time = get_local_now()
        duration_seconds = (train_end_time - train_start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"训练总耗时: {format_training_duration(duration_seconds)}")
        logger.info("训练完成!")
        logger.info(f"模型路径: {output_dir}")

        print("\n" + "=" * 60)
        print("三模型下订预测训练完成（试驾后下订商谈阶段）")
        print("=" * 60)
        print(f"训练耗时: {format_training_duration(duration_seconds)}")
        print("\n各模型 ROC-AUC:")
        for r in all_results:
            roc_auc_display = f"{r['roc_auc']:.4f}" if r.get('roc_auc') is not None else 'N/A'
            print(f"  {r['target']}: {roc_auc_display}")
        print(f"\n模型目录: {output_dir}")
        print(f"日志文件: {log_file}")

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))

        if output_dir.exists():
            logger.info(f"清理失败的模型目录: {output_dir}")
            try:
                shutil.rmtree(output_dir)
            except Exception as cleanup_error:
                logger.warning(f"清理失败: {cleanup_error}")

        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
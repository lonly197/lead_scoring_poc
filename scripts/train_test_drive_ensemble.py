"""
三模型试驾预测训练脚本

训练预测线索 7/14/21 天内试驾概率的三个模型。
用于后续基于业务规则推导 HAB 评级。

使用方法：
    uv run python scripts/train_test_drive_ensemble.py \
        --data-path ./data/202602~03.tsv \
        --preset high_quality
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

TASK_NAME = "train_test_drive_ensemble"

# 时间窗口定义（对应 H/A/B 业务规则）
TIME_WINDOWS = ["7天", "14天", "21天"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="三模型试驾预测训练")

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
        default=0.8,
        help="最大内存使用比例",
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
        target_label: 目标变量名（如 "试驾标签_7天"）
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

    predictor = LeadScoringPredictor(
        label=target_label,
        output_path=str(model_dir),
        eval_metric="roc_auc",
        problem_type="binary",
        sample_weight="balance_weight",
        weight_evaluation=False,
        max_memory_usage_ratio=args.max_memory_ratio,
        excluded_model_types=excluded_model_types,
        num_folds_parallel=args.num_folds_parallel,
        included_model_types=included_model_types,
        # Bagging/Stacking 参数
        num_bag_folds=getattr(args, "num_bag_folds", None),
        num_stack_levels=getattr(args, "num_stack_levels", None),
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
    """
    为并行训练准备数据缓存

    使用 Parquet 格式保存数据（安全的列式存储格式），
    供子进程独立加载。Parquet 不存在 pickle 的安全风险。

    Args:
        train_df: 训练数据
        test_df: 测试数据
        cache_dir: 缓存目录

    Returns:
        (train_path, test_path) 缓存文件路径
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "train_df.parquet"
    test_path = cache_dir / "test_df.parquet"

    # 使用 Parquet 格式（安全，不支持任意代码执行）
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
    """
    并行训练的安全包装函数

    使用文件路径而非 DataFrame 对象，避免序列化大型数据结构。
    子进程通过 Parquet 文件加载数据，确保内存隔离和安全性。

    Args:
        target_label: 目标变量名
        train_path: 训练数据缓存文件路径（Parquet 格式）
        test_path: 测试数据缓存文件路径（Parquet 格式）
        output_dir: 模型输出目录
        args_dict: 参数字典

    Returns:
        训练结果字典
    """
    import pandas as pd

    # 从 Parquet 文件加载数据（安全格式，无代码执行风险）
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 创建简单的 args 替代对象
    class ArgsWrapper:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = ArgsWrapper(args_dict)

    # 构建模型目录
    model_dir = Path(output_dir) / target_label
    model_dir.mkdir(parents=True, exist_ok=True)

    excluded_columns = get_excluded_columns(target_label)

    excluded_model_types = None
    if getattr(args, "exclude_memory_heavy_models", False):
        excluded_model_types = ["KNN", "RF", "XT"]

    # 指定训练的模型类型（白名单模式）
    included_model_types = getattr(args, "included_model_types", None)

    predictor = LeadScoringPredictor(
        label=target_label,
        output_path=str(model_dir),
        eval_metric="roc_auc",
        problem_type="binary",
        sample_weight="balance_weight",
        weight_evaluation=False,
        max_memory_usage_ratio=getattr(args, "max_memory_ratio", 0.8),
        excluded_model_types=excluded_model_types,
        num_folds_parallel=getattr(args, "num_folds_parallel", None),
        included_model_types=included_model_types,
        # Bagging/Stacking 参数
        num_bag_folds=getattr(args, "num_bag_folds", None),
        num_stack_levels=getattr(args, "num_stack_levels", None),
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

    # 保存模型
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
    """
    并行训练多个模型

    使用 Parquet 文件传递数据，安全且高效。

    Args:
        targets: 目标变量列表
        train_df: 训练数据
        test_df: 测试数据
        output_dir: 输出目录
        args: 命令行参数
        max_workers: 最大并行进程数
        cache_dir: 缓存目录（可选）

    Returns:
        List[dict] 各模型训练结果
    """
    # 创建临时缓存目录
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="ensemble_cache_"))

    # 缓存数据文件（Parquet 格式，安全）
    train_path, test_path = prepare_parallel_cache(train_df, test_df, cache_dir)
    logger.info(f"数据缓存: {cache_dir}")

    # 准备轻量级参数（只传递路径和字符串，不传递复杂对象）
    args_dict = {
        "preset": args.preset,
        "time_limit": args.time_limit,
        "max_memory_ratio": args.max_memory_ratio,
        "exclude_memory_heavy_models": getattr(args, "exclude_memory_heavy_models", False),
        "num_folds_parallel": getattr(args, "num_folds_parallel", None),
        "included_model_types": args.included_model_types,
        # Bagging/Stacking 参数
        "num_bag_folds": getattr(args, "num_bag_folds", None),
        "num_stack_levels": getattr(args, "num_stack_levels", None),
        "dynamic_stacking": getattr(args, "dynamic_stacking", None),
    }

    task_args = [
        (target, train_path, test_path, str(output_dir), args_dict)
        for target in targets
    ]

    results = []

    # 使用 spawn 启动方法（安全最佳实践）
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
                logger.info(f"完成: {target} (ROC-AUC: {result['roc_auc']:.4f if result['roc_auc'] else 'N/A'})")
            except Exception as e:
                logger.error(f"训练失败: {target}, 错误: {e}")
                # 清理临时缓存
                shutil.rmtree(cache_dir, ignore_errors=True)
                raise

    # 清理临时缓存
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info(f"清理缓存: {cache_dir}")

    return results


def main():
    """主函数"""
    args = parse_args()

    # 判断数据加载模式
    train_path = args.train_path or config.data.train_data_path
    test_path = args.test_path or config.data.test_data_path
    use_split_data = train_path and test_path
    data_path = args.data_path or config.data.data_path

    output_dir = Path(args.output_dir or config.output.models_dir) / "test_drive_ensemble"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
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
        preset=args.preset,
        output_dir=str(output_dir),
    )

    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())

    train_start_time = get_local_now()

    # 启动信息
    logger.info("=" * 60)
    logger.info("三模型试驾预测训练")
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

    # 磁盘空间检查
    required_gb = get_preset_disk_requirement(args.preset) * len(args.time_windows)
    disk_info = check_disk_space("./", required_gb=required_gb)
    logger.info(f"磁盘状态: 剩余 {disk_info['free_gb']}G / 需要 {required_gb}G")

    if not disk_info["sufficient"]:
        logger.warning("磁盘空间不足！建议使用 medium_quality preset 或清理磁盘空间")

    all_results = []

    try:
        # 1. 加载数据
        logger.info("步骤 1/4: 加载数据")

        if use_split_data:
            # 模式一：提前拆分的数据文件
            logger.info("使用提前拆分的数据文件")
            train_loader = DataLoader(train_path, auto_adapt=True)
            test_loader = DataLoader(test_path, auto_adapt=True)
            df_train = train_loader.load()
            df_test = test_loader.load()

            # 2. 特征工程（训练集 fit_transform，测试集 transform）
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

            # 保存数据划分信息
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
            # 模式二：动态拆分
            loader = DataLoader(data_path, auto_adapt=True)
            df = loader.load()
            quality = check_data_quality(df)
            logger.info(f"数据: {quality['total_rows']} 行, {quality['total_columns']} 列")

            # 2. 特征工程
            logger.info("步骤 2/4: 特征工程")
            feature_engineer = FeatureEngineer(
                time_columns=config.feature.time_columns,
                numeric_columns=config.feature.numeric_features,
            )
            df_processed, feature_metadata = feature_engineer.process(df)

            # 3. 数据划分（使用第一个目标变量做分层）
            first_target = f"试驾标签_{args.time_windows[0]}"
            logger.info(f"步骤 3/4: 数据划分（分层目标: {first_target}）")
            train_df, test_df = split_data(
                df_processed,
                target_label=first_target,
                test_size=args.test_size,
                stratify=True,
            )
            logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

            # 保存数据划分信息
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

        # 保存特征元数据
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_metadata_path = output_dir / "feature_metadata.json"
        with open(feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"特征元数据已保存: {feature_metadata_path}")

        # 4. 循环训练各时间窗口模型
        logger.info("步骤 4/4: 训练模型")

        # 判断是否并行训练（命令行参数或环境变量）
        parallel_mode = args.parallel or config.ensemble.parallel_training
        max_workers = args.max_workers or config.ensemble.max_workers

        if parallel_mode:
            logger.info(f"并行训练模式（max_workers={max_workers}）")
            # 检查内存是否足够（建议 32GB+）
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                if available_gb < 24:
                    logger.warning(f"可用内存仅 {available_gb:.1f}GB，并行训练可能触发 OOM，建议使用顺序训练")
            except ImportError:
                logger.warning("未安装 psutil，无法检查内存状态")

            all_results = train_models_parallel(
                targets=[f"试驾标签_{w}" for w in args.time_windows],
                train_df=train_df,
                test_df=test_df,
                output_dir=output_dir,
                args=args,
                max_workers=max_workers,
            )
        else:
            logger.info("顺序训练模式")
            for window in args.time_windows:
                target_label = f"试驾标签_{window}"
                result = train_single_model(
                    target_label=target_label,
                    train_df=train_df,
                    test_df=test_df,
                    output_dir=output_dir,
                    args=args,
                )
                all_results.append(result)

        # 保存整体训练元数据
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

        # 完成信息
        train_end_time = get_local_now()
        duration_seconds = (train_end_time - train_start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"训练总耗时: {format_training_duration(duration_seconds)}")
        logger.info("训练完成!")
        logger.info(f"模型路径: {output_dir}")

        # 输出汇总
        print("\n" + "=" * 60)
        print("三模型训练完成")
        print("=" * 60)
        print(f"训练耗时: {format_training_duration(duration_seconds)}")
        print("\n各模型 ROC-AUC:")
        for r in all_results:
            print(f"  {r['target']}: {r['roc_auc']:.4f if r['roc_auc'] else 'N/A'}")
        print(f"\n模型目录: {output_dir}")
        print(f"日志文件: {log_file}")

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

        # 关闭 Ray
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
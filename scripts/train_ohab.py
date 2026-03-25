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
from typing import Iterable

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.loader import DataLoader, FeatureEngineer, smart_split_data, split_data_oot_three_way
from src.data.label_policy import (
    apply_ohab_label_policy,
    build_ohab_label_policy,
    filter_to_effective_ohab_labels,
)
from src.evaluation.ohab_metrics import (
    apply_hab_decision_policy,
    classification_report_dict,
    compute_hab_bucket_summary,
    compute_class_ranking_report,
    compute_threshold_report,
    confusion_matrix_frame,
    check_hab_monotonicity,
    optimize_hab_decision_policy,
)
from src.models.predictor import LeadScoringPredictor
from src.evaluation.business_logic import calculate_dimension_contribution, BUSINESS_DIMENSION_MAP
from src.utils.visualization import plot_feature_importance, plot_dimension_contribution
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

TASK_NAME = "train_ohab"


def _parse_report_topk(raw_value: str) -> tuple[float, ...]:
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item) / 100)
    return tuple(values or [0.05, 0.10, 0.20])


def _dump_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _candidate_prefixes_for_family(family: str) -> tuple[str, ...]:
    family = family.lower()
    if family == "gbm":
        return ("LightGBM", "LightGBMXT", "LightGBMLarge")
    if family == "cat":
        return ("CatBoost",)
    if family == "xgb":
        return ("XGBoost",)
    return ()


def _select_baseline_model_from_leaderboard(leaderboard_df: pd.DataFrame, family: str) -> dict:
    non_ensemble_df = leaderboard_df[
        ~leaderboard_df["model"].astype(str).str.startswith("WeightedEnsemble")
    ].copy()
    if non_ensemble_df.empty:
        raise ValueError("leaderboard 中未找到可用的非集成子模型")

    candidate_df = pd.DataFrame()
    for prefix in _candidate_prefixes_for_family(family):
        candidate_df = non_ensemble_df[
            non_ensemble_df["model"].astype(str).str.startswith(prefix)
        ].copy()
        if not candidate_df.empty:
            break

    fallback_reason = ""
    if candidate_df.empty:
        candidate_df = non_ensemble_df.copy()
        fallback_reason = "requested_family_not_found"

    ranking_column = "score_test" if "score_test" in candidate_df.columns else "score_val"
    candidate_df = candidate_df.sort_values(ranking_column, ascending=False)
    baseline_row = candidate_df.iloc[0]

    return {
        "baseline_family_requested": family,
        "baseline_model_name": str(baseline_row["model"]),
        "selection_metric": ranking_column,
        "selection_score": float(baseline_row[ranking_column]),
        "fallback_reason": fallback_reason,
    }


def _build_model_decision_policy(
    predictor: LeadScoringPredictor,
    dataset: pd.DataFrame,
    target_label: str,
    label_mode: str,
    model_name: str,
) -> dict:
    if label_mode != "hab" or len(dataset) == 0:
        return {"strategy": "argmax"}
    valid_proba = predictor.predict_proba(dataset, model=model_name).reset_index(drop=True)
    decision_policy = optimize_hab_decision_policy(
        dataset[target_label].reset_index(drop=True),
        valid_proba,
    )
    decision_policy["model_name"] = model_name
    return decision_policy


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
    parser.add_argument(
        "--o-merge-threshold",
        type=int,
        default=50,
        help="训练集 O 级样本低于该阈值时自动合并",
    )
    parser.add_argument(
        "--o-merge-target",
        type=str,
        default="H",
        help="O 级样本不足时合并到的目标等级",
    )
    parser.add_argument(
        "--report-topk",
        type=str,
        default="5,10,20",
        help="多分类排序分析 Top 比例，逗号分隔的百分比值",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="hab",
        choices=["hab", "ohab"],
        help="标签模式：hab=仅输出 H/A/B；ohab=保留 O/H/A/B",
    )
    parser.add_argument(
        "--enable-model-comparison",
        action="store_true",
        help="保留基线子模型并输出 baseline vs best 的对比配置",
    )
    parser.add_argument(
        "--baseline-family",
        type=str,
        default="gbm",
        choices=["gbm", "cat", "xgb", "auto"],
        help="模型对比时的基线家族",
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

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置路径
    data_path = args.data_path or config.data.data_path
    target_label = args.target
    output_dir = Path(args.output_dir or config.output.models_dir) / "ohab_model"
    top_ratios = _parse_report_topk(args.report_topk)
    
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
    logger.info("OHAB 评级模型训练 (统一自适应版)")
    logger.info("=" * 60)
    logger.info(f"训练开始时间: {train_start_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"目标变量: {target_label}")

    # 1. 加载数据
    try:
        logger.info("步骤 1/6: 加载数据")
        # 统一开启 auto_adapt=True 以支持 2-3 月新 SQL 格式
        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()
        adaptation_metadata = loader.get_adaptation_metadata()

        if target_label in df.columns:
            df = df[df[target_label] != "Unknown"].copy()
            logger.info(f"过滤 Unknown 后: {len(df):,} 行")

        # 2. 智能数据切分
        logger.info("步骤 2/6: 执行智能数据切分")
        if args.train_end and args.valid_end:
            logger.info(f"采用手动 OOT 切分: {args.train_end} / {args.valid_end}")
            train_df, valid_df, test_df = split_data_oot_three_way(
                df, target_label, "线索创建时间", args.train_end, args.valid_end
            )
            split_info = {"mode": "oot_manual", "train_end": args.train_end, "valid_end": args.valid_end}
        else:
            train_df, valid_df, test_df, split_info = smart_split_data(df, target_label)

        label_policy = build_ohab_label_policy(
            train_df,
            target_label=target_label,
            label_mode=args.label_mode,
            o_merge_threshold=args.o_merge_threshold,
            merge_target=args.o_merge_target,
        )
        logger.info(f"训练标签策略: {label_policy}")

        train_df = apply_ohab_label_policy(train_df, target_label, label_policy)
        valid_df = apply_ohab_label_policy(valid_df, target_label, label_policy)
        test_df = apply_ohab_label_policy(test_df, target_label, label_policy)
        train_df = filter_to_effective_ohab_labels(train_df, target_label, label_policy)
        valid_df = filter_to_effective_ohab_labels(valid_df, target_label, label_policy)
        test_df = filter_to_effective_ohab_labels(test_df, target_label, label_policy)

        # 3. 特征工程（先 fit 训练集，再 transform 验证/测试集）
        logger.info("步骤 3/6: 特征工程")
        feature_engineer = FeatureEngineer(
            time_columns=config.feature.time_columns,
            numeric_columns=config.feature.numeric_features,
        )
        train_df, feature_metadata = feature_engineer.fit_transform(train_df)
        valid_df, _ = feature_engineer.transform(
            valid_df,
            interaction_context=feature_metadata.get("interaction_context"),
        )
        test_df, _ = feature_engineer.transform(
            test_df,
            interaction_context=feature_metadata.get("interaction_context"),
        )

        feature_metadata["split_info"] = split_info
        feature_metadata["label_policy"] = label_policy
        feature_metadata["label_mode"] = args.label_mode
        feature_metadata["schema_contract"] = adaptation_metadata.get("schema_contract", {})
        feature_metadata["feature_names_version"] = "ohab_schema_contract_v2"
        if adaptation_metadata.get("json_feature_source"):
            feature_metadata["json_feature_source"] = adaptation_metadata["json_feature_source"]

        # 4. 训练模型
        logger.info("步骤 4/6: 模型训练")
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
            eval_metric="log_loss",
            problem_type="multiclass",
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
        best_model_name = predictor.get_model_info().get("best_model")
        comparison_config = {
            "enabled": False,
            "baseline_family_requested": args.baseline_family,
            "best_model_name": best_model_name,
            "baseline_model_name": None,
            "models": {},
        }

        if args.enable_model_comparison and len(valid_df) > 0:
            valid_leaderboard_df = predictor.get_leaderboard(valid_df, silent=True)
            valid_leaderboard_df.to_csv(output_dir / "leaderboard_valid.csv", index=False, encoding="utf-8-sig")
            baseline_selection = _select_baseline_model_from_leaderboard(valid_leaderboard_df, args.baseline_family)
            baseline_model_name = baseline_selection["baseline_model_name"]
            comparison_config.update(
                {
                    "enabled": True,
                    "baseline_model_name": baseline_model_name,
                    "baseline_selection": baseline_selection,
                    "comparison_model_names": list(dict.fromkeys([baseline_model_name, best_model_name])),
                }
            )
            for model_name, role in [
                (baseline_model_name, "baseline"),
                (best_model_name, "best"),
            ]:
                comparison_config["models"][model_name] = {
                    "role": role,
                    "decision_policy": _build_model_decision_policy(
                        predictor=predictor,
                        dataset=valid_df,
                        target_label=target_label,
                        label_mode=args.label_mode,
                        model_name=model_name,
                    ),
                }
            logger.info(f"模型对比配置: {comparison_config}")
        elif args.enable_model_comparison:
            logger.warning("启用了模型对比，但验证集为空，跳过 baseline 配置生成")

        # 5. 模型评估
        logger.info("步骤 5/6: 模型评估与可解释性分析")
        raw_evaluation_results = predictor.evaluate(test_df)
        logger.info(f"原始评估结果: {raw_evaluation_results}")

        decision_policy = {"strategy": "argmax"}
        if comparison_config["enabled"]:
            decision_policy = comparison_config["models"][best_model_name]["decision_policy"]
        elif args.label_mode == "hab" and len(valid_df) > 0:
            valid_proba = predictor.predict_proba(valid_df).reset_index(drop=True)
            decision_policy = optimize_hab_decision_policy(
                valid_df[target_label].reset_index(drop=True),
                valid_proba,
            )
            logger.info(f"验证集阈值策略: {decision_policy}")

        y_true = test_df[target_label].reset_index(drop=True)
        y_proba = predictor.predict_proba(test_df, model=best_model_name).reset_index(drop=True)
        if args.label_mode == "hab":
            y_pred = apply_hab_decision_policy(y_proba, decision_policy)
        else:
            y_pred = predictor.predict(test_df, model=best_model_name)

        confusion_df = confusion_matrix_frame(y_true, y_pred)
        classification_dict = classification_report_dict(y_true, y_pred)
        class_ranking_report = compute_class_ranking_report(
            y_true,
            y_proba,
            top_ratios=top_ratios,
        )
        b_threshold_report = None
        if "B" in y_proba.columns:
            b_threshold_report = compute_threshold_report(
                y_true,
                y_proba["B"],
                positive_label="B",
            )
        bucket_summary_df = pd.DataFrame()
        monotonicity_result = {"passed": False, "metric": None, "message": "未启用 HAB 桶验证"}
        if args.label_mode == "hab":
            evaluation_frame = test_df.reset_index(drop=True).copy()
            evaluation_frame["真实标签"] = y_true.values
            evaluation_frame["预测标签"] = y_pred.values if hasattr(y_pred, "values") else y_pred
            for column in y_proba.columns:
                evaluation_frame[f"概率_{column}"] = y_proba[column].values
            bucket_summary_df = compute_hab_bucket_summary(evaluation_frame, label_column="预测标签")
            monotonicity_result = check_hab_monotonicity(bucket_summary_df)
            logger.info(f"HAB 行为分层检查: {monotonicity_result}")

        # 计算特征重要性
        logger.info("计算特征重要性...")
        importance_df = predictor.get_feature_importance(test_df)
        leaderboard_df = predictor.get_leaderboard(test_df, silent=True)

        # 可视化特征重要性
        plot_feature_importance(
            importance_df, 
            output_path=str(output_dir / "feature_importance.png")
        )

        # 保存特征重要性原始数据
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        importance_df.to_json(output_dir / "feature_importance.json", orient="records", force_ascii=False, indent=2)
        
        # 统计业务维度贡献
        feature_importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
        dimension_contribution = calculate_dimension_contribution(feature_importance_dict)
        
        logger.info("业务维度贡献分析:")
        for dim, score in dimension_contribution.items():
            logger.info(f"  - {dim}: {score:.4f}")
            
        # 可视化业务维度贡献
        plot_dimension_contribution(
            dimension_contribution,
            output_path=str(output_dir / "business_dimension_contribution.png")
        )
        
        # 保存业务维度贡献结果
        with open(output_dir / "business_dimension_contribution.json", "w", encoding="utf-8") as f:
            json.dump(dimension_contribution, f, ensure_ascii=False, indent=2)

        leaderboard_df.to_csv(output_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
        confusion_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        _dump_json(output_dir / "classification_report.json", classification_dict)
        _dump_json(output_dir / "class_ranking_report.json", class_ranking_report)
        if not bucket_summary_df.empty:
            bucket_summary_df.to_csv(output_dir / "hab_bucket_summary.csv", index=False, encoding="utf-8-sig")
            _dump_json(output_dir / "hab_bucket_summary.json", bucket_summary_df.to_dict(orient="records"))
        _dump_json(output_dir / "monotonicity_check.json", monotonicity_result)

        if b_threshold_report is not None:
            b_threshold_report.to_csv(output_dir / "b_threshold_report.csv", index=False, encoding="utf-8-sig")

        predictions_df = test_df[["线索唯一ID"]].copy() if "线索唯一ID" in test_df.columns else test_df[[target_label]].copy()
        if "线索唯一ID" not in predictions_df.columns:
            predictions_df.insert(0, "样本索引", test_df.index)
        predictions_df["真实标签"] = y_true.values
        predictions_df["预测标签"] = y_pred.values if hasattr(y_pred, "values") else y_pred
        for col in y_proba.columns:
            predictions_df[f"概率_{col}"] = y_proba[col].values
        predictions_df.to_csv(output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

        evaluation_summary = {
            "metrics": classification_dict,
            "raw_metrics": raw_evaluation_results,
            "label_policy": label_policy,
            "label_mode": args.label_mode,
            "decision_policy": decision_policy,
            "model_comparison": comparison_config,
            "split_info": split_info,
            "top_ratios": list(top_ratios),
            "monotonicity_check": monotonicity_result,
        }
        _dump_json(output_dir / "evaluation_summary.json", evaluation_summary)
        feature_metadata["decision_policy"] = decision_policy
        feature_metadata["label_mode"] = args.label_mode
        feature_metadata["model_comparison"] = comparison_config
        _dump_json(output_dir / "feature_metadata.json", feature_metadata)
        _dump_json(output_dir / "model_comparison_config.json", comparison_config)
            
        # 自动生成 Top-K 列表
        try:
            from scripts.generate_topk import generate_topk_from_predictor
            topk_path = Path(config.output.topk_dir) / f"topk_ohab_{get_timestamp()}.csv"
            generate_topk_from_predictor(predictor, test_df, str(topk_path))
            logger.info(f"Top-K 列表已生成: {topk_path}")
        except ImportError:
            logger.warning("scripts.generate_topk 未找到，跳过 Top-K 生成")
            
        # 6. 保存与清理
        logger.info("步骤 6/6: 保存与清理")
        predictor.save(
            extra_metadata={
                "label_policy": label_policy,
                "label_mode": args.label_mode,
                "decision_policy": decision_policy,
                "model_comparison": comparison_config,
                "schema_contract": feature_metadata.get("schema_contract", {}),
                "interaction_context": feature_metadata.get("interaction_context", {}),
            }
        )
        if comparison_config["enabled"]:
            predictor.cleanup(
                keep_best_only=False,
                keep_model_names=comparison_config.get("comparison_model_names"),
            )
        else:
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

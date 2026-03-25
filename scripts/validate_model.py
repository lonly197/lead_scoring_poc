#!/usr/bin/env python3
"""
模型验证脚本

验证训练好的 OHAB 模型效果，包括：
1. 加载模型和新数据
2. 预测并评估
3. 生成详细报告
"""

import argparse
import json
import logging
import pickle  # noqa: S403 - 加载 AutoGluon 模型需要
import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config, get_excluded_columns
from src.data.label_policy import apply_ohab_label_policy, filter_to_effective_ohab_labels
from src.data.loader import DataLoader, FeatureEngineer
from src.evaluation.business_logic import build_bucket_summary_text, build_lead_action_record
from src.evaluation.ohab_metrics import (
    apply_hab_decision_policy,
    classification_report_dict,
    compute_hab_bucket_summary,
    compute_class_ranking_report,
    compute_threshold_report,
    confusion_matrix_frame,
    check_hab_monotonicity,
)
from src.utils.helpers import setup_logging


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def dump_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

logger = logging.getLogger(__name__)
FINAL_ORDERED_COLUMN = "is_final_ordered"


def parse_args():
    parser = argparse.ArgumentParser(description="验证 OHAB 模型")

    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/ohab_model",
        help="模型路径",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="测试数据路径（默认使用训练数据）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="线索评级_试驾前",
        help="目标变量名",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation",
        help="输出目录",
    )
    parser.add_argument(
        "--oot-test",
        action="store_true",
        help="仅评估 OOT 测试集（时间 >= valid_end），避免数据泄露",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2026-03-11",
        help="训练集截止日期（仅 --oot-test 模式使用）",
    )
    parser.add_argument(
        "--valid-end",
        type=str,
        default="2026-03-16",
        help="验证集截止日期（仅 --oot-test 模式使用，测试集为时间 >= valid_end）",
    )
    parser.add_argument(
        "--report-topk",
        type=str,
        default="5,10,20",
        help="多分类排序分析 Top 比例，逗号分隔的百分比值",
    )

    return parser.parse_args()


def load_model(model_path: Path):
    """加载 AutoGluon 模型，处理版本兼容性问题"""
    # 方式1：尝试使用 require_py_version_match 参数
    try:
        predictor = TabularPredictor.load(
            str(model_path),
            require_version_match=False,
            require_py_version_match=False,
        )
        return predictor
    except TypeError:
        # 旧版本可能不支持 require_py_version_match
        pass
    except Exception as e:
        logger.warning(f"标准加载失败: {e}")

    # 方式2：直接加载
    try:
        predictor = TabularPredictor.load(str(model_path))
        return predictor
    except Exception as e:
        logger.warning(f"直接加载失败: {e}")

    # 方式3：使用 pickle 手动加载
    predictor_path = model_path / "predictor.pkl"
    if predictor_path.exists():
        logger.info("使用 pickle 手动加载...")
        with open(predictor_path, "rb") as f:
            predictor = pickle.load(f)
        return predictor

    raise RuntimeError(f"无法加载模型: {model_path}")


def find_available_model(default_path: Path) -> Path:
    """查找可用的模型目录，优先使用统一训练输出目录"""
    if default_path.exists():
        return default_path

    # 搜索 outputs/models/ 下的模型目录
    models_dir = Path("outputs/models")
    if models_dir.exists():
        # 优先级：ohab_model > ohab_oot > 其他
        for name in ["ohab_model", "ohab_oot"]:
            candidate = models_dir / name
            if candidate.exists() and (candidate / "predictor.pkl").exists():
                return candidate

        # 查找其他有效模型目录
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "predictor.pkl").exists():
                return model_dir

    return default_path  # 返回默认路径，让后续代码报错


def main():
    args = parse_args()
    setup_logging(level=logging.INFO)

    model_path = Path(args.model_path)
    # 如果指定的模型路径不存在，尝试自动检测
    if not model_path.exists():
        detected_path = find_available_model(model_path)
        if detected_path.exists():
            logger.info(f"模型路径 {model_path} 不存在，使用检测到的模型: {detected_path}")
            model_path = detected_path
        else:
            logger.error(f"未找到可用模型，请先运行训练脚本")
            logger.error(f"  uv run python scripts/train_ohab.py")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ratios = tuple(float(item.strip()) / 100 for item in args.report_topk.split(",") if item.strip())

    # 1. 加载模型
    logger.info("=" * 60)
    logger.info("加载模型")
    logger.info("=" * 60)

    predictor = load_model(model_path)
    logger.info(f"模型加载完成: {model_path}")
    logger.info(f"标签: {predictor.label}")
    logger.info(f"评估指标: {predictor.eval_metric}")
    logger.info(f"问题类型: {predictor.problem_type}")
    logger.info(f"最佳模型: {predictor.model_best}")

    # 2. 加载测试数据
    logger.info("\n" + "=" * 60)
    logger.info("加载测试数据")
    logger.info("=" * 60)

    data_path = args.data_path or "./data/20260308-v2.csv"

    loader = DataLoader(data_path, auto_adapt=True)
    df = loader.load()

    # 过滤 Unknown
    target = predictor.label
    if target in df.columns:
        df = df[df[target] != "Unknown"].copy()
        logger.info(f"过滤 Unknown 后: {len(df)} 行")

    # 自动识别防泄漏切分标记 (Smart Test Set Filtering)
    metadata = load_feature_metadata(model_path)
    split_info = metadata.get("split_info", {})
    label_policy = metadata.get("label_policy", {})
    label_mode = metadata.get("label_mode", "ohab")
    interaction_context = metadata.get("interaction_context", {})
    decision_policy = metadata.get("decision_policy", {"strategy": "argmax"})
    
    if split_info:
        logger.info(f"读取到模型切分元数据，模式: {split_info.get('mode')}")
        mode = split_info.get("mode")
        
        if mode in ["oot", "oot_manual"]:
            time_column = "线索创建时间"
            if time_column not in df.columns:
                logger.error(f"OOT 模式需要 '{time_column}' 列，但数据中不存在")
                sys.exit(1)
            
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            valid_end_str = split_info.get("valid_end")
            valid_end = pd.Timestamp(valid_end_str)

            df = df[df[time_column] >= valid_end].copy()
            logger.info(f"OOT 测试集自动切分: 时间 >= {valid_end_str}")
            
        elif mode == "random":
            # 降级情况下的防泄漏过滤
            if "test_ids" in split_info and "id_column" in split_info:
                id_col = split_info["id_column"]
                test_ids = set(split_info["test_ids"])
                df = df[df[id_col].isin(test_ids)].copy()
                logger.info(f"随机切分防泄漏: 根据 {id_col} 提取了 {len(test_ids)} 个测试集样本")
            elif "test_indices" in split_info:
                test_indices = split_info["test_indices"]
                # 注意：物理索引过滤前提是 df 未被重新排序或清洗改变行号
                df = df.iloc[test_indices].copy()
                logger.info(f"随机切分防泄漏: 根据物理索引提取了 {len(test_indices)} 个测试集样本")
            else:
                logger.warning("模型为随机切分但未找到测试集标记，可能存在数据泄漏风险！")
                
    elif args.oot_test:
        # 向后兼容：手动指定 OOT 测试集
        time_column = "线索创建时间"
        if time_column not in df.columns:
            logger.error(f"OOT 模式需要 '{time_column}' 列，但数据中不存在")
            sys.exit(1)

        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        valid_end = pd.Timestamp(args.valid_end)

        # 测试集：时间 >= valid_end
        df = df[df[time_column] >= valid_end].copy()
        logger.info(f"OOT 测试集切分(手动参数): 时间 >= {args.valid_end}")
    
    else:
        logger.warning("未启用防泄漏过滤，当前验证集可能包含训练数据！")

    logger.info(f"验证集最终样本量: {len(df)} 行")

    if len(df) == 0:
        logger.error("测试集为空，请检查数据时间范围或切分逻辑")
        sys.exit(1)

    final_ordered = None
    if FINAL_ORDERED_COLUMN in df.columns:
        final_ordered = df[FINAL_ORDERED_COLUMN].copy()
        logger.info(f"检测到 {FINAL_ORDERED_COLUMN}，将仅用于业务转化验证")

    if label_policy:
        df = apply_ohab_label_policy(df, target, label_policy)
        df = filter_to_effective_ohab_labels(df, target, label_policy)
        logger.info(f"应用训练标签策略: {label_policy}")

    business_metric_columns = [
        "到店标签_7天",
        "到店标签_14天",
        "到店标签_30天",
        "试驾标签_7天",
        "试驾标签_14天",
        "试驾标签_30天",
        FINAL_ORDERED_COLUMN,
    ]
    business_metric_frame = df[[col for col in business_metric_columns if col in df.columns]].copy()

    # 排除不需要的列（但保留目标列用于评估）
    excluded_columns = get_excluded_columns(target)
    cols_to_drop = [col for col in excluded_columns if col in df.columns and col != target]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"排除 {len(cols_to_drop)} 列: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")

    # 特征工程（与训练时相同的处理）
    logger.info("执行特征工程...")
    feature_engineer = FeatureEngineer(
        time_columns=config.feature.time_columns,
        numeric_columns=config.feature.numeric_features,
        interaction_context=interaction_context,
    )

    # 注意：AutoGluon 自动处理类别编码和缺失值
    # FeatureEngineer 只处理时间特征提取和数值类型转换
    df_processed, _ = feature_engineer.transform(df, interaction_context=interaction_context)

    # 保存目标值用于评估
    y_true = df_processed[target].values

    # 3. 预测
    logger.info("\n" + "=" * 60)
    logger.info("执行预测")
    logger.info("=" * 60)

    raw_pred = predictor.predict(df_processed)
    y_proba = predictor.predict_proba(df_processed)
    if label_mode == "hab":
        y_pred = apply_hab_decision_policy(y_proba, decision_policy)
    else:
        y_pred = raw_pred

    logger.info(f"预测完成: {len(y_pred)} 个样本")

    # 4. 评估
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        matthews_corrcoef,
    )

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"MCC: {mcc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n混淆矩阵:\n{cm}")

    report = classification_report(y_true, y_pred)
    logger.info(f"\n分类报告:\n{report}")
    confusion_df = confusion_matrix_frame(y_true, y_pred)
    classification_dict = classification_report_dict(y_true, y_pred)
    class_ranking_report = compute_class_ranking_report(y_true, y_proba, top_ratios=top_ratios)
    b_threshold_report = None
    if "B" in y_proba.columns:
        b_threshold_report = compute_threshold_report(y_true, y_proba["B"], positive_label="B")

    # 5. 按类别分析
    logger.info("\n" + "=" * 60)
    logger.info("各类别详细分析")
    logger.info("=" * 60)

    classes = df_processed[target].unique()
    for cls in sorted(classes):
        mask = y_true == cls
        correct = (y_pred[mask] == cls).sum()
        total = mask.sum()
        acc = correct / total if total > 0 else 0
        logger.info(f"类别 {cls}: {correct}/{total} 正确 ({acc:.1%})")

    # 6. 保存结果
    logger.info("\n" + "=" * 60)
    logger.info("保存结果")
    logger.info("=" * 60)

    # 预测结果
    results_df = pd.DataFrame({
        "真实标签": y_true,
        "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred,
    })

    if final_ordered is not None:
        final_ordered = final_ordered.loc[df_processed.index].fillna(0).astype(int)
        results_df["实际下定"] = final_ordered.values

    # 添加概率
    for col in y_proba.columns:
        results_df[f"概率_{col}"] = y_proba[col].values

    results_path = output_dir / "predictions.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info(f"预测结果已保存: {results_path}")

    bucket_summary_df = pd.DataFrame()
    monotonicity_result = {"passed": False, "metric": None, "message": "未启用 HAB 桶验证"}
    lead_actions_df = pd.DataFrame()
    if label_mode == "hab":
        bucket_input_df = pd.DataFrame({
            "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred,
            "真实标签": y_true,
        })
        for metric_column in [
            "到店标签_7天",
            "到店标签_14天",
            "到店标签_30天",
            "试驾标签_7天",
            "试驾标签_14天",
            "试驾标签_30天",
            FINAL_ORDERED_COLUMN,
        ]:
            if metric_column in business_metric_frame.columns:
                bucket_input_df[metric_column] = business_metric_frame.loc[df_processed.index, metric_column].values
        bucket_summary_df = compute_hab_bucket_summary(bucket_input_df, label_column="预测标签")
        monotonicity_result = check_hab_monotonicity(bucket_summary_df)
        if not bucket_summary_df.empty:
            bucket_summary_df.to_csv(output_dir / "hab_bucket_summary.csv", index=False, encoding="utf-8-sig")
            dump_json(output_dir / "hab_bucket_summary.json", bucket_summary_df.to_dict(orient="records"))
            dump_json(output_dir / "monotonicity_check.json", monotonicity_result)

        lead_action_rows = []
        for idx, row in df_processed.reset_index(drop=True).iterrows():
            probability_map = {label: float(y_proba.iloc[idx].get(label, 0.0)) for label in y_proba.columns}
            lead_action_rows.append(
                build_lead_action_record(
                    row=row,
                    predicted_label=str(y_pred.iloc[idx] if hasattr(y_pred, "iloc") else y_pred[idx]),
                    probability_map=probability_map,
                )
            )
        lead_actions_df = pd.DataFrame(lead_action_rows)
        if not lead_actions_df.empty:
            lead_actions_df.to_csv(output_dir / "lead_actions.csv", index=False, encoding="utf-8-sig")

    confusion_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    dump_json(output_dir / "classification_report.json", classification_dict)
    dump_json(output_dir / "class_ranking_report.json", class_ranking_report)
    if b_threshold_report is not None:
        b_threshold_report.to_csv(output_dir / "b_threshold_report.csv", index=False, encoding="utf-8-sig")
    dump_json(
        output_dir / "evaluation_summary.json",
        {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "mcc": mcc,
            "label_policy": label_policy,
            "label_mode": label_mode,
            "decision_policy": decision_policy,
            "split_info": split_info,
            "top_ratios": list(top_ratios),
            "monotonicity_check": monotonicity_result,
        },
    )

    # 评估报告
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("OHAB 模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"样本数量: {len(y_true)}\n")
        f.write(f"最佳模型: {predictor.model_best}\n\n")
        f.write("评估指标\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n\n")
        if label_mode == "hab":
            f.write(f"Label Mode: {label_mode}\n")
            f.write(f"Decision Policy: {json.dumps(decision_policy, ensure_ascii=False)}\n\n")
        f.write("混淆矩阵\n")
        f.write("-" * 40 + "\n")
        f.write(f"{cm}\n\n")
        f.write("分类报告\n")
        f.write("-" * 40 + "\n")
        f.write(report)
        if not bucket_summary_df.empty:
            f.write("\n\nHAB 桶业务表现\n")
            f.write("-" * 40 + "\n")
            for line in build_bucket_summary_text(bucket_summary_df.to_dict(orient="records")):
                f.write(f"- {line}\n")
            f.write(f"\n单调性检查: {monotonicity_result.get('message')}\n")

        # === 追加业务转化漏斗验证 (终态 O 验证) ===
        if final_ordered is not None:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("🎯 业务转化效果验证 (AI 评级 vs 最终下定)\n")
            f.write("=" * 60 + "\n")
            f.write("模型预测出的各级别线索中，最终实际下定 (O级) 的转化率对比：\n\n")
            
            print("\n" + "=" * 60)
            print("🎯 业务转化效果验证 (AI 评级 vs 最终下定)")
            print("=" * 60)
            
            # 使用预测标签和实际隔离标签做聚合
            final_df = pd.DataFrame({
                "predicted_level": y_pred,
                "is_ordered": final_ordered.values,
            })
            
            # 按 H, A, B 的顺序输出
            for level in ["H", "A", "B"]:
                subset = final_df[final_df["predicted_level"] == level]
                count_total = len(subset)
                if count_total > 0:
                    count_ordered = subset["is_ordered"].sum()
                    conversion_rate = count_ordered / count_total
                    
                    report_chunk = (
                        f"【AI 预测为 {level} 级 的线索】\n"
                        f"- 命中人数: {count_total} 人\n"
                        f"- 实际下定人数: {count_ordered} 人\n"
                        f"- 下定转化率: {conversion_rate:.2%}\n\n"
                    )
                    f.write(report_chunk)
                    print(report_chunk.strip())

    logger.info(f"评估报告已保存: {report_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

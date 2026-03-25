#!/usr/bin/env python3
"""
模型验证脚本

验证训练好的 OHAB 模型效果，包括：
1. 加载模型和新数据
2. 预测并评估
3. 生成详细报告
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import pickle  # noqa: S403 - 加载 AutoGluon 模型需要
import subprocess
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
    classification_report_text,
    compute_hab_bucket_summary,
    compute_class_ranking_report,
    compute_threshold_report,
    confusion_matrix_frame,
    check_hab_monotonicity,
)
from src.utils.helpers import (
    complete_process_if_running,
    format_training_duration,
    get_timestamp,
    get_local_now,
    format_timestamp,
    save_process_info,
    setup_logging,
    update_process_status,
)


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate_model_artifacts(metadata: dict, model_path: Path) -> None:
    """校验模型目录是否为统一训练入口产出的完整模型。"""
    artifact_status = metadata.get("artifact_status")
    if not isinstance(artifact_status, dict):
        raise RuntimeError(
            f"模型目录缺少 artifact_status: {model_path}。"
            "该模型可能来自旧训练流程或训练未完成，请重新运行 train_ohab.py。"
        )

    if artifact_status.get("training_complete") is not True:
        raise RuntimeError(
            f"模型目录尚未完成训练: {model_path}。"
            "请重新运行 train_ohab.py，等待核心产物全部写出后再执行 validate_model.py。"
        )

    comparison_expected = bool(artifact_status.get("comparison_expected"))
    comparison_config = metadata.get("model_comparison", {})
    comparison_models = (
        comparison_config.get("models", {})
        if isinstance(comparison_config, dict)
        else {}
    )
    if comparison_expected and not comparison_models:
        raise RuntimeError(
            f"模型目录缺少 baseline/best 对比元数据: {model_path}。"
            "当前模型不可用于正式汇报，请重新运行 train_ohab.py。"
        )


def dump_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

logger = logging.getLogger(__name__)
TASK_NAME = "validate_model"
FINAL_ORDERED_COLUMN = "is_final_ordered"
ROLE_DISPLAY = {
    "baseline": "基线模型",
    "best": "最优模型",
}


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_strict_hab_monotonic(bucket_summary_df: pd.DataFrame, metric_column: str) -> bool | None:
    if bucket_summary_df.empty or metric_column not in bucket_summary_df.columns:
        return None

    ordered_df = bucket_summary_df.set_index("bucket")
    if not {"H", "A", "B"}.issubset(set(ordered_df.index)):
        return None

    h_value = _safe_float(ordered_df.loc["H", metric_column])
    a_value = _safe_float(ordered_df.loc["A", metric_column])
    b_value = _safe_float(ordered_df.loc["B", metric_column])
    return h_value > a_value > b_value


def build_client_layering_summary(bucket_summary_df: pd.DataFrame) -> dict:
    arrive_monotonic = _is_strict_hab_monotonic(bucket_summary_df, "到店标签_14天_rate")
    drive_monotonic = _is_strict_hab_monotonic(bucket_summary_df, "试驾标签_14天_rate")

    if arrive_monotonic is True and drive_monotonic is True:
        client_message = "已形成清晰分层效果"
    elif arrive_monotonic is True or drive_monotonic is True:
        client_message = "已形成初步分层效果，H/A 边界仍需二期优化"
    else:
        client_message = "POC 已验证建模与 SOP 联动可行，分层边界仍需结合更多行为特征继续校准"

    return {
        "arrive_monotonic": arrive_monotonic,
        "drive_monotonic": drive_monotonic,
        "client_message": client_message,
    }


def compute_business_kpis(bucket_input_df: pd.DataFrame, bucket_summary_df: pd.DataFrame) -> dict:
    overall_arrive_rate = (
        _safe_float(pd.to_numeric(bucket_input_df["到店标签_14天"], errors="coerce").fillna(0).mean())
        if "到店标签_14天" in bucket_input_df.columns and len(bucket_input_df) > 0
        else 0.0
    )
    overall_drive_rate = (
        _safe_float(pd.to_numeric(bucket_input_df["试驾标签_14天"], errors="coerce").fillna(0).mean())
        if "试驾标签_14天" in bucket_input_df.columns and len(bucket_input_df) > 0
        else 0.0
    )

    ordered_df = (
        bucket_summary_df.set_index("bucket")
        if not bucket_summary_df.empty and "bucket" in bucket_summary_df.columns
        else pd.DataFrame()
    )

    def bucket_rate(bucket: str, metric: str) -> float:
        if ordered_df.empty or bucket not in ordered_df.index or metric not in ordered_df.columns:
            return 0.0
        return _safe_float(ordered_df.loc[bucket, metric])

    h_arrive_rate = bucket_rate("H", "到店标签_14天_rate")
    a_arrive_rate = bucket_rate("A", "到店标签_14天_rate")
    b_arrive_rate = bucket_rate("B", "到店标签_14天_rate")
    h_drive_rate = bucket_rate("H", "试驾标签_14天_rate")
    a_drive_rate = bucket_rate("A", "试驾标签_14天_rate")
    b_drive_rate = bucket_rate("B", "试驾标签_14天_rate")
    b_bucket_share = bucket_rate("B", "sample_ratio")

    ha_arrive_capture = 0.0
    if "到店标签_14天" in bucket_input_df.columns and "预测标签" in bucket_input_df.columns:
        arrive_positive = pd.to_numeric(bucket_input_df["到店标签_14天"], errors="coerce").fillna(0) > 0
        if int(arrive_positive.sum()) > 0:
            ha_arrive_capture = _safe_float(
                bucket_input_df.loc[arrive_positive, "预测标签"].isin(["H", "A"]).mean()
            )

    ha_drive_capture = 0.0
    if "试驾标签_14天" in bucket_input_df.columns and "预测标签" in bucket_input_df.columns:
        drive_positive = pd.to_numeric(bucket_input_df["试驾标签_14天"], errors="coerce").fillna(0) > 0
        if int(drive_positive.sum()) > 0:
            ha_drive_capture = _safe_float(
                bucket_input_df.loc[drive_positive, "预测标签"].isin(["H", "A"]).mean()
            )

    layering_summary = build_client_layering_summary(bucket_summary_df)
    return {
        "overall_arrive_14d_rate": overall_arrive_rate,
        "overall_drive_14d_rate": overall_drive_rate,
        "h_arrive_14d_rate": h_arrive_rate,
        "a_arrive_14d_rate": a_arrive_rate,
        "b_arrive_14d_rate": b_arrive_rate,
        "h_drive_14d_rate": h_drive_rate,
        "a_drive_14d_rate": a_drive_rate,
        "b_drive_14d_rate": b_drive_rate,
        "h_arrive_lift": (h_arrive_rate / overall_arrive_rate) if overall_arrive_rate > 0 else 0.0,
        "h_drive_lift": (h_drive_rate / overall_drive_rate) if overall_drive_rate > 0 else 0.0,
        "ha_arrive_capture": ha_arrive_capture,
        "ha_drive_capture": ha_drive_capture,
        "b_bucket_share": b_bucket_share,
        "client_layering_message": layering_summary["client_message"],
        "arrive_monotonic": layering_summary["arrive_monotonic"],
        "drive_monotonic": layering_summary["drive_monotonic"],
    }


def select_business_recommended_row(
    comparison_df: pd.DataFrame,
    macro_f1_tolerance: float = 0.01,
) -> tuple[pd.Series, dict]:
    if comparison_df.empty:
        raise ValueError("comparison_df 为空，无法选择业务推荐模型")

    ranked_df = comparison_df.copy()
    for metric in ["macro_f1", "balanced_accuracy", "b_recall", "h_arrive_14d_rate", "h_drive_14d_rate"]:
        if metric not in ranked_df.columns:
            ranked_df[metric] = 0.0
        ranked_df[metric] = pd.to_numeric(ranked_df[metric], errors="coerce").fillna(0.0)

    max_macro_f1 = _safe_float(ranked_df["macro_f1"].max())
    candidate_threshold = max_macro_f1 - macro_f1_tolerance
    candidate_df = ranked_df[ranked_df["macro_f1"] >= candidate_threshold].copy()
    if candidate_df.empty:
        candidate_df = ranked_df.copy()

    candidate_df = candidate_df.sort_values(
        ["balanced_accuracy", "b_recall", "h_arrive_14d_rate", "h_drive_14d_rate"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    selected_row = candidate_df.iloc[0]

    selection_reason = {
        "selected_model_name": selected_row["model_name"],
        "selected_role": selected_row["role"],
        "candidate_model_names": candidate_df["model_name"].tolist(),
        "criteria": {
            "macro_f1_tolerance": macro_f1_tolerance,
            "max_macro_f1": max_macro_f1,
            "candidate_threshold": candidate_threshold,
            "tie_break_order": [
                "balanced_accuracy",
                "b_recall",
                "h_arrive_14d_rate",
                "h_drive_14d_rate",
            ],
        },
    }
    return selected_row, selection_reason


def run_model_predictions(
    predictor,
    data: pd.DataFrame,
    label_mode: str,
    decision_policy: dict,
    model_name: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """执行指定模型预测并应用 HAB 决策层。"""
    raw_pred = predictor.predict(data, model=model_name) if model_name else predictor.predict(data)
    y_proba = predictor.predict_proba(data, model=model_name) if model_name else predictor.predict_proba(data)
    if label_mode == "hab":
        y_pred = apply_hab_decision_policy(y_proba, decision_policy)
    else:
        y_pred = raw_pred
    return y_pred, y_proba


def parse_args():
    parser = argparse.ArgumentParser(description="验证 OHAB 模型")

    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="后台运行模式",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径",
    )
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


def _strip_daemon_flags(argv: list[str]) -> list[str]:
    daemon_flags = {"--daemon", "-d"}
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in daemon_flags:
            continue
        stripped.append(arg)
    return stripped


def run_background(args: argparse.Namespace) -> int:
    """后台启动验证脚本。"""
    log_dir = Path("./outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else log_dir / f"{TASK_NAME}_{get_timestamp()}.log"

    cmd = [sys.executable, __file__, *_strip_daemon_flags(sys.argv[1:])]
    if args.log_file is None:
        cmd.extend(["--log-file", str(log_file)])
    cmd_str = " ".join(cmd)

    print(f"启动后台任务: {TASK_NAME}")
    print(f"日志文件: {log_file}")
    print(f"命令: {cmd_str}")

    start_time = get_local_now()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"启动时间: {format_timestamp(start_time)}\n")
        f.write(f"命令: {cmd_str}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "LEAD_SCORING_DISABLE_CONSOLE_LOG": "1"},
        )

    print(f"进程 ID: {process.pid}")
    print("\n查看状态: uv run python scripts/monitor.py status")
    print(f"查看日志: uv run python scripts/monitor.py log {TASK_NAME}")
    print(f"持续跟踪: tail -f {log_file}")

    return process.pid


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
    if args.daemon:
        pid = run_background(args)
        print(f"\n✅ 后台任务已启动 (PID: {pid})")
        return

    data_path = args.data_path or config.data.data_path

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_dir = Path("./outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{TASK_NAME}_{get_timestamp()}.log"

    setup_logging(log_file=str(log_file), level=logging.INFO)

    save_process_info(
        task_name=TASK_NAME,
        pid=os.getpid(),
        command=" ".join(sys.argv),
        log_file=str(log_file),
        data_path=data_path,
        target=args.target,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )
    atexit.register(complete_process_if_running, TASK_NAME, os.getpid())
    evaluation_start_time = get_local_now()

    logger.info("=" * 60)
    logger.info("OHAB 模型验证")
    logger.info("=" * 60)
    logger.info(f"评估开始时间: {evaluation_start_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据路径: {data_path}")

    try:
        model_path = Path(args.model_path)
        # 如果指定的模型路径不存在，尝试自动检测
        if not model_path.exists():
            detected_path = find_available_model(model_path)
            if detected_path.exists():
                logger.info(f"模型路径 {model_path} 不存在，使用检测到的模型: {detected_path}")
                model_path = detected_path
            else:
                logger.error("未找到可用模型，请先运行训练脚本")
                logger.error("  uv run python scripts/train_ohab.py")
                sys.exit(1)

        metadata = load_feature_metadata(model_path)
        validate_model_artifacts(metadata, model_path)
        supplemental_failures = metadata.get("artifact_status", {}).get("supplemental_failures", [])
        if supplemental_failures:
            logger.warning(
                "模型存在补充产物缺失: %s。"
                "不影响 baseline vs best 验证，但会影响特征贡献/Top 特征/Top-K 等扩展产物。",
                supplemental_failures,
            )

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

        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()

        # 过滤 Unknown
        target = predictor.label
        if target in df.columns:
            df = df[df[target] != "Unknown"].copy()
            logger.info(f"过滤 Unknown 后: {len(df)} 行")

        # 自动识别防泄漏切分标记 (Smart Test Set Filtering)
        split_info = metadata.get("split_info", {})
        label_policy = metadata.get("label_policy", {})
        label_mode = metadata.get("label_mode", "ohab")
        interaction_context = metadata.get("interaction_context", {})
        decision_policy = metadata.get("decision_policy", {"strategy": "argmax"})
        comparison_config = metadata.get("model_comparison", {"enabled": False})
    
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

        # 3. 预测与评估
        logger.info("\n" + "=" * 60)
        logger.info("执行预测与评估")
        logger.info("=" * 60)

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            matthews_corrcoef,
        )

        if comparison_config.get("enabled") and comparison_config.get("models"):
            model_specs = list(comparison_config["models"].items())
        else:
            model_specs = [
                (
                    predictor.model_best,
                    {
                        "role": "best",
                        "decision_policy": decision_policy,
                    },
                )
            ]

        comparison_rows = []
        per_model_outputs = {}

        for model_name, model_cfg in model_specs:
            role = model_cfg.get("role", model_name)
            current_policy = model_cfg.get("decision_policy", decision_policy)
            y_pred, y_proba = run_model_predictions(
                predictor=predictor,
                data=df_processed,
                label_mode=label_mode,
                decision_policy=current_policy,
                model_name=model_name,
            )

            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report_text(y_true, y_pred)
            confusion_df = confusion_matrix_frame(y_true, y_pred)
            classification_dict = classification_report_dict(y_true, y_pred)
            class_ranking_report = compute_class_ranking_report(y_true, y_proba, top_ratios=top_ratios)
            b_threshold_report = None
            if "B" in y_proba.columns:
                b_threshold_report = compute_threshold_report(y_true, y_proba["B"], positive_label="B")

            results_df = pd.DataFrame({
                "真实标签": y_true,
                "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred,
            })
            if final_ordered is not None:
                final_ordered = final_ordered.loc[df_processed.index].fillna(0).astype(int)
                results_df["实际下定"] = final_ordered.values
            for col in y_proba.columns:
                results_df[f"概率_{col}"] = y_proba[col].values

            bucket_summary_df = pd.DataFrame()
            monotonicity_result = {"passed": False, "metric": None, "message": "未启用 HAB 桶验证"}
            lead_actions_df = pd.DataFrame()
            business_kpis = {
                "overall_arrive_14d_rate": 0.0,
                "overall_drive_14d_rate": 0.0,
                "h_arrive_14d_rate": 0.0,
                "a_arrive_14d_rate": 0.0,
                "b_arrive_14d_rate": 0.0,
                "h_drive_14d_rate": 0.0,
                "a_drive_14d_rate": 0.0,
                "b_drive_14d_rate": 0.0,
                "h_arrive_lift": 0.0,
                "h_drive_lift": 0.0,
                "ha_arrive_capture": 0.0,
                "ha_drive_capture": 0.0,
                "b_bucket_share": 0.0,
                "client_layering_message": "POC 已验证建模与 SOP 联动可行，分层边界仍需结合更多行为特征继续校准",
                "arrive_monotonic": None,
                "drive_monotonic": None,
            }
            if label_mode == "hab":
                bucket_input_df = pd.DataFrame({
                    "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred,
                    "真实标签": y_true,
                })
                for metric_column in business_metric_columns:
                    if metric_column in business_metric_frame.columns:
                        bucket_input_df[metric_column] = business_metric_frame.loc[df_processed.index, metric_column].values
                bucket_summary_df = compute_hab_bucket_summary(bucket_input_df, label_column="预测标签")
                monotonicity_result = check_hab_monotonicity(bucket_summary_df)
                business_kpis = compute_business_kpis(bucket_input_df, bucket_summary_df)

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

            suffix = role
            results_df.to_csv(output_dir / f"predictions_{suffix}.csv", index=False, encoding="utf-8-sig")
            confusion_df.to_csv(output_dir / f"confusion_matrix_{suffix}.csv", encoding="utf-8-sig")
            dump_json(output_dir / f"classification_report_{suffix}.json", classification_dict)
            dump_json(output_dir / f"class_ranking_report_{suffix}.json", class_ranking_report)
            if b_threshold_report is not None:
                b_threshold_report.to_csv(output_dir / f"b_threshold_report_{suffix}.csv", index=False, encoding="utf-8-sig")
            if not bucket_summary_df.empty:
                bucket_summary_df.to_csv(output_dir / f"hab_bucket_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
                dump_json(output_dir / f"hab_bucket_summary_{suffix}.json", bucket_summary_df.to_dict(orient="records"))
                dump_json(output_dir / f"monotonicity_check_{suffix}.json", monotonicity_result)
            if not lead_actions_df.empty:
                lead_actions_df.to_csv(output_dir / f"lead_actions_{suffix}.csv", index=False, encoding="utf-8-sig")

            comparison_row = {
                "model_name": model_name,
                "role": role,
                "模型角色": ROLE_DISPLAY.get(role, role),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "mcc": float(mcc),
                "macro_f1": float(classification_dict.get("macro avg", {}).get("f1-score", 0.0)),
                "b_recall": float(classification_dict.get("B", {}).get("recall", 0.0)),
                "monotonicity_passed": bool(monotonicity_result.get("passed", False)),
                "monotonicity_metric": monotonicity_result.get("metric"),
                **business_kpis,
            }
            comparison_rows.append(comparison_row)
            per_model_outputs[role] = {
                "model_name": model_name,
                "decision_policy": current_policy,
                "results_df": results_df,
                "confusion_df": confusion_df,
                "classification_dict": classification_dict,
                "class_ranking_report": class_ranking_report,
                "bucket_summary_df": bucket_summary_df,
                "monotonicity_result": monotonicity_result,
                "lead_actions_df": lead_actions_df,
                "business_kpis": business_kpis,
                "report": report,
                "cm": cm,
            }

        comparison_df = pd.DataFrame(comparison_rows)
        if not comparison_df.empty:
            baseline_rows = comparison_df[comparison_df["role"] == "baseline"]
            if not baseline_rows.empty:
                baseline_row = baseline_rows.iloc[0]
                for metric in [
                    "accuracy",
                    "balanced_accuracy",
                    "mcc",
                    "macro_f1",
                    "b_recall",
                ]:
                    comparison_df[f"delta_vs_baseline_{metric}"] = comparison_df[metric] - float(baseline_row[metric])
            comparison_df.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
            dump_json(output_dir / "model_comparison.json", comparison_df.to_dict(orient="records"))

        primary_row, selection_reason = select_business_recommended_row(comparison_df)
        primary_role = str(primary_row["role"])
        primary_output = per_model_outputs[primary_role]
        technical_best_model = predictor.model_best
        business_recommended_model = primary_output["model_name"]

        primary_output["results_df"].to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        primary_output["confusion_df"].to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        dump_json(output_dir / "classification_report.json", primary_output["classification_dict"])
        dump_json(output_dir / "class_ranking_report.json", primary_output["class_ranking_report"])
        if not primary_output["bucket_summary_df"].empty:
            primary_output["bucket_summary_df"].to_csv(output_dir / "hab_bucket_summary.csv", index=False, encoding="utf-8-sig")
            dump_json(output_dir / "hab_bucket_summary.json", primary_output["bucket_summary_df"].to_dict(orient="records"))
            dump_json(output_dir / "monotonicity_check.json", primary_output["monotonicity_result"])
        if not primary_output["lead_actions_df"].empty:
            primary_output["lead_actions_df"].to_csv(output_dir / "lead_actions.csv", index=False, encoding="utf-8-sig")

        dump_json(
            output_dir / "evaluation_summary.json",
            {
                "accuracy": comparison_df.loc[comparison_df["role"] == primary_role, "accuracy"].iloc[0],
                "balanced_accuracy": comparison_df.loc[comparison_df["role"] == primary_role, "balanced_accuracy"].iloc[0],
                "mcc": comparison_df.loc[comparison_df["role"] == primary_role, "mcc"].iloc[0],
                "macro_f1": comparison_df.loc[comparison_df["role"] == primary_role, "macro_f1"].iloc[0],
                "b_recall": comparison_df.loc[comparison_df["role"] == primary_role, "b_recall"].iloc[0],
                "label_policy": label_policy,
                "label_mode": label_mode,
                "decision_policy": per_model_outputs[primary_role]["decision_policy"],
                "split_info": split_info,
                "top_ratios": list(top_ratios),
                "technical_best_model": technical_best_model,
                "business_recommended_model": business_recommended_model,
                "report_primary_role": primary_role,
                "selection_reason": selection_reason,
                "primary_metrics": {
                    "accuracy": comparison_df.loc[comparison_df["role"] == primary_role, "accuracy"].iloc[0],
                    "balanced_accuracy": comparison_df.loc[comparison_df["role"] == primary_role, "balanced_accuracy"].iloc[0],
                    "mcc": comparison_df.loc[comparison_df["role"] == primary_role, "mcc"].iloc[0],
                    "macro_f1": comparison_df.loc[comparison_df["role"] == primary_role, "macro_f1"].iloc[0],
                    "b_recall": comparison_df.loc[comparison_df["role"] == primary_role, "b_recall"].iloc[0],
                },
                "business_kpis": primary_output["business_kpis"],
                "monotonicity_check": primary_output["monotonicity_result"],
                "model_comparison": comparison_config,
                "comparison_rows": comparison_df.to_dict(orient="records"),
            },
        )

        report_path = output_dir / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("OHAB 模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"数据路径: {data_path}\n")
            f.write(f"样本数量: {len(y_true)}\n")
            f.write(f"技术最优模型: {technical_best_model}\n")
            f.write(f"业务推荐模型: {business_recommended_model}\n\n")
            if not comparison_df.empty and len(comparison_df) > 1:
                f.write("模型对比\n")
                f.write("-" * 40 + "\n")
                for row in comparison_df.to_dict(orient="records"):
                    f.write(
                        f"{ROLE_DISPLAY.get(row['role'], row['role'])}（{row['model_name']}）: "
                        f"平衡准确率 Balanced Accuracy={row['balanced_accuracy']:.4f}, "
                        f"宏平均 F1 Macro F1={row['macro_f1']:.4f}, "
                        f"B 类召回率 B Recall={row['b_recall']:.4f}\n"
                    )
                f.write("\n")

            f.write("主模型评估指标\n")
            f.write("-" * 40 + "\n")
            f.write(f"当前主输出角色: {ROLE_DISPLAY.get(primary_role, primary_role)}\n")
            f.write(f"准确率 Accuracy: {comparison_df.loc[comparison_df['role'] == primary_role, 'accuracy'].iloc[0]:.4f}\n")
            f.write(f"平衡准确率 Balanced Accuracy: {comparison_df.loc[comparison_df['role'] == primary_role, 'balanced_accuracy'].iloc[0]:.4f}\n")
            f.write(f"马修斯相关系数 MCC: {comparison_df.loc[comparison_df['role'] == primary_role, 'mcc'].iloc[0]:.4f}\n\n")
            if label_mode == "hab":
                f.write(f"标签模式: {label_mode}\n")
                f.write(f"决策策略: {json.dumps(primary_output['decision_policy'], ensure_ascii=False)}\n\n")
            f.write("混淆矩阵\n")
            f.write("-" * 40 + "\n")
            f.write(f"{primary_output['cm']}\n\n")
            f.write("分类报告\n")
            f.write("-" * 40 + "\n")
            f.write(primary_output["report"])
            if not primary_output["bucket_summary_df"].empty:
                f.write("\n\nHAB 桶业务表现\n")
                f.write("-" * 40 + "\n")
                for line in build_bucket_summary_text(primary_output["bucket_summary_df"].to_dict(orient="records")):
                    f.write(f"- {line}\n")
                f.write(f"\n单调性检查: {primary_output['monotonicity_result'].get('message')}\n")
                f.write(f"客户口径分层结论: {primary_output['business_kpis'].get('client_layering_message')}\n")

        logger.info(f"评估报告已保存: {report_path}")

        evaluation_end_time = get_local_now()
        duration_seconds = (evaluation_end_time - evaluation_start_time).total_seconds()
        duration_text = format_training_duration(duration_seconds)
        update_process_status(
            TASK_NAME,
            os.getpid(),
            "completed",
            duration_seconds=duration_seconds,
            duration_human=duration_text,
            output_dir=str(output_dir),
        )

        logger.info("=" * 60)
        logger.info(f"评估总耗时: {duration_text}")
        logger.info(f"评估结束时间: {evaluation_end_time.strftime('%Y-%m-%d %H:%M:%S%z')}")
        logger.info(f"评估完成! 结果已保存至: {output_dir}")

        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)
        primary_row = comparison_df[comparison_df["role"] == primary_role].iloc[0]
        print(f"准确率 Accuracy: {primary_row['accuracy']:.4f}")
        print(f"平衡准确率 Balanced Accuracy: {primary_row['balanced_accuracy']:.4f}")
        print(f"马修斯相关系数 MCC: {primary_row['mcc']:.4f}")
        print(f"评估总耗时: {duration_text}")
        print(f"\n结果保存在: {output_dir}")
    except Exception as e:
        update_process_status(TASK_NAME, os.getpid(), "failed", error=str(e))
        logger.exception("评估失败")
        raise


if __name__ == "__main__":
    main()

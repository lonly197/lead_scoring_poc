#!/usr/bin/env python3
"""
模型验证脚本

验证训练好的 OHAB 模型效果，包括：
1. 加载模型和新数据
2. 预测并评估
3. 生成详细报告
"""

import argparse
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
from src.data.loader import DataLoader, FeatureEngineer
from src.utils.helpers import setup_logging


def load_feature_metadata(model_path: Path) -> dict:
    """加载训练时保存的特征工程元数据"""
    metadata_path = model_path / "feature_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="验证 OHAB 模型")

    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/ohab_oot",
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
    """查找可用的模型目录，优先使用 OOT 版本"""
    if default_path.exists():
        return default_path

    # 搜索 outputs/models/ 下的模型目录
    models_dir = Path("outputs/models")
    if models_dir.exists():
        # 优先级：ohab_oot > ohab_model > 其他
        for name in ["ohab_oot", "ohab_model"]:
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
            logger.error(f"  uv run python scripts/train_ohab_oot.py")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 自动检测数据格式，新格式数据需要 auto_adapt=True 来计算目标变量
    from pathlib import Path as PathLib
    data_file = PathLib(data_path)
    suffix = data_file.suffix.lower()

    # 检查是否为新格式数据（无表头的 TSV）
    use_auto_adapt = False
    if suffix == ".tsv":
        use_auto_adapt = True
        logger.info("检测到 TSV 格式，启用 auto_adapt 模式")
    else:
        # 尝试检测是否有表头（CSV 第一行包含中文）
        try:
            import csv
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader)
                # 如果第一行没有中文，可能是无表头数据
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in str(first_row))
                if not has_chinese:
                    use_auto_adapt = True
                    logger.info("检测到无表头数据，启用 auto_adapt 模式")
        except Exception:
            pass

    loader = DataLoader(data_path, auto_adapt=use_auto_adapt)
    df = loader.load()

    # 过滤 Unknown
    target = predictor.label
    if target in df.columns:
        df = df[df[target] != "Unknown"].copy()
        logger.info(f"过滤 Unknown 后: {len(df)} 行")

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
    )

    # 注意：AutoGluon 自动处理类别编码和缺失值
    # FeatureEngineer 只处理时间特征提取和数值类型转换
    df_processed, _ = feature_engineer.process(df)

    # 保存目标值用于评估
    y_true = df_processed[target].values

    # 3. 预测
    logger.info("\n" + "=" * 60)
    logger.info("执行预测")
    logger.info("=" * 60)

    y_pred = predictor.predict(df_processed)
    y_proba = predictor.predict_proba(df_processed)

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

    # 添加概率
    for col in y_proba.columns:
        results_df[f"概率_{col}"] = y_proba[col].values

    results_path = output_dir / "predictions.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info(f"预测结果已保存: {results_path}")

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
        f.write("混淆矩阵\n")
        f.write("-" * 40 + "\n")
        f.write(f"{cm}\n\n")
        f.write("分类报告\n")
        f.write("-" * 40 + "\n")
        f.write(report)

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
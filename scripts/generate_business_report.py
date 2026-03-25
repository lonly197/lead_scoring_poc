"""
客户版业务报告生成脚本。

职责：
1. 读取训练与验证阶段输出的结构化结果。
2. 生成可直接用于 POC 汇报的 Markdown 报告。

本脚本不负责：
- 推断业务规则
- 计算模型指标
- 伪造概率或解释
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.business_logic import build_bucket_summary_text, summarize_top_dimensions
from src.utils.helpers import get_timestamp, setup_logging

logger = logging.getLogger(__name__)

ROLE_DISPLAY = {
    "baseline": "基线模型",
    "best": "最优模型",
}


def parse_args():
    parser = argparse.ArgumentParser(description="生成客户版业务解释报告")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models/ohab_model",
        help="模型目录",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="outputs/validation",
        help="验证结果目录",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/reports/hab_poc_report.md",
        help="输出报告路径",
    )
    return parser.parse_args()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def generate_report() -> None:
    args = parse_args()
    setup_logging(level=logging.INFO)

    model_dir = Path(args.model_dir)
    validation_dir = Path(args.validation_dir)
    output_path = Path(args.output_path)

    evaluation_summary = _load_json(validation_dir / "evaluation_summary.json", {})
    bucket_summary = _load_json(validation_dir / "hab_bucket_summary.json", [])
    monotonicity_check = _load_json(validation_dir / "monotonicity_check.json", {})
    model_comparison = _load_json(validation_dir / "model_comparison.json", [])
    dimension_contribution = _load_json(model_dir / "business_dimension_contribution.json", {})
    lead_actions_df = _load_dataframe(validation_dir / "lead_actions.csv")
    feature_importance_df = _load_dataframe(model_dir / "feature_importance.csv")

    top_dimensions = summarize_top_dimensions(dimension_contribution)
    top_features = feature_importance_df.head(5).to_dict(orient="records") if not feature_importance_df.empty else []
    bucket_lines = build_bucket_summary_text(bucket_summary)
    sample_actions = lead_actions_df.head(5).to_dict(orient="records") if not lead_actions_df.empty else []

    metrics = evaluation_summary.get("metrics", {})
    balanced_accuracy = metrics.get("balanced_accuracy")
    macro_avg = metrics.get("macro avg", {})
    macro_f1 = macro_avg.get("f1-score")

    report_lines = [
        "# HAB 线索评级 POC 业务报告",
        "",
        f"> 生成时间: {get_timestamp()}",
        "> 报告口径: H/A/B 智能评级 + SOP 下发",
        "",
        "## 1. 核心结论",
        "",
    ]

    if bucket_lines:
        report_lines.extend([f"- {line}" for line in bucket_lines])
    else:
        report_lines.append("- 当前未找到 HAB 桶业务摘要，请先运行 validate_model.py。")

    if balanced_accuracy is not None:
        report_lines.append(f"- 平衡准确率 Balanced Accuracy: {float(balanced_accuracy):.4f}")
    if macro_f1 is not None:
        report_lines.append(f"- 宏平均 F1 Macro F1: {float(macro_f1):.4f}")
    if monotonicity_check:
        report_lines.append(f"- 分层检查: {monotonicity_check.get('message', '无')}")

    if model_comparison:
        report_lines.extend([
            "",
            "## 2. 基线模型 vs 最优模型",
            "",
            "| 模型角色 | 模型名称 | 平衡准确率 Balanced Accuracy | 宏平均 F1 Macro F1 | B 类召回率 B Recall |",
            "|----------|----------|-------------------------------|--------------------|--------------------|",
        ])
        for row in model_comparison:
            report_lines.append(
                f"| {ROLE_DISPLAY.get(row.get('role', ''), row.get('role', ''))} | {row.get('model_name', '')} | "
                f"{float(row.get('balanced_accuracy', 0.0)):.4f} | "
                f"{float(row.get('macro_f1', 0.0)):.4f} | "
                f"{float(row.get('b_recall', 0.0)):.4f} |"
            )

    report_lines.extend([
        "",
        "## 3. 五大业务维度贡献",
        "",
        "| 维度 | 贡献值 |",
        "|------|--------|",
    ])
    for dimension, score in top_dimensions:
        report_lines.append(f"| {dimension} | {float(score):.4f} |")
    if not top_dimensions:
        report_lines.append("| 暂无 | 0.0000 |")

    report_lines.extend([
        "",
        "## 4. Top 特征信号",
        "",
        "| 特征 | 重要性 |",
        "|------|--------|",
    ])
    for row in top_features:
        report_lines.append(f"| {row.get('feature', '')} | {float(row.get('importance', 0.0)):.4f} |")
    if not top_features:
        report_lines.append("| 暂无 | 0.0000 |")

    report_lines.extend([
        "",
        "## 5. 一线下发示例",
        "",
        "| 线索ID | 预测HAB | 建议SOP | 原因1 |",
        "|--------|---------|---------|-------|",
    ])
    for row in sample_actions:
        report_lines.append(
            f"| {row.get('线索唯一ID', '')} | {row.get('预测HAB', '')} | "
            f"{row.get('建议SOP', '')} | {row.get('原因1', '')} |"
        )
    if not sample_actions:
        report_lines.append("| 暂无 | - | - | - |")

    report_lines.extend([
        "",
        "## 6. 使用说明",
        "",
        "- 销售前台只展示 H/A/B、建议 SOP 和原因摘要，不展示模型内部指标。",
        "- 运营和项目组通过附录中的模型指标和行为分层结果判断 POC 是否成立。",
        "- `O` 视为已成交/已锁单状态，不进入常规 HAB 分桶。",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"业务报告已生成: {output_path}")


if __name__ == "__main__":
    generate_report()

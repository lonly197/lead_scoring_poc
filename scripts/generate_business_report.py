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

CLIENT_ROLE_DISPLAY = {
    "baseline": "业务推荐模型",
    "best": "自动寻优候选模型",
}


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_pct(value: Any) -> str:
    return f"{_safe_float(value):.1%}"


def _format_lift(value: Any) -> str:
    return f"{_safe_float(value):.2f}x"


def _select_comparison_row(rows: list[dict], model_name: str | None, fallback_role: str | None) -> dict:
    if model_name:
        for row in rows:
            if row.get("model_name") == model_name:
                return row
    if fallback_role:
        for row in rows:
            if row.get("role") == fallback_role:
                return row
    return rows[0] if rows else {}


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
        default="outputs/validation/ohab_validation",
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

    primary_metrics = evaluation_summary.get("primary_metrics", {})
    business_kpis = evaluation_summary.get("business_kpis", {})
    business_model_name = evaluation_summary.get("business_recommended_model")
    technical_model_name = evaluation_summary.get("technical_best_model")
    report_primary_role = evaluation_summary.get("report_primary_role")
    business_row = _select_comparison_row(model_comparison, business_model_name, report_primary_role)
    technical_row = _select_comparison_row(model_comparison, technical_model_name, "best")
    balanced_accuracy = primary_metrics.get("balanced_accuracy", evaluation_summary.get("balanced_accuracy"))
    macro_f1 = primary_metrics.get("macro_f1", evaluation_summary.get("macro_f1"))
    client_layering_message = business_kpis.get(
        "client_layering_message",
        monotonicity_check.get("message", "POC 已验证建模与 SOP 联动可行，分层边界仍需结合更多行为特征继续校准"),
    )

    report_lines = [
        "# HAB 线索评级 POC 业务报告",
        "",
        f"> 生成时间: {get_timestamp()}",
        "> 报告口径: H/A/B 智能评级 + SOP 下发",
        "",
        "## 1. POC 结论",
        "",
    ]

    if business_model_name:
        report_lines.append(f"- 推荐采用 `{business_model_name}` 作为当前版本的业务推荐模型。")
    if client_layering_message:
        report_lines.append(f"- {client_layering_message}")
    if business_row:
        report_lines.append(
            f"- H 桶 14天到店率 {_format_pct(business_row.get('h_arrive_14d_rate'))}，"
            f"相对整体提升 {_format_lift(business_row.get('h_arrive_lift'))}；"
            f"14天试驾率 {_format_pct(business_row.get('h_drive_14d_rate'))}，"
            f"相对整体提升 {_format_lift(business_row.get('h_drive_lift'))}。"
        )
    if business_kpis:
        report_lines.append(
            f"- 若优先跟进 H+A，可覆盖 {_format_pct(business_kpis.get('ha_arrive_capture'))} 的 14天到店线索，"
            f"{_format_pct(business_kpis.get('ha_drive_capture'))} 的 14天试驾线索。"
        )
        report_lines.append(
            f"- B 桶占比 {_format_pct(business_kpis.get('b_bucket_share'))}，可作为低频培育池承接自动化运营。"
        )

    report_lines.extend([
        "",
        "## 2. 业务价值闭环",
        "",
        f"- 高意向优先触达：H 桶 14天到店 Lift {_format_lift(business_kpis.get('h_arrive_lift'))}，"
        f"14天试驾 Lift {_format_lift(business_kpis.get('h_drive_lift'))}。",
        f"- H+A 转化覆盖率：可覆盖 {_format_pct(business_kpis.get('ha_arrive_capture'))} 的 14天到店，"
        f"{_format_pct(business_kpis.get('ha_drive_capture'))} 的 14天试驾。",
        f"- B 桶降频空间：B 桶占比 {_format_pct(business_kpis.get('b_bucket_share'))}，"
        "适合进入低频培育与内容运营节奏。",
    ])

    report_lines.extend([
        "",
        "## 3. 推荐分层与 SOP",
        "",
    ])
    if bucket_lines:
        report_lines.extend([f"- {line}" for line in bucket_lines if not str(line).startswith("O 桶")])
    else:
        report_lines.append("- 当前未找到 HAB 桶业务摘要，请先运行 validate_model.py。")

    report_lines.extend([
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
        "## 4. 模型说明",
        "",
    ])
    if business_row:
        report_lines.append(
            f"- 业务推荐模型：`{business_row.get('model_name', '')}`，"
            f"平衡准确率 {_safe_float(business_row.get('balanced_accuracy')):.4f}，"
            f"宏平均 F1 {_safe_float(business_row.get('macro_f1')):.4f}，"
            f"B 类召回率 {_safe_float(business_row.get('b_recall')):.4f}。"
        )
    if technical_row:
        report_lines.append(
            f"- 自动寻优候选模型：`{technical_row.get('model_name', '')}`。"
            "该模型是 AutoML 在内部概率损失口径下的最优结果，用于技术参考。"
        )
    report_lines.append(
        "- 客户版主模型采用业务推荐模型，因为它更适合当前“高意向优先触达 + 分层运营”的落地目标。"
    )
    if balanced_accuracy is not None or macro_f1 is not None:
        report_lines.append(
            f"- 当前主模型关键指标：平衡准确率 {_safe_float(balanced_accuracy):.4f}，"
            f"宏平均 F1 {_safe_float(macro_f1):.4f}。"
        )

    report_lines.extend([
        "",
        "## 5. 落地建议",
        "",
        "- 建议以业务推荐模型作为 POC 演示口径，先推动单车型场景试点。",
        "- 销售前台只展示 H/A/B、建议 SOP 和原因摘要，不展示模型内部技术指标。",
        "- 二期优化重点放在 H/A 边界校准与更多行为特征补充，不影响当前立项与试点推进。",
        "",
        "### 关键业务维度",
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
        "### Top 特征信号",
        "",
        "| 特征 | 重要性 |",
        "|------|--------|",
    ])
    for row in top_features:
        report_lines.append(f"| {row.get('feature', '')} | {float(row.get('importance', 0.0)):.4f} |")
    if not top_features:
        report_lines.append("| 暂无 | 0.0000 |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"业务报告已生成: {output_path}")


if __name__ == "__main__":
    generate_report()

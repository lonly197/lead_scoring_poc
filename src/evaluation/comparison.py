from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.evaluation.business_logic import build_lead_action_record
from src.evaluation.ohab_metrics import (
    classification_report_dict,
    classification_report_text,
    compute_class_ranking_report,
    compute_hab_bucket_summary,
    compute_threshold_report,
    confusion_matrix_frame,
    check_hab_monotonicity,
)


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_client_layering_summary(bucket_summary_df: pd.DataFrame) -> dict:
    def is_monotonic(metric_column: str) -> bool | None:
        if bucket_summary_df.empty or metric_column not in bucket_summary_df.columns:
            return None
        ordered_df = bucket_summary_df.set_index("bucket")
        if not {"H", "A", "B"}.issubset(set(ordered_df.index)):
            return None
        h_value = _safe_float(ordered_df.loc["H", metric_column])
        a_value = _safe_float(ordered_df.loc["A", metric_column])
        b_value = _safe_float(ordered_df.loc["B", metric_column])
        return h_value > a_value > b_value

    arrive_monotonic = is_monotonic("到店标签_14天_rate")
    drive_monotonic = is_monotonic("试驾标签_14天_rate")

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
            ha_arrive_capture = _safe_float(bucket_input_df.loc[arrive_positive, "预测标签"].isin(["H", "A"]).mean())

    ha_drive_capture = 0.0
    if "试驾标签_14天" in bucket_input_df.columns and "预测标签" in bucket_input_df.columns:
        drive_positive = pd.to_numeric(bucket_input_df["试驾标签_14天"], errors="coerce").fillna(0) > 0
        if int(drive_positive.sum()) > 0:
            ha_drive_capture = _safe_float(bucket_input_df.loc[drive_positive, "预测标签"].isin(["H", "A"]).mean())

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


def build_comparator_bundle(
    *,
    comparator_name: str,
    role: str,
    y_true,
    y_pred,
    y_proba: pd.DataFrame,
    df_processed: pd.DataFrame,
    business_metric_frame: pd.DataFrame,
    business_metric_columns: Iterable[str],
    top_ratios: Iterable[float],
    label_mode: str,
    final_ordered: pd.Series | None = None,
    decision_policy: dict | None = None,
) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    report = classification_report_text(y_true, y_pred)
    confusion_df = confusion_matrix_frame(y_true, y_pred)
    classification_dict = classification_report_dict(y_true, y_pred)
    class_ranking_report = compute_class_ranking_report(y_true, y_proba, top_ratios=top_ratios)
    b_threshold_report = compute_threshold_report(y_true, y_proba["B"], positive_label="B") if "B" in y_proba.columns else None

    results_df = pd.DataFrame({"真实标签": y_true, "预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred})
    if final_ordered is not None:
        ordered_series = final_ordered.loc[df_processed.index].fillna(0).astype(int)
        results_df["实际下定"] = ordered_series.values
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
        bucket_input_df = pd.DataFrame({"预测标签": y_pred.values if hasattr(y_pred, "values") else y_pred, "真实标签": y_true})
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

    comparison_row = {
        "model_name": comparator_name,
        "role": role,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "macro_f1": float(classification_dict.get("macro avg", {}).get("f1-score", 0.0)),
        "b_recall": float(classification_dict.get("B", {}).get("recall", 0.0)),
        "monotonicity_passed": bool(monotonicity_result.get("passed", False)),
        "monotonicity_metric": monotonicity_result.get("metric"),
        **business_kpis,
    }

    return {
        "model_name": comparator_name,
        "role": role,
        "comparison_row": comparison_row,
        "decision_policy": decision_policy or {"strategy": "argmax"},
        "results_df": results_df,
        "confusion_df": confusion_df,
        "classification_dict": classification_dict,
        "class_ranking_report": class_ranking_report,
        "b_threshold_report": b_threshold_report,
        "bucket_summary_df": bucket_summary_df,
        "monotonicity_result": monotonicity_result,
        "lead_actions_df": lead_actions_df,
        "business_kpis": business_kpis,
        "report": report,
        "cm": confusion_matrix(y_true, y_pred),
    }

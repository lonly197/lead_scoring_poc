"""OHAB 多分类评估与排序分析工具。"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from src.data.label_policy import ordered_hab_labels, ordered_ohab_labels


def classification_report_text(y_true: Sequence[str], y_pred: Sequence[str]) -> str:
    labels = ordered_ohab_labels(list(y_true) + list(y_pred))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )


def classification_report_dict(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    labels = ordered_ohab_labels(list(y_true) + list(y_pred))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_frame(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    labels = ordered_ohab_labels(list(y_true) + list(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_class_ranking_report(
    y_true: Sequence[str],
    proba_df: pd.DataFrame,
    top_ratios: Iterable[float] = (0.05, 0.10, 0.20),
) -> list[Dict[str, float]]:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    total_samples = len(y_true_series)
    report: list[Dict[str, float]] = []

    if total_samples == 0:
        return report

    for label in ordered_ohab_labels(proba_df.columns):
        if label not in proba_df.columns:
            continue

        baseline_rate = float((y_true_series == label).mean())
        label_scores = proba_df[label].reset_index(drop=True)

        for ratio in top_ratios:
            top_n = max(1, int(total_samples * ratio))
            top_indices = label_scores.sort_values(ascending=False).head(top_n).index
            hit_rate = float((y_true_series.loc[top_indices] == label).mean()) if top_n else 0.0
            lift = hit_rate / baseline_rate if baseline_rate > 0 else 0.0
            report.append(
                {
                    "class": label,
                    "top_ratio": float(ratio),
                    "top_n": int(top_n),
                    "baseline_rate": baseline_rate,
                    "hit_rate": hit_rate,
                    "lift": lift,
                }
            )

    return report


def compute_threshold_report(
    y_true: Sequence[str],
    positive_scores: Sequence[float],
    positive_label: str = "B",
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    score_series = pd.Series(positive_scores).reset_index(drop=True)
    thresholds = list(thresholds or [round(step, 2) for step in [0.1, 0.2, 0.3, 0.4, 0.5]])

    rows = []
    truth = (y_true_series == positive_label).astype(int)

    for threshold in thresholds:
        prediction = (score_series >= threshold).astype(int)
        predicted_positive = int(prediction.sum())
        true_positive = int(((prediction == 1) & (truth == 1)).sum())
        actual_positive = int(truth.sum())
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = f1_score(truth, prediction, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive": predicted_positive,
                "actual_positive": actual_positive,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": float(f1),
            }
        )

    return pd.DataFrame(rows)


def apply_hab_decision_policy(
    proba_df: pd.DataFrame,
    decision_policy: Dict[str, object] | None = None,
) -> pd.Series:
    """
    将多分类概率映射为 HAB 业务标签。

    规则：
    1. P(H) >= h_threshold -> H
    2. 否则 P(B) >= b_threshold -> B
    3. 否则 -> A
    """
    if proba_df.empty:
        return pd.Series(dtype="object")

    policy = decision_policy or {}
    if policy.get("strategy") != "hab_threshold":
        return proba_df.idxmax(axis=1)

    h_threshold = float(policy.get("h_threshold", 0.50))
    b_threshold = float(policy.get("b_threshold", 0.30))

    predictions = []
    for _, row in proba_df.iterrows():
        h_score = float(row.get("H", 0.0))
        b_score = float(row.get("B", 0.0))
        if h_score >= h_threshold:
            predictions.append("H")
        elif b_score >= b_threshold:
            predictions.append("B")
        elif "A" in proba_df.columns:
            predictions.append("A")
        else:
            # 兜底逻辑：若无 A 则取概率最大的类
            predictions.append(row.idxmax())

    return pd.Series(predictions, index=proba_df.index)


def optimize_hab_decision_policy(
    y_true: Sequence[str],
    proba_df: pd.DataFrame,
    h_thresholds: Iterable[float] | None = None,
    b_thresholds: Iterable[float] | None = None,
    min_predicted_b_rate: float = 0.03,
) -> Dict[str, object]:
    """在验证集上搜索 HAB 阈值策略。"""
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    proba_df = proba_df.reset_index(drop=True)

    if y_true_series.empty or not {"H", "A", "B"}.issubset(set(proba_df.columns)):
        return {"strategy": "argmax"}

    h_candidates = list(h_thresholds or [round(x, 2) for x in [0.45, 0.50, 0.55, 0.60, 0.65]])
    b_candidates = list(b_thresholds or [round(x, 2) for x in [0.20, 0.25, 0.30, 0.35, 0.40]])

    best_policy: Dict[str, object] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for h_threshold in h_candidates:
        for b_threshold in b_candidates:
            policy = {
                "strategy": "hab_threshold",
                "h_threshold": float(h_threshold),
                "b_threshold": float(b_threshold),
                "fallback_label": "A",
            }
            y_pred = apply_hab_decision_policy(proba_df, policy)
            predicted_b_rate = float((y_pred == "B").mean())
            balanced_acc = float(balanced_accuracy_score(y_true_series, y_pred))
            b_recall = float(recall_score(y_true_series == "B", y_pred == "B", zero_division=0))
            macro_f1 = float(f1_score(y_true_series, y_pred, average="macro", zero_division=0))
            score = (
                float(predicted_b_rate >= min_predicted_b_rate),
                balanced_acc,
                b_recall,
                macro_f1,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_policy = {
                    **policy,
                    "predicted_b_rate": predicted_b_rate,
                    "balanced_accuracy": balanced_acc,
                    "b_recall": b_recall,
                    "macro_f1": macro_f1,
                    "min_predicted_b_rate": float(min_predicted_b_rate),
                }

    return best_policy or {"strategy": "argmax"}


def compute_hab_bucket_summary(
    prediction_df: pd.DataFrame,
    label_column: str = "预测标签",
) -> pd.DataFrame:
    """统计 HAB 桶的业务表现。"""
    bucket_order = ordered_hab_labels(prediction_df[label_column].dropna().unique())
    rows: list[Dict[str, float | int | str]] = []
    total = len(prediction_df)

    for label in bucket_order:
        subset = prediction_df[prediction_df[label_column] == label].copy()
        if subset.empty:
            continue
        row: Dict[str, float | int | str] = {
            "bucket": label,
            "sample_count": int(len(subset)),
            "sample_ratio": float(len(subset) / total) if total else 0.0,
        }
        for metric_column in [
            "到店标签_7天",
            "到店标签_14天",
            "到店标签_30天",
            "试驾标签_7天",
            "试驾标签_14天",
            "试驾标签_30天",
            "is_final_ordered",
        ]:
            if metric_column in subset.columns:
                row[f"{metric_column}_rate"] = float(pd.to_numeric(subset[metric_column], errors="coerce").fillna(0).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def check_hab_monotonicity(
    bucket_summary_df: pd.DataFrame,
    metric_candidates: Iterable[str] = ("试驾标签_14天_rate", "到店标签_14天_rate"),
) -> Dict[str, object]:
    """检查 HAB 三桶是否形成 H > A > B 的单调分层。"""
    if bucket_summary_df.empty:
        return {"passed": False, "metric": None, "message": "桶摘要为空"}

    ordered_df = bucket_summary_df.set_index("bucket")
    for metric in metric_candidates:
        if metric not in ordered_df.columns:
            continue
        if not {"H", "A", "B"}.issubset(set(ordered_df.index)):
            continue
        h_value = float(ordered_df.loc["H", metric])
        a_value = float(ordered_df.loc["A", metric])
        b_value = float(ordered_df.loc["B", metric])
        passed = h_value > a_value > b_value
        return {
            "passed": passed,
            "metric": metric,
            "values": {"H": h_value, "A": a_value, "B": b_value},
            "message": "H/A/B 分层单调" if passed else "H/A/B 分层未形成单调",
        }

    return {"passed": False, "metric": None, "message": "缺少可用的行为验证指标"}
